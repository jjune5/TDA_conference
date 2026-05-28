"""DNP driver: run a per-node-PH variant through the reused diffusion_features LP
pipeline. Reports the §14 diagnostic (collapse gate) and 50-trial LP AUC vs
no-PI / exact-PI baselines. Variant 'A'|'C'|'B'."""
from __future__ import annotations
import os, sys, json, csv, time, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import diffusion_features as DF          # reuse the whole §15 pipeline
import node_ph_features as NPH


def load_dataset_ext(name):
    """Load any of the LP datasets. Cora/Chameleon via the §15 loader (keeps the exact
    features/transform the Phase-1 gate used); the rest via the main loaddatas loader
    (Citeseer/PubMed/Computers/Photo/Squirrel/Texas/Cornell/Wisconsin/ChChMiner)."""
    if name in ('Cora', 'Chameleon'):
        return DF.load_dataset(name)
    import loaddatas as lds
    return lds.loaddatas(name)


def compute_variant(data, variant, K, hop, max_nodes, verbose):
    if variant == 'A':
        return NPH.phi_A(data, K=K, hop=hop, max_nodes=max_nodes, verbose=verbose)
    if variant == 'C':
        return NPH.phi_C(data, hop=hop, max_nodes=max_nodes, verbose=verbose)
    if variant == 'B':
        return NPH.phi_B(data, K=K, hop=hop, max_nodes=max_nodes, verbose=verbose)
    raise ValueError(variant)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=['Cora', 'Chameleon'])
    ap.add_argument('--variant', choices=['A', 'C', 'B'], default='A')
    ap.add_argument('--trials', type=int, default=50)
    ap.add_argument('--K', type=int, default=5)
    ap.add_argument('--hop', type=int, default=2)
    ap.add_argument('--max_nodes', type=int, default=300)
    ap.add_argument('--outdir', default=None)
    args = ap.parse_args()
    outdir = args.outdir or f'results/dnp_{args.variant}'
    os.makedirs(outdir, exist_ok=True)
    print(f'DNP variant={args.variant}  datasets={args.datasets}  '
          f'trials={args.trials}  K={args.K} hop={args.hop} max_nodes={args.max_nodes}')

    lp_raw, diag_out = {}, {}
    for name in args.datasets:
        print(f'\n{"="*64}\n{name}\n{"="*64}')
        raw = load_dataset_ext(name)[0]
        in_feats = raw.x.size(1)
        data, splits = DF.prepare_data(raw)
        bounds, total = DF.segment_bounds(data)
        print('  segments: ' + ', '.join(f'{k}={v[1]-v[0]}' for k, v in bounds.items()))

        t0 = time.time()
        print(f'  computing per-node PH (variant {args.variant})...')
        phi = compute_variant(data, args.variant, args.K, args.hop, args.max_nodes, True)
        print(f'    Phi shape={phi.shape}  ({time.time()-t0:.0f}s)')
        feat = DF.build_edge_features(phi, data.total_edges)       # 3*D per edge
        fdim = feat.shape[1]

        # exact-PI cache is optional (graceful skip if absent)
        try:
            exact_pi = DF.load_pi_cache(name, total)
        except (FileNotFoundError, ValueError) as e:
            exact_pi = None
            print(f'  [exact-PI cache unavailable: {e}] -> skipping exact-PI baseline')

        # ---- §14 diagnostic = collapse gate (reuse feature_separability_auc) ----
        (tr, trf, va, vaf, te, tef) = splits
        y_te = np.concatenate([np.ones(len(te)), np.zeros(len(tef))])
        X_te = DF.build_edge_features(phi, np.concatenate([te, tef]))
        test_auc = float(np.mean([DF.feature_separability_auc(X_te, y_te, seed=s)
                                  for s in range(5)]))
        pi_test_auc = float('nan')
        if exact_pi is not None:
            lo_p, hi_p = bounds['test_pos']; lo_n, hi_n = bounds['test_neg']
            Xpi_te = np.concatenate([exact_pi[lo_p:hi_p], exact_pi[lo_n:hi_n]])
            pi_test_auc = float(np.mean([DF.feature_separability_auc(Xpi_te, y_te, seed=s)
                                         for s in range(5)]))
        gate = 'PASS' if test_auc > 0.55 else 'COLLAPSE'
        diag_out[name] = {'dnp_test_auc': test_auc,
                          'exactPI_test_auc': pi_test_auc, 'gate': gate}
        print(f'  COLLAPSE GATE: DNP test-AUC={test_auc:.4f}  '
              f'(exact-PI test-AUC={pi_test_auc:.4f})  -> {gate}')

        # ---- LP: no-PI / exact-PI / DNP ----
        pi_dim = exact_pi.shape[1] if exact_pi is not None else fdim
        modes = [('none', None, pi_dim, 'no-PI')]
        if exact_pi is not None:
            modes.append(('pi', exact_pi, pi_dim, 'exact-PI'))
        modes.append(('hks', feat, fdim, f'DNP_{args.variant}'))
        for mode, arr, d, tag in modes:
            aucs = [DF.run_one_trial(data, arr, d, mode, in_feats, seed=s, dev=DF.device)
                    for s in range(args.trials)]
            lp_raw[(name, tag)] = aucs
            print(f'  LP [{tag}] mean={np.mean(aucs):.4f} std={np.std(aucs):.4f}')

    with open(os.path.join(outdir, 'lp_auc.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['dataset', 'variant', 'mean_auc', 'std_auc', 'n'])
        for (name, tag), a in lp_raw.items():
            a = np.array(a)
            w.writerow([name, tag, f'{a.mean():.6f}', f'{a.std():.6f}', len(a)])
    with open(os.path.join(outdir, 'collapse_gate.json'), 'w') as f:
        json.dump(diag_out, f, indent=2)
    print(f'\nOutputs -> {outdir}/')


if __name__ == '__main__':
    main()
