"""Stage-2: does chromatic (kernel) persistence beat controls on classification?

First the synthetic positive-control (chromatic MUST win by construction), then real
hetero datasets (Task 11). Controls: none / chromatic / achromatic / shuffled / random.

run_synthetic uses EXACT chromatic labels (not the neural engine) to first prove the
SIGNAL exists and is separable; the neural-engine path is exercised on real data."""
from __future__ import annotations
import argparse
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hetero.synth_chromatic import make_synth_chromatic
from hetero.chromatic_labels import compute_chromatic_0dim
from node_ph_features import PI_RES
from sg2dgm import PersistenceImager as pimg_mod

_LEGS = ('image', 'kernel', 'cokernel')


def _leg_pi(lab, leg, imager):
    pts = lab[leg]
    return (np.asarray(imager.transform(pts)).reshape(-1) if pts.shape[0]
            else np.zeros(PI_RES * PI_RES))


def run_synthetic(n_per_class=200, seed=0):
    """Logistic-regression AUC of chromatic (3-leg) vs achromatic (ordinary) vs
    kernel-only on the synthetic positive control. Expect chromatic/kernel ~1.0,
    achromatic ~0.5 (same-type topology identical across classes)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    imager = pimg_mod.PersistenceImager(resolution=PI_RES)
    X = {'chromatic': [], 'kernel_only': [], 'achromatic': []}
    yv = []
    rng = np.random.RandomState(seed)
    for _ in range(n_per_class):
        for lab in (0, 1):
            gd = make_synth_chromatic(label=lab, seed=int(rng.randint(1 << 30)))
            # fv ~ U[0,1); global cap max_f=1.0 so essential classes born at the data
            # max are still visible (per-ego max would drop top-born kernels).
            L = compute_chromatic_0dim(gd['fv'], gd['ei'], gd['is_L'], max_f=1.0)
            X['chromatic'].append(np.concatenate([_leg_pi(L, lg, imager) for lg in _LEGS]))
            X['kernel_only'].append(_leg_pi(L, 'kernel', imager))
            X['achromatic'].append(_leg_pi(L, 'ordinary', imager))
            yv.append(lab)
    yv = np.array(yv)
    print(f"=== synthetic positive control (n={2 * n_per_class}) ===")
    aucs = {}
    for name, Xl in X.items():
        Xl = np.array(Xl)
        if np.allclose(Xl.std(0).sum(), 0):
            aucs[name] = 0.5
            print(f"  {name:12s} AUC = 0.500  (degenerate/constant feature)")
            continue
        sc = cross_val_score(LogisticRegression(max_iter=2000), Xl, yv, cv=5, scoring='roc_auc')
        aucs[name] = float(sc.mean())
        print(f"  {name:12s} AUC = {sc.mean():.3f} +- {sc.std():.3f}")
    ok = aucs['kernel_only'] > aucs['achromatic'] + 0.15
    print(f"GATE chromatic/kernel >> achromatic: {'PASS' if ok else 'FAIL'} "
          f"(kernel {aucs['kernel_only']:.3f} vs achromatic {aucs['achromatic']:.3f})")
    return aucs


def _znorm(x):
    x = np.asarray(x, dtype=np.float64)
    return ((x - x.mean(0)) / (x.std(0) + 1e-8)).astype(np.float32)


def run_real(dataset='ACM', backbones=('HAN', 'HGT'), K=2, hop=2, max_nodes=60,
             samples_n=400, epochs=30, trials=10, seed=0, outdir=None):
    """Stage-2 real-data NC: controls none/chromatic/achromatic/shuffled/random across
    backbones. chromatic = neural 3-leg PI; achromatic = type-blind unified ordinary EPD
    (run_han_hgt.compute_epd 'unified'); shuffled = chromatic rows permuted;
    random = chromatic PI from a random node filtration. Verdict: chromatic must beat all
    four, consistently across backbones (Idea-2 fluke lesson)."""
    import os, csv, torch
    from hetero.metapath_graph import load_hgb, TARGET, METAPATHS, build_metapath_graph
    from hetero.unified_filter import build_homo
    from hetero.pdgnn_metapath import _graph_hks
    from hetero.chromatic_pdgnn import (gen_chromatic_samples, train_chromatic_pdgnn,
                                        predict_node_chromatic_pi)
    from hetero.run_han_hgt import (HANNet, HGTNet, run_once, _mp_edge_index, compute_epd,
                                    RAND_DIM, device)
    outdir = outdir or f'results/chromatic_{dataset}'
    os.makedirs(outdir, exist_ok=True)
    d = load_hgb(dataset); tgt = TARGET[dataset]; mps = list(METAPATHS[dataset].keys())
    g, ntype, n_t, y_u, masks_u = build_homo(d, dataset)
    rng0 = np.random.RandomState(seed)

    y = d[tgt].y.numpy(); multilabel = (y.ndim > 1)
    n_cls = y.shape[1] if multilabel else int(y.max()) + 1
    masks = {k: getattr(d[tgt], f'{k}_mask').numpy()
             for k in ('train', 'val', 'test') if hasattr(d[tgt], f'{k}_mask')}
    if 'val' not in masks:
        tr = masks['train'].copy(); idx = np.where(tr)[0]; cut = idx[int(0.85 * len(idx)):]
        masks['val'] = np.zeros_like(tr); masks['val'][cut] = True; masks['train'][cut] = False
    n_tgt = int(d[tgt].num_nodes)
    feats = {}
    for t in d.node_types:
        x = getattr(d[t], 'x', None)
        feats[t] = (x.numpy().astype(np.float32) if x is not None
                    else rng0.randn(int(d[t].num_nodes), RAND_DIM).astype(np.float32))

    # --- chromatic feature (neural 3-leg PI on the unified graph) ---
    hks = _graph_hks(g, K)
    chrom_model = train_chromatic_pdgnn(
        gen_chromatic_samples(g, hks, ntype, 0, hop, max_nodes, samples_n, seed=seed),
        hidden=32, layers=3, epochs=epochs)
    CHROM = _znorm(predict_node_chromatic_pi(chrom_model, g, hks, ntype, 0, hop, max_nodes)[:n_tgt])
    # random-filter control: chromatic PI of a random node filtration
    rfilt = rng0.randn(g.number_of_nodes(), K).astype(np.float32)
    rmodel = train_chromatic_pdgnn(
        gen_chromatic_samples(g, rfilt, ntype, 0, hop, max_nodes, samples_n, seed=seed + 1),
        hidden=32, layers=3, epochs=epochs)
    RANDF = _znorm(predict_node_chromatic_pi(rmodel, g, rfilt, ntype, 0, hop, max_nodes)[:n_tgt])
    SHUF = CHROM[rng0.permutation(n_tgt)]
    # achromatic: type-blind unified ordinary EPD (the Idea-2 feature)
    ACHRO, _ = compute_epd(d, dataset, 'unified', mps[0], K, hop, max_nodes, samples_n, epochs, n_tgt)
    ACHRO = _znorm(ACHRO)
    print(f'[{dataset}] CHROM{CHROM.shape} ACHRO{ACHRO.shape} '
          f'distinct(chrom)={len(np.unique(np.round(CHROM,3),axis=0))}/{n_tgt}')

    feat_by_variant = {'none': None, 'chromatic': CHROM, 'achromatic': ACHRO,
                       'shuffled': SHUF, 'random': RANDF}

    def target_feat(variant):
        base = feats[tgt]
        ex = feat_by_variant[variant]
        return base if ex is None else np.concatenate([base, ex], axis=1)

    ei_full = {et: d[et].edge_index.to(device) for et in d.edge_types}
    ei_han = {('T', mp, 'T'): _mp_edge_index(build_metapath_graph(d, mp)[0], n_tgt) for mp in mps}
    meta_han = (['T'], list(ei_han.keys())); meta_full = d.metadata()

    variants = ['none', 'chromatic', 'achromatic', 'shuffled', 'random']
    results = {}
    for bb in backbones:
        for variant in variants:
            xt = target_feat(variant); accs = []
            for s in range(trials):
                if bb == 'HAN':
                    x_dict = {'T': torch.tensor(xt, dtype=torch.float32, device=device)}
                    model = HANNet(xt.shape[1], n_cls, meta_han).to(device)
                    accs.append(run_once(model, (x_dict,), ei_han, y, masks, multilabel, s))
                else:
                    x_dict = {t: torch.tensor(feats[t] if t != tgt else xt,
                                              dtype=torch.float32, device=device)
                              for t in d.node_types}
                    in_dims = {t: x_dict[t].shape[1] for t in d.node_types}
                    model = HGTNet(in_dims, n_cls, meta_full, tgt).to(device)
                    accs.append(run_once(model, (x_dict,), ei_full, y, masks, multilabel, s))
            results[(bb, variant)] = accs
            print(f'  [{bb} {variant:10}] {"MacroF1" if multilabel else "acc"} = '
                  f'{np.mean(accs):.4f} ± {np.std(accs):.4f}')

    with open(os.path.join(outdir, 'nc_acc.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['dataset', 'backbone', 'variant', 'mean', 'std', 'n'])
        for (bb, v), accs in results.items():
            a = np.array(accs); w.writerow([dataset, bb, v, f'{a.mean():.6f}', f'{a.std():.6f}', len(a)])
    print(f'Outputs -> {outdir}/nc_acc.csv')
    return results


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['synthetic', 'real'], default='synthetic')
    ap.add_argument('--n_per_class', type=int, default=200)
    ap.add_argument('--dataset', default='ACM')
    ap.add_argument('--backbones', nargs='+', default=['HAN', 'HGT'])
    ap.add_argument('--K', type=int, default=2)
    ap.add_argument('--hop', type=int, default=2)
    ap.add_argument('--max_nodes', type=int, default=60)
    ap.add_argument('--samples_n', type=int, default=400)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--trials', type=int, default=10)
    a = ap.parse_args()
    if a.mode == 'synthetic':
        run_synthetic(n_per_class=a.n_per_class)
    else:
        run_real(a.dataset, a.backbones, a.K, a.hop, a.max_nodes, a.samples_n, a.epochs, a.trials)
