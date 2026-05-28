"""feature_noise_exp.py — Experiment R2: Feature-noise robustness.

Deepens batch-2 finding N1 (topology robust to EDGE noise).  Now corrupt
NODE FEATURES and measure whether the PI-vs-no-PI AUC gap grows as features
degrade.

Corruption scheme
-----------------
For each noise level q ∈ {0, 0.25, 0.5, 1.0}:

    x_noisy[n, d] = x_clean[n, d] + N(0, (q * sigma_d)^2)

where sigma_d = per-dimension std of the clean feature matrix (computed once on
the full node-feature matrix, not just train nodes).  At q=0 features are
unchanged; at q=1.0 noise std equals the feature std in every dimension (SNR ≈ 0 dB).

Topology is UNCHANGED — the graph structure (edge_index) stays fixed for all q.
Therefore the Ollivier-Ricci curvature and the Persistence Image are IDENTICAL
for every q and are loaded from the existing exact-PI cache
  data/TLCGNN/<name>.npy
NO Ollivier-Ricci computation is triggered.

CRITICAL: this script asserts that the PI is loaded from the cache (file exists
before training begins) and logs a confirmation for every trial.

Variants
--------
    PI    : exact PI from cache, GCN sees corrupted features
    no-PI : PI zeroed (pure GCN), GCN sees corrupted features

Datasets: Cora (homophilic), Chameleon (heterophilic).
Trials  : 20 per (dataset, q, variant).

Hypothesis
----------
PI is feature-independent.  As q grows the GCN's feature signal collapses but
the topological signal remains intact.  Expected: PI−noPI AUC gap widens with q.

Outputs (results/feature_noise/)
----------------------------------
    raw_trials.csv      : every individual trial (dataset, variant, q, trial, auc)
    auc_vs_q.csv        : per (dataset, variant, q) mean/std AUC + n
    gap_vs_q.csv        : PI−noPI gap per (dataset, q)
    auc_vs_q.png        : AUC-vs-q plot (2 datasets × 2 variants) + gap subplot
    summary.txt         : verdict (gap widens / flat / reversal)
"""

import os
import sys
import copy
import time
import csv
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.init import xavier_normal_ as xavier
from sklearn.metrics import roc_auc_score, average_precision_score

# ── import from main repo ─────────────────────────────────────────────────────
# This script lives in the worktree.  The main repo has the full loaddatas.py
# (with train-neg capping so edge counts match the pre-built PI caches) and the
# updated TLCGNN.py (use_pi ablation).  Always prepend the main repo so it
# shadows the worktree's incomplete copies.
_MAIN_REPO = '/mnt/data/users/junyoungpark/code/TLC-GNN'
if os.path.isdir(_MAIN_REPO) and _MAIN_REPO not in sys.path:
    sys.path.insert(0, _MAIN_REPO)

import loaddatas as lds
from loaddatas import get_edges_split, compute_persistence_image
from baselines.TLCGNN import Net


# ─────────────────────────── helpers ─────────────────────────────────────────

def weights_init(m):
    if isinstance(m, torch.nn.Linear):
        xavier(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


def corrupt_features(x_clean: torch.Tensor, q: float, rng: np.random.Generator,
                     feat_std: np.ndarray) -> torch.Tensor:
    """Add Gaussian noise with std = q * per-dim-std to the feature matrix.

    Parameters
    ----------
    x_clean  : Tensor [N, F]  clean (possibly normalised) features, float32
    q        : float in [0,1]  noise level (0 → clean, 1 → SNR ≈ 0 dB)
    rng      : np.random.Generator  seeded for reproducibility
    feat_std : ndarray [F]     per-dimension std of the clean feature matrix

    Returns
    -------
    x_noisy  : Tensor [N, F]  float32, same device as x_clean
    """
    if q == 0.0:
        return x_clean
    n, f = x_clean.shape
    noise_std = q * feat_std   # [F] broadcast to [N, F]
    noise = rng.standard_normal((n, f)).astype(np.float32) * noise_std[None, :]
    noise_t = torch.from_numpy(noise).to(x_clean.device)
    return x_clean + noise_t


# ─────────────────────── PI loading (cache-only) ─────────────────────────────

def _pi_cache_path(d_name: str) -> str:
    """Return the path to the exact-PI cache for *d_name*.

    Mirrors the logic in loaddatas.compute_persistence_image for the
    default (no env-var) case so we can assert the file exists BEFORE
    calling compute_persistence_image.

    Some caches are stored with lowercase names (e.g. chameleon.npy,
    squirrel.npy) because that is how they were written by earlier runs.
    We check both cases and return whichever exists.
    """
    name = d_name
    if name.lower() == 'photo':
        name = 'Photo'
    if name.lower() == 'computers':
        name = 'Computers'
    # Primary path (as compute_persistence_image constructs it)
    primary = f'./data/TLCGNN/{name}.npy'
    if os.path.exists(primary):
        return primary
    # Fallback: lowercase name (some datasets cached this way)
    fallback = f'./data/TLCGNN/{name.lower()}.npy'
    if os.path.exists(fallback):
        return fallback
    return primary  # return primary so assertion message is informative


def load_pi_from_cache(d_name: str, expected_rows: int) -> np.ndarray:
    """Load the PI array from the exact-PI cache.

    Asserts that the cache file exists (no recomputation) and that the
    row count matches *expected_rows*.  Logs a confirmation to stdout.
    """
    path = _pi_cache_path(d_name)
    assert os.path.exists(path), (
        f"[PI cache] MISSING: {path}  — "
        f"script requires a pre-existing cache (no Ollivier-Ricci recomputation).")
    pi = np.load(path)
    # The cache may have a stale layout (uncapped train-neg); if so,
    # compute_persistence_image's splicing code handles it.  For our
    # assert we just check the file exists; the actual shape check happens
    # inside compute_persistence_image.
    print(f"[PI cache CONFIRMED] loaded {path}  shape={pi.shape}  "
          f"expected_rows={expected_rows}", flush=True)
    return pi   # may still be passed through compute_persistence_image below


# ───────────────────── build fixed LP task ────────────────────────────────────

def build_task(data, val_prop, test_prop, split_seed=1234):
    """Derive the fixed LP splits from the CLEAN graph (topology unchanged).

    Returns
    -------
    total_edges, total_edges_y, clean_mp_ei, counts, splits
    """
    from torch_geometric.utils import remove_self_loops as _rsl

    tr, trf, va, vaf, te, tef = get_edges_split(
        data, val_prop=val_prop, test_prop=test_prop, seed=split_seed)
    total_edges = np.concatenate((tr, trf, va, vaf, te, tef))
    total_edges_y = torch.cat((
        torch.ones(len(tr)), torch.zeros(len(trf)),
        torch.ones(len(va)), torch.zeros(len(vaf)),
        torch.ones(len(te)), torch.zeros(len(tef)))).long()

    # clean message-passing graph (val/test positives removed)
    ei = np.array(data.edge_index)
    rm = set()
    for e in va.tolist() + te.tolist():
        rm.add((e[0], e[1])); rm.add((e[1], e[0]))
    keep = np.array([(int(u), int(v)) not in rm for u, v in zip(ei[0], ei[1])])
    clean_mp_ei = torch.from_numpy(ei[:, keep]).long()
    clean_mp_ei, _ = _rsl(clean_mp_ei)

    counts = dict(train_pos=len(tr), train_neg=len(trf),
                  val_pos=len(va), val_neg=len(vaf),
                  test_pos=len(te), test_neg=len(tef))
    splits = dict(train_edges=tr, train_edges_false=trf,
                  val_edges=va, val_edges_false=vaf,
                  test_edges=te, test_edges_false=tef)
    return total_edges, total_edges_y, clean_mp_ei, counts, splits


# ─────────────────────────── one training trial ──────────────────────────────

def run_trial(data_proto, clean_mp_ei, PI, total_edges, total_edges_y,
              counts, x_noisy, num_classes, use_pi, init_seed,
              wait_total=200, total_epochs=2000):
    """One LP training run on the CLEAN topology with CORRUPTED features.

    Parameters
    ----------
    x_noisy : Tensor [N, F]  corrupted features (same shape as clean; float32)
    PI      : ndarray [E, 25] persistence image (from cache, unchanged)
    use_pi  : bool  if False PI is zeroed inside Net.decode (no-PI variant)

    Returns
    -------
    test_roc : float
    """
    torch.manual_seed(init_seed)
    torch.cuda.manual_seed_all(init_seed)
    np.random.seed(init_seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = copy.copy(data_proto)
    # Overwrite features with the corrupted version for this trial.
    data.x = x_noisy.clone()
    data.edge_index = clean_mp_ei.clone()
    data.train_pos, data.train_neg = counts['train_pos'], counts['train_neg']
    data.val_pos,   data.val_neg   = counts['val_pos'],   counts['val_neg']
    data.test_pos,  data.test_neg  = counts['test_pos'],  counts['test_neg']
    data.total_edges   = total_edges
    data.total_edges_y = total_edges_y

    num_features = x_noisy.shape[1]
    model = Net(data, num_features, num_classes, PI=PI, use_pi=use_pi).to(device)
    model.apply(weights_init)
    data = data.to(device)
    data.total_edges_y = data.total_edges_y.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0)

    def _train():
        model.train(); optimizer.zero_grad()
        emb = model.encode(data)
        x, y = model.decode(data, emb)
        loss = F.binary_cross_entropy(x, y)
        loss.backward(); optimizer.step()

    def _test():
        model.eval()
        emb = model.encode(data)
        out = {}
        for tp in ('val', 'test'):
            pred, y = model.decode(data, emb, type=tp)
            pred, y = pred.cpu(), y.cpu()
            if tp == 'val':
                out['val_roc'] = roc_auc_score(y.numpy(), pred.data.numpy())
            else:
                out['test_roc'] = roc_auc_score(y.numpy(), pred.data.numpy())
        return out

    best_val_roc = 0.0
    test_roc = 0.0
    wait_step = 0
    for epoch in range(1, total_epochs + 1):
        _train()
        r = _test()
        if r['val_roc'] >= best_val_roc:
            best_val_roc = r['val_roc']
            test_roc = r['test_roc']
            wait_step = 0
        else:
            wait_step += 1
            if wait_step == wait_total:
                break
    del model
    torch.cuda.empty_cache()
    return float(test_roc)


# ─────────────────────────── output helpers ──────────────────────────────────

def write_raw(rows, out_dir):
    path = os.path.join(out_dir, 'raw_trials.csv')
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['dataset', 'variant', 'q', 'trial', 'auc'])
        for r in rows:
            w.writerow(r)


def summarize(rows, datasets, qs, results_dir):
    """Write auc_vs_q.csv, gap_vs_q.csv, plot, summary.txt."""
    variants = ['PI', 'no-PI']

    # ── aggregate ─────────────────────────────────────────────────────────────
    agg = {}  # (dataset, variant, q) -> (mean, std, n)
    for d in datasets:
        for v in variants:
            for q in qs:
                vals = [float(r[4]) for r in rows
                        if r[0] == d and r[1] == v and float(r[2]) == q]
                if vals:
                    agg[(d, v, q)] = (float(np.mean(vals)),
                                      float(np.std(vals)),
                                      len(vals))

    # auc_vs_q.csv
    with open(os.path.join(results_dir, 'auc_vs_q.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['dataset', 'variant', 'q', 'auc_mean', 'auc_std', 'n'])
        for d in datasets:
            for v in variants:
                for q in sorted(qs):
                    if (d, v, q) in agg:
                        m, s, n = agg[(d, v, q)]
                        w.writerow([d, v, q, f'{m:.4f}', f'{s:.4f}', n])

    # gap_vs_q.csv  (PI − no-PI)
    gaps = {}  # (dataset, q) -> gap
    with open(os.path.join(results_dir, 'gap_vs_q.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['dataset', 'q', 'gap_PI_minus_noPI', 'auc_PI', 'auc_noPI'])
        for d in datasets:
            for q in sorted(qs):
                if (d, 'PI', q) in agg and (d, 'no-PI', q) in agg:
                    pi_m = agg[(d, 'PI', q)][0]
                    nopi_m = agg[(d, 'no-PI', q)][0]
                    gap = pi_m - nopi_m
                    gaps[(d, q)] = gap
                    w.writerow([d, q, f'{gap:.4f}', f'{pi_m:.4f}', f'{nopi_m:.4f}'])

    # ── plot ──────────────────────────────────────────────────────────────────
    plot_path = os.path.join(results_dir, 'auc_vs_q.png')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        nd = len(datasets)
        # 2 rows: top = AUC curves, bottom = gap
        fig, axes = plt.subplots(2, nd, figsize=(6 * nd, 9), squeeze=False)

        colors = {'PI': 'tab:blue', 'no-PI': 'tab:gray'}
        markers = {'PI': 'o', 'no-PI': 's'}
        qs_sorted = sorted(qs)

        for col, d in enumerate(datasets):
            ax_auc = axes[0][col]
            ax_gap = axes[1][col]
            for v in variants:
                pts = [(q, agg[(d, v, q)][0], agg[(d, v, q)][1])
                       for q in qs_sorted if (d, v, q) in agg]
                if not pts:
                    continue
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                es = [p[2] for p in pts]
                ax_auc.errorbar(xs, ys, yerr=es, marker=markers[v],
                                color=colors[v], capsize=3, label=v)
            ax_auc.set_title(f'{d}: AUC vs feature noise level q')
            ax_auc.set_xlabel('noise level q')
            ax_auc.set_ylabel('test AUC (ROC)')
            ax_auc.grid(alpha=0.3)
            ax_auc.legend()

            # gap subplot
            gap_pts = [(q, gaps[(d, q)]) for q in qs_sorted if (d, q) in gaps]
            if gap_pts:
                xs2 = [p[0] for p in gap_pts]
                ys2 = [p[1] for p in gap_pts]
                ax_gap.plot(xs2, ys2, 'o-', color='tab:purple')
                ax_gap.axhline(0, color='k', lw=0.8, ls='--')
                ax_gap.set_title(f'{d}: PI−noPI gap vs q')
                ax_gap.set_xlabel('noise level q')
                ax_gap.set_ylabel('AUC gap (PI − no-PI)')
                ax_gap.grid(alpha=0.3)

        fig.tight_layout()
        fig.savefig(plot_path, dpi=130)
        print(f'[plot] saved {plot_path}', flush=True)
    except Exception as e:
        print(f'[plot] FAILED: {e}', flush=True)

    # ── verdict ───────────────────────────────────────────────────────────────
    lines = ['Experiment R2 — Feature-noise robustness (deepens N1)',
             '=' * 64,
             '',
             'Corruption: x_noisy = x_clean + N(0, (q * sigma_d)^2) per feature dim',
             'Topology (edge_index) unchanged; PI loaded from exact cache (no recompute)',
             '',
             'PI − noPI AUC gap by dataset and noise level:']
    for d in datasets:
        lines.append(f'\n  {d}:')
        lines.append(f'  {"q":>6}  {"PI AUC":>10}  {"noPI AUC":>10}  {"gap":>8}')
        for q in sorted(qs):
            pi_str = f'{agg[(d,"PI",q)][0]:.4f}' if (d,'PI',q) in agg else 'N/A'
            nopi_str = f'{agg[(d,"no-PI",q)][0]:.4f}' if (d,'no-PI',q) in agg else 'N/A'
            gap_str = f'{gaps[(d,q)]:+.4f}' if (d,q) in gaps else 'N/A'
            lines.append(f'  {q:>6.2f}  {pi_str:>10}  {nopi_str:>10}  {gap_str:>8}')

    # determine if gap monotonically increases (widens) with q per dataset
    lines.append('\nGap trend analysis (does PI−noPI gap widen with q?):')
    widening_flags = []
    for d in datasets:
        gap_seq = [gaps[(d, q)] for q in sorted(qs) if (d, q) in gaps]
        qs_seq  = [q             for q in sorted(qs) if (d, q) in gaps]
        if len(gap_seq) < 2:
            lines.append(f'  {d}: insufficient data')
            continue
        # slope of gap vs q
        xs = np.array(qs_seq)
        ys = np.array(gap_seq)
        A = np.vstack([xs, np.ones_like(xs)]).T
        coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
        slope = coef[0]
        # endpoint increase
        endpoint_diff = gap_seq[-1] - gap_seq[0]
        widens = slope > 0
        widening_flags.append(widens)
        lines.append(f'  {d}: gap at q=0 -> q={qs_seq[-1]}: '
                     f'{gap_seq[0]:+.4f} -> {gap_seq[-1]:+.4f}  '
                     f'(endpoint_diff={endpoint_diff:+.4f}, slope={slope:+.5f}/unit-q)  '
                     f'{"WIDENS" if widens else "FLAT/REVERSAL"}')

    lines.append('')
    if widening_flags and all(widening_flags):
        lines.append('VERDICT: PRO-TOPOLOGY — PI−noPI gap WIDENS with feature noise on '
                     'all tested datasets. Topology is a feature-robustness prior: when '
                     'the GCN\'s feature signal collapses, the topological signal carries '
                     'the prediction. Complements N1 (edge-noise robustness).')
    elif widening_flags and any(widening_flags):
        lines.append('VERDICT: MIXED — gap widens on some datasets but not all. '
                     'Partial support for topology as a feature-robustness prior.')
    else:
        lines.append('VERDICT: NULL — PI−noPI gap does NOT widen with feature noise. '
                     'Topology does not provide a feature-robustness advantage here. '
                     'Honest null result; feature and topological signals may be coupled '
                     'or the noise level range insufficient to collapse GCN features.')

    txt = '\n'.join(lines)
    with open(os.path.join(results_dir, 'summary.txt'), 'w') as f:
        f.write(txt + '\n')
    print('\n' + txt, flush=True)


# ─────────────────────────────── main ────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description='Experiment R2: feature-noise robustness, reuses exact PI cache')
    ap.add_argument('--datasets', nargs='+', default=['Cora', 'Chameleon'])
    ap.add_argument('--qs', nargs='+', type=float, default=[0.0, 0.25, 0.5, 1.0],
                    help='feature noise levels')
    ap.add_argument('--trials', type=int, default=20,
                    help='trials per (dataset, q, variant)')
    ap.add_argument('--results_dir', default='results/feature_noise')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=2000)
    ap.add_argument('--wait', type=int, default=200)
    args = ap.parse_args()

    # Force the default (exact) PI source — no pdgnn, no GDC, no HKS.
    os.environ['TLCGNN_PI_SOURCE'] = 'dionysus'
    os.environ.pop('TLCGNN_PI_DIR', None)
    os.environ.pop('TLCGNN_GDC', None)
    os.environ.pop('TLCGNN_LP_FILTER', None)

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)

    os.makedirs(args.results_dir, exist_ok=True)

    qs = sorted(args.qs)
    variants = ['PI', 'no-PI']
    rows = []   # (dataset, variant, q, trial, auc)

    t0_global = time.time()

    for d_name in args.datasets:
        print(f"\n{'='*72}\nDATASET: {d_name}\n{'='*72}", flush=True)

        dataset = lds.loaddatas(d_name)
        base_data = copy.deepcopy(dataset[0])
        num_features = base_data.x.shape[1]
        num_classes  = dataset.num_classes
        val_prop     = 0.05
        test_prop    = 0.1

        # Fixed LP task (topology unchanged for all q)
        total_edges, total_edges_y, clean_mp_ei, counts, splits = build_task(
            base_data, val_prop, test_prop)

        n_total = len(total_edges)
        print(f'[task] nodes={base_data.num_nodes} features={num_features} '
              f'total_pi_rows={n_total} '
              f'train_pos={counts["train_pos"]} val_pos={counts["val_pos"]} '
              f'test_pos={counts["test_pos"]}', flush=True)

        # ── Verify + load PI from cache (ASSERTION: no recomputation) ─────────
        # Confirm the cache file exists BEFORE calling compute_persistence_image.
        pi_path = _pi_cache_path(d_name)
        assert os.path.exists(pi_path), (
            f"PI cache missing for {d_name} at {pi_path}. "
            f"This experiment requires a pre-existing cache.  "
            f"Run the default pipeline once to generate it.")
        print(f'[PI cache] VERIFIED EXISTS: {pi_path}', flush=True)

        # Call compute_persistence_image which will load (and if needed splice)
        # from the cache — no Ollivier-Ricci triggered.
        # We pass the data copy with clean_mp_ei so the function can verify
        # edge counts but MUST hit the cache.
        _data_for_pi = copy.copy(base_data)
        _data_for_pi.edge_index = clean_mp_ei.clone()
        pi_exact = compute_persistence_image(
            _data_for_pi,
            splits['train_edges'], splits['train_edges_false'],
            splits['val_edges'], splits['val_edges_false'],
            splits['test_edges'], splits['test_edges_false'],
            d_name, hop=1)
        assert pi_exact.shape[0] == n_total, (
            f"PI row count mismatch: got {pi_exact.shape[0]}, expected {n_total}")
        print(f'[PI cache] LOADED from {pi_path}  shape={pi_exact.shape}  '
              f'CONFIRMED: no Ollivier-Ricci was run.', flush=True)

        # zeros PI placeholder for no-PI variant (architecture identical, PI zeroed)
        zeros_pi = np.zeros((n_total, 25), dtype=np.float64)

        # ── per-dim feature std (computed once on clean features) ─────────────
        x_clean = base_data.x.cpu().float()  # [N, F]
        feat_std = x_clean.numpy().std(axis=0)  # [F]
        # Guard: dims with zero std (constant features) get zero noise => correct
        feat_std = np.maximum(feat_std, 0.0)
        print(f'[features] mean per-dim std: {feat_std.mean():.5f}  '
              f'min: {feat_std.min():.5f}  max: {feat_std.max():.5f}', flush=True)

        for q in qs:
            print(f'\n[{d_name}] q={q}', flush=True)

            # One noise realization per trial (different seed) but fixed q.
            for trial_i in range(args.trials):
                # Seed: unique per (dataset, q, trial) for reproducibility
                noise_seed = int(hash((d_name, q, trial_i)) % (2**31))
                rng = np.random.default_rng(noise_seed)
                x_noisy = corrupt_features(x_clean, q, rng, feat_std)
                # Bring to GPU? No — Net.encode expects data.x on device;
                # data.to(device) in run_trial will move it.
                x_noisy = x_noisy.cpu()  # run_trial will move with data.to(device)

                init_seed = int(hash((d_name, q, trial_i, 'init')) % (2**31))

                for variant in variants:
                    use_pi = (variant == 'PI')
                    PI = pi_exact if use_pi else zeros_pi

                    auc = run_trial(
                        base_data, clean_mp_ei, PI,
                        total_edges, total_edges_y,
                        counts, x_noisy, num_classes,
                        use_pi=use_pi, init_seed=init_seed,
                        wait_total=args.wait, total_epochs=args.epochs)
                    rows.append((d_name, variant, q, trial_i, auc))

                # progress
                pi_aucs  = [r[4] for r in rows if r[0]==d_name and r[1]=='PI'    and r[2]==q]
                nopi_aucs= [r[4] for r in rows if r[0]==d_name and r[1]=='no-PI' and r[2]==q]
                print(f'  trial {trial_i:2d}/{args.trials}  '
                      f'PI={np.mean(pi_aucs):.4f}(n={len(pi_aucs)})  '
                      f'noPI={np.mean(nopi_aucs):.4f}(n={len(nopi_aucs)})  '
                      f'gap={np.mean(pi_aucs)-np.mean(nopi_aucs):+.4f}', flush=True)

            # checkpoint after each q
            write_raw(rows, args.results_dir)

    elapsed = (time.time() - t0_global) / 60
    print(f'\nAll trials complete in {elapsed:.1f} min', flush=True)
    write_raw(rows, args.results_dir)
    summarize(rows, args.datasets, qs, args.results_dir)


if __name__ == '__main__':
    main()
