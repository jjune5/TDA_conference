"""noise_robust_exp.py — Experiment N1: topology robustness to graph STRUCTURAL noise.

Question
--------
Does adding topology (Persistence Image / GDC-PI) make link prediction MORE robust
to structural perturbation of the graph than plain GCN (no-PI)?  This is a direct
test of TDA's core "robust to noise" claim.

Protocol
--------
The link-prediction *task* (which node-pairs are positives / negatives in the
train / val / test splits) is fixed from the CLEAN graph (seed=1234, identical to
the default pipeline).  We then PERTURB the *observed* message-passing graph that
the model is allowed to see:

    For perturbation level p in {0, 5, 10, 20} %:
        * remove p% of the observed (undirected) edges
        * add an equal number of random non-edges
      (seeded RNG, symmetric, no self-loops)

The SAME perturbed edge_index feeds BOTH the GCN message passing AND the Ollivier-
Ricci + Persistence Image computation.  So as p grows, both the GCN's neighbourhood
structure and the topological descriptor degrade together; we measure how fast test
AUC falls for each of the three variants.

    PI      : exact Persistence Image (dionysus), recomputed on the perturbed graph
    no-PI   : PI features zeroed (plain 2-layer GCN + edge MLP)
    GDC-PI  : Persistence Image computed on the GDC heat-kernel diffused perturbed graph

CRITICAL: the PI is recomputed on the PERTURBED graph (never a clean cache).  We use a
unique cache directory per (dataset, p, graph_seed) via TLCGNN_PI_DIR so it never
collides with the clean/default caches and is forced to recompute.

This script is fully self-contained: it imports loaddatas / baselines internals and
orchestrates the pipeline, injecting the perturbation between "split derivation" and
"PI computation + training".  It does NOT change default pipeline behaviour.

Outputs (in --results_dir):
    auc_vs_p.csv         : per (dataset, variant, p) mean/std AUC + n
    slopes.csv           : degradation slope (Delta AUC per +10% noise) per variant
    auc_vs_p.png         : AUC-vs-p plot, 3 lines per dataset
    raw_trials.csv       : every individual trial AUC
    summary.txt          : human-readable verdict
"""

import os
import sys
import copy
import time
import json
import argparse

import numpy as np
import networkx as nx
import torch
import torch.nn.functional as F
from torch.nn.init import xavier_normal_ as xavier
from torch_geometric.utils import remove_self_loops
from sklearn.metrics import roc_auc_score, average_precision_score

import loaddatas as lds
from loaddatas import get_edges_split, compute_persistence_image
from baselines.TLCGNN import Net


# ───────────────────────────── perturbation ──────────────────────────────────

def perturb_edge_index(edge_index, num_nodes, p, rng, forbidden=None):
    """Remove p-fraction of undirected edges and add an equal number of random
    non-edges.  Returns a symmetric, self-loop-free directed edge_index.

    Parameters
    ----------
    edge_index : LongTensor [2, E]   (symmetric directed; observed graph)
    num_nodes  : int
    p          : float in [0,1]       fraction of edges to remove (and add back as noise)
    rng        : np.random.Generator  seeded
    forbidden  : set[(u,v)] | None    undirected (u<v) edges that added noise must
                                      NOT coincide with.  Pass the FULL clean-graph
                                      edge set (incl. held-out val/test positives) so
                                      injected noise can never re-introduce a true /
                                      label edge into the observed graph (no leakage).

    Returns
    -------
    new_edge_index : LongTensor [2, E']
    stats : dict   (n_undirected, n_removed, n_added)
    """
    ei = edge_index.cpu().numpy()
    # undirected edge set (u < v), de-duplicated
    und = set()
    for u, v in zip(ei[0], ei[1]):
        u, v = int(u), int(v)
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        und.add((a, b))
    und = list(und)
    m = len(und)
    n_remove = int(round(p * m))

    if n_remove == 0:
        # p == 0: identity (still rebuild symmetric, self-loop-free for parity)
        kept = und
        added = []
    else:
        perm = rng.permutation(m)
        remove_idx = set(perm[:n_remove].tolist())
        kept = [e for i, e in enumerate(und) if i not in remove_idx]

        # sample n_remove random NON-edges (u<v, not currently an edge, no self-loop).
        # `existing` excludes both the current observed edges AND any forbidden (true)
        # edges so noise never coincides with a real or held-out edge.
        existing = set(und)
        if forbidden:
            existing |= forbidden
        added = []
        added_set = set()
        # rejection sampling — sparse graphs => collision prob tiny
        max_tries = n_remove * 200 + 10000
        tries = 0
        while len(added) < n_remove and tries < max_tries:
            tries += 1
            u = int(rng.integers(0, num_nodes))
            v = int(rng.integers(0, num_nodes))
            if u == v:
                continue
            a, b = (u, v) if u < v else (v, u)
            if (a, b) in existing or (a, b) in added_set:
                continue
            added.append((a, b))
            added_set.add((a, b))
        if len(added) < n_remove:
            print(f"  [warn] only added {len(added)}/{n_remove} non-edges "
                  f"after {tries} tries (graph nearly complete?)")

    final_und = kept + added
    if len(final_und) == 0:
        new_ei = torch.zeros((2, 0), dtype=torch.long)
    else:
        arr = np.array(final_und, dtype=np.int64).T  # [2, K]
        # symmetrise
        sym = np.concatenate([arr, arr[::-1, :]], axis=1)
        new_ei = torch.from_numpy(sym).long()
    new_ei, _ = remove_self_loops(new_ei)
    stats = dict(n_undirected=m, n_removed=n_remove, n_added=len(added))
    return new_ei, stats


# ──────────────────────── pipeline orchestration ─────────────────────────────

def weights_init(m):
    if isinstance(m, torch.nn.Linear):
        xavier(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


def build_fixed_task(data, val_prop, test_prop, split_seed=1234):
    """Derive the FIXED LP task from the CLEAN graph and the clean message-passing
    graph (val/test positives removed).  Returns everything the model needs that
    does NOT depend on the perturbation."""
    tr, trf, va, vaf, te, tef = get_edges_split(
        data, val_prop=val_prop, test_prop=test_prop, seed=split_seed)
    total_edges = np.concatenate((tr, trf, va, vaf, te, tef))
    total_edges_y = torch.cat((
        torch.ones(len(tr)), torch.zeros(len(trf)),
        torch.ones(len(va)), torch.zeros(len(vaf)),
        torch.ones(len(te)), torch.zeros(len(tef)))).long()

    # FULL clean undirected edge set (incl. val/test positives) — used as the
    # "forbidden" set so injected noise never re-introduces a true/label edge.
    _ei_full = np.array(data.edge_index)
    forbidden = set()
    for u, v in zip(_ei_full[0], _ei_full[1]):
        u, v = int(u), int(v)
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        forbidden.add((a, b))

    # clean message-passing graph with val/test positives removed (mirror call())
    _ei = np.array(data.edge_index)
    _rm = set()
    for e in va.tolist() + te.tolist():
        _rm.add((e[0], e[1])); _rm.add((e[1], e[0]))
    _keep = np.array([(int(u), int(v)) not in _rm for u, v in zip(_ei[0], _ei[1])])
    clean_mp_ei = torch.from_numpy(_ei[:, _keep]).long()
    clean_mp_ei, _ = remove_self_loops(clean_mp_ei)

    counts = dict(train_pos=len(tr), train_neg=len(trf),
                  val_pos=len(va), val_neg=len(vaf),
                  test_pos=len(te), test_neg=len(tef))
    splits = dict(train_edges=tr, train_edges_false=trf,
                  val_edges=va, val_edges_false=vaf,
                  test_edges=te, test_edges_false=tef)
    return total_edges, total_edges_y, clean_mp_ei, counts, splits, forbidden


def compute_pi_for_perturbed(data, perturbed_ei, splits, data_name, pi_dir, hop, use_gdc):
    """Compute (or load cached) Persistence Image on the *perturbed* graph.

    Sets data.edge_index = perturbed_ei and routes through the real
    compute_persistence_image, using TLCGNN_PI_DIR=pi_dir so the cache is unique
    per perturbation realization and never collides with the clean cache.

    Note: compute_persistence_image internally re-removes val/test positive edges
    from data.edge_index (set-based, idempotent).  Our perturbation only *adds*
    non-edges of the clean graph, and val/test positives are clean edges, so they
    are never re-introduced as noise — the internal removal stays a safe no-op for
    the added edges and correctly strips any val/test positive that survived.
    """
    d = copy.copy(data)
    d.edge_index = perturbed_ei.clone()
    os.makedirs(pi_dir, exist_ok=True)
    prev_dir = os.environ.get('TLCGNN_PI_DIR', '')
    prev_gdc = os.environ.get('TLCGNN_GDC', '')
    os.environ['TLCGNN_PI_DIR'] = pi_dir
    os.environ['TLCGNN_GDC'] = '1' if use_gdc else '0'
    try:
        pi = compute_persistence_image(
            d,
            splits['train_edges'], splits['train_edges_false'],
            splits['val_edges'], splits['val_edges_false'],
            splits['test_edges'], splits['test_edges_false'],
            data_name, hop=hop)
    finally:
        if prev_dir:
            os.environ['TLCGNN_PI_DIR'] = prev_dir
        else:
            os.environ.pop('TLCGNN_PI_DIR', None)
        if prev_gdc:
            os.environ['TLCGNN_GDC'] = prev_gdc
        else:
            os.environ.pop('TLCGNN_GDC', None)
    return pi


def run_training_trial(data_proto, perturbed_ei, PI, total_edges, total_edges_y,
                       counts, num_features, num_classes, use_pi, dropout, init_seed,
                       wait_total=200, total_epochs=2000):
    """One LP training run on the perturbed graph; returns best test AUC (ROC)."""
    torch.manual_seed(init_seed)
    torch.cuda.manual_seed_all(init_seed)
    np.random.seed(init_seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = copy.copy(data_proto)
    # the encoder must see the PERTURBED graph
    data.edge_index = perturbed_ei.clone()
    data.train_pos, data.train_neg = counts['train_pos'], counts['train_neg']
    data.val_pos, data.val_neg = counts['val_pos'], counts['val_neg']
    data.test_pos, data.test_neg = counts['test_pos'], counts['test_neg']
    data.total_edges = total_edges
    data.total_edges_y = total_edges_y

    model = Net(data, num_features, num_classes, PI=PI, use_pi=use_pi).to(device)
    # override the default 0.5 dropout in encode() is hard-coded; dropout arg kept
    # for parity with pipelines.py signature but encode() uses p=0.5 internally.
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
        for tp in ["val", "test"]:
            pred, y = model.decode(data, emb, type=tp)
            pred, y = pred.cpu(), y.cpu()
            if tp == "val":
                out['val_loss'] = F.binary_cross_entropy(pred, y).item()
                out['val_roc'] = roc_auc_score(y.data.numpy(), pred.data.numpy())
            else:
                out['test_roc'] = roc_auc_score(y.data.numpy(), pred.data.numpy())
                out['test_ap'] = average_precision_score(y.data.numpy(), pred.data.numpy())
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


# ──────────────────────────────── driver ─────────────────────────────────────

VARIANTS = {
    'PI':     dict(use_pi=True,  use_gdc=False),
    'no-PI':  dict(use_pi=False, use_gdc=False),
    'GDC-PI': dict(use_pi=True,  use_gdc=True),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=['Chameleon', 'Cora'])
    ap.add_argument('--ps', nargs='+', type=float, default=[0, 5, 10, 20],
                    help='perturbation percentages')
    ap.add_argument('--variants', nargs='+', default=['PI', 'no-PI', 'GDC-PI'])
    ap.add_argument('--graph_seeds', type=int, default=5,
                    help='number of distinct perturbation realizations per (dataset,p)')
    ap.add_argument('--init_per_graph', type=int, default=5,
                    help='model-init trials per perturbation realization')
    ap.add_argument('--cores', type=int, default=64, help='cpu cores for PI compute')
    ap.add_argument('--results_dir', type=str,
                    default='results/noise_robust')
    ap.add_argument('--dropout', type=float, default=0.5)
    ap.add_argument('--epochs', type=int, default=2000)
    ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args()

    os.environ['TLCGNN_CORES'] = str(args.cores)
    os.environ['TLCGNN_PI_SOURCE'] = 'dionysus'
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)

    os.makedirs(args.results_dir, exist_ok=True)
    pi_cache_root = 'data'  # noise_p{p}_s{seed}_TLCGNN dirs created under here

    raw_rows = []   # (dataset, variant, p, graph_seed, init_seed, auc)
    t_start = time.time()

    for d_name in args.datasets:
        print(f"\n{'='*70}\nDATASET: {d_name}\n{'='*70}", flush=True)
        dataset = lds.loaddatas(d_name)
        base_data = copy.deepcopy(dataset[0])
        num_features = base_data.x.size(1)
        num_classes = dataset.num_classes
        val_prop = 0.05
        test_prop = 0.1
        hop = 2 if d_name in ['PubMed'] else 1

        # FIXED task derived from clean graph
        total_edges, total_edges_y, clean_mp_ei, counts, splits, forbidden = build_fixed_task(
            base_data, val_prop, test_prop)
        num_nodes = base_data.num_nodes
        print(f"[task] nodes={num_nodes} clean_mp_edges(dir)={clean_mp_ei.shape[1]} "
              f"train_pos={counts['train_pos']} val_pos={counts['val_pos']} "
              f"test_pos={counts['test_pos']} total_pi_edges={len(total_edges)}",
              flush=True)

        for p in args.ps:
            pf = p / 100.0
            for gseed in range(args.graph_seeds):
                rng = np.random.default_rng(10_000 * int(round(p)) + gseed)
                perturbed_ei, pstats = perturb_edge_index(
                    clean_mp_ei, num_nodes, pf, rng, forbidden=forbidden)
                print(f"\n[{d_name} p={p}% gseed={gseed}] "
                      f"undirected={pstats['n_undirected']} "
                      f"removed={pstats['n_removed']} added={pstats['n_added']} "
                      f"-> perturbed_mp_edges(dir)={perturbed_ei.shape[1]}", flush=True)

                # Precompute PI variants needed for this realization (shared across init seeds)
                pi_cache = {}
                for variant in args.variants:
                    cfg = VARIANTS[variant]
                    if not cfg['use_pi']:
                        continue  # no-PI needs no PI
                    tag = 'GDC' if cfg['use_gdc'] else 'PI'
                    # At p==0 the perturbation is identity for every graph_seed, so
                    # the PI is identical — share a single cache (s0) across seeds to
                    # avoid recomputing the same clean-graph PI graph_seeds times.
                    cache_seed = 0 if p == 0 else gseed
                    pi_dir = os.path.join(
                        pi_cache_root,
                        f"noise_{tag}_p{int(round(p))}_s{cache_seed}_TLCGNN")
                    t0 = time.time()
                    pi = compute_pi_for_perturbed(
                        base_data, perturbed_ei, splits, d_name, pi_dir, hop,
                        use_gdc=cfg['use_gdc'])
                    pi_cache[variant] = pi
                    print(f"  [PI {variant}] shape={pi.shape} dir={pi_dir} "
                          f"({time.time()-t0:.0f}s)", flush=True)

                # zeros placeholder for no-PI
                zeros_pi = np.zeros((len(total_edges), 25), dtype=np.float64)

                for variant in args.variants:
                    cfg = VARIANTS[variant]
                    PI = pi_cache[variant] if cfg['use_pi'] else zeros_pi
                    for it in range(args.init_per_graph):
                        init_seed = 1000 * gseed + it
                        auc = run_training_trial(
                            base_data, perturbed_ei, PI, total_edges, total_edges_y,
                            counts, num_features, num_classes,
                            use_pi=cfg['use_pi'], dropout=args.dropout,
                            init_seed=init_seed, total_epochs=args.epochs)
                        raw_rows.append((d_name, variant, p, gseed, init_seed, auc))
                    # progress line per variant/realization
                    last = [r[-1] for r in raw_rows
                            if r[0] == d_name and r[1] == variant and r[2] == p
                            and r[3] == gseed]
                    print(f"  [{variant}] p={p}% gseed={gseed} "
                          f"AUC={np.mean(last):.4f}+-{np.std(last):.4f} "
                          f"(n={len(last)})", flush=True)

                # checkpoint raw after each realization (crash-safe)
                _write_raw(raw_rows, args.results_dir)

    print(f"\nAll trials done in {(time.time()-t_start)/60:.1f} min", flush=True)
    _write_raw(raw_rows, args.results_dir)
    summarize(raw_rows, args)


def _write_raw(raw_rows, results_dir):
    import csv
    path = os.path.join(results_dir, 'raw_trials.csv')
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['dataset', 'variant', 'p', 'graph_seed', 'init_seed', 'auc'])
        for r in raw_rows:
            w.writerow(r)


def summarize(raw_rows, args):
    import csv
    arr = np.array([(r[0], r[1], float(r[2]), float(r[5])) for r in raw_rows],
                   dtype=object)
    datasets = args.datasets
    variants = args.variants
    ps = sorted(set(float(p) for p in args.ps))

    # auc_vs_p.csv
    agg = {}  # (dataset, variant, p) -> (mean, std, n)
    with open(os.path.join(args.results_dir, 'auc_vs_p.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['dataset', 'variant', 'p', 'auc_mean', 'auc_std', 'n'])
        for d in datasets:
            for v in variants:
                for p in ps:
                    vals = [float(r[5]) for r in raw_rows
                            if r[0] == d and r[1] == v and float(r[2]) == p]
                    if not vals:
                        continue
                    m, s, n = float(np.mean(vals)), float(np.std(vals)), len(vals)
                    agg[(d, v, p)] = (m, s, n)
                    w.writerow([d, v, p, f'{m:.4f}', f'{s:.4f}', n])

    # slopes.csv: linear fit AUC ~ p (per 10% noise) + endpoint drop
    slopes = {}
    with open(os.path.join(args.results_dir, 'slopes.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['dataset', 'variant', 'slope_per_10pct', 'auc_at_0', 'auc_at_max_p',
                    'total_drop', 'r2'])
        for d in datasets:
            for v in variants:
                pts = [(p, agg[(d, v, p)][0]) for p in ps if (d, v, p) in agg]
                if len(pts) < 2:
                    continue
                xs = np.array([pp for pp, _ in pts])
                ys = np.array([aa for _, aa in pts])
                # slope per 1% then scale to per 10%
                A = np.vstack([xs, np.ones_like(xs)]).T
                coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
                slope1 = coef[0]
                slope10 = slope1 * 10.0
                pred = A @ coef
                ss_res = float(np.sum((ys - pred) ** 2))
                ss_tot = float(np.sum((ys - ys.mean()) ** 2))
                r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else float('nan')
                auc0 = agg[(d, v, 0.0)][0] if (d, v, 0.0) in agg else ys[0]
                aucmax = ys[-1]
                slopes[(d, v)] = slope10
                w.writerow([d, v, f'{slope10:.5f}', f'{auc0:.4f}', f'{aucmax:.4f}',
                            f'{auc0 - aucmax:.4f}', f'{r2:.3f}'])

    # plot
    plot_path = os.path.join(args.results_dir, 'auc_vs_p.png')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        nd = len(datasets)
        fig, axes = plt.subplots(1, nd, figsize=(6 * nd, 5), squeeze=False)
        colors = {'PI': 'tab:blue', 'no-PI': 'tab:gray', 'GDC-PI': 'tab:red'}
        markers = {'PI': 'o', 'no-PI': 's', 'GDC-PI': '^'}
        for ax, d in zip(axes[0], datasets):
            for v in variants:
                pts = [(p, agg[(d, v, p)][0], agg[(d, v, p)][1])
                       for p in ps if (d, v, p) in agg]
                if not pts:
                    continue
                xs = [pp for pp, _, _ in pts]
                ys = [mm for _, mm, _ in pts]
                es = [ss for _, _, ss in pts]
                sl = slopes.get((d, v), float('nan'))
                ax.errorbar(xs, ys, yerr=es, marker=markers.get(v, 'o'),
                            color=colors.get(v), capsize=3,
                            label=f'{v} (slope {sl:+.4f}/10%)')
            ax.set_title(f'{d}: LP AUC vs structural noise')
            ax.set_xlabel('edges removed & re-added as noise (%)')
            ax.set_ylabel('test AUC (ROC)')
            ax.grid(alpha=0.3)
            ax.legend(loc='lower left')
        fig.tight_layout()
        fig.savefig(plot_path, dpi=130)
        print(f"[plot] saved {plot_path}", flush=True)
    except Exception as e:
        print(f"[plot] FAILED: {e}", flush=True)

    # verdict
    lines = []
    lines.append("Experiment N1 — Topology robustness to graph structural noise")
    lines.append("=" * 64)
    lines.append(f"datasets={datasets} ps={ps} variants={variants}")
    _gs, _ipg = args.graph_seeds, args.init_per_graph
    if isinstance(_gs, int) and isinstance(_ipg, int):
        _ntr = f"=> {_gs*_ipg} trials per (dataset,p,variant)"
    else:
        _ntr = "(see per-cell n in auc_vs_p.csv)"
    lines.append(f"graph_seeds={_gs} init_per_graph={_ipg} {_ntr}")
    lines.append("")
    lines.append("Degradation slope (Delta AUC per +10% noise; less negative = more robust):")
    for d in datasets:
        lines.append(f"\n  {d}:")
        for v in variants:
            if (d, v) in slopes:
                lines.append(f"    {v:8s}: {slopes[(d,v)]:+.5f} per 10%")
        # verdict per dataset
        if (d, 'no-PI') in slopes:
            base = slopes[(d, 'no-PI')]
            for v in ['PI', 'GDC-PI']:
                if (d, v) in slopes:
                    flatter = slopes[(d, v)] > base  # less negative
                    diff = slopes[(d, v)] - base
                    verdict = ("MORE robust (flatter)" if flatter
                               else "NOT more robust")
                    lines.append(f"    -> {v} vs no-PI: {verdict} "
                                 f"(slope diff {diff:+.5f}/10%)")
    lines.append("")
    # overall verdict
    pro_topo = []
    for d in datasets:
        if (d, 'no-PI') not in slopes:
            continue
        base = slopes[(d, 'no-PI')]
        for v in ['PI', 'GDC-PI']:
            if (d, v) in slopes:
                pro_topo.append(slopes[(d, v)] > base)
    if pro_topo and all(pro_topo):
        lines.append("VERDICT: PRO-TOPOLOGY — PI/GDC-PI degrade SLOWER than no-PI on all "
                     "tested (dataset, variant) pairs. First pro-topology LP robustness result.")
    elif pro_topo and any(pro_topo):
        lines.append("VERDICT: MIXED — topology is more robust in some (dataset, variant) "
                     "pairs but not all. See per-dataset breakdown.")
    else:
        lines.append("VERDICT: NULL — topology does NOT degrade slower than plain GCN. "
                     "No support for the TDA structural-robustness claim in LP here.")

    txt = "\n".join(lines)
    with open(os.path.join(args.results_dir, 'summary.txt'), 'w') as f:
        f.write(txt + "\n")
    print("\n" + txt, flush=True)


if __name__ == '__main__':
    main()
