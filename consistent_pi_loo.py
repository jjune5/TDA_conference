"""Leave-one-out PI control (results doc S14 centerpiece).

CLAIM (to prove directly): TLC-GNN's exact persistence-image (PI) signal on
train_pos edges is a *train-graph-membership artifact*. A train_pos edge (u,v)
gets PI >> 0 only because the (u,v) edge is present in its vicinity; remove the
edge from the graph and the vicinity persistence collapses to ~= 0 -- which is
exactly why test_pos edges (whose edges are held out from the graph) already
have PI ~= 0 in the cache.

EXPERIMENT: "leave-one-out PI" on sampled train_pos edges (Photo + Chameleon).
For each sampled train_pos edge (u,v):
  * WITH-edge   PI = vicinity PI computed on the training graph G_full
                    (edge present). This is the train-time / cached condition.
  * WITHOUT-edge PI = vicinity PI computed on G_full minus the (u,v) edge.
                    This is the consistent / test-like / held-out condition
                    (exactly how test_pos edges are treated).
We compare the two and report the fraction of train_pos edges whose PI
"collapses to ~0" once their own edge is removed.

FIDELITY: both conditions use the IDENTICAL PI math the cache uses --
sg2dgm.filtration.build_fv (weighted, normed) -> perturb_filter_function ->
Union_find -> Accelerate_PD -> PersistenceImager, hop=1 intersection vicinity,
descriptor='sum', resolution=5 -- which is exactly sg2dgm.parallel_pi._compute_one.
The ONLY difference between the two conditions is whether the (u,v) edge is in
the graph. Ollivier-Ricci curvatures for the vicinity edges are computed
*exactly* (globally, on the full G_full resp. G_minus) via
OllivierRicci.compute_ricci_curvature_edges, localized to the vicinity edges for
tractability (the curvature of an edge depends only on its endpoints' 1-hop mass
distributions + pairwise shortest paths, computed on the full graph either way).

We also sanity-check WITH-edge PI against the actual cached train_pos rows in
data/TLCGNN/<name>.npy (they should match closely).
"""
from __future__ import annotations
import os
import sys
import time
import argparse
import numpy as np
import networkx as nx
import torch
from torch_geometric.utils import remove_self_loops

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import loaddatas as lds
import pi_artifact_analysis as pa
from sg2dgm.riccidist2dgm import filtration as _filtration
from sg2dgm.accelerated_PD import (perturb_filter_function, Union_find,
                                   Accelerate_PD)
import sg2dgm.PersistenceImager as _pimg
from GraphRicciCurvature.OllivierRicci import OllivierRicci

RES = 5
HOP = 1
DESC = 'sum'
NORM = True
EGO_RADIUS = 5  # local ego-ball radius for the without-edge Ricci recompute
                # (>= HOP+3; verified to match the global cache to L1 ~0)


def build_train_graph(name):
    """Reproduce exactly the graph compute_persistence_image builds:
    full edge_index with val_pos + test_pos edges removed, self-loops removed,
    all original nodes present (incl. isolated).

    CRITICAL: the cache (baselines/TLCGNN.call -> get_edges_split) derives the
    train/val/test split from adj = nx.adjacency_matrix(g) where g is built node-
    by-node then edge-by-edge. The triu().nonzero() ordering of THAT adjacency
    (and hence the seeded shuffle, and hence which edges are train_pos and in
    what order) differs from a coo_matrix built straight off edge_index. To make
    cache row i correspond to train_pos[i] EXACTLY we must reproduce the cache's
    split via get_edges_split (verified: with-edge recompute then == cache)."""
    ds = lds.loaddatas(name)
    data = ds[0]
    ei = np.array(data.edge_index)
    # Exact cache split (same code path baselines/TLCGNN.call used).
    tr, trf, va, vaf, te, tef = lds.get_edges_split(
        data, val_prop=0.05, test_prop=0.1)
    _mask = set()
    for e in va.tolist():
        _mask.add((e[0], e[1])); _mask.add((e[1], e[0]))
    for e in te.tolist():
        _mask.add((e[0], e[1])); _mask.add((e[1], e[0]))
    _keep = np.array([(int(u), int(v)) not in _mask
                      for u, v in zip(ei[0], ei[1])])
    ei2 = ei[:, _keep]
    ei2t, _ = remove_self_loops(torch.from_numpy(ei2).long())
    g = nx.Graph()
    g.add_nodes_from(range(data.num_nodes))
    g.add_edges_from([(int(ei2t[0, i]), int(ei2t[1, i]))
                      for i in range(ei2t.shape[1])])
    return data, g, tr  # tr = train_pos edges (original labels), cache order


def vicinity_nodes(g, u, v, hop=HOP):
    """hop-intersection vicinity, exactly as parallel_pi._compute_one."""
    nodes_u = [u] + [x for _, x in nx.bfs_edges(g, u, depth_limit=hop)]
    nodes_v = [v] + [x for _, x in nx.bfs_edges(g, v, depth_limit=hop)]
    return list(set(nodes_u) & set(nodes_v))


def pi_for_edge(g, u, v, ricci_dict, hop=HOP):
    """Vicinity PI for edge (u,v) on graph g with curvature dict ricci_dict
    (keyed both directions on g's node labels). Mirrors _compute_one EXACTLY.

    ricci_dict must contain a curvature for every edge in the vicinity subgraph
    (both directions). Returns flat (RES*RES,) PI vector. Returns zeros for the
    degenerate cases the cache also zeros (empty / disconnected vicinity, or any
    exception) -- identical to parallel_pi._compute_one's behavior."""
    try:
        nodes = vicinity_nodes(g, u, v, hop=hop)
        if len(nodes) == 0:
            return np.zeros(RES * RES), 'empty_vicinity', len(nodes)
        subgraph = g.subgraph(nodes)
        if not nx.is_connected(subgraph):
            return np.zeros(RES * RES), 'disconnected', len(nodes)
        # filtration.build_fv (weight_graph=True) finds dijkstra paths using the
        # edge 'weight' attribute and sums ricci_curv[(a,b)]+1 along them. To
        # match parallel_pi.compute_pi_parallel EXACTLY we must set
        # weight = curvature+1 on the (copied) subgraph edges, and pass the RAW
        # curvature dict (build_fv adds the +1 itself).
        subgraph = nx.Graph(subgraph)  # mutable copy
        for a, b in subgraph.edges():
            subgraph[a][b]['weight'] = ricci_dict.get((a, b), 0.0) + 1.0
        fil = _filtration(subgraph, u, v, hop, ricci_curv=ricci_dict)
        gg = fil.build_fv(weight_graph=True, norm=NORM)
        sf = perturb_filter_function(gg, descriptor=DESC)
        PD0, Pos_edges, Neg_edges = Union_find(sf)
        PD1 = Accelerate_PD(Pos_edges, Neg_edges, sf)
        pers = _pimg.PersistenceImager(resolution=RES)
        diag = np.array(list(PD0) + list(PD1))
        pi = pers.transform(diag).reshape(-1)
        return pi, 'ok', len(nodes)
    except BaseException as ex:
        return np.zeros(RES * RES), f'exc:{type(ex).__name__}', -1


def ricci_dict_for_edges(g, edge_list, proc):
    """Exact Ollivier-Ricci curvature (alpha=0.5, Sinkhorn) for `edge_list`,
    computed globally on g (full-graph distances + neighborhoods), returned as a
    dict keyed both directions. Matches loaddatas.compute_ricci_curvature math,
    but only for the requested edges (tractable)."""
    if len(edge_list) == 0:
        return {}
    orc = OllivierRicci(g, alpha=0.5, method='Sinkhorn', proc=proc,
                        verbose='ERROR')
    out = orc.compute_ricci_curvature_edges(edge_list=list(edge_list))
    rd = {}
    for (a, b), c in out.items():
        rd[(a, b)] = c
        rd[(b, a)] = c
    return rd


def ego_subgraph(g, centers, radius):
    """Union of `radius`-hop ego graphs around the given center nodes."""
    nodes = set()
    for c in centers:
        if c in g:
            nodes |= set(nx.single_source_shortest_path_length(
                g, c, cutoff=radius).keys())
    return g.subgraph(nodes)


def ricci_dict_local(g, edge_list, centers, proc, ego_radius):
    """Ollivier-Ricci curvature for `edge_list` computed on the ego subgraph
    around `centers` (radius=ego_radius), instead of the full graph.

    EXACTNESS: Ollivier-Ricci curvature of an edge (a,b) depends only on the
    1-hop mass distributions m_a (on N(a)+{a}) and m_b (on N(b)+{b}) and the
    pairwise shortest-path distances among those nodes. For vicinity edges --
    whose endpoints lie within HOP of u or v -- those neighbor nodes lie within
    HOP+1 of {u,v}, and the shortest paths between near-adjacent node pairs are
    short. With ego_radius >= HOP+3 the ego subgraph contains those nodes AND
    their connecting shortest paths, so the curvature equals the full-graph
    value (verified empirically against the global cache to L1 ~0). This makes
    the per-edge APSP O(ego size) instead of O(|V|), the key speedup for the
    leave-one-out without-edge recompute (one tiny APSP per removed edge)."""
    if len(edge_list) == 0:
        return {}
    sg = nx.Graph(ego_subgraph(g, centers, ego_radius))  # mutable copy
    # ensure the edges we want curvature for are present (they are, since they
    # are vicinity edges inside the ego ball)
    orc = OllivierRicci(sg, alpha=0.5, method='Sinkhorn', proc=proc,
                        verbose='ERROR')
    out = orc.compute_ricci_curvature_edges(edge_list=list(edge_list))
    rd = {}
    for (a, b), c in out.items():
        rd[(a, b)] = c
        rd[(b, a)] = c
    return rd


# ── Non-daemonic process pool ──────────────────────────────────────────────
# Our per-edge worker calls OllivierRicci.compute_ricci_curvature_edges, which
# itself spawns an internal multiprocessing Pool. A normal Pool's workers are
# DAEMONIC and may not have children -> AssertionError. So the outer pool must
# use non-daemonic worker processes. Standard recipe: override Process.daemon.
import multiprocessing as _mp
import multiprocessing.pool as _mpp

_FORK_CTX = _mp.get_context('fork')


class _NoDaemonProcess(_FORK_CTX.Process):
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass


class _NoDaemonContext(type(_FORK_CTX)):
    Process = _NoDaemonProcess


class NoDaemonPool(_mpp.Pool):
    def __init__(self, *args, **kwargs):
        kwargs['context'] = _NoDaemonContext()
        super().__init__(*args, **kwargs)


# ── Per-edge parallel worker ───────────────────────────────────────────────
# The per-edge PI (build_fv dijkstra-from-every-vicinity-node + accelerated PD)
# is single-process and, for high-degree HUB edges, the vicinity (intersection
# of 1-hop neighborhoods) can be large -> one such edge can take minutes. The
# cache itself was computed edge-parallel, so individual slow edges were
# absorbed. We do the same: a process pool over the sampled edges, each worker
# computing BOTH the with-edge and without-edge PI for one edge. Workers hold
# the full graph + with-edge Ricci dict as globals (set once per worker via the
# initializer, not pickled per task).
_WG = None          # full training graph (edge present)
_RD_WITH = None     # with-edge Ricci dict (raw curvature, both directions)
_EGO_R = None       # ego radius for without-edge Ricci


def _init_edge_worker(graph, rd_with, ego_r):
    global _WG, _RD_WITH, _EGO_R
    _WG, _RD_WITH, _EGO_R = graph, rd_with, ego_r


def _edge_worker(args):
    """Compute with-edge and without-edge vicinity PI for one sampled edge."""
    i, u, v, has_edge = args
    g = _WG
    # WITH-edge PI (uses pre-computed global with-edge Ricci)
    pi_with, st_with, nvic_with = pi_for_edge(g, u, v, _RD_WITH, HOP)
    # WITHOUT-edge: build the edge-removed local ego ball, recompute its Ricci,
    # then the vicinity PI on g-minus-(u,v).  We mutate only a small local copy
    # (the ego ball), never the shared global graph.
    ego = nx.Graph(ego_subgraph(g, (u, v), _EGO_R))  # mutable local copy
    if ego.has_edge(u, v):
        ego.remove_edge(u, v)
    # without-edge vicinity (intersection of 1-hop nbrs in the edge-removed ball)
    vn_wo = vicinity_nodes(ego, u, v, HOP)
    vic_edges_wo = list(ego.subgraph(vn_wo).edges())
    if vic_edges_wo:
        orc = OllivierRicci(ego, alpha=0.5, method='Sinkhorn', proc=1,
                            verbose='ERROR')
        out = orc.compute_ricci_curvature_edges(edge_list=list(vic_edges_wo))
        rd_wo = {}
        for (a, b), c in out.items():
            rd_wo[(a, b)] = c
            rd_wo[(b, a)] = c
    else:
        rd_wo = {}
    pi_wo, st_wo, nvic_wo = pi_for_edge(ego, u, v, rd_wo, HOP)
    return i, u, v, has_edge, pi_with, st_with, nvic_with, pi_wo, st_wo, nvic_wo


def run_dataset(name, n_sample, seed, proc, outdir):
    t0 = time.time()
    print(f'\n=== {name}: leave-one-out PI ===', flush=True)
    data, g, train_pos = build_train_graph(name)
    print(f'  train graph: {g.number_of_nodes()} nodes, '
          f'{g.number_of_edges()} edges; #train_pos={len(train_pos)}',
          flush=True)

    # cached train_pos PI rows (cache layout starts with train_pos)
    cache_path = None
    for cand in (f'data/TLCGNN/{name}.npy', f'data/TLCGNN/{name.lower()}.npy'):
        if os.path.exists(cand):
            cache_path = cand
            break
    cache = np.load(cache_path) if cache_path else None
    print(f'  cache: {cache_path} shape={None if cache is None else cache.shape}',
          flush=True)

    rng = np.random.RandomState(seed)
    # sample distinct train_pos indices (these index both train_pos[] and the
    # first len(train_pos) cache rows, since train_pos is the first segment)
    n_sample = min(n_sample, len(train_pos))
    idx = rng.choice(len(train_pos), size=n_sample, replace=False)
    idx.sort()

    rows = []
    # Gather the WITH-edge vicinity edges across all sampled edges, so we can
    # batch their exact (global-graph) Ricci curvature in ONE OllivierRicci call.
    print('  gathering with-edge vicinity edges...', flush=True)
    with_edges_needed = set()
    edge_specs = []  # (i, u, v, has_edge)
    for i in idx:
        u, v = int(train_pos[i][0]), int(train_pos[i][1])
        has_edge = g.has_edge(u, v)  # train_pos edges are kept in g
        for a, b in g.subgraph(vicinity_nodes(g, u, v, HOP)).edges():
            with_edges_needed.add((a, b))
        edge_specs.append((i, u, v, has_edge))

    # WITH-edge curvature (exact, global graph) -- one batched call.
    print(f'  computing WITH-edge Ricci for {len(with_edges_needed)} vicinity '
          f'edges...', flush=True)
    rd_with = ricci_dict_for_edges(g, list(with_edges_needed), proc)

    # Per-edge PI in PARALLEL over the sampled edges (each worker computes both
    # with-edge and without-edge PI; the without-edge Ricci runs single-threaded
    # on the small edge-removed ego ball). This distributes the cost of slow
    # hub-edge vicinities across `proc` workers.
    print(f'  computing per-edge PI (with & without) in parallel '
          f'({proc} workers)...', flush=True)
    results = {}
    done = 0
    with NoDaemonPool(processes=proc, initializer=_init_edge_worker,
                      initargs=(g, rd_with, EGO_RADIUS)) as pool:
        for res in pool.imap_unordered(_edge_worker, edge_specs, chunksize=1):
            (i, u, v, has_edge, pi_with, st_with, nvic_with,
             pi_wo, st_wo, nvic_wo) = res
            results[i] = res
            done += 1
            if done % 25 == 0 or done == len(edge_specs):
                print(f'    {done}/{len(edge_specs)} '
                      f'({(time.time()-t0)/60:.1f} min)', flush=True)

    # Assemble rows in the sampled-index order.
    for i, u, v, has_edge in edge_specs:
        (_, _, _, _, pi_with, st_with, nvic_with,
         pi_wo, st_wo, nvic_wo) = results[i]
        l1_with = float(np.abs(pi_with).sum())
        l1_wo = float(np.abs(pi_wo).sum())
        l1_diff = float(np.abs(pi_with - pi_wo).sum())
        cache_l1 = (float(np.abs(cache[i]).sum())
                    if cache is not None and i < len(cache) else np.nan)
        cache_match_l1 = (float(np.abs(pi_with - cache[i]).sum())
                          if cache is not None and i < len(cache) else np.nan)
        rows.append(dict(
            idx=int(i), u=u, v=v, has_edge=int(has_edge),
            nvic_with=nvic_with, nvic_without=nvic_wo,
            l1_with=l1_with, l1_without=l1_wo, l1_with_minus_without=l1_diff,
            status_with=st_with, status_without=st_wo,
            cache_l1=cache_l1, recompute_vs_cache_l1=cache_match_l1,
        ))

    # ---- write per-edge CSV ----
    os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(outdir, f'loo_pi_{name}.csv')
    cols = ['idx', 'u', 'v', 'has_edge', 'nvic_with', 'nvic_without',
            'l1_with', 'l1_without', 'l1_with_minus_without',
            'status_with', 'status_without', 'cache_l1',
            'recompute_vs_cache_l1']
    with open(csv_path, 'w') as f:
        f.write(','.join(cols) + '\n')
        for r in rows:
            f.write(','.join(str(r[c]) for c in cols) + '\n')
    print(f'  wrote {csv_path}', flush=True)

    # ---- summary stats ----
    l1_with = np.array([r['l1_with'] for r in rows])
    l1_wo = np.array([r['l1_without'] for r in rows])
    # collapse threshold: PI L1 drops to <= 5% of its with-edge value
    # (and also report an absolute-threshold variant)
    nz_with = l1_with > 1e-6
    frac_collapse_rel = float(np.mean(
        (l1_wo <= 0.05 * np.maximum(l1_with, 1e-12))[nz_with])) if nz_with.any() else float('nan')
    frac_collapse_abs = float(np.mean(l1_wo < 1e-3))
    frac_near_zero_with = float(np.mean(~nz_with))
    cache_match = np.array([r['recompute_vs_cache_l1'] for r in rows
                            if not np.isnan(r['recompute_vs_cache_l1'])])

    summary = dict(
        name=name, n_sample=len(rows), seed=seed,
        mean_l1_with=float(l1_with.mean()), mean_l1_without=float(l1_wo.mean()),
        median_l1_with=float(np.median(l1_with)),
        median_l1_without=float(np.median(l1_wo)),
        mean_l1_with_minus_without=float(np.mean([r['l1_with_minus_without'] for r in rows])),
        frac_collapse_rel95=frac_collapse_rel,   # of edges that HAD signal, frac dropping >=95%
        frac_collapse_abs=frac_collapse_abs,      # frac whose without-edge L1 < 1e-3
        frac_with_zero=frac_near_zero_with,
        mean_recompute_vs_cache_l1=float(cache_match.mean()) if len(cache_match) else float('nan'),
        median_recompute_vs_cache_l1=float(np.median(cache_match)) if len(cache_match) else float('nan'),
        elapsed_min=(time.time() - t0) / 60,
    )
    return rows, summary, l1_with, l1_wo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=['Photo', 'Chameleon'])
    ap.add_argument('--n', type=int, default=150)
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--proc', type=int, default=16)
    ap.add_argument('--outdir', default='results/consistent_pi')
    args = ap.parse_args()

    all_summ = []
    all_l1 = {}
    for name in args.datasets:
        rows, summ, l1w, l1wo = run_dataset(name, args.n, args.seed, args.proc,
                                            args.outdir)
        all_summ.append(summ)
        all_l1[name] = (l1w, l1wo)
        print('  SUMMARY:', summ, flush=True)

    # ---- summary.txt ----
    os.makedirs(args.outdir, exist_ok=True)
    sp = os.path.join(args.outdir, 'summary.txt')
    with open(sp, 'w') as f:
        f.write('Leave-one-out PI control (results doc S14 centerpiece)\n')
        f.write('=' * 70 + '\n')
        f.write('Claim: exact-PI train_pos signal is a train-graph-membership\n')
        f.write('artifact. For each sampled train_pos edge (u,v) we compute its\n')
        f.write('vicinity PI WITH the (u,v) edge in the graph (train/cache\n')
        f.write('condition) vs WITHOUT it (held-out / test-like condition).\n')
        f.write('Identical PI math (sg2dgm hop=1 intersection, descriptor=sum,\n')
        f.write('resolution=5, weighted+normed); exact per-vicinity Ollivier-\n')
        f.write('Ricci curvature both conditions. Only the edge presence differs.\n\n')
        for s in all_summ:
            f.write(f"[{s['name']}]  (n={s['n_sample']}, seed={s['seed']})\n")
            f.write(f"  mean PI L1  WITH-edge    : {s['mean_l1_with']:.4f}\n")
            f.write(f"  mean PI L1  WITHOUT-edge : {s['mean_l1_without']:.4f}\n")
            f.write(f"  median PI L1 with / without: {s['median_l1_with']:.4f} / {s['median_l1_without']:.4f}\n")
            f.write(f"  mean L1(with - without)  : {s['mean_l1_with_minus_without']:.4f}\n")
            f.write(f"  collapse frac (>=95% drop, of edges w/ signal): {s['frac_collapse_rel95']:.3f}\n")
            f.write(f"  collapse frac (without-edge L1 < 1e-3)        : {s['frac_collapse_abs']:.3f}\n")
            f.write(f"  frac with-edge already ~0                     : {s['frac_with_zero']:.3f}\n")
            f.write(f"  recompute-vs-cache mean/median L1 (sanity)    : "
                    f"{s['mean_recompute_vs_cache_l1']:.4f} / {s['median_recompute_vs_cache_l1']:.4f}\n")
            verdict = ('MEMBERSHIP ARTIFACT CONFIRMED -- removing the edge '
                       'collapses train_pos PI'
                       if s['frac_collapse_rel95'] >= 0.8 or s['mean_l1_without'] <= 0.1 * s['mean_l1_with']
                       else 'NOT confirmed -- surrounding structure retains PI')
            f.write(f"  VERDICT: {verdict}\n\n")
        f.write('Cross-check: cached test_pos PI is ~0 for these datasets '
                '(see pi_artifact_analysis.py): exactly the without-edge regime.\n')
    print(f'wrote {sp}', flush=True)

    # ---- histogram PNG ----
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        nplt = len(all_l1)
        fig, axes = plt.subplots(1, nplt, figsize=(6 * nplt, 4.5), squeeze=False)
        for ax, (name, (l1w, l1wo)) in zip(axes[0], all_l1.items()):
            mx = max(float(l1w.max()), float(l1wo.max()), 1e-6)
            bins = np.linspace(0, mx, 40)
            ax.hist(l1w, bins=bins, alpha=0.6, label='WITH edge (train/cache)',
                    color='tab:blue')
            ax.hist(l1wo, bins=bins, alpha=0.6,
                    label='WITHOUT edge (held-out / test-like)',
                    color='tab:red')
            ax.set_title(f'{name}: train_pos vicinity PI L1\n'
                         f'mean {l1w.mean():.2f} -> {l1wo.mean():.2f}')
            ax.set_xlabel('PI L1 mass'); ax.set_ylabel('# train_pos edges')
            ax.legend()
        fig.tight_layout()
        pp = os.path.join(args.outdir, 'loo_pi_hist.png')
        fig.savefig(pp, dpi=130)
        print(f'wrote {pp}', flush=True)
    except Exception as ex:
        print(f'[hist skipped] {ex}', flush=True)


if __name__ == '__main__':
    main()
