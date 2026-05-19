"""Parallel persistence-image computation using multiprocessing (process-based).

The original `graph2pi.get_pimg_for_all_edges` used `multiprocessing.dummy`
(ThreadPool) which is GIL-bound — so 16 "cores" gave at best 1.5–3× speedup.
The PI computation is pure-Python (networkx, dict, set), so true process pools
give roughly Nx speedup.

This module exposes `compute_pi_parallel(graph, ricci_curv, edges, hop, ...)`
that uses `multiprocessing.Pool` with a worker initializer holding the graph
and curvature dict as globals (pickled once per worker, not per task).
"""

from __future__ import annotations
import os
import time
import numpy as np
import networkx as nx
from multiprocessing import Pool

# Lazy local imports to keep this module light at import time
from sg2dgm.accelerated_PD import (perturb_filter_function, Union_find,
                                    Accelerate_PD)
from sg2dgm.riccidist2dgm import filtration as _filtration
import sg2dgm.PersistenceImager as _pimg


# Worker-global state set by `_init_worker`.
_W_GRAPH: nx.Graph | None = None
_W_RICCI: dict | None = None
_W_DICT_NODE: dict | None = None


def _init_worker(graph: nx.Graph, ricci_curv: dict, dict_node: dict) -> None:
    global _W_GRAPH, _W_RICCI, _W_DICT_NODE
    _W_GRAPH = graph
    _W_RICCI = ricci_curv
    _W_DICT_NODE = dict_node


def _compute_one(args) -> tuple[int, np.ndarray]:
    """Compute the persistence image for a single (u, v) edge."""
    cnt, u, v, hop, norm, extended_flag, resolution, descriptor = args
    if u < 0 or v < 0:
        # sentinel for missing dict_node lookup (isolated node)
        return cnt, np.zeros(resolution * resolution)
    try:
        ru = _W_DICT_NODE[u]
        rv = _W_DICT_NODE[v]
        # Intersection of hop-neighborhoods of u and v.
        nodes_u = [ru] + [x for _, x in nx.bfs_edges(_W_GRAPH, ru, depth_limit=hop)]
        nodes_v = [rv] + [x for _, x in nx.bfs_edges(_W_GRAPH, rv, depth_limit=hop)]
        nodes = list(set(nodes_u) & set(nodes_v))
        if len(nodes) == 0:
            return cnt, np.zeros(resolution * resolution)
        subgraph = _W_GRAPH.subgraph(nodes)
        if not nx.is_connected(subgraph):
            # The original code only handles the connected case.
            return cnt, np.zeros(resolution * resolution)
        fil = _filtration(subgraph, ru, rv, hop, ricci_curv=_W_RICCI)
        g = fil.build_fv(weight_graph=True, norm=norm)
        sf = perturb_filter_function(g, descriptor=descriptor)
        PD0, Pos_edges, Neg_edges = Union_find(sf)
        if extended_flag:
            PD1 = Accelerate_PD(Pos_edges, Neg_edges, sf)
        else:
            PD1 = []
        pers_imager = _pimg.PersistenceImager(resolution=resolution)
        diag = np.array(list(PD0) + list(PD1))
        pi = pers_imager.transform(diag).reshape(-1)
        return cnt, pi
    except BaseException:
        return cnt, np.zeros(resolution * resolution)


def compute_pi_parallel(graph: nx.Graph, ricci_curv_list: list,
                        dict_node: dict, edges: np.ndarray, *, hop: int = 1,
                        norm: bool = True, extended_flag: bool = True,
                        resolution: int = 5, descriptor: str = 'sum',
                        n_workers: int | None = None,
                        chunksize: int = 32) -> np.ndarray:
    """Compute PI for every edge in `edges` (shape (E, 2)) in parallel.

    `ricci_curv_list` is the list-of-triples format returned by
    `loaddatas.compute_ricci_curvature`. We convert it once to a dict keyed by
    relabeled (u, v) using `dict_node` (orig -> relabeled).
    """
    n_workers = n_workers or max(1, os.cpu_count() // 2)
    print(f'[parallel_pi] computing PI for {len(edges)} edges with {n_workers} workers, hop={hop}', flush=True)

    # Build relabeled ricci_curv dict expected by `filtration.build_fv`
    rcd: dict = {}
    for (u, v, c) in ricci_curv_list:
        ru, rv = dict_node[u], dict_node[v]
        rcd[(ru, rv)] = c
        rcd[(rv, ru)] = c

    # Make graph weighted with (curvature + 1)
    g_weighted = graph.copy()
    for (u, v) in g_weighted.edges():
        c = rcd.get((u, v), 0.0)
        g_weighted[u][v]['weight'] = c + 1.0

    pi_sg = np.zeros((len(edges), resolution * resolution))
    t0 = time.time()
    params = [(cnt, int(e[0]), int(e[1]), hop, norm, extended_flag, resolution, descriptor)
              for cnt, e in enumerate(edges)]

    with Pool(processes=n_workers, initializer=_init_worker,
              initargs=(g_weighted, rcd, dict_node)) as pool:
        done = 0
        log_every = max(1, len(params) // 50)
        for cnt, pi in pool.imap_unordered(_compute_one, params,
                                            chunksize=chunksize):
            pi_sg[cnt] = pi
            done += 1
            if done % log_every == 0:
                elapsed = time.time() - t0
                rate = done / max(elapsed, 1e-6)
                eta = (len(params) - done) / max(rate, 1e-6)
                print(f'[parallel_pi] {done}/{len(params)} '
                      f'({100*done/len(params):.1f}%) rate={rate:.1f} edges/s '
                      f'eta={eta/60:.1f} min', flush=True)
    print(f'[parallel_pi] done in {(time.time()-t0)/60:.1f} min', flush=True)
    return pi_sg
