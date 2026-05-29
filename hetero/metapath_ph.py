"""Per-target-node exact persistence image on a meta-path graph.

For each target node v: take its k-hop ego-graph in the meta-path graph and compute
exact extended persistence (gudhi), vectorized to a 5x5 persistence image. Two node
filters:
  - 'hks' (default): multi-scale Heat-Kernel-Signature filter (DNP variant A,
    node_ph_features.phi_A). HKS varies smoothly per node -> DISTINCT per-node PI
    (verified: 1538/3025 distinct on ACM PAP, vs only 32/3025 for degree). Output
    (N, 25*K).
  - 'degree': weighted-degree filter (coarse; near-constant PI -> kept only as a
    control/baseline). Output (N, 25).
Exact-first: prove the signal is genuine before approximating with PDGNN (Phase 2).
"""
from __future__ import annotations
import os, sys
import numpy as np
import networkx as nx
import torch
from types import SimpleNamespace
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from node_ph_features import _ego_sublevel_pi, PI_RES, phi_A   # reuse DNP helpers
from sg2dgm import PersistenceImager as pimg_mod


def _graph_to_data(g: nx.Graph):
    """networkx graph -> SimpleNamespace(edge_index, num_nodes) for phi_A."""
    n = g.number_of_nodes()
    ei = np.array(list(g.edges())).T if g.number_of_edges() else np.zeros((2, 0), int)
    if ei.size:
        ei = np.concatenate([ei, ei[[1, 0]]], axis=1)
    return SimpleNamespace(edge_index=torch.tensor(ei, dtype=torch.long), num_nodes=n)


def metapath_node_pi(g: nx.Graph, filter: str = 'hks', K: int = 3, hop: int = 1,
                     max_nodes: int = 200, verbose: bool = False) -> np.ndarray:
    """(N, 25*K) HKS-filtered, or (N, 25) degree-filtered, per-node exact PI."""
    if filter == 'hks':
        return phi_A(_graph_to_data(g), K=K, hop=hop, max_nodes=max_nodes, verbose=verbose)
    elif filter == 'degree':
        imager = pimg_mod.PersistenceImager(resolution=PI_RES)
        n = g.number_of_nodes()
        wdeg = {u: float(sum(dd['weight'] for _, dd in g[u].items())) for u in g.nodes()}
        out = np.zeros((n, PI_RES * PI_RES), dtype=np.float64)
        for v in range(n):
            out[v] = _ego_sublevel_pi(g, v, hop, wdeg, imager, max_nodes)
            if verbose and (v + 1) % 1000 == 0:
                print(f'    metapath_pi(deg) {v+1}/{n}')
        return out
    raise ValueError(filter)


def random_filter_node_pi(g: nx.Graph, K: int = 3, hop: int = 1,
                          max_nodes: int = 200, seed: int = 0) -> np.ndarray:
    """(N, 25*K) control: K random per-node scalar filters -> sublevel PI.
    Tests whether the HKS (structure-derived) filter beats a meaningless filter."""
    rng = np.random.RandomState(seed)
    imager = pimg_mod.PersistenceImager(resolution=PI_RES)
    n = g.number_of_nodes()
    out = np.zeros((n, PI_RES * PI_RES * K), dtype=np.float64)
    for k in range(K):
        rfilt = {u: float(rng.rand()) for u in g.nodes()}
        for v in range(n):
            out[v, k * 25:(k + 1) * 25] = _ego_sublevel_pi(g, v, hop, rfilt, imager, max_nodes)
    return out
