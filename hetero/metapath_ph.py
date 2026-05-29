"""Per-target-node exact persistence image on a meta-path graph.

For each target node v: take its k-hop ego-graph in the (weighted) meta-path
graph, use weighted node degree as the sublevel filter, compute exact extended
persistence (gudhi) and vectorize to a 5x5=25-dim persistence image. Reuses the
DNP machinery in node_ph_features.py (_ego_sublevel_pi, PI_RES). Exact-first:
prove the signal is genuine before approximating with PDGNN (Phase 2).
"""
from __future__ import annotations
import os, sys
import numpy as np
import networkx as nx
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from node_ph_features import _ego_sublevel_pi, PI_RES   # reuse DNP helpers
from sg2dgm import PersistenceImager as pimg_mod


def metapath_node_pi(g: nx.Graph, hop: int = 1, max_nodes: int = 200,
                     verbose: bool = False) -> np.ndarray:
    """(N, 25) per-node exact PI on the meta-path graph; filter = weighted degree."""
    imager = pimg_mod.PersistenceImager(resolution=PI_RES)
    n = g.number_of_nodes()
    # weighted degree as the node filter value (sum of incident meta-path counts)
    wdeg = {u: float(sum(dd['weight'] for _, dd in g[u].items())) for u in g.nodes()}
    out = np.zeros((n, PI_RES * PI_RES), dtype=np.float64)
    for v in range(n):
        out[v] = _ego_sublevel_pi(g, v, hop, wdeg, imager, max_nodes)
        if verbose and (v + 1) % 1000 == 0:
            print(f'    metapath_pi {v+1}/{n}')
    return out
