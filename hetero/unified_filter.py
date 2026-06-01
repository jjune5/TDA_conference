"""Idea 2: unified type-aware filtration on the WHOLE heterogeneous graph (PDGNN-EPD).

Faithful core of the Notion 'Idea 2'. Instead of collapsing to one type via meta-paths
(Idea 1), keep ALL node types in a single graph and build a UNIFIED filtration whose
per-node filter is (2-3) CALIBRATED across types by within-type quantile normalization,
so cross-type ordering is comparable (PH stability theorem: monotone within-type ->
relative order preserved). PDGNN then predicts each TARGET node's local EPD under this
unified multi-type filtration.

Notes / honest scope:
- The fully end-to-end LEARNABLE type-MLP filter (2-1) is a further extension; here the
  per-node base value is multi-scale HKS on the homogeneous graph, then within-type
  quantile-calibrated -- a fixed, deterministic, faithful proxy for the type-aware filter.
- The only change vs Idea 1 is the SOURCE of the topological feature (whole multi-type
  graph ego, type-calibrated) instead of a collapsed meta-path subgraph. Backbone GCN and
  controls are identical, isolating Idea 2's contribution.
"""
from __future__ import annotations
import os, sys
import numpy as np
import networkx as nx
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hetero.metapath_graph import TARGET


def build_homo(d, dataset: str):
    """HeteroData -> (nx.Graph over ALL nodes (target type first, global-indexed),
    ntype (Ntotal,) int type id, n_target, y(target), masks(target)).
    Target type is placed FIRST so target globals = arange(n_target)."""
    tgt = TARGET[dataset]
    node_types = [tgt] + [t for t in d.node_types if t != tgt]
    offset, off = {}, 0
    for t in node_types:
        offset[t] = off
        off += int(d[t].num_nodes)
    Ntot = off
    ntype = np.zeros(Ntot, dtype=np.int64)
    for ti, t in enumerate(node_types):
        ntype[offset[t]:offset[t] + int(d[t].num_nodes)] = ti
    g = nx.Graph()
    g.add_nodes_from(range(Ntot))
    for et in d.edge_types:
        s, _, t = et
        ei = d[et].edge_index.numpy()
        us = ei[0] + offset[s]
        vs = ei[1] + offset[t]
        g.add_edges_from(zip(us.tolist(), vs.tolist()))
    n_target = int(d[tgt].num_nodes)
    y = d[tgt].y.numpy()
    masks = {k: getattr(d[tgt], f'{k}_mask').numpy()
             for k in ('train', 'val', 'test') if hasattr(d[tgt], f'{k}_mask')}
    if 'val' not in masks:
        tr = masks['train'].copy(); idx = np.where(tr)[0]; cut = idx[int(0.85 * len(idx)):]
        masks['val'] = np.zeros_like(tr); masks['val'][cut] = True; masks['train'][cut] = False
    return g, ntype, n_target, y, masks


def calibrated_filter(g: nx.Graph, ntype: np.ndarray, K: int = 3) -> np.ndarray:
    """(Ntotal, K) within-type quantile-calibrated multi-scale HKS on the homo graph.
    HKS gives a rich per-node base value; quantile-normalizing WITHIN each node type
    makes cross-type filter values comparable (Idea 2's calibration)."""
    from hetero.pdgnn_metapath import _graph_hks
    hks = _graph_hks(g, K)                                  # (Ntotal, K) raw HKS
    cal = np.zeros_like(hks)
    for t in np.unique(ntype):
        idx = np.where(ntype == t)[0]
        for k in range(K):
            v = hks[idx, k]
            order = v.argsort()
            ranks = np.empty(len(v)); ranks[order] = np.arange(len(v))
            cal[idx, k] = ranks / max(len(v) - 1, 1)        # quantile -> [0,1] within type
    return cal
