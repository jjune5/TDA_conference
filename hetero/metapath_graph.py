"""Heterogeneous graph -> meta-path induced weighted homogeneous graph.

A meta-path P = e1 e2 ... ek (a sequence of relation types that starts and ends
at the TARGET node type) induces a homogeneous graph over the target nodes whose
weighted adjacency is the product of the per-relation adjacency matrices
(A_e1 @ A_e2 @ ... ). Off-diagonal entry (i,j) = number of meta-path instances
connecting target nodes i and j (= relation strength). The diagonal (self
co-occurrence count) is removed. This is the hetero->homo step (CS224W lec09).
"""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
import networkx as nx
import torch

# canonical meta-paths per dataset (target node type first/last). Each entry is a
# list of edge_type triples in HGBDataset naming.
METAPATHS = {
    'ACM': {
        'PAP': [('paper', 'to', 'author'), ('author', 'to', 'paper')],
        'PSP': [('paper', 'to', 'subject'), ('subject', 'to', 'paper')],  # leak-audit
    },
    'DBLP': {
        # target = author
        'APA': [('author', 'to', 'paper'), ('paper', 'to', 'author')],
        'APCPA': [('author', 'to', 'paper'), ('paper', 'to', 'term'),
                  ('term', 'to', 'paper'), ('paper', 'to', 'author')],  # leak-audit
    },
}
TARGET = {'ACM': 'paper', 'DBLP': 'author'}


def load_hgb(name: str):
    from torch_geometric.datasets import HGBDataset
    return HGBDataset(root=f'./data/HGB_{name}', name=name)[0]


def _edge_adj(d, etype) -> sp.csr_matrix:
    """Sparse adjacency (src_count x dst_count) for one edge type."""
    src, _, dst = etype
    ei = d[etype].edge_index.numpy()
    n_src = int(d[src].num_nodes)
    n_dst = int(d[dst].num_nodes)
    data = np.ones(ei.shape[1], dtype=np.float64)
    return sp.csr_matrix((data, (ei[0], ei[1])), shape=(n_src, n_dst))


def compose_metapath_adj(adjs):
    """Multiply a list of sparse adjacencies left-to-right -> target x target."""
    W = adjs[0]
    for A in adjs[1:]:
        W = W @ A
    return W.tocsr()


def build_metapath_graph(d, metapath_name: str):
    """Return (nx weighted graph over target nodes, y (N,), masks dict)."""
    # infer dataset name from which METAPATHS table contains metapath_name
    ds_name = next(k for k, v in METAPATHS.items() if metapath_name in v)
    etypes = METAPATHS[ds_name][metapath_name]
    adjs = [_edge_adj(d, et) for et in etypes]
    W = compose_metapath_adj(adjs)
    W = W.tocoo()
    tgt = TARGET[ds_name]
    n = int(d[tgt].num_nodes)
    g = nx.Graph()
    g.add_nodes_from(range(n))
    for i, j, w in zip(W.row, W.col, W.data):
        if i < j and w > 0:                  # upper triangle, drop diagonal
            g.add_edge(int(i), int(j), weight=float(w))
    y = d[tgt].y.numpy()
    masks = {k: getattr(d[tgt], f'{k}_mask').numpy()
             for k in ('train', 'val', 'test') if hasattr(d[tgt], f'{k}_mask')}
    # HGB ACM has train/test masks; synthesize val from train tail if absent
    if 'val' not in masks:
        tr = masks['train'].copy()
        idx = np.where(tr)[0]
        cut = idx[int(0.85 * len(idx)):]
        masks['val'] = np.zeros_like(tr); masks['val'][cut] = True
        masks['train'][cut] = False
    return g, y, masks
