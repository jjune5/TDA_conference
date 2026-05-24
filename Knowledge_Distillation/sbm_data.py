# Knowledge_Distillation/sbm_data.py
"""Stochastic Block Model graph generator for density × heterophily sweep.

Generates synthetic graphs where we control:
- density (via p_in, p_out)
- heterophily (via p_out / (p_in + p_out) ratio)

The full graph and a PyG-compatible Data wrapper are exposed for downstream
pipelines.py / pipelines.SBM_<config>.
"""
from __future__ import annotations
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data


def generate_sbm(n_per_block: int, n_blocks: int, p_in: float, p_out: float,
                 seed: int = 1234) -> nx.Graph:
    """Generate an SBM graph with n_blocks communities of n_per_block nodes.

    p_in: probability of edge within a block (community).
    p_out: probability of edge between blocks (heterophilic).
    Returns: undirected networkx Graph with node attribute 'block'.
    """
    rng = np.random.RandomState(seed)
    sizes = [n_per_block] * n_blocks
    probs = [[p_in if i == j else p_out for j in range(n_blocks)] for i in range(n_blocks)]
    g = nx.stochastic_block_model(sizes, probs, seed=seed)
    return g


def sbm_to_pyg(g: nx.Graph, n_blocks: int, feat_dim: int = 16,
               seed: int = 1234) -> Data:
    """Wrap SBM graph as PyG Data object.

    Features: random gaussian per node (deterministic via seed).
    Labels: block id.
    """
    rng = np.random.RandomState(seed)
    n = g.number_of_nodes()
    # features: random gaussian
    x = torch.from_numpy(rng.randn(n, feat_dim).astype(np.float32))
    # labels from networkx node attribute 'block'
    blocks = nx.get_node_attributes(g, 'block')
    y = torch.tensor([blocks[i] for i in range(n)], dtype=torch.long)
    # symmetric edge_index
    ei = np.array(list(g.edges()), dtype=np.int64).T  # (2, E)
    if ei.size:
        ei = np.concatenate([ei, ei[[1, 0]]], axis=1)
    else:
        ei = np.zeros((2, 0), dtype=np.int64)
    edge_index = torch.from_numpy(ei).long()
    return Data(x=x, edge_index=edge_index, y=y, num_classes=n_blocks)
