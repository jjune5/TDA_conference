# tests/test_sbm_data.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np
from Knowledge_Distillation.sbm_data import generate_sbm, sbm_to_pyg


def test_generate_sbm_shape():
    """500-node, 5-block SBM should produce networkx graph with expected stats."""
    g = generate_sbm(n_per_block=100, n_blocks=5, p_in=0.1, p_out=0.01, seed=1234)
    assert g.number_of_nodes() == 500
    # density should be roughly p_in/5 + p_out*4/5 = 0.02 + 0.008 = 0.028
    density = g.number_of_edges() * 2 / (500 * 499)
    assert 0.01 < density < 0.06, f"density {density} out of range"


def test_sbm_to_pyg_data():
    """sbm_to_pyg returns a Data-like object with .x, .edge_index, .y."""
    g = generate_sbm(n_per_block=20, n_blocks=3, p_in=0.3, p_out=0.05, seed=1234)
    data = sbm_to_pyg(g, n_blocks=3, feat_dim=16)
    assert data.x.shape == (60, 16)
    assert data.edge_index.dim() == 2 and data.edge_index.size(0) == 2
    # symmetric (undirected)
    assert data.edge_index.size(1) == g.number_of_edges() * 2
    # labels per block
    assert data.y.shape == (60,) and data.y.max().item() == 2


def test_sbm_reproducible():
    """Same seed → same graph."""
    g1 = generate_sbm(n_per_block=50, n_blocks=4, p_in=0.2, p_out=0.05, seed=42)
    g2 = generate_sbm(n_per_block=50, n_blocks=4, p_in=0.2, p_out=0.05, seed=42)
    assert sorted(g1.edges()) == sorted(g2.edges())
