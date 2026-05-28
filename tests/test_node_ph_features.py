import os, sys
import numpy as np
import networkx as nx
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_ego_sublevel_pi_triangle_is_finite_25vec():
    """A triangle (3-cycle) with a non-constant node filter yields a finite (25,) PI."""
    from node_ph_features import _ego_sublevel_pi
    from sg2dgm import PersistenceImager as pimg_mod
    imager = pimg_mod.PersistenceImager(resolution=5)
    g = nx.cycle_graph(3)                      # nodes 0,1,2 ; edges (0,1),(1,2),(2,0)
    node_filt = {0: 0.0, 1: 0.5, 2: 1.0}
    pi = _ego_sublevel_pi(g, center=0, hop=2, node_filt=node_filt,
                          imager=imager, max_nodes=200)
    assert pi.shape == (25,)
    assert np.all(np.isfinite(pi))
    assert pi.sum() >= 0.0                     # PI is non-negative


def test_phi_A_shape_and_nonzero_on_synthetic():
    """phi_A returns (N, 25*K) and is not all-zero on a connected graph."""
    import torch
    from types import SimpleNamespace
    from node_ph_features import phi_A
    # 2 triangles joined by an edge -> 5 nodes
    edges = [(0,1),(1,2),(2,0),(2,3),(3,4),(4,2)]
    ei = torch.tensor([[a for a,b in edges]+[b for a,b in edges],
                       [b for a,b in edges]+[a for a,b in edges]], dtype=torch.long)
    data = SimpleNamespace(edge_index=ei, num_nodes=5)
    phi = phi_A(data, K=3, hop=2, max_nodes=200)
    assert phi.shape == (5, 75)                # 25 * K(=3)
    assert np.isfinite(phi).all()
    assert phi.sum() > 0.0
