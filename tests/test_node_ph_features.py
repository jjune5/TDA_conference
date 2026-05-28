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


def test_phi_C_shape_and_finite_on_synthetic():
    import torch
    from types import SimpleNamespace
    from node_ph_features import phi_C
    edges = [(0,1),(1,2),(2,0),(2,3),(3,4),(4,2)]
    ei = torch.tensor([[a for a,b in edges]+[b for a,b in edges],
                       [b for a,b in edges]+[a for a,b in edges]], dtype=torch.long)
    data = SimpleNamespace(edge_index=ei, num_nodes=5)
    phi = phi_C(data, hop=2, max_nodes=200)
    assert phi.shape == (5, 25)
    assert np.isfinite(phi).all()


def test_phi_A_does_not_collapse_on_edge_removal():
    """Removing ONE edge incident to a node changes its phi_A only slightly
    (the §14 fix: per-node ego features are robust to single-edge deletion).

    Uses a non-regular (Barabasi-Albert) graph — the realistic regime. NOTE: on
    *regular* graphs phi_A is unstable, but that is an eigenvector-degeneracy
    pathology of HKS (repeated eigenvalues -> ill-defined eigenvectors), not a
    property of real benchmark graphs (Cora/Chameleon are non-regular, 0
    degenerate eigenvalues). Measured rel on BA graphs: ~0.002-0.013."""
    import torch
    from types import SimpleNamespace
    from node_ph_features import phi_A
    g = nx.barabasi_albert_graph(80, 3, seed=1)         # non-regular, realistic degrees
    v = max(g.degree, key=lambda x: x[1])[0]            # hub: most incident edges (worst case)
    e0 = next(iter(g.edges(v)))
    def mk(gg):
        es = list(gg.edges())
        ei = torch.tensor([[a for a,b in es]+[b for a,b in es],
                           [b for a,b in es]+[a for a,b in es]], dtype=torch.long)
        return SimpleNamespace(edge_index=ei, num_nodes=gg.number_of_nodes())
    g2 = g.copy(); g2.remove_edge(*e0)
    phi_full = phi_A(mk(g), K=3, hop=2, max_nodes=300)[v]
    phi_drop = phi_A(mk(g2), K=3, hop=2, max_nodes=300)[v]
    # feature stays nonzero (NOT a collapse-to-zero like vicinity-PI) AND stable
    assert phi_full.sum() > 0 and phi_drop.sum() > 0
    rel = np.linalg.norm(phi_full - phi_drop) / (np.linalg.norm(phi_full) + 1e-9)
    assert rel < 0.3, f'phi_A changed too much on single-edge removal: rel={rel:.3f}'
