# tests/test_mol_data.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from Knowledge_Distillation.mol_data import load_tudataset, graph_to_pi


def test_load_mutag():
    """MUTAG has 188 graphs, binary labels."""
    graphs, labels = load_tudataset('MUTAG')
    assert len(graphs) == 188
    assert set(labels.tolist()) <= {0, 1}
    import networkx as nx
    assert isinstance(graphs[0], nx.Graph)


def test_graph_to_pi_shape():
    """graph_to_pi returns a flat 25-dim vector per graph."""
    import networkx as nx
    g = nx.path_graph(6)
    pi = graph_to_pi(g)
    assert pi.shape == (25,), f"expected (25,), got {pi.shape}"
    assert np.isfinite(pi).all()
