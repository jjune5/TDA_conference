"""Smoke tests for hetero/chromatic_pdgnn.py (shapes, forward/backward, train, predict)."""
import numpy as np
import torch
import networkx as nx
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hetero.chromatic_pdgnn import (ChromaticPDGNN, gen_chromatic_samples,
                                    train_chromatic_pdgnn, predict_node_chromatic_pi)


def test_forward_shapes_and_backward():
    torch.manual_seed(0)
    m, E = 6, 8
    filt = torch.rand(m, 1)
    color = torch.randint(0, 2, (m, 1)).float()
    ei = torch.randint(0, m, (2, E))
    model = ChromaticPDGNN(hidden_dim=16, num_layers=2)
    out = model(filt, color, ei)
    assert set(out.keys()) == {'image', 'kernel', 'cokernel'}
    for leg in out.values():
        assert leg.shape == (E, 2)
    loss = sum(v.sum() for v in out.values())
    loss.backward()
    assert next(model.parameters()).grad is not None


def test_train_and_predict_tiny():
    g = nx.Graph(); g.add_edges_from([(0, 2), (1, 2), (3, 2)])
    ntype = np.array([0, 0, 1, 0]); K = 2
    hks = np.random.RandomState(0).rand(4, K).astype(np.float32)
    samples = gen_chromatic_samples(g, hks, ntype, 0, hop=2, max_nodes=20, n_samples=4, seed=0)
    assert len(samples) >= 1
    model = train_chromatic_pdgnn(samples, hidden=16, layers=2, epochs=5, verbose=False)
    pi = predict_node_chromatic_pi(model, g, hks, ntype, 0, hop=2, max_nodes=20)
    assert pi.shape == (4, 3 * 25 * K)
