import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Knowledge_Distillation.pdgnn_modern import PDGNN, PDGNNLayer


def test_forward_synthetic_chain():
    """5-node chain. PDGNN should run forward without error and return (E, 2)."""
    n = 5
    # chain: 0-1-2-3-4 (undirected)
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4],
                                [1, 0, 2, 1, 3, 2, 4, 3]], dtype=torch.long)
    filt = torch.arange(n, dtype=torch.float).view(-1, 1)
    model = PDGNN(hidden_dim=16, num_layers=2)
    out = model(filt, edge_index)
    assert out.shape == (edge_index.size(1), 2), f"expected ({edge_index.size(1)}, 2), got {out.shape}"


def test_backward_synthetic_chain():
    """Backward pass should produce gradients on all model params."""
    n = 5
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4],
                                [1, 0, 2, 1, 3, 2, 4, 3]], dtype=torch.long)
    filt = torch.arange(n, dtype=torch.float).view(-1, 1)
    target = torch.zeros(edge_index.size(1), 2)
    model = PDGNN(hidden_dim=16, num_layers=2)
    out = model(filt, edge_index)
    loss = (out - target).pow(2).mean()
    loss.backward()
    grads_ok = all(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in model.parameters())
    assert grads_ok, "some params have None or NaN gradient"
