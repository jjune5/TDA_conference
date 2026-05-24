# tests/test_adaptive_gating.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np
from baselines.TLCGNN_gated import GatingNet, gated_decode


def test_gating_net_output_shape():
    """GatingNet takes per-edge features and outputs gate in [0,1]."""
    n_edges = 10
    feat_dim = 3
    gnet = GatingNet(in_dim=feat_dim, hidden=16)
    edge_feats = torch.randn(n_edges, feat_dim)
    gates = gnet(edge_feats)
    assert gates.shape == (n_edges,), f"expected ({n_edges},), got {gates.shape}"
    assert (gates >= 0).all() and (gates <= 1).all(), "gates out of [0,1]"


def test_gating_net_extremes():
    """Train a tiny gate to output 1 for one input and 0 for another → learnable."""
    torch.manual_seed(0)
    gnet = GatingNet(in_dim=1, hidden=8)
    opt = torch.optim.Adam(gnet.parameters(), lr=0.05)
    pos = torch.tensor([[1.0]] * 4)  # should → 1
    neg = torch.tensor([[-1.0]] * 4)  # should → 0
    for _ in range(200):
        opt.zero_grad()
        loss = (1 - gnet(pos)).mean() + gnet(neg).mean()
        loss.backward()
        opt.step()
    assert gnet(pos).mean() > 0.7, f"pos gate {gnet(pos).mean()} should be >0.7"
    assert gnet(neg).mean() < 0.3, f"neg gate {gnet(neg).mean()} should be <0.3"


def test_gated_decode_zero_gate_eq_no_pi():
    """When gate=0, gated_decode should equal a no-PI decode (PI contribution zeroed)."""
    torch.manual_seed(0)
    from baselines.TLCGNN_gated import gated_decode
    sqdist = torch.randn(5, 16)
    PI = torch.randn(5, 25)
    gates_zero = torch.zeros(5)
    gates_one = torch.ones(5)
    # With gate=0, the PI part of concat should be zero
    feat_zero = gated_decode(sqdist, PI, gates_zero)
    feat_one = gated_decode(sqdist, PI, gates_one)
    # PI columns (16:41) should be 0 for zero gate
    assert torch.allclose(feat_zero[:, 16:], torch.zeros(5, 25))
    # PI columns should equal PI for gate=1
    assert torch.allclose(feat_one[:, 16:], PI)
