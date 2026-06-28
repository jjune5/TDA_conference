"""Phase-1 RED tests: HeteroTopoLinkPredictor forward for all 5 topology modes
+ attention masking / normalisation."""
import torch
import pytest

from hetero_pdg.data import PairBatch
from hetero_pdg.models import HeteroTopoLinkPredictor

NODE_DIMS = {"paper": 12}
B, M, TD, TH, UD = 6, 2, 6, 16, 5


def _nf():
    return {"paper": torch.randn(10, 12)}


def _pb():
    return PairBatch(torch.randint(0, 10, (B,)), torch.randint(0, 10, (B,)),
                     torch.ones(B), "paper", "paper")


def test_modes_constant():
    assert set(HeteroTopoLinkPredictor.MODES) == {
        "no_topology", "collapsed_topology", "metapath_topology_concat",
        "metapath_topology_attention", "unified_filter_topology"}


def test_invalid_mode_raises():
    with pytest.raises(ValueError):
        HeteroTopoLinkPredictor(NODE_DIMS, mode="bogus")


@pytest.mark.parametrize("mode", ["no_topology", "collapsed_topology",
                                  "unified_filter_topology"])
def test_forward_single_or_none_topo(mode):
    m = HeteroTopoLinkPredictor(NODE_DIMS, mode=mode, hidden_dim=16,
                                topo_dim=TD, unified_dim=UD)
    if mode == "no_topology":
        topo = None
    elif mode == "unified_filter_topology":
        topo = torch.randn(B, UD)
    else:
        topo = torch.randn(B, TD)
    out = m(_nf(), _pb(), topo=topo)
    assert out.shape == (B,) and torch.isfinite(out).all()


@pytest.mark.parametrize("mode", ["metapath_topology_concat",
                                  "metapath_topology_attention"])
def test_forward_metapath_modes(mode):
    m = HeteroTopoLinkPredictor(NODE_DIMS, mode=mode, hidden_dim=16, topo_dim=TD,
                                topo_hidden_dim=TH, n_metapaths=M)
    topo = torch.randn(B, M, TD)
    mask = torch.ones(B, M, dtype=torch.bool)
    out = m(_nf(), _pb(), topo=topo, topo_mask=mask)
    assert out.shape == (B,) and torch.isfinite(out).all()


def test_attention_weights_sum_to_one():
    m = HeteroTopoLinkPredictor(NODE_DIMS, mode="metapath_topology_attention",
                                hidden_dim=16, topo_dim=TD, topo_hidden_dim=TH, n_metapaths=M)
    w = m.compute_attention(torch.randn(B, M, TD), torch.ones(B, M, dtype=torch.bool))
    assert w.shape == (B, M)
    assert torch.allclose(w.sum(-1), torch.ones(B), atol=1e-5)


def test_attention_masks_missing_metapath():
    m = HeteroTopoLinkPredictor(NODE_DIMS, mode="metapath_topology_attention",
                                hidden_dim=16, topo_dim=TD, topo_hidden_dim=TH, n_metapaths=M)
    mask = torch.ones(B, M, dtype=torch.bool); mask[:, 1] = False
    w = m.compute_attention(torch.randn(B, M, TD), mask)
    assert torch.allclose(w[:, 1], torch.zeros(B), atol=1e-6)
    assert torch.allclose(w.sum(-1), torch.ones(B), atol=1e-5)
