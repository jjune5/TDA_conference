"""Phase-3 tests for typed_cycle PROXY descriptors.

These are *proxy* descriptors that try to preserve relation/type semantics that an
untyped persistence diagram would lose. They are explicitly NOT exact persistent
homology, NOT real PDGNN/EPD. Tests check shape, determinism, finiteness,
missing-meta-path handling, and the combine helper.
"""
import networkx as nx
import numpy as np
import torch
import pytest

from hetero_pdg.data import make_toy_hetero_graph, PairBatch
from hetero_pdg.metapath import (
    DEFAULT_METAPATHS,
    project_all_metapaths,
    ProjectedGraphBundle,
)
from hetero_pdg.typed_cycle import (
    TYPED_CYCLE_BACKEND,
    typed_cycle_dim,
    typed_cycle_descriptor_names,
    typed_cycle_signature_features,
    combine_topology_with_typed_cycle_signature,
)


def _toy():
    d = make_toy_hetero_graph(seed=0)
    bundles = project_all_metapaths(
        d, [DEFAULT_METAPATHS["PFP"], DEFAULT_METAPATHS["PCP"]]
    )
    pb = PairBatch(
        torch.tensor([0, 1, 2, 3]),
        torch.tensor([1, 2, 3, 4]),
        torch.ones(4),
        "paper",
        "paper",
    )
    return d, bundles, pb


def test_backend_label_is_proxy_not_exact():
    # honesty: never claim exact PH / real PDGNN / EPD in the constant name
    low = TYPED_CYCLE_BACKEND.lower()
    assert "proxy" in low
    assert "epd" not in low
    assert "pdgnn" not in low
    assert "persistent" not in low


def test_dim_and_names_agree():
    names = typed_cycle_descriptor_names(["PFP", "PCP"], include_density=True)
    assert len(names) == typed_cycle_dim(2, include_density=True)
    # node-type + edge-type histograms must be present
    assert any(n.startswith("node_type_hist") for n in names)
    assert any(n.startswith("edge_type_hist") for n in names)
    # per-meta-path channels present
    assert any("PFP" in n for n in names)
    assert any("PCP" in n for n in names)


def test_output_shape():
    d, bundles, pb = _toy()
    feats = typed_cycle_signature_features(d, pb, bundles, k=1)
    assert feats.shape == (4, typed_cycle_dim(len(bundles), include_density=True))
    assert feats.dtype == torch.float32


def test_deterministic_two_runs_equal():
    d, bundles, pb = _toy()
    a = typed_cycle_signature_features(d, pb, bundles, k=1)
    b = typed_cycle_signature_features(d, pb, bundles, k=1)
    assert torch.equal(a, b)


def test_all_finite():
    d, bundles, pb = _toy()
    feats = typed_cycle_signature_features(d, pb, bundles, k=2)
    assert torch.isfinite(feats).all()


def test_missing_metapath_pair_is_zero_block_and_mask_false():
    # build a projected bundle whose endpoints 0 and 1 are isolated -> zero block
    d = make_toy_hetero_graph(seed=0)
    n_paper = int(d["paper"].num_nodes)
    g = nx.Graph()
    g.add_nodes_from(range(n_paper))  # everyone isolated
    iso = ProjectedGraphBundle("ISO", "paper", n_paper, None, g)
    real = project_all_metapaths(d, [DEFAULT_METAPATHS["PCP"]])["PCP"]
    bundles = {"ISO": iso, "PCP": real}
    pb = PairBatch(
        torch.tensor([0]), torch.tensor([1]), torch.ones(1), "paper", "paper"
    )
    feats, mask = typed_cycle_signature_features(
        d, pb, bundles, k=1, return_mask=True
    )
    assert feats.shape == (1, typed_cycle_dim(2, include_density=True))
    assert torch.isfinite(feats).all()
    # ISO channel must be masked invalid (its per-meta-path block is zero)
    assert mask.shape == (1, 2)
    iso_col = list(bundles).index("ISO")
    assert bool(mask[0, iso_col]) is False


def test_out_of_target_type_pair_does_not_crash():
    # author/paper pair: paper-targeted meta-paths can't apply -> zero per-mp block
    d, bundles, _ = _toy()
    pb = PairBatch(
        torch.tensor([0, 1]), torch.tensor([0, 1]), torch.ones(2), "author", "paper"
    )
    feats = typed_cycle_signature_features(d, pb, bundles, k=1)
    assert feats.shape == (2, typed_cycle_dim(len(bundles), include_density=True))
    assert torch.isfinite(feats).all()


def test_combine_concatenates_right_shape():
    d, bundles, pb = _toy()
    tc = typed_cycle_signature_features(d, pb, bundles, k=1)
    topo = torch.randn(4, 6)
    combined = combine_topology_with_typed_cycle_signature(topo, tc)
    assert combined.shape == (4, 6 + tc.shape[1])
    assert torch.allclose(combined[:, :6], topo)
    assert torch.allclose(combined[:, 6:], tc)


def test_combine_batch_mismatch_raises():
    d, bundles, pb = _toy()
    tc = typed_cycle_signature_features(d, pb, bundles, k=1)
    with pytest.raises(ValueError):
        combine_topology_with_typed_cycle_signature(torch.randn(3, 6), tc)
