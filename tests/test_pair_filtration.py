"""RED tests for the experimental pair-conditioned filtration module.

These exercise the public API of ``hetero_pdg.pair_filtration``:
  * per-node pair-conditioned filtration values (one scalar per target node);
  * a batched helper that turns a PairBatch into an LP topology tensor [B, feat_dim];
  * output shape, finiteness, determinism under a fixed seed, and -- the key
    property of this module -- PAIR-DEPENDENCE (different (u,v) -> different values).

CPU only, small toy graph. No TLC-GNN / gudhi / PYTHONPATH dependency.
"""
import numpy as np
import torch
import pytest

from hetero_pdg.data import (
    make_toy_hetero_graph,
    PairBatch,
)
from hetero_pdg.metapath import DEFAULT_METAPATHS, project_metapath, project_all_metapaths
from hetero_pdg.pair_filtration import (
    PairConditionedFiltrationMLP,
    pair_conditioned_node_filtration,
    compute_pair_conditioned_lp_features,
    PairConditionedFeatures,
    PAIR_CONDITIONED_FILTRATION_LABEL,
    PAIR_CONDITIONED_FEATURE_NAMES,
    FILTRATION_MODE,
    NO_PATH_DISTANCE,
)


# meta-paths over papers (PFP, PCP) -- target type 'paper'
PAPER_SPECS = [DEFAULT_METAPATHS["PFP"], DEFAULT_METAPATHS["PCP"]]


def _build(seed=0):
    torch.manual_seed(seed)
    data = make_toy_hetero_graph(seed=seed)
    mlp = PairConditionedFiltrationMLP(data)
    return data, mlp


def test_honesty_label_and_mode_constants():
    # exact honesty wording must be present and non-misleading
    assert "experimental pair-conditioned filtration" in PAIR_CONDITIONED_FILTRATION_LABEL
    assert "NOT a reproduction" in PAIR_CONDITIONED_FILTRATION_LABEL
    low = PAIR_CONDITIONED_FILTRATION_LABEL.lower()
    # must never claim to be exact PH / real PDGNN / EPD
    for forbidden in ("exact persistent homology", "real pdgnn", "epd feature"):
        assert forbidden not in low
    assert FILTRATION_MODE == "pair_conditioned"


def test_node_filtration_shape_and_finite():
    data, mlp = _build()
    bundle = project_metapath(data, DEFAULT_METAPATHS["PCP"])
    vals = pair_conditioned_node_filtration(mlp, data, bundle, u=0, v=1, relation_id=0)
    assert vals.shape == (bundle.num_nodes,)
    assert torch.isfinite(vals).all()


def test_node_filtration_determinism():
    data1, mlp1 = _build(seed=0)
    data2, mlp2 = _build(seed=0)
    b1 = project_metapath(data1, DEFAULT_METAPATHS["PCP"])
    b2 = project_metapath(data2, DEFAULT_METAPATHS["PCP"])
    v1 = pair_conditioned_node_filtration(mlp1, data1, b1, u=2, v=5, relation_id=0)
    v2 = pair_conditioned_node_filtration(mlp2, data2, b2, u=2, v=5, relation_id=0)
    assert torch.allclose(v1, v2)


def test_node_filtration_pair_dependence():
    # THE defining property: changing the conditioning pair (u,v) must be able to
    # change the per-node filtration values.
    data, mlp = _build()
    bundle = project_metapath(data, DEFAULT_METAPATHS["PCP"])
    va = pair_conditioned_node_filtration(mlp, data, bundle, u=0, v=1, relation_id=0)
    vb = pair_conditioned_node_filtration(mlp, data, bundle, u=10, v=20, relation_id=0)
    assert not torch.allclose(va, vb)


def test_node_filtration_relation_dependence():
    # different relation id should be able to change values (learnable rel embedding)
    data = make_toy_hetero_graph(seed=0)
    torch.manual_seed(0)
    mlp = PairConditionedFiltrationMLP(data, n_relations=2)
    bundle = project_metapath(data, DEFAULT_METAPATHS["PCP"])
    v0 = pair_conditioned_node_filtration(mlp, data, bundle, u=0, v=1, relation_id=0)
    v1 = pair_conditioned_node_filtration(mlp, data, bundle, u=0, v=1, relation_id=1)
    assert not torch.allclose(v0, v1)


def test_custom_node_embeddings_accepted():
    data, mlp = _build()
    bundle = project_metapath(data, DEFAULT_METAPATHS["PCP"])
    emb = mlp.node_embeddings(data)
    assert "paper" in emb and emb["paper"].shape[0] == data["paper"].num_nodes
    vals = pair_conditioned_node_filtration(
        mlp, data, bundle, u=0, v=1, relation_id=0, node_embeddings=emb)
    assert vals.shape == (bundle.num_nodes,)
    assert torch.isfinite(vals).all()


def _pair_batch(data, seed=0):
    rng = np.random.RandomState(seed)
    n = int(data["paper"].num_nodes)
    src = torch.tensor(rng.randint(0, n, size=8), dtype=torch.long)
    dst = torch.tensor(rng.randint(0, n, size=8), dtype=torch.long)
    label = torch.zeros(8)
    return PairBatch(src, dst, label, "paper", "paper")


def test_lp_features_shape_finite_and_mask():
    data, mlp = _build()
    bundle = project_metapath(data, DEFAULT_METAPATHS["PCP"])
    pb = _pair_batch(data)
    out = compute_pair_conditioned_lp_features(mlp, data, bundle, pb, relation_id=0)
    assert isinstance(out, PairConditionedFeatures)
    B = pb.src.numel()
    feat_dim = len(PAIR_CONDITIONED_FEATURE_NAMES)
    assert out.features.shape == (B, feat_dim)
    assert out.mask.shape == (B,)
    assert torch.isfinite(out.features).all()
    assert out.mask.dtype == torch.bool


def test_lp_features_determinism():
    d1, m1 = _build(seed=0)
    d2, m2 = _build(seed=0)
    b1 = project_metapath(d1, DEFAULT_METAPATHS["PCP"])
    b2 = project_metapath(d2, DEFAULT_METAPATHS["PCP"])
    pb1, pb2 = _pair_batch(d1), _pair_batch(d2)
    o1 = compute_pair_conditioned_lp_features(m1, d1, b1, pb1, relation_id=0)
    o2 = compute_pair_conditioned_lp_features(m2, d2, b2, pb2, relation_id=0)
    assert torch.allclose(o1.features, o2.features)


def test_lp_features_pair_dependence():
    # two batches with different pairs should not give identical feature matrices
    data, mlp = _build()
    bundle = project_metapath(data, DEFAULT_METAPATHS["PCP"])
    pb_a = _pair_batch(data, seed=1)
    pb_b = _pair_batch(data, seed=2)
    fa = compute_pair_conditioned_lp_features(mlp, data, bundle, pb_a, relation_id=0).features
    fb = compute_pair_conditioned_lp_features(mlp, data, bundle, pb_b, relation_id=0).features
    assert not torch.allclose(fa, fb)


def test_lp_features_plug_into_predictor():
    # the produced tensor should be consumable as the 'topo' tensor of the model
    from hetero_pdg.models import HeteroTopoLinkPredictor
    data, mlp = _build()
    bundle = project_metapath(data, DEFAULT_METAPATHS["PCP"])
    pb = _pair_batch(data)
    out = compute_pair_conditioned_lp_features(mlp, data, bundle, pb, relation_id=0)
    feat_dim = len(PAIR_CONDITIONED_FEATURE_NAMES)
    node_in = {nt: data[nt].x.size(1) for nt in ("author", "paper", "field")}
    model = HeteroTopoLinkPredictor(node_in, mode="collapsed_topology", topo_dim=feat_dim)
    node_feats = {nt: data[nt].x for nt in ("author", "paper", "field")}
    scores = model(node_feats, pb, topo=out.features)
    assert scores.shape == (pb.src.numel(),)
    assert torch.isfinite(scores).all()


def test_no_path_sentinel_used_for_disconnected():
    # a node with no path to u/v should receive the NO_PATH sentinel distance;
    # smoke-check the helper exposes the constant and an isolated pair is handled.
    data, mlp = _build()
    bundle = project_metapath(data, DEFAULT_METAPATHS["PCP"])
    pb = PairBatch(torch.tensor([0]), torch.tensor([1]), torch.zeros(1), "paper", "paper")
    out = compute_pair_conditioned_lp_features(mlp, data, bundle, pb, relation_id=0)
    assert NO_PATH_DISTANCE == -1.0
    assert out.features.shape[0] == 1


def test_invalid_pair_zero_filled_and_masked():
    data, mlp = _build()
    bundle = project_metapath(data, DEFAULT_METAPATHS["PCP"])
    big = bundle.num_nodes + 100  # out of range -> invalid
    pb = PairBatch(torch.tensor([big]), torch.tensor([big]), torch.zeros(1), "paper", "paper")
    out = compute_pair_conditioned_lp_features(mlp, data, bundle, pb, relation_id=0)
    assert out.mask.numel() == 1
    assert bool(out.mask[0]) is False
    assert torch.allclose(out.features[0], torch.zeros_like(out.features[0]))
