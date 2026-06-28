"""Phase-1 RED tests: type-aware filtration (relation degrees, per-type MLP,
quantile calibration, lexicographic ordering, homogeneous view)."""
import numpy as np
import torch
import pytest

from hetero_pdg.data import make_toy_hetero_graph, NODE_TYPES
from hetero_pdg.filtration import (
    compute_relation_degree_vectors,
    TypeAwareFiltrationMLP,
    type_wise_quantile_calibration,
    lexicographic_node_ordering,
    build_homogeneous_view_with_type_ids,
    DEFAULT_TYPE_PRIORITY,
)


def test_relation_degree_vectors_shapes_and_finite():
    d = make_toy_hetero_graph(seed=0)
    rdv = compute_relation_degree_vectors(d)
    assert set(rdv) == set(NODE_TYPES)
    for nt in NODE_TYPES:
        assert rdv[nt].shape[0] == d[nt].num_nodes and rdv[nt].shape[1] >= 1
        assert torch.isfinite(rdv[nt]).all()


def test_homogeneous_view_has_type_ids_and_valid_edges():
    d = make_toy_hetero_graph(seed=0)
    view = build_homogeneous_view_with_type_ids(d)
    N = sum(int(d[nt].num_nodes) for nt in NODE_TYPES)
    assert view["num_nodes"] == N
    assert view["node_type_ids"].shape == (N,)
    assert view["edge_index"].shape[0] == 2
    if view["edge_index"].shape[1] > 0:
        assert int(view["edge_index"].max()) < N


def test_filtration_mlp_one_scalar_per_node_no_nan_and_per_type():
    d = make_toy_hetero_graph(seed=0)
    mlp = TypeAwareFiltrationMLP(d, hidden=16)
    out = mlp(d)
    assert set(out) == set(NODE_TYPES)
    for nt in NODE_TYPES:
        assert out[nt].shape == (int(d[nt].num_nodes),)
        assert torch.isfinite(out[nt]).all()
    assert mlp.mlps["paper"] is not mlp.mlps["author"]   # genuinely per-type


def test_filtration_handles_featureless_type_with_placeholder():
    d = make_toy_hetero_graph(seed=0)
    d["field"].x = None                                   # featureless type
    mlp = TypeAwareFiltrationMLP(d, hidden=8)
    out = mlp(d)
    assert out["field"].shape == (int(d["field"].num_nodes),)
    assert torch.isfinite(out["field"]).all()


def test_quantile_calibration_unit_range_and_rank_preserving():
    vals = {"paper": torch.tensor([3.0, 1.0, 2.0, 2.5])}
    cal = type_wise_quantile_calibration(vals)
    c = np.asarray(cal["paper"])
    assert c.min() >= -1e-6 and c.max() <= 1 + 1e-6
    assert list(np.argsort(c)) == list(np.argsort(vals["paper"].numpy()))


def test_lexicographic_value_primary_priority_secondary():
    order = list(lexicographic_node_ordering(np.array([0.5, 0.5, 0.4, 0.6]),
                                             np.array([2, 0, 5, 1])))
    assert order == [2, 1, 0, 3]


def test_lexicographic_value_dominates_not_big_constant():
    order = list(lexicographic_node_ordering(np.array([0.5, 0.5 + 1e-9]),
                                             np.array([10, 0])))
    assert order[0] == 0


def test_default_type_priority_covers_all_types():
    assert set(DEFAULT_TYPE_PRIORITY) == set(NODE_TYPES)
