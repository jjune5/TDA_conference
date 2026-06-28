"""Phase-1 RED tests: deterministic FALLBACK topology descriptors (explicitly NOT
PDGNN / EPD), per-meta-path features + masks, and target-edge leakage removal."""
import networkx as nx
import numpy as np
import torch
import pytest

from hetero_pdg.data import make_toy_hetero_graph, PairBatch
from hetero_pdg.metapath import DEFAULT_METAPATHS, project_all_metapaths, ProjectedGraphBundle
from hetero_pdg.topology_features import (
    FALLBACK_TOPOLOGY_DESCRIPTOR_NAMES,
    TOPOLOGY_BACKEND,
    NO_PATH_DISTANCE,
    fallback_topology_descriptors,
    compute_fallback_features_by_metapath,
    TopologyFeatures,
)

_IDX = {n: i for i, n in enumerate(FALLBACK_TOPOLOGY_DESCRIPTOR_NAMES)}


def test_backend_is_clearly_labeled_fallback():
    assert TOPOLOGY_BACKEND == "fallback_topology_descriptors"
    assert "pdgnn" not in TOPOLOGY_BACKEND.lower() and "epd" not in TOPOLOGY_BACKEND.lower()
    assert len(FALLBACK_TOPOLOGY_DESCRIPTOR_NAMES) == 6


def test_descriptor_shape_and_finite():
    f = fallback_topology_descriptors(nx.path_graph(5), 0, 4)
    assert f.shape == (6,) and np.isfinite(f).all()


def test_descriptor_removes_target_edge_and_restores():
    g = nx.Graph([(0, 1), (0, 2), (1, 2)])  # triangle
    f_rm = fallback_topology_descriptors(g, 0, 1, remove_target=True)
    f_keep = fallback_topology_descriptors(g, 0, 1, remove_target=False)
    assert f_rm[_IDX["src_degree"]] == 1 and f_keep[_IDX["src_degree"]] == 2
    assert f_rm[_IDX["common_neighbors"]] == 1
    assert g.has_edge(0, 1)  # restored, no permanent mutation


def test_descriptor_disconnected_uses_sentinel():
    g = nx.Graph([(0, 1)]); g.add_node(3)
    f = fallback_topology_descriptors(g, 0, 3)
    assert f[_IDX["shortest_path_distance"]] == NO_PATH_DISTANCE


def test_features_by_metapath_shapes_mask_and_backend():
    d = make_toy_hetero_graph(seed=0)
    bundles = project_all_metapaths(d, [DEFAULT_METAPATHS["PFP"], DEFAULT_METAPATHS["PCP"]])
    pb = PairBatch(torch.tensor([0, 1, 2, 3]), torch.tensor([1, 2, 3, 4]),
                   torch.ones(4), "paper", "paper")
    tf = compute_fallback_features_by_metapath(bundles, pb, k=2)
    assert isinstance(tf, TopologyFeatures)
    assert set(tf.features_by_metapath) == {"PFP", "PCP"}
    assert tf.backend == "fallback_topology_descriptors"
    for name in ("PFP", "PCP"):
        assert tf.features_by_metapath[name].shape == (4, 6)
        assert torch.isfinite(tf.features_by_metapath[name]).all()
        assert tf.mask_by_metapath[name].shape == (4,)
        assert tf.mask_by_metapath[name].dtype == torch.bool


def test_invalid_pair_returns_zero_features_and_mask_false():
    g = nx.Graph(); g.add_nodes_from(range(4)); g.add_edge(0, 1)  # 2,3 isolated
    bundle = ProjectedGraphBundle("X", "paper", 4, None, g)
    pb = PairBatch(torch.tensor([2]), torch.tensor([3]), torch.zeros(1), "paper", "paper")
    tf = compute_fallback_features_by_metapath({"X": bundle}, pb)
    assert bool(tf.mask_by_metapath["X"][0]) is False
    assert torch.allclose(tf.features_by_metapath["X"][0], torch.zeros(6))
