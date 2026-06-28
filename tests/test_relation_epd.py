"""Phase-4 RED tests: relation-aware 0-dim EPD where edge TYPE actually changes the
persistence events (typed != untyped). Standalone (own persistence image; reuses
edge_filtration's relation_edge_filtration + zero_dim_sublevel_persistence).

This is NOT exact extended persistence and NOT PDGNN; it is a relation-aware 0-dim
sublevel EPD -> persistence image. No performance claims.
"""
import networkx as nx
import numpy as np
import torch

from hetero_pdg.data import make_toy_hetero_graph, PairBatch
from hetero_pdg.metapath import DEFAULT_METAPATHS, project_all_metapaths
from hetero_pdg.relation_epd import (
    RELATION_EPD_HONESTY_LABEL,
    PI_DIM,
    persistence_image,
    build_relation_tagged_graph,
    RelationAwareEPDTopology,
    compute_relation_epd_features,
)


def test_honesty_label_not_overclaiming():
    lab = RELATION_EPD_HONESTY_LABEL.lower()
    assert "not exact" in lab or "not pdgnn" in lab or "0-dim" in lab
    assert "pdgnn" not in lab.replace("not pdgnn", "")  # never claims to BE pdgnn


def test_persistence_image_shape_finite_deterministic():
    pairs = [(0.0, 1.0), (0.2, 0.9), (0.5, float("inf"))]
    a = persistence_image(pairs)
    b = persistence_image(pairs)
    assert a.shape == (PI_DIM,)
    assert np.isfinite(a).all()
    np.testing.assert_array_equal(a, b)           # deterministic
    assert np.allclose(persistence_image([]), 0.0)  # empty -> zeros


def test_build_relation_tagged_graph_has_relation_ids():
    d = make_toy_hetero_graph(seed=0)
    bundles = project_all_metapaths(d, [DEFAULT_METAPATHS["PFP"], DEFAULT_METAPATHS["PCP"]])
    g, n_rel = build_relation_tagged_graph(bundles)
    assert isinstance(g, nx.Graph) and n_rel == 2
    rels = {data["rel"] for _, _, data in g.edges(data=True)}
    assert rels.issubset({0, 1}) and len(rels) >= 1


def test_pair_pi_shape_finite_and_leakage_restored():
    d = make_toy_hetero_graph(seed=0)
    bundles = project_all_metapaths(d, [DEFAULT_METAPATHS["PFP"], DEFAULT_METAPATHS["PCP"]])
    g, n_rel = build_relation_tagged_graph(bundles)
    topo = RelationAwareEPDTopology(g, n_rel, seed=0)
    u, v = next(iter(g.edges()))
    vec = topo.pair_pi(u, v, mode="relation_delay")
    assert vec.shape == (PI_DIM,) and np.isfinite(vec).all()
    assert g.has_edge(u, v)                        # target edge restored


def test_edge_type_changes_events_typed_differs_from_untyped():
    # THE point of Phase 4: with distinct per-relation delays, the typed EPD must be
    # able to differ from the untyped (max) EPD on at least one pair.
    d = make_toy_hetero_graph(seed=0)
    bundles = project_all_metapaths(d, [DEFAULT_METAPATHS["PFP"], DEFAULT_METAPATHS["PCP"]])
    g, n_rel = build_relation_tagged_graph(bundles)
    topo = RelationAwareEPDTopology(g, n_rel, delays=torch.tensor([0.0, 4.0]), seed=0)
    edges = list(g.edges())[:40]
    any_diff = False
    for (u, v) in edges:
        a = topo.pair_pi(u, v, mode="max")
        b = topo.pair_pi(u, v, mode="relation_delay")
        if not np.allclose(a, b):
            any_diff = True
            break
    assert any_diff, "relation delays did not change any EPD -> edge type has no effect"


def test_compute_features_batch_typed_vs_untyped():
    d = make_toy_hetero_graph(seed=0)
    bundles = project_all_metapaths(d, [DEFAULT_METAPATHS["PFP"], DEFAULT_METAPATHS["PCP"]])
    pb = PairBatch(torch.tensor([0, 1, 2]), torch.tensor([1, 2, 3]), torch.ones(3), "paper", "paper")
    feats_typed = compute_relation_epd_features(bundles, pb, mode="relation_delay")
    feats_unty = compute_relation_epd_features(bundles, pb, mode="max")
    assert feats_typed.shape == (3, PI_DIM)
    assert torch.isfinite(feats_typed).all() and torch.isfinite(feats_unty).all()
