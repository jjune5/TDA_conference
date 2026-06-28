"""Phase-1 RED tests: meta-path projection (adjacency multiplication) + validation."""
import networkx as nx
import numpy as np
import torch
import pytest
from torch_geometric.data import HeteroData

from hetero_pdg.data import make_toy_hetero_graph
from hetero_pdg.metapath import (
    MetaPathSpec,
    ProjectedGraphBundle,
    DEFAULT_METAPATHS,
    validate_metapath,
    project_metapath,
    project_all_metapaths,
    build_collapsed_baseline_graph,
)


def _hand() -> HeteroData:
    d = HeteroData()
    d["author"].x = torch.zeros(2, 3); d["author"].num_nodes = 2
    d["paper"].x = torch.zeros(2, 3); d["paper"].num_nodes = 2
    d["field"].x = torch.zeros(1, 2); d["field"].num_nodes = 1
    writes = torch.tensor([[0, 1, 1], [0, 0, 1]])
    d["author", "writes", "paper"].edge_index = writes
    d["paper", "written_by", "author"].edge_index = writes.flip(0)
    ht = torch.tensor([[0, 1], [0, 0]])
    d["paper", "has_topic", "field"].edge_index = ht
    d["field", "rev_has_topic", "paper"].edge_index = ht.flip(0)
    d["paper", "cites", "paper"].edge_index = torch.tensor([[0], [1]])
    return d


def _dense(A):
    return A.toarray() if hasattr(A, "toarray") else np.asarray(A)


def test_spec_target_type():
    assert DEFAULT_METAPATHS["APA"].target_type == "author"
    assert DEFAULT_METAPATHS["PFP"].target_type == "paper"
    assert DEFAULT_METAPATHS["PCP"].target_type == "paper"


def test_validate_raises_on_type_mismatch():
    bad = MetaPathSpec("BAD", [("author", "writes", "paper"),
                               ("field", "rev_has_topic", "paper")])
    with pytest.raises(ValueError):
        validate_metapath(bad, _hand())


def test_validate_raises_on_missing_edge_type():
    bad = MetaPathSpec("BAD", [("author", "writes", "journal"),
                               ("journal", "x", "author")])
    with pytest.raises(ValueError):
        validate_metapath(bad, _hand())


def test_validate_raises_when_not_closed_loop():
    bad = MetaPathSpec("BAD", [("author", "writes", "paper")])  # author != paper
    with pytest.raises(ValueError):
        validate_metapath(bad, _hand())


def test_project_apa_hand_example():
    b = project_metapath(_hand(), DEFAULT_METAPATHS["APA"])
    assert isinstance(b, ProjectedGraphBundle)
    assert b.target_type == "author" and b.num_nodes == 2
    np.testing.assert_array_equal(_dense(b.adjacency), np.array([[1, 1], [1, 2]]))
    assert isinstance(b.graph, nx.Graph)
    assert b.graph.has_edge(0, 1)            # off-diagonal co-authorship
    assert not b.graph.has_edge(0, 0)        # diagonal dropped in the graph


def test_project_pcp_is_cites():
    b = project_metapath(_hand(), DEFAULT_METAPATHS["PCP"])
    np.testing.assert_array_equal(_dense(b.adjacency), np.array([[0, 1], [0, 0]]))


def test_dense_fallback_matches_sparse():
    s = project_metapath(_hand(), DEFAULT_METAPATHS["APA"], use_sparse=True)
    d = project_metapath(_hand(), DEFAULT_METAPATHS["APA"], use_sparse=False)
    np.testing.assert_array_equal(_dense(s.adjacency), _dense(d.adjacency))


def test_project_all_and_collapsed_union():
    data = make_toy_hetero_graph(seed=0)
    paper_mps = [DEFAULT_METAPATHS["PFP"], DEFAULT_METAPATHS["PCP"]]
    bundles = project_all_metapaths(data, paper_mps)
    assert set(bundles) == {"PFP", "PCP"}
    assert all(b.num_nodes == data["paper"].num_nodes for b in bundles.values())
    collapsed = build_collapsed_baseline_graph(data, paper_mps)
    assert isinstance(collapsed, ProjectedGraphBundle)
    assert collapsed.num_nodes == data["paper"].num_nodes
    assert collapsed.graph.number_of_edges() >= max(
        b.graph.number_of_edges() for b in bundles.values())
