"""Phase-1 RED tests: toy HeteroData + type-constrained LP data utilities."""
import torch
import pytest

from hetero_pdg.data import (
    make_toy_hetero_graph,
    NODE_TYPES,
    EDGE_TYPES,
    LinkSplit,
    PairBatch,
    split_target_edges,
    typed_negative_sampling,
    make_pair_batches,
    remove_target_edges_for_split,
)

CITES = ("paper", "cites", "paper")


def test_toy_types_edges_and_nonempty_projection_preconditions():
    d = make_toy_hetero_graph(seed=0)
    assert set(d.node_types) == set(NODE_TYPES) == {"author", "paper", "field"}
    for et in EDGE_TYPES:
        assert et in d.edge_types
    # preconditions so APA / PFP / PCP projections are non-empty
    assert d["author", "writes", "paper"].edge_index.numel() > 0     # APA
    assert d["paper", "has_topic", "field"].edge_index.numel() > 0   # PFP
    assert d["paper", "cites", "paper"].edge_index.numel() > 0       # PCP


def test_reverse_edges_consistent():
    d = make_toy_hetero_graph(seed=0)
    assert torch.equal(d["paper", "written_by", "author"].edge_index,
                       d["author", "writes", "paper"].edge_index.flip(0))
    assert torch.equal(d["field", "rev_has_topic", "paper"].edge_index,
                       d["paper", "has_topic", "field"].edge_index.flip(0))


def test_split_target_edges_disjoint_and_complete():
    d = make_toy_hetero_graph(seed=0)
    sp = split_target_edges(d, CITES, val_frac=0.2, test_frac=0.2, seed=0)
    assert isinstance(sp, LinkSplit)
    tr = {(int(a), int(b)) for a, b in sp.train_pos.t()}
    va = {(int(a), int(b)) for a, b in sp.val_pos.t()}
    te = {(int(a), int(b)) for a, b in sp.test_pos.t()}
    assert tr and va and te
    assert tr.isdisjoint(va) and tr.isdisjoint(te) and va.isdisjoint(te)
    assert len(tr) + len(va) + len(te) == d[CITES].edge_index.shape[1]


def test_typed_negative_sampling_constrained_and_disjoint():
    d = make_toy_hetero_graph(seed=0)
    pos = {(int(a), int(b)) for a, b in d[CITES].edge_index.t()}
    neg = typed_negative_sampling(d, CITES, num_samples=30, seed=1)
    assert neg.shape == (2, 30)
    npap = int(d["paper"].num_nodes)
    for a, b in neg.t():
        a, b = int(a), int(b)
        assert 0 <= a < npap and 0 <= b < npap
        assert a != b and (a, b) not in pos


def test_remove_target_edges_does_not_mutate_original():
    d = make_toy_hetero_graph(seed=0)
    before = d[CITES].edge_index.clone()
    rm = d[CITES].edge_index[:, :3]
    d2 = remove_target_edges_for_split(d, rm, CITES)
    assert torch.equal(d[CITES].edge_index, before)  # original untouched
    s2 = {(int(a), int(b)) for a, b in d2[CITES].edge_index.t()}
    for a, b in rm.t():
        assert (int(a), int(b)) not in s2


def test_make_pair_batches_labels_and_types():
    d = make_toy_hetero_graph(seed=0)
    pos = d[CITES].edge_index[:, :5]
    neg = typed_negative_sampling(d, CITES, num_samples=5, seed=2)
    batches = make_pair_batches(pos, neg, src_type="paper", dst_type="paper",
                                batch_size=100, seed=0)
    assert len(batches) >= 1 and isinstance(batches[0], PairBatch)
    b = batches[0]
    assert b.src.shape == b.dst.shape == b.label.shape
    assert int(b.label.sum().item()) == 5
    assert b.src_type == "paper" and b.dst_type == "paper"


def test_determinism():
    a = make_toy_hetero_graph(seed=7)
    b = make_toy_hetero_graph(seed=7)
    assert torch.equal(a[CITES].edge_index, b[CITES].edge_index)
