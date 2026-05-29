import os, sys
import numpy as np
import networkx as nx
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_metapath_adjacency_counts_cooccurrence():
    """PAP 류 meta-path: A1 @ A2 가 공동출현 횟수(가중치)를 준다."""
    import scipy.sparse as sp
    from hetero.metapath_graph import compose_metapath_adj
    # paper(2) - author(2): p0-a0, p0-a1, p1-a1
    pa = sp.csr_matrix(np.array([[1, 1], [0, 1]], dtype=np.float64))  # paper x author
    ap = pa.T.tocsr()                                                 # author x paper
    W = compose_metapath_adj([pa, ap])                                # paper x paper
    W = np.asarray(W.todense())
    # p0,p1 share author a1 -> off-diagonal = 1
    assert W[0, 1] == 1 and W[1, 0] == 1
    # diagonal = #authors per paper (p0 has 2) -> will be zeroed by builder later
    assert W[0, 0] == 2 and W[1, 1] == 1


def test_build_metapath_graph_acm_pap_smoke():
    """ACM PAP meta-path -> paper 동종 weighted nx graph (no self loops)."""
    from hetero.metapath_graph import load_hgb, build_metapath_graph
    d = load_hgb('ACM')
    g, y, masks = build_metapath_graph(d, 'PAP')
    assert g.number_of_nodes() == int(d['paper'].num_nodes)   # 3025
    assert g.number_of_edges() > 0
    assert not any(u == v for u, v in g.edges())              # diagonal removed
    # weights are positive (co-authored paper counts)
    w = [dd['weight'] for _, _, dd in g.edges(data=True)]
    assert min(w) >= 1
    assert y.shape[0] == 3025 and int(y.max()) == 2           # 3 classes
    assert masks['train'].sum() > 0 and masks['test'].sum() > 0


def test_leakage_audit_runs():
    """meta-path 그래프 구조만으로 라벨 예측(LP) 정확도를 반환한다."""
    from hetero.metapath_graph import load_hgb, build_metapath_graph
    from hetero.leakage_audit import structure_only_label_acc
    d = load_hgb('ACM')
    g, y, masks = build_metapath_graph(d, 'PAP')
    acc = structure_only_label_acc(g, y, masks)
    assert 0.0 <= acc <= 1.0
