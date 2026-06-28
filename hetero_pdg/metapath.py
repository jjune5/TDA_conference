"""Meta-path projection: heterogeneous graph -> homogeneous weighted graph over a
single (target) node type, via adjacency multiplication.

A meta-path is a closed sequence of edge types ``[(s0,r0,d0),...,(sk,rk,dk)]`` with
``d_i == s_{i+1}`` and ``s0 == d_k`` (starts and ends at the target type). Its
projected adjacency is ``A_0 @ A_1 @ ... @ A_k`` (``A_i`` = biadjacency of step i);
entry (u,v) counts meta-path instances joining target nodes u and v.

Sparse (scipy) is used when available; a correct dense (numpy) fallback is provided
for toy/smoke use. Invalid meta-paths raise ValueError (never silently ignored).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import networkx as nx
import numpy as np
from torch_geometric.data import HeteroData

try:
    import scipy.sparse as sp
    _HAS_SCIPY = True
except Exception:  # pragma: no cover - scipy is normally present
    _HAS_SCIPY = False

EdgeType = Tuple[str, str, str]


@dataclass
class MetaPathSpec:
    """Named meta-path = ordered list of edge-type triples (closed loop)."""
    name: str
    edge_types: List[EdgeType]

    @property
    def target_type(self) -> str:
        return self.edge_types[0][0]


@dataclass
class ProjectedGraphBundle:
    """Result of projecting one meta-path onto its target node type."""
    name: str
    target_type: str
    num_nodes: int
    adjacency: object      # scipy.csr_matrix or np.ndarray (raw composed counts)
    graph: nx.Graph        # undirected, weighted, diagonal dropped (for descriptors)


DEFAULT_METAPATHS: Dict[str, MetaPathSpec] = {
    "APA": MetaPathSpec("APA", [("author", "writes", "paper"),
                                ("paper", "written_by", "author")]),
    "PFP": MetaPathSpec("PFP", [("paper", "has_topic", "field"),
                                ("field", "rev_has_topic", "paper")]),
    "PCP": MetaPathSpec("PCP", [("paper", "cites", "paper")]),
}


def validate_metapath(spec: MetaPathSpec, data: HeteroData) -> None:
    """Raise ValueError with a clear message if the meta-path is malformed."""
    ets = spec.edge_types
    if not ets:
        raise ValueError(f"meta-path {spec.name!r}: empty edge-type list")
    data_ets = set(data.edge_types)
    for et in ets:
        if len(et) != 3:
            raise ValueError(f"meta-path {spec.name!r}: edge type {et!r} is not a triple")
        if tuple(et) not in data_ets:
            raise ValueError(
                f"meta-path {spec.name!r}: edge type {tuple(et)!r} not present in data "
                f"(have {sorted(data_ets)})")
    for a, b in zip(ets[:-1], ets[1:]):
        if a[2] != b[0]:
            raise ValueError(
                f"meta-path {spec.name!r}: incompatible step {tuple(a)!r} -> {tuple(b)!r} "
                f"(dst type {a[2]!r} != src type {b[0]!r})")
    if ets[0][0] != ets[-1][2]:
        raise ValueError(
            f"meta-path {spec.name!r}: must start and end at the same node type "
            f"({ets[0][0]!r} != {ets[-1][2]!r})")


def _biadjacency(data: HeteroData, et: EdgeType, use_sparse: bool):
    s, _, d = et
    n_src, n_dst = int(data[s].num_nodes), int(data[d].num_nodes)
    ei = data[et].edge_index.cpu().numpy()
    if use_sparse and _HAS_SCIPY:
        return sp.csr_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(n_src, n_dst))
    M = np.zeros((n_src, n_dst), dtype=np.float64)
    if ei.shape[1]:
        np.add.at(M, (ei[0], ei[1]), 1.0)
    return M


def _graph_from_adjacency(adj, num_nodes: int) -> nx.Graph:
    """Undirected weighted graph: connected if a path exists either way, diagonal dropped."""
    g = nx.Graph()
    g.add_nodes_from(range(num_nodes))
    if _HAS_SCIPY and sp.issparse(adj):
        u = sp.triu(adj.maximum(adj.T), k=1).tocoo()
        for i, j, w in zip(u.row, u.col, u.data):
            if w > 0:
                g.add_edge(int(i), int(j), weight=float(w))
    else:
        A = np.asarray(adj)
        U = np.maximum(A, A.T)
        ii, jj = np.triu_indices(U.shape[0], k=1)
        for i, j in zip(ii, jj):
            w = U[i, j]
            if w > 0:
                g.add_edge(int(i), int(j), weight=float(w))
    return g


def project_metapath(data: HeteroData, spec: MetaPathSpec,
                     use_sparse: bool = True) -> ProjectedGraphBundle:
    """Project one (validated) meta-path; return adjacency + undirected nx graph."""
    validate_metapath(spec, data)
    W = _biadjacency(data, spec.edge_types[0], use_sparse)
    for et in spec.edge_types[1:]:
        W = W @ _biadjacency(data, et, use_sparse)
    n = int(data[spec.target_type].num_nodes)
    return ProjectedGraphBundle(spec.name, spec.target_type, n, W,
                                _graph_from_adjacency(W, n))


def project_all_metapaths(data: HeteroData, specs: Sequence[MetaPathSpec],
                          use_sparse: bool = True) -> Dict[str, ProjectedGraphBundle]:
    return {s.name: project_metapath(data, s, use_sparse) for s in specs}


def build_collapsed_baseline_graph(data: HeteroData, specs: Sequence[MetaPathSpec],
                                   use_sparse: bool = True) -> ProjectedGraphBundle:
    """Collapse several same-target meta-paths into one type-blind homogeneous graph."""
    tt = specs[0].target_type
    for s in specs:
        if s.target_type != tt:
            raise ValueError(
                f"collapsed baseline needs one target type, got {s.target_type!r} vs {tt!r}")
    bundles = [project_metapath(data, s, use_sparse) for s in specs]
    n = bundles[0].num_nodes
    g = nx.Graph(); g.add_nodes_from(range(n))
    for b in bundles:
        for u, v, d in b.graph.edges(data=True):
            w = d.get("weight", 1.0)
            if g.has_edge(u, v):
                g[u][v]["weight"] += w
            else:
                g.add_edge(u, v, weight=w)
    adj = bundles[0].adjacency
    for b in bundles[1:]:
        adj = adj + b.adjacency
    return ProjectedGraphBundle("collapsed", tt, n, adj, g)
