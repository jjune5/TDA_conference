"""Type-aware unified filtration (the UnifiedFilter-PDGNN path).

Heterogeneous nodes live in incomparable feature spaces, so a single global filter
function is ill-typed. We instead:

  1. give every node type its OWN MLP whose input is
     ``[raw node feature  ||  relation-degree vector]`` (placeholder features are
     synthesised deterministically for featureless types);
  2. map each type's scalar outputs to a common ``[0, 1]`` range by type-wise
     quantile (rank) calibration;
  3. order nodes lexicographically by ``(calibrated_value, node_type_priority)`` --
     NOT by folding the type into the value with a large multiplicative constant.

NOTE: type-wise quantile calibration is a *scale-alignment heuristic* to make
different node types' filter values comparable. It is NOT a complete theoretical
guarantee from persistent-homology stability theory; it simply aligns marginals.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData

from hetero_pdg.data import NODE_TYPES, EDGE_TYPES

DEFAULT_TYPE_PRIORITY: Dict[str, int] = {"author": 0, "paper": 1, "field": 2}
RelationColumn = Tuple[Tuple[str, str, str], str]


def relation_schema(node_type: str) -> List[RelationColumn]:
    """Ordered (edge_type, role in {'src','dst'}) columns in which the type participates."""
    cols: List[RelationColumn] = []
    for et in EDGE_TYPES:
        s, _, d = et
        if s == node_type:
            cols.append((et, "src"))
        if d == node_type:
            cols.append((et, "dst"))
    return cols


def compute_relation_degree_vectors(data: HeteroData) -> Dict[str, torch.Tensor]:
    """Per node type: ``(n, n_relation_columns)`` integer relation degrees as a float tensor."""
    out: Dict[str, torch.Tensor] = {}
    for nt in NODE_TYPES:
        cols = relation_schema(nt)
        n = int(data[nt].num_nodes)
        m = np.zeros((n, len(cols)), dtype=np.float64)
        for c, (et, role) in enumerate(cols):
            ei = data[et].edge_index.cpu().numpy()
            endpoint = ei[0] if role == "src" else ei[1]
            m[:, c] = np.bincount(endpoint, minlength=n)
        out[nt] = torch.tensor(m, dtype=torch.float32)
    return out


def _placeholder_features(n: int, dim: int = 4) -> torch.Tensor:
    """Deterministic positional features for a featureless node type."""
    idx = torch.arange(n, dtype=torch.float32).unsqueeze(1)
    freqs = torch.arange(1, dim + 1, dtype=torch.float32)
    return torch.sin(idx * freqs * 0.1)


def build_homogeneous_view_with_type_ids(data: HeteroData) -> dict:
    """Type-blind homogeneous view that retains node type ids.

    Concatenates node types in NODE_TYPES order into one ``0..N-1`` index space;
    returns node_type_ids, a unified (type-blind) edge_index, per-type offsets, N.
    """
    offsets: Dict[str, Tuple[int, int]] = {}
    type_ids: List[int] = []
    cur = 0
    for tid, nt in enumerate(NODE_TYPES):
        n = int(data[nt].num_nodes)
        offsets[nt] = (cur, cur + n)
        type_ids.extend([tid] * n)
        cur += n
    edges = []
    for et in data.edge_types:
        s, _, d = et
        ei = data[et].edge_index.cpu().numpy()
        os_, od = offsets[s][0], offsets[d][0]
        for a, b in zip(ei[0], ei[1]):
            edges.append((os_ + int(a), od + int(b)))
    edge_index = (torch.tensor(edges, dtype=torch.long).t().contiguous()
                  if edges else torch.zeros((2, 0), dtype=torch.long))
    return {
        "num_nodes": cur,
        "node_type_ids": torch.tensor(type_ids, dtype=torch.long),
        "edge_index": edge_index,
        "offsets": offsets,
    }


class TypeAwareFiltrationMLP(nn.Module):
    """One MLP per node type: ``[raw feat || relation-degree] -> scalar``."""

    def __init__(self, data: HeteroData, hidden: int = 16, placeholder_dim: int = 4):
        super().__init__()
        self.placeholder_dim = placeholder_dim
        self.mlps = nn.ModuleDict()
        self._featureless: Dict[str, bool] = {}
        for nt in NODE_TYPES:
            x = getattr(data[nt], "x", None)
            featureless = x is None
            self._featureless[nt] = featureless
            raw_dim = placeholder_dim if featureless else int(x.size(1))
            rd_dim = len(relation_schema(nt))
            self.mlps[nt] = nn.Sequential(
                nn.Linear(raw_dim + rd_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, 1),
            )

    def forward(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        rdv = compute_relation_degree_vectors(data)
        dev = next(self.parameters()).device
        out: Dict[str, torch.Tensor] = {}
        for nt in NODE_TYPES:
            x = getattr(data[nt], "x", None)
            if x is None:
                x = _placeholder_features(int(data[nt].num_nodes), self.placeholder_dim)
            inp = torch.cat([x.to(dev), rdv[nt].to(dev)], dim=-1)
            out[nt] = self.mlps[nt](inp).squeeze(-1)
        return out


def type_wise_quantile_calibration(
    values_by_type: Dict[str, torch.Tensor]
) -> Dict[str, np.ndarray]:
    """Rank-normalise each type's scalar values to [0, 1] (ties averaged)."""
    out: Dict[str, np.ndarray] = {}
    for nt, values in values_by_type.items():
        v = np.asarray(values.detach().cpu() if torch.is_tensor(values) else values,
                       dtype=np.float64).reshape(-1)
        n = v.size
        if n == 0:
            out[nt] = v
        elif n == 1:
            out[nt] = np.array([0.5])
        else:
            uniq, inv, counts = np.unique(v, return_inverse=True, return_counts=True)
            start = np.zeros(uniq.size); start[1:] = np.cumsum(counts)[:-1]
            avg_rank = start + (counts - 1) / 2.0
            out[nt] = avg_rank[inv] / (n - 1)
    return out


def lexicographic_node_ordering(
    calibrated_values: np.ndarray, node_type_priority: np.ndarray
) -> np.ndarray:
    """Order nodes by (calibrated_value primary, type_priority secondary).

    Uses ``np.lexsort`` (last key primary) so a tiny value difference is never
    overridden by the priority -- unlike a large-constant additive key.
    """
    cv = np.asarray(calibrated_values, dtype=np.float64).reshape(-1)
    tp = np.asarray(node_type_priority).reshape(-1)
    return np.lexsort((tp, cv))
