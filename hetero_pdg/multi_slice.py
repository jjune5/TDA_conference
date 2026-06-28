"""Multi-slice (vector-valued) filtration: a SLICED FILTRATION APPROXIMATION.

HONESTY LABEL: this module implements a
``sliced filtration approximation (NOT exact multi-parameter persistent homology)``.
It is NOT exact persistent homology, NOT a real PDGNN, and NOT
extended-persistence-diagram (EPD) features. No performance claims are made.

Motivation
----------
A single scalar filtration is restrictive on a heterogeneous graph. Here we build a
VECTOR-valued node filtration ``F(x) in R^4`` and then *observe* topology through ``K``
scalar projections ("slices"). Each slice is a convex combination of the four
components, so a slice is itself an ordinary scalar filtration; looking through several
slices at once is a cheap, practical stand-in for multi-parameter persistence -- it is
explicitly NOT the exact multi-parameter object.

Pipeline (``topology_mode = "multi_slice_topology"``)
-----------------------------------------------------
1.  Vector filtration ``F(x)`` with 4 per-node components (all min-max normalised to
    ``[0, 1]`` for comparability):
      - ``type_aware_score``     : scalar from a per-type :class:`TypeAwareFiltrationMLP`.
      - ``relation_degree_score``: mean of the node's relation-degree vector.
      - ``centrality_proxy``     : degree centrality in the type-blind homogeneous view
                                   (a no-pair proxy for a shortest-path / distance score).
      - ``type_priority_score``  : ``DEFAULT_TYPE_PRIORITY`` of the node type (constant
                                   within a single type).
2.  ``K`` scalar slices ``f_k(x) = lambda_k^T F(x)`` with a learnable ``(K, 4)`` matrix
    pushed through a per-row softmax, so every ``lambda_k >= 0`` and ``sum(lambda_k) = 1``.
3.  Per slice, simple per-pair vicinity statistics of ``f_k`` on each projected
    meta-path graph (endpoint values, |difference|, neighbour means, k-hop-vicinity
    mean), averaged across the supplied meta-path graphs -> a per-slice feature vector.
4.  The ``K`` slice-feature vectors are fused with a (per-pair) attention over slices
    -> a final ``Tensor[B, fused_dim]``.

Everything is seed-controlled and reproducible. Imports stay within
``torch / numpy / networkx / scipy / hetero_pdg.*`` only.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hetero_pdg.data import PairBatch
from hetero_pdg.filtration import (
    DEFAULT_TYPE_PRIORITY,
    TypeAwareFiltrationMLP,
    build_homogeneous_view_with_type_ids,
    compute_relation_degree_vectors,
)

# Honest, non-overclaiming identifiers. These deliberately avoid the words
# "persistent homology", "PDGNN" and "EPD".
MULTI_SLICE_BACKEND = (
    "sliced filtration approximation "
    "(NOT exact multi-parameter persistent homology)"
)
FILTRATION_COMPONENT_NAMES: List[str] = [
    "type_aware_score",
    "relation_degree_score",
    "centrality_proxy",
    "type_priority_score",
]
NUM_FILTRATION_COMPONENTS = len(FILTRATION_COMPONENT_NAMES)

# Per-slice descriptor layout (length = PER_SLICE_DIM).
SLICE_DESCRIPTOR_NAMES: List[str] = [
    "src_value",
    "dst_value",
    "abs_value_diff",
    "src_neighbor_mean",
    "dst_neighbor_mean",
    "vicinity_mean",
]
PER_SLICE_DIM = len(SLICE_DESCRIPTOR_NAMES)

_EPS = 1e-12


def _minmax_normalize(col: torch.Tensor) -> torch.Tensor:
    """Min-max scale a 1-D tensor to ``[0, 1]``; constant columns map to 0.5.

    ``min``/``max`` are detached so this stays differentiable w.r.t. learnable
    components without back-propagating through the (data-dependent) extremes.
    """
    lo = col.min().detach()
    hi = col.max().detach()
    rng = hi - lo
    if float(rng) <= _EPS:
        return torch.full_like(col, 0.5)
    return (col - lo) / rng


class MultiSliceFiltration(nn.Module):
    """Vector filtration observed through ``K`` learnable convex slices.

    Parameters
    ----------
    num_slices:
        Number of scalar slices ``K`` (default 3).
    filt_hidden:
        Hidden width of the per-type filtration MLP (built lazily on first use).
    attn_hidden:
        Hidden width of the per-pair attention over slices.
    k:
        Hop radius for the k-hop vicinity descriptor.
    seed:
        Seed controlling all parameter initialisation (reproducible).
    """

    def __init__(
        self,
        num_slices: int = 3,
        filt_hidden: int = 16,
        attn_hidden: int = 16,
        k: int = 2,
        seed: int = 0,
    ):
        super().__init__()
        if num_slices < 1:
            raise ValueError(f"num_slices must be >= 1, got {num_slices}")
        self.num_slices = int(num_slices)
        self.filt_hidden = int(filt_hidden)
        self.attn_hidden = int(attn_hidden)
        self.k = int(k)
        self.seed = int(seed)
        self.per_slice_dim = PER_SLICE_DIM
        self.fused_dim = PER_SLICE_DIM
        self.backend = MULTI_SLICE_BACKEND

        torch.manual_seed(self.seed)
        # (K, 4) raw logits -> per-row softmax gives convex lambda_k.
        self.raw_lambda = nn.Parameter(
            0.01 * torch.randn(self.num_slices, NUM_FILTRATION_COMPONENTS)
        )
        # Per-pair attention over the K slice-feature vectors.
        self.attn_transform = nn.Linear(self.per_slice_dim, self.attn_hidden)
        self.attn_score = nn.Linear(self.attn_hidden, 1)

        # Per-type filtration MLP is built lazily once we see the data, because
        # node-feature dims differ per type.
        self.filt_mlp: Optional[TypeAwareFiltrationMLP] = None

    # ----------------------------------------------------------------- slices
    def slice_weights(self) -> torch.Tensor:
        """Return the convex slice matrix ``lambda`` of shape ``(K, 4)``.

        Each row satisfies ``lambda_k >= 0`` and ``sum(lambda_k) == 1`` (softmax).
        """
        return F.softmax(self.raw_lambda, dim=1)

    # -------------------------------------------------------------- filtration
    def _ensure_filt_mlp(self, data) -> TypeAwareFiltrationMLP:
        if self.filt_mlp is None:
            torch.manual_seed(self.seed)
            self.filt_mlp = TypeAwareFiltrationMLP(data, hidden=self.filt_hidden)
        return self.filt_mlp

    def _centrality_proxy(self, data, node_type: str) -> torch.Tensor:
        """Degree centrality of ``node_type`` nodes in the homogeneous view.

        A deterministic no-pair stand-in for a shortest-path / distance score.
        """
        view = build_homogeneous_view_with_type_ids(data)
        n_total = int(view["num_nodes"])
        ei = view["edge_index"]
        deg = torch.zeros(n_total, dtype=torch.float32)
        if ei.numel() > 0:
            ones = torch.ones(ei.shape[1], dtype=torch.float32)
            deg.scatter_add_(0, ei[0], ones)
            deg.scatter_add_(0, ei[1], ones)
        start, end = view["offsets"][node_type]
        denom = max(n_total - 1, 1)
        return deg[start:end] / float(denom)

    def vector_filtration(self, data, node_type: str) -> torch.Tensor:
        """Compute the vector filtration ``F(x)`` of shape ``(N, 4)`` for ``node_type``.

        Columns follow :data:`FILTRATION_COMPONENT_NAMES`; each is min-max
        normalised to ``[0, 1]`` so the components are comparable before slicing.
        """
        mlp = self._ensure_filt_mlp(data)
        scores_by_type = mlp(data)
        type_aware = scores_by_type[node_type].reshape(-1)  # (N,) differentiable

        rdv = compute_relation_degree_vectors(data)[node_type]  # (N, R)
        relation_degree = rdv.mean(dim=1) if rdv.numel() else torch.zeros(rdv.shape[0])

        centrality = self._centrality_proxy(data, node_type)  # (N,)

        n = int(data[node_type].num_nodes)
        prio = float(DEFAULT_TYPE_PRIORITY.get(node_type, 0))
        type_priority = torch.full((n,), prio, dtype=torch.float32)

        cols = [
            _minmax_normalize(type_aware),
            _minmax_normalize(relation_degree),
            _minmax_normalize(centrality),
            _minmax_normalize(type_priority),
        ]
        return torch.stack(cols, dim=1)  # (N, 4)

    def slice_fields(self, data, node_type: str) -> torch.Tensor:
        """Return the ``K`` scalar slice fields ``f_k(x)`` as a ``(K, N)`` tensor."""
        Fmat = self.vector_filtration(data, node_type)  # (N, 4)
        lam = self.slice_weights()                      # (K, 4)
        return lam @ Fmat.t()                           # (K, N)

    # ------------------------------------------------------------- descriptors
    def _slice_descriptors_for_graph(
        self,
        f_all: torch.Tensor,
        graph: nx.Graph,
        src: Sequence[int],
        dst: Sequence[int],
        num_nodes: int,
    ) -> torch.Tensor:
        """Per-pair, per-slice vicinity descriptors on one graph.

        Returns ``(B, K, PER_SLICE_DIM)``. ``f_all`` is ``(K, N)`` and indexing it
        keeps gradients flowing into the slice fields.
        """
        K = f_all.size(0)
        rows: List[torch.Tensor] = []
        for u, v in zip(src, dst):
            u, v = int(u), int(v)
            if not (0 <= u < num_nodes and 0 <= v < num_nodes
                    and u in graph and v in graph):
                rows.append(torch.zeros(K, PER_SLICE_DIM))
                continue
            su = f_all[:, u]
            sv = f_all[:, v]
            diff = (su - sv).abs()
            nu = list(graph.neighbors(u))
            nv = list(graph.neighbors(v))
            mu = f_all[:, nu].mean(dim=1) if nu else su
            mv = f_all[:, nv].mean(dim=1) if nv else sv
            vic = set(nx.single_source_shortest_path_length(graph, u, cutoff=self.k))
            vic |= set(nx.single_source_shortest_path_length(graph, v, cutoff=self.k))
            vic |= {u, v}
            mvic = f_all[:, sorted(vic)].mean(dim=1)
            rows.append(torch.stack([su, sv, diff, mu, mv, mvic], dim=-1))  # (K, D)
        return torch.stack(rows, dim=0)  # (B, K, D)

    # ----------------------------------------------------------------- fusion
    def _fuse_slices(self, slice_feats: torch.Tensor) -> torch.Tensor:
        """Attention-fuse ``(B, K, D)`` over the K slices -> ``(B, D)``."""
        h = torch.tanh(self.attn_transform(slice_feats))   # (B, K, attn_hidden)
        scores = self.attn_score(h).squeeze(-1)            # (B, K)
        w = F.softmax(scores, dim=1)                       # (B, K)
        return (w.unsqueeze(-1) * slice_feats).sum(dim=1)  # (B, D)

    def forward(
        self,
        data,
        bundles: Dict[str, object],
        pair_batch: PairBatch,
    ) -> torch.Tensor:
        """Fused per-pair multi-slice features, shape ``(B, fused_dim)``.

        ``bundles`` maps meta-path name -> ``ProjectedGraphBundle`` (same target
        type as ``pair_batch.src_type``). Per-slice descriptors are computed on
        every bundle graph and averaged across meta-paths.
        """
        node_type = pair_batch.src_type
        if pair_batch.dst_type != node_type:
            raise ValueError(
                "multi-slice filtration expects same-type pairs "
                f"(got src_type={node_type!r}, dst_type={pair_batch.dst_type!r})"
            )
        f_all = self.slice_fields(data, node_type)  # (K, N)
        src = pair_batch.src.tolist()
        dst = pair_batch.dst.tolist()

        per_graph: List[torch.Tensor] = []
        for bundle in bundles.values():
            if bundle.target_type != node_type:
                continue
            per_graph.append(
                self._slice_descriptors_for_graph(
                    f_all, bundle.graph, src, dst, int(bundle.num_nodes)
                )
            )
        if not per_graph:
            raise ValueError(
                f"no meta-path bundle has target_type {node_type!r}; cannot build "
                "multi-slice features."
            )
        slice_feats = torch.stack(per_graph, dim=0).mean(dim=0)  # (B, K, D)
        return self._fuse_slices(slice_feats)                    # (B, D)


def compute_multi_slice_features(
    data,
    bundles: Dict[str, object],
    pair_batch: PairBatch,
    num_slices: int = 3,
    filt_hidden: int = 16,
    attn_hidden: int = 16,
    k: int = 2,
    seed: int = 0,
) -> torch.Tensor:
    """Convenience: build a :class:`MultiSliceFiltration` and return ``(B, fused_dim)``.

    Seed-controlled and reproducible. This is the
    ``sliced filtration approximation (NOT exact multi-parameter persistent
    homology)`` -- no exact-PH / PDGNN / EPD claims.
    """
    model = MultiSliceFiltration(
        num_slices=num_slices,
        filt_hidden=filt_hidden,
        attn_hidden=attn_hidden,
        k=k,
        seed=seed,
    )
    model.eval()
    with torch.no_grad():
        return model(data, bundles, pair_batch)
