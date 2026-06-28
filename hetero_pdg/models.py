"""HeteroTopoLinkPredictor: minimal typed link predictor with switchable topology.

Per node type a small input-projection MLP produces node embeddings. For a PairBatch
we gather z_src, z_dst and score with an MLP over
``concat([z_src, z_dst, z_src*z_dst, |z_src - z_dst|, topology_features])``.

Topology modes:
  no_topology                  -- node embeddings only.
  collapsed_topology           -- one descriptor vector from the collapsed graph.
  metapath_topology_concat     -- per-meta-path descriptors, masked + concatenated.
  metapath_topology_attention  -- per-meta-path descriptors, attention-pooled over
                                  valid meta-paths (weights sum to 1; missing masked).
  unified_filter_topology      -- one vector from the type-aware unified filtration.

No HAN/HGT/GTN/real-PDGNN here (Phase 1). The model only consumes precomputed
topology tensors, so a Phase-2 PDGNN adapter can feed it unchanged.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hetero_pdg.data import PairBatch


class HeteroTopoLinkPredictor(nn.Module):
    MODES = (
        "no_topology",
        "collapsed_topology",
        "metapath_topology_concat",
        "metapath_topology_attention",
        "unified_filter_topology",
    )

    def __init__(self, node_in_dims: Dict[str, int], mode: str,
                 hidden_dim: int = 32, topo_dim: int = 6, topo_hidden_dim: int = 32,
                 n_metapaths: int = 2, unified_dim: int = 5):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"unknown mode {mode!r}; expected one of {self.MODES}")
        self.mode = mode
        self.hidden_dim = hidden_dim
        self.topo_dim = topo_dim
        self.topo_hidden_dim = topo_hidden_dim
        self.n_metapaths = n_metapaths
        self.unified_dim = unified_dim

        self.proj = nn.ModuleDict({
            nt: nn.Sequential(nn.Linear(dim, hidden_dim), nn.ReLU(),
                              nn.Linear(hidden_dim, hidden_dim))
            for nt, dim in node_in_dims.items()
        })

        if mode == "no_topology":
            topo_contrib = 0
        elif mode == "collapsed_topology":
            topo_contrib = topo_dim
        elif mode == "metapath_topology_concat":
            topo_contrib = topo_dim * n_metapaths
        elif mode == "metapath_topology_attention":
            self.attn_transform = nn.Linear(topo_dim, topo_hidden_dim)
            self.attn_score = nn.Linear(topo_hidden_dim, 1)
            topo_contrib = topo_hidden_dim
        else:  # unified_filter_topology
            topo_contrib = unified_dim

        self.scorer = nn.Sequential(
            nn.Linear(4 * hidden_dim + topo_contrib, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def encode(self, node_features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {nt: self.proj[nt](x) for nt, x in node_features.items() if nt in self.proj}

    def compute_attention(self, topo: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Attention weights over meta-paths. topo (B,M,topo_dim), mask (B,M) -> (B,M)."""
        scores = self.attn_score(torch.tanh(self.attn_transform(topo))).squeeze(-1)  # (B,M)
        scores = scores.masked_fill(~mask, float("-inf"))
        w = F.softmax(scores, dim=-1)
        return torch.nan_to_num(w, nan=0.0)  # fully-masked rows -> 0

    def _topo_part(self, topo, mask) -> torch.Tensor:
        if self.mode == "metapath_topology_concat":
            if mask is not None:
                topo = topo * mask.unsqueeze(-1).float()
            return topo.reshape(topo.size(0), -1)               # (B, M*topo_dim)
        if self.mode == "metapath_topology_attention":
            if mask is None:
                mask = torch.ones(topo.shape[:2], dtype=torch.bool, device=topo.device)
            w = self.compute_attention(topo, mask)              # (B, M)
            h = torch.tanh(self.attn_transform(topo))           # (B, M, topo_hidden)
            return (w.unsqueeze(-1) * h).sum(dim=1)             # (B, topo_hidden)
        return topo                                             # collapsed / unified (B, dim)

    def forward(self, node_features: Dict[str, torch.Tensor], pair_batch: PairBatch,
                topo: Optional[torch.Tensor] = None,
                topo_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        z = self.encode(node_features)
        z_src = z[pair_batch.src_type][pair_batch.src]
        z_dst = z[pair_batch.dst_type][pair_batch.dst]
        parts = [z_src, z_dst, z_src * z_dst, (z_src - z_dst).abs()]
        if self.mode != "no_topology":
            if topo is None:
                raise ValueError(f"mode {self.mode!r} requires a topology tensor")
            parts.append(self._topo_part(topo, topo_mask))
        return self.scorer(torch.cat(parts, dim=-1)).squeeze(-1)
