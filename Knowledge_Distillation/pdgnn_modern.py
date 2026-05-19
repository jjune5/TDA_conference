"""Modern PyG 2.x reimplementation of PDGNN (Yan et al., NeurIPS 2022).

The original code in this repo (Teacher_model.py, gat_conv.py, message_passing.py)
depends on torch-geometric 1.6 internals that no longer exist in PyG 2.x.
This file rewrites the model cleanly on top of PyG 2.x MessagePassing.

PDGNN approximates the 1D extended persistence diagram of a filtered graph by
predicting, for every edge (u,v), a (birth, death) coordinate pair. The pairing
is supervised via the ground-truth extended persistence diagram.

Architecture (paper §4.2):
- Input: filter value f(v) ∈ R per node.
- L PDGNN layers, each:
    h_u^l = AGG^l({MSG^l(h_v^{l-1}), v ∈ N(u)}, h_u^{l-1})
    AGG^l = SUM ⊕ MIN     (concat sum-pool and min-pool aggregations)
    MSG^l(h_v^{l-1}) = PReLU(alpha_{uv} * (h_u^{l-1} ⊕ h_v^{l-1} W^l))
    alpha_{uv} = edge weight (Ollivier-Ricci based, taken from the graph).
- After L layers: H ∈ R^{|V| × d_h}.
- For each edge (u,v): W_edge(h_u ⊕ h_v) → (b, d) ∈ R^2.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax
from torch_scatter import scatter_min, scatter_add


class PDGNNLayer(MessagePassing):
    """Single PDGNN message-passing layer with AGG = SUM ⊕ MIN."""

    def __init__(self, in_dim: int, out_dim: int):
        # We override aggregate, so the 'aggr' arg is unused.
        super().__init__(aggr=None, flow='source_to_target')
        self.lin_self = nn.Linear(in_dim, out_dim, bias=False)
        self.lin_msg = nn.Linear(2 * in_dim, out_dim, bias=False)
        self.lin_combine = nn.Linear(2 * out_dim + in_dim, out_dim, bias=True)
        self.act_msg = nn.PReLU(out_dim)
        self.act_out = nn.PReLU(out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: torch.Tensor | None = None) -> torch.Tensor:
        # propagate calls message() per edge then aggregate() per node.
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=x.device)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             dim_size=x.size(0))
        # Combine with self.
        combined = torch.cat([out, self.lin_self(x)], dim=-1)
        h = self.act_out(self.lin_combine(combined))
        return h

    def message(self, x_i: torch.Tensor, x_j: torch.Tensor,
                edge_weight: torch.Tensor) -> torch.Tensor:
        # MSG^k(h_v^{k-1}) = PReLU(alpha * (h_u || h_v W))
        msg = self.lin_msg(torch.cat([x_i, x_j], dim=-1))
        msg = self.act_msg(msg)
        return edge_weight.view(-1, 1) * msg

    def aggregate(self, inputs: torch.Tensor, index: torch.Tensor,
                  dim_size: int | None = None) -> torch.Tensor:
        # SUM ⊕ MIN concatenation
        sum_agg = scatter_add(inputs, index, dim=0, dim_size=dim_size)
        min_agg, _ = scatter_min(inputs, index, dim=0, dim_size=dim_size)
        # Replace +inf with 0 where there is no incoming edge.
        min_agg = torch.where(torch.isinf(min_agg), torch.zeros_like(min_agg), min_agg)
        return torch.cat([sum_agg, min_agg], dim=-1)


class PDGNN(nn.Module):
    """PDGNN: predict 1-dim extended persistence pairs for every edge."""

    def __init__(self, hidden_dim: int = 32, num_layers: int = 3, dropout: float = 0.0):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(PDGNNLayer(1, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(PDGNNLayer(hidden_dim, hidden_dim))
        self.dropout = dropout

        # Per-edge MLP for (birth, death) prediction.
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.PReLU(hidden_dim),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, filt_value: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: torch.Tensor | None = None,
                pred_edges: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            filt_value: (N, 1) filter values per node.
            edge_index: (2, E) edge index (assumed to include self-loops + reverse edges).
            edge_weight: optional (E,) edge weights (alpha_uv).
            pred_edges: optional (2, E_pred) edges to predict for. Defaults to edge_index.
        Returns:
            pred_pd: (E_pred, 2) tensor of (birth, death) predictions.
        """
        x = filt_value
        for layer in self.layers:
            x = layer(x, edge_index, edge_weight)
            if self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)
        if pred_edges is None:
            pred_edges = edge_index
        h_u = x[pred_edges[0]]
        h_v = x[pred_edges[1]]
        pd = self.edge_mlp(torch.cat([h_u, h_v], dim=-1))
        return pd
