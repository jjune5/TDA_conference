# baselines/TLCGNN_gated.py
"""Adaptive PI Gating variant of TLC-GNN.

Identical to TLCGNN.Net except for a GatingNet that emits a per-edge
gate ∈ [0,1] multiplied into the persistence-image contribution before
the final MLP. When gate=0, the model degenerates to no-PI; when gate=1,
it is identical to TLC-GNN exact.

Gate inputs (per-edge features, computed in decode):
  [clustering coefficient of u, clustering coefficient of v, |emb_u−emb_v|_2]
"""
from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import Softmax
from torch_geometric.nn import GCNConv


class GatingNet(nn.Module):
    """Tiny MLP that maps per-edge features → gate in [0, 1]."""

    def __init__(self, in_dim: int = 3, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, edge_feats: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(edge_feats)).squeeze(-1)


def gated_decode(sqdist: torch.Tensor, PI: torch.Tensor,
                 gates: torch.Tensor) -> torch.Tensor:
    """Combine sqdist (E, 16) and PI (E, 25) with per-edge gates (E,) → (E, 41)."""
    gated_PI = gates.unsqueeze(-1) * PI  # broadcast
    return torch.cat([sqdist, gated_PI], dim=-1)


class Net(nn.Module):
    """TLC-GNN.Net with optional adaptive gating.

    Same encoder (2-layer GCN: in→100→16). Decoder concatenates
    [sqdist(emb_u−emb_v), gate × PI(u,v)] then MLP.
    """

    def __init__(self, data, num_features: int, num_classes: int, PI,
                 dimension: int = 5, clustering: Optional[np.ndarray] = None):
        super().__init__()
        self.conv1 = GCNConv(num_features, 100, cached=True)
        self.conv2 = GCNConv(100, 16, cached=True)
        self.PI = PI
        self.clustering = clustering  # (N,) per-node clustering coef
        self.leakyrelu = nn.LeakyReLU(0.2, True)
        self.linear_1 = nn.Linear(dimension * dimension + 16, dimension * dimension, bias=True)
        self.linear = nn.Linear(dimension * dimension, 1, bias=True)
        self.softmax = Softmax(dim=1)
        self.gate_net = GatingNet(in_dim=3, hidden=16)

    def encode(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        return x

    def _edge_features_for_gate(self, total_edges, emb_in, emb_out):
        """Build per-edge feature tensor (E, 3)."""
        E = total_edges.shape[0]
        device = emb_in.device
        if self.clustering is not None:
            u_idx = total_edges[:, 0]
            v_idx = total_edges[:, 1]
            cl_u = torch.from_numpy(self.clustering[u_idx]).float().to(device)
            cl_v = torch.from_numpy(self.clustering[v_idx]).float().to(device)
        else:
            cl_u = torch.zeros(E, device=device)
            cl_v = torch.zeros(E, device=device)
        emb_dist = (emb_in - emb_out).norm(dim=-1)
        return torch.stack([cl_u, cl_v, emb_dist], dim=-1)

    def decode(self, data, emb, type='train'):
        if type == 'train':
            edges_pos = data.total_edges[:data.train_pos]
            index = np.random.randint(0, data.train_neg, data.train_pos)
            edges_neg = data.total_edges[data.train_pos:data.train_pos + data.train_neg][index]
            total_edges = np.concatenate((edges_pos, edges_neg))
            edges_y = torch.cat((data.total_edges_y[:data.train_pos],
                                  data.total_edges_y[data.train_pos:data.train_pos + data.train_neg][index]))
            PI = np.concatenate(
                (self.PI[:data.train_pos], self.PI[data.train_pos:data.train_pos + data.train_neg][index]))
        elif type == 'val':
            total_edges = data.total_edges[data.train_pos+data.train_neg:data.train_pos+data.train_neg+data.val_pos+data.val_neg]
            edges_y = data.total_edges_y[data.train_pos+data.train_neg:data.train_pos+data.train_neg+data.val_pos+data.val_neg]
            PI = self.PI[data.train_pos+data.train_neg:data.train_pos+data.train_neg+data.val_pos+data.val_neg]
        elif type == 'test':
            total_edges = data.total_edges[data.train_pos+data.train_neg+data.val_pos+data.val_neg:]
            edges_y = data.total_edges_y[data.train_pos+data.train_neg+data.val_pos+data.val_neg:]
            PI = self.PI[data.train_pos+data.train_neg+data.val_pos+data.val_neg:]

        emb = emb.renorm(2, 0, 1)
        new_x = torch.tensor(PI.reshape((len(total_edges), -1)), dtype=torch.float, device=emb.device)
        emb_in = emb[total_edges[:, 0]]
        emb_out = emb[total_edges[:, 1]]
        sqdist = (emb_in - emb_out).pow(2)
        edge_feats = self._edge_features_for_gate(total_edges, emb_in, emb_out)
        gates = self.gate_net(edge_feats)
        feats = gated_decode(sqdist, new_x, gates)
        feats = self.leakyrelu(self.linear_1(feats))
        feats = torch.abs(self.linear(feats)).reshape(-1)
        feats = torch.clamp(feats, min=0, max=40)
        prob = 1. / (torch.exp((feats - 2.0) / 1.0) + 1.0)
        return prob, edges_y.float()


def call(data, name, num_features, num_classes, data_cnt, use_pi: bool = True):
    """Drop-in replacement for TLCGNN.call that returns the gated Net."""
    from baselines.TLCGNN import call as orig_call
    model, data = orig_call(data, name, num_features, num_classes, data_cnt, use_pi=use_pi)
    # Replace the model with gated version
    import networkx as nx
    # Build clustering coefficient
    g = nx.Graph()
    g.add_nodes_from(range(data.num_nodes))
    ei = data.edge_index.cpu().numpy()
    g.add_edges_from(((int(ei[0, i]), int(ei[1, i])) for i in range(ei.shape[1])))
    clustering_dict = nx.clustering(g)
    cl_arr = np.array([clustering_dict.get(i, 0.0) for i in range(data.num_nodes)], dtype=np.float32)
    gated_model = Net(data, num_features, num_classes, PI=model.PI, clustering=cl_arr).to(data.x.device)
    return gated_model, data
