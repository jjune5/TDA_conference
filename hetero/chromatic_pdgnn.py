"""ChromaticPDGNN: extends PDGNN to predict the 0-dim image/kernel/cokernel
persistence of the target-type-subgraph inclusion L into K. Node TYPE enters the
MECHANISM via a per-node color indicator concatenated to the HKS filter (input dim 2),
and three edge-heads emit one (birth,death) diagram per leg.

Reuses Knowledge_Distillation.pdgnn_modern.PDGNNLayer (SUM(+)MIN union-find emulation)
and the ego machinery from hetero.pdgnn_metapath. Exact labels come from
hetero.chromatic_labels (PDGNN-only: exact never used as a downstream feature).
Gudhi-free (the 'ordinary' achromatic baseline also comes from chromatic_labels).
"""
from __future__ import annotations
import os, sys
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Knowledge_Distillation.pdgnn_modern import PDGNNLayer
from Knowledge_Distillation.train_pdgnn_lp import _bipartite_loss
from hetero.pdgnn_metapath import _ego_filt_edges
from hetero.chromatic_labels import compute_chromatic_0dim
from node_ph_features import PI_RES
from sg2dgm import PersistenceImager as pimg_mod

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_LEGS = ('image', 'kernel', 'cokernel')


class ChromaticPDGNN(nn.Module):
    def __init__(self, hidden_dim: int = 32, num_layers: int = 3, dropout: float = 0.0):
        super().__init__()
        self.dropout = dropout
        dims = [2] + [hidden_dim] * num_layers          # input = [filt, color]
        self.layers = nn.ModuleList(
            PDGNNLayer(dims[i], dims[i + 1]) for i in range(num_layers))

        def _head():
            return nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim), nn.PReLU(hidden_dim),
                nn.Linear(hidden_dim, 2))
        self.head_image = _head()
        self.head_kernel = _head()
        self.head_cokernel = _head()

    def forward(self, filt_value: torch.Tensor, color: torch.Tensor,
                edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor] = None,
                pred_edges: Optional[torch.Tensor] = None):
        x = torch.cat([filt_value, color], dim=-1)      # (N,2)
        for layer in self.layers:
            x = layer(x, edge_index, edge_weight)
            if self.dropout > 0:
                x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        if pred_edges is None:
            pred_edges = edge_index
        h = torch.cat([x[pred_edges[0]], x[pred_edges[1]]], dim=-1)
        return {'image': self.head_image(h),
                'kernel': self.head_kernel(h),
                'cokernel': self.head_cokernel(h)}


def gen_chromatic_samples(g, hks, ntype, target_type, hop, max_nodes, n_samples, seed=0):
    """Sample (node, scale) egos on the UNIFIED graph -> per-leg exact labels.
    Returns [(filt(m,1), color(m,1), ei(2,E), {leg: gt(P,2)}), ...]."""
    rng = np.random.RandomState(seed)
    n, K = hks.shape
    ntype = np.asarray(ntype)
    gmax = [float(hks[:, k].max()) for k in range(K)]   # global per-scale cap so essential
    # (cross-type-only) kernel classes stay visible & comparable across egos (the
    # synthetic positive control showed per-ego max drops top-born kernels).
    nodes = rng.choice(n, size=min(n_samples, n), replace=False)
    samples = []
    for v in nodes:
        for k in range(K):
            node_filt = {nd: float(hks[nd, k]) for nd in g.nodes()}
            res = _ego_filt_edges(g, int(v), hop, node_filt, max_nodes)
            if res is None:
                continue
            filt, ei, nodelist = res
            if ei.shape[1] == 0:
                continue
            is_L = (ntype[np.array(nodelist)] == target_type)
            labels = compute_chromatic_0dim(filt, ei, is_L, max_f=gmax[k])
            if all(labels[L].shape[0] == 0 for L in _LEGS):
                continue
            color = is_L.astype(np.float32).reshape(-1, 1)
            samples.append((filt.reshape(-1, 1).astype(np.float32), color,
                            ei.astype(np.int64),
                            {L: labels[L].astype(np.float32) for L in _LEGS}))
    return samples


def train_chromatic_pdgnn(samples, hidden=32, layers=3, epochs=30, lr=1e-3,
                          seed=1234, verbose=True):
    """Train ChromaticPDGNN on per-leg exact labels (sum of per-leg Hungarian)."""
    torch.manual_seed(seed); np.random.seed(seed)
    model = ChromaticPDGNN(hidden_dim=hidden, num_layers=layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(1, epochs + 1):
        model.train(); order = np.random.permutation(len(samples)); losses = []
        for idx in order:
            filt, color, ei, gts = samples[idx]
            ft = torch.tensor(filt, device=device)
            ct = torch.tensor(color, device=device)
            et = torch.tensor(ei, device=device)
            opt.zero_grad()
            pred = model(ft, ct, et)
            loss = 0.0
            for L in _LEGS:
                gt = gts[L]
                if gt.shape[0] == 0:
                    continue
                loss = loss + _bipartite_loss(pred[L], torch.tensor(gt, device=device))
            if not torch.is_tensor(loss) or not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); losses.append(float(loss))
        if verbose and (ep % 5 == 0 or ep == 1):
            mu = np.mean(losses) if losses else float('nan')
            print(f'    chromatic-pdgnn ep {ep:3d} loss={mu:.4f}')
    return model


@torch.no_grad()
def predict_node_chromatic_pi(model, g, hks, ntype, target_type, hop, max_nodes,
                              verbose=False):
    """(N, 3*25*K): per-node image/kernel/cokernel PI under unified-graph ego. No exact compute."""
    model.eval()
    imager = pimg_mod.PersistenceImager(resolution=PI_RES)
    n, K = hks.shape
    ntype = np.asarray(ntype)
    per_leg = PI_RES * PI_RES
    out = np.zeros((n, 3 * per_leg * K), dtype=np.float64)
    filts_by_k = [{nd: float(hks[nd, k]) for nd in g.nodes()} for k in range(K)]
    for v in range(n):
        for k in range(K):
            res = _ego_filt_edges(g, v, hop, filts_by_k[k], max_nodes)
            if res is None:
                continue
            filt, ei, nodelist = res
            if ei.shape[1] == 0:
                continue
            color = (ntype[np.array(nodelist)] == target_type).astype(np.float32).reshape(-1, 1)
            ft = torch.tensor(filt.reshape(-1, 1).astype(np.float32), device=device)
            ct = torch.tensor(color, device=device)
            et = torch.tensor(ei.astype(np.int64), device=device)
            pred = model(ft, ct, et)
            for li, L in enumerate(_LEGS):
                pts = pred[L].cpu().numpy()
                pts = pts[pts[:, 1] > pts[:, 0]]
                base = (li * K + k) * per_leg
                if pts.size:
                    out[v, base:base + per_leg] = np.asarray(
                        imager.transform(pts.astype(np.float64))).reshape(-1)
        if verbose and (v + 1) % 1000 == 0:
            print(f'    chromatic_pi {v+1}/{n}')
    return out
