"""PDGNN-predicted EPD features on meta-path graphs (option 2: retrain PDGNN).

Workflow (no exact PI as a FEATURE; exact EPD used only as one-time training labels,
which is intrinsic to PDGNN as a neural EPD approximator):
  1. gen_training_samples: sample target nodes from a meta-path graph; for each node
     v and HKS scale k, build its k-hop ego-graph with HKS node filter, compute the
     EXACT extended persistence diagram (gudhi, lower-star) as the training LABEL.
  2. train_pdgnn_metapath: train PDGNN to map (node filter, edge_index) -> per-edge
     (birth, death), supervised by the exact EPD via the Hungarian/bipartite loss.
  3. predict_node_pi: run the trained PDGNN per node-ego -> predicted EPD -> 5x5 PI.
     Output (N, 25*K). At inference NO exact computation is used.

The trained PDGNN approximates node_ph_features.phi_A; for large/dense meta-path
graphs (ogbn-mag, PSP) where exact PH OOMs, only this PDGNN path is feasible.
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import networkx as nx
import torch
import gudhi
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Knowledge_Distillation.pdgnn_modern import PDGNN
from Knowledge_Distillation.train_pdgnn_lp import _bipartite_loss
from node_ph_features import _diagram_points, PI_RES
from sg2dgm import PersistenceImager as pimg_mod
from types import SimpleNamespace

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _graph_hks(g: nx.Graph, K: int):
    """(N, K) multi-scale HKS on the meta-path graph (reuse compute_hks_features)."""
    from diffusion_features import compute_hks_features
    n = g.number_of_nodes()
    ei = np.array(list(g.edges())).T if g.number_of_edges() else np.zeros((2, 0), int)
    if ei.size:
        ei = np.concatenate([ei, ei[[1, 0]]], axis=1)
    data = SimpleNamespace(edge_index=torch.tensor(ei, dtype=torch.long), num_nodes=n)
    hks, _ = compute_hks_features(data, K=K, verbose=False)
    return hks


def _ego_filt_edges(g: nx.Graph, center: int, hop: int, node_filt: dict, max_nodes: int):
    """Return (filt (m,1) float, edge_index (2,E) long, node_list) for the ego-graph,
    with local re-indexing. node_filt: {global_node: scalar}."""
    ego = nx.ego_graph(g, center, radius=hop)
    if ego.number_of_nodes() > max_nodes:
        keep = sorted(ego.nodes(), key=lambda nd: node_filt.get(nd, 0.0))[:max_nodes]
        ego = ego.subgraph(keep).copy()
    nodes = list(ego.nodes())
    if len(nodes) == 0:
        return None
    remap = {nd: i for i, nd in enumerate(nodes)}
    filt = np.array([node_filt.get(nd, 0.0) for nd in nodes], dtype=np.float64)
    if ego.number_of_edges() == 0:
        ei = np.zeros((2, 0), dtype=np.int64)
    else:
        e = np.array([(remap[u], remap[v]) for u, v in ego.edges()], dtype=np.int64).T
        ei = np.concatenate([e, e[[1, 0]]], axis=1)   # symmetric for PDGNN
    return filt, ei, nodes


def _exact_epd(filt: np.ndarray, ei: np.ndarray):
    """Lower-star exact EPD (birth,death) points for a small ego (training label)."""
    st = gudhi.SimplexTree()
    for i, f in enumerate(filt):
        st.insert([int(i)], filtration=float(f))
    # undirected unique edges from symmetric ei
    seen = set()
    for a, b in zip(ei[0], ei[1]):
        e = (int(min(a, b)), int(max(a, b)))
        if e in seen:
            continue
        seen.add(e)
        st.insert([e[0], e[1]], filtration=float(max(filt[e[0]], filt[e[1]])))
    max_filt = float(filt.max()) if filt.size else 1.0
    return np.array(_diagram_points(st, max_filt), dtype=np.float64)


def gen_training_samples(g: nx.Graph, hks: np.ndarray, hop: int, max_nodes: int,
                         n_samples: int, seed: int = 0):
    """Sample (node, scale) egos -> [(filt(m,1), ei(2,E), gt_epd(P,2)), ...]."""
    rng = np.random.RandomState(seed)
    n, K = hks.shape
    nodes = rng.choice(n, size=min(n_samples, n), replace=False)
    samples = []
    for v in nodes:
        for k in range(K):
            node_filt = {nd: float(hks[nd, k]) for nd in g.nodes()}
            res = _ego_filt_edges(g, int(v), hop, node_filt, max_nodes)
            if res is None:
                continue
            filt, ei, _ = res
            if ei.shape[1] == 0:
                continue
            gt = _exact_epd(filt, ei)
            if gt.size == 0:
                continue
            samples.append((filt.reshape(-1, 1).astype(np.float32),
                            ei.astype(np.int64), gt.astype(np.float32)))
    return samples


def train_pdgnn_metapath(samples, hidden=32, layers=3, epochs=30, lr=1e-3,
                         seed=1234, verbose=True):
    """Train PDGNN on meta-path ego EPD labels (Hungarian loss). Returns model."""
    torch.manual_seed(seed); np.random.seed(seed)
    model = PDGNN(hidden_dim=hidden, num_layers=layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(1, epochs + 1):
        model.train()
        order = np.random.permutation(len(samples))
        losses = []
        for idx in order:
            filt, ei, gt = samples[idx]
            ft = torch.tensor(filt, device=device)
            et = torch.tensor(ei, device=device)
            gtt = torch.tensor(gt, device=device)
            opt.zero_grad()
            pred = model(ft, et)
            loss = _bipartite_loss(pred, gtt)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)  # stabilize (loss-spike guard)
            opt.step()
            losses.append(loss.item())
        if verbose and (ep % 5 == 0 or ep == 1):
            print(f'    pdgnn ep {ep:3d} loss={np.mean(losses):.4f}')
    return model


@torch.no_grad()
def predict_node_pi(model, g: nx.Graph, hks: np.ndarray, hop: int, max_nodes: int,
                    verbose: bool = False) -> np.ndarray:
    """(N, 25*K) PDGNN-predicted EPD -> PI per node, per HKS scale. No exact compute."""
    model.eval()
    imager = pimg_mod.PersistenceImager(resolution=PI_RES)
    n, K = hks.shape
    out = np.zeros((n, PI_RES * PI_RES * K), dtype=np.float64)
    filts_by_k = [{nd: float(hks[nd, k]) for nd in g.nodes()} for k in range(K)]
    for v in range(n):
        for k in range(K):
            res = _ego_filt_edges(g, v, hop, filts_by_k[k], max_nodes)
            if res is None:
                continue
            filt, ei, _ = res
            if ei.shape[1] == 0:
                continue
            ft = torch.tensor(filt.reshape(-1, 1).astype(np.float32), device=device)
            et = torch.tensor(ei.astype(np.int64), device=device)
            pred = model(ft, et).cpu().numpy()
            pred = pred[pred[:, 1] > pred[:, 0]]          # keep death>birth
            if pred.size:
                out[v, k * 25:(k + 1) * 25] = np.asarray(
                    imager.transform(pred.astype(np.float64))).reshape(-1)
        if verbose and (v + 1) % 1000 == 0:
            print(f'    pdgnn_pi {v+1}/{n}')
    return out
