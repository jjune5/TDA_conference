"""diffusion_features.py — Variant A of the diffusion-features LP experiment.

Spec: docs/superpowers/specs/2026-05-28-diffusion-features-lp-design.md

Hypothesis (§14 follow-up): exact vicinity-PI is a train-graph-membership
artifact — for a candidate edge (u,v) the persistence is >> 0 only when (u,v)
is *in* the graph (train_pos); at test the edge is removed → vicinity collapses
→ PI ≈ 0 → no genuine test signal.

Variant A instead uses a NODE-LEVEL multi-scale Heat Kernel Signature (HKS),
computed from the GLOBAL graph Laplacian on the leakage-free training graph
(val/test positives removed, exactly as the PI cache is built). Removing one
candidate edge barely changes a node's global HKS, so an edge feature built from
node HKS should NOT collapse at test → it can supply genuine test-time signal.

    HKS_t(i) = sum_k exp(-t * lambda_k) * phi_k(i)^2

from the normalized Laplacian L = I - D^{-1/2} A D^{-1/2} (eigh on GPU), for K
log-spaced diffusion times t (local -> global). Per candidate edge (u,v):

    feat(u,v) = concat[ HKS(u) (K), HKS(v) (K), |HKS(u) - HKS(v)| (K) ]  -> 3K-dim

Two measurements, both on Cora (homophilic) + Chameleon (heterophilic):

 1. §14 DIAGNOSTIC: per-segment (train_pos / test_pos / test_neg), does the
    feature alone discriminate test_pos from test_neg? Logistic-regression
    stratified-CV AUC on the feature over the *test* edges (mirrors the S3
    pi_separability per-segment pattern in pi_artifact_analysis.py). Prediction:
    A discriminates at test (AUC > 0.5), unlike exact PI which collapses to ~0.5.

 2. LP: GCN encoder + a decoder that concatenates the 3K diffusion feature
    (mirrors how baselines/TLCGNN.py concatenates the PI). 50 trials,
    no-PI vs exact-PI vs A(node-HKS).

Outputs -> results/diffusion_feat_A/:
    diagnostic_auc.csv  — §14 per-segment feature-only CV-AUC (A vs exact-PI)
    lp_auc.csv          — end-to-end LP AUC (no-PI / exact-PI / A), mean±std
    raw_scores.json     — all raw per-trial AUCs
    summary.txt         — human-readable tables + verdict
"""

from __future__ import annotations

import os
import sys
import csv
import copy
import json
import time
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
import torch_geometric.datasets
import scipy.sparse as sp
import networkx as nx

from torch_geometric.nn import GCNConv
from torch.nn.init import xavier_normal_ as xavier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score

# ── repo root so relative paths (./data/...) work regardless of cwd ───────────
_REPO = os.path.dirname(os.path.abspath(__file__))
os.chdir(_REPO)

# ── hyperparameters (mirror gnn_backbone_exp.py / baselines/TLCGNN.py) ────────
HIDDEN = 100
OUT = 16
DROPOUT = 0.5
LR = 0.005
EPOCHS = 2000
EARLY_STOP = 200
PI_DIM = 5            # exact PI is 5x5 = 25-dim
K_SCALES = 5         # number of HKS diffusion scales (local -> global)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ── data loading (mirror loaddatas.py / gnn_backbone_exp.py) ──────────────────

def load_dataset(name):
    if name == 'Cora':
        return torch_geometric.datasets.Planetoid(
            './data/Cora', 'Cora', transform=T.NormalizeFeatures())
    elif name == 'Chameleon':
        return torch_geometric.datasets.WikipediaNetwork('./data/Chameleon', 'chameleon')
    else:
        raise ValueError(f'Unknown dataset: {name}')


def get_edges_split(data, val_prop=0.05, test_prop=0.1, seed=1234):
    """Deterministic edge split (seed=1234) — identical to gnn_backbone_exp.py,
    which is what regenerated the current data/TLCGNN/<name>.npy PI caches
    (capped train_neg). Layout: [train_pos|train_neg|val_pos|val_neg|test_pos|test_neg]."""
    np.random.seed(seed)
    g = nx.Graph()
    g.add_nodes_from(range(len(data.y)))
    ei = np.array(data.edge_index)
    g.add_edges_from([(int(ei[0, i]), int(ei[1, i])) for i in range(ei.shape[1])])
    adj = nx.adjacency_matrix(g)

    x, y = sp.triu(adj).nonzero()
    pos_edges = np.stack([x, y], axis=1)
    np.random.shuffle(pos_edges)

    N = adj.shape[0]
    adj_bool = adj.astype(bool).toarray()
    upper = np.triu(np.ones((N, N), dtype=bool), k=1)
    neg_mask = (~adj_bool) & upper
    nx_idx, ny_idx = np.where(neg_mask)
    neg_edges = np.stack([nx_idx, ny_idx], axis=1)
    np.random.shuffle(neg_edges)

    m_pos = len(pos_edges)
    n_val = int(m_pos * val_prop)
    n_test = int(m_pos * test_prop)
    val_edges = pos_edges[:n_val]
    test_edges = pos_edges[n_val:n_test + n_val]
    train_edges = pos_edges[n_test + n_val:]
    val_edges_false = neg_edges[:n_val]
    test_edges_false = neg_edges[n_val:n_test + n_val]
    cap = min(len(neg_edges), max(int(len(train_edges) * 5), 1024))
    train_edges_false = neg_edges[n_test + n_val: n_test + n_val + cap]

    return train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false


def prepare_data(raw_data):
    """Edge split + remove val/test positives from the training graph.
    Returns a data copy carrying the segment counts + total_edges layout, plus
    the raw split tuple."""
    data = copy.deepcopy(raw_data)
    (train_edges, train_edges_false,
     val_edges, val_edges_false,
     test_edges, test_edges_false) = get_edges_split(data)

    total_edges = np.concatenate(
        [train_edges, train_edges_false, val_edges, val_edges_false,
         test_edges, test_edges_false])
    data.train_pos = len(train_edges); data.train_neg = len(train_edges_false)
    data.val_pos = len(val_edges);     data.val_neg = len(val_edges_false)
    data.test_pos = len(test_edges);   data.test_neg = len(test_edges_false)
    data.total_edges = total_edges
    data.total_edges_y = torch.cat([
        torch.ones(len(train_edges)), torch.zeros(len(train_edges_false)),
        torch.ones(len(val_edges)),   torch.zeros(len(val_edges_false)),
        torch.ones(len(test_edges)),  torch.zeros(len(test_edges_false)),
    ]).long()

    # remove val/test positive edges from the training graph (leakage-free)
    ei = np.array(data.edge_index)
    remove = set()
    for e in val_edges.tolist():
        remove.add((e[0], e[1])); remove.add((e[1], e[0]))
    for e in test_edges.tolist():
        remove.add((e[0], e[1])); remove.add((e[1], e[0]))
    keep = np.array([(int(ei[0, i]), int(ei[1, i])) not in remove
                     for i in range(ei.shape[1])])
    data.edge_index = torch.from_numpy(ei[:, keep]).long()

    return data, (train_edges, train_edges_false, val_edges, val_edges_false,
                  test_edges, test_edges_false)


def segment_bounds(data):
    """[train_pos|train_neg|val_pos|val_neg|test_pos|test_neg] -> {name:(lo,hi)}."""
    counts = [data.train_pos, data.train_neg, data.val_pos, data.val_neg,
              data.test_pos, data.test_neg]
    names = ['train_pos', 'train_neg', 'val_pos', 'val_neg', 'test_pos', 'test_neg']
    bounds, c = {}, 0
    for n, k in zip(names, counts):
        bounds[n] = (c, c + k); c += k
    return bounds, c


# ── multi-scale HKS node features (GPU eigendecomp) ───────────────────────────

def compute_hks_features(data, K=K_SCALES, dev=None, verbose=True):
    """Compute the (n, K) multi-scale HKS node-feature matrix on the (already
    leakage-free) training graph carried by `data.edge_index`.

        HKS_t(i) = sum_k exp(-t * lambda_k) * phi_k(i)^2

    Eigendecomposition of the normalized Laplacian L = I - D^{-1/2} A D^{-1/2}
    via torch.linalg.eigh on GPU. K diffusion times t are log-spaced over the
    spectrum (local -> global): t ranges from 1/lambda_max (fast, local detail)
    to 1/lambda_min_pos (slow, global structure), the standard HKS scale window.
    Each scale column is z-normalized (mean 0, std 1) across nodes for a clean,
    comparable per-edge feature.
    """
    dev = dev or device
    n = int(data.num_nodes)
    ei = np.array(data.edge_index.cpu())

    # build symmetric adjacency on GPU (training graph: val/test pos removed)
    A = torch.zeros((n, n), dtype=torch.float64, device=dev)
    src = torch.from_numpy(ei[0]).long().to(dev)
    dst = torch.from_numpy(ei[1]).long().to(dev)
    A[src, dst] = 1.0
    A[dst, src] = 1.0  # force symmetry
    A.fill_diagonal_(0.0)

    deg = A.sum(dim=1)
    dinv = torch.where(deg > 0, deg.pow(-0.5), torch.zeros_like(deg))
    Dinv = torch.diag(dinv)
    # L_norm = I - D^{-1/2} A D^{-1/2}   (isolated nodes -> row/col of zeros + 1 on diag)
    L = torch.eye(n, dtype=torch.float64, device=dev) - Dinv @ A @ Dinv

    t0 = time.time()
    lams, phis = torch.linalg.eigh(L)           # ascending eigenvalues, symmetric
    lams = torch.clamp(lams, min=0.0)           # clip tiny negatives (PSD)
    if verbose:
        print(f'    eigh: n={n} done in {time.time()-t0:.2f}s  '
              f'lambda range [{float(lams.min()):.4f}, {float(lams.max()):.4f}]')

    # choose K log-spaced diffusion times across the positive spectrum
    pos = lams[lams > 1e-8]
    if pos.numel() == 0:
        return np.zeros((n, K), dtype=np.float64), {'degenerate': True}
    lam_min = float(pos.min())
    lam_max = float(lams.max())
    # t_small -> resolves the fast (high-lambda) modes -> local;
    # t_large -> only the slow (low-lambda) modes survive -> global.
    t_small = 1.0 / lam_max
    t_large = 1.0 / lam_min
    if t_large <= t_small:
        t_large = t_small * 10.0
    ts = np.geomspace(t_small, t_large, K)
    ts_t = torch.tensor(ts, dtype=torch.float64, device=dev)        # (K,)

    phis2 = phis ** 2                                                # (n, n) phi_k(i)^2 in col k
    weights = torch.exp(-torch.outer(ts_t, lams))                   # (K, n_eig)
    hks = phis2 @ weights.t()                                       # (n, K)

    # z-normalize each scale across nodes
    mu = hks.mean(dim=0, keepdim=True)
    sd = hks.std(dim=0, keepdim=True)
    sd = torch.where(sd > 1e-12, sd, torch.ones_like(sd))
    hks = (hks - mu) / sd

    meta = {'degenerate': False, 'ts': ts.tolist(),
            'lam_min_pos': lam_min, 'lam_max': lam_max}
    return hks.cpu().numpy().astype(np.float64), meta


def build_edge_features(hks, edges):
    """Per-edge 3K-dim feature: concat[ HKS(u), HKS(v), |HKS(u)-HKS(v)| ]."""
    u = edges[:, 0]; v = edges[:, 1]
    hu = hks[u]; hv = hks[v]
    return np.concatenate([hu, hv, np.abs(hu - hv)], axis=1)


# ── §14 diagnostic: feature-only stratified-CV AUC (mirror pi_separability) ───

def feature_separability_auc(X, y, seed=0, n_splits=5):
    """Logistic-regression stratified-CV ROC-AUC of `X` (feature alone) vs label
    `y`. Mirrors the S3 pi_separability idea: can the feature, on its own,
    discriminate the two classes? Returns mean CV-AUC.

    Guards: needs both classes present and enough samples per fold; if a feature
    column is constant the scaler/LR still handle it (z-score -> 0)."""
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return float('nan')
    # cap folds so the minority class has >= 2 per fold
    min_class = min(np.bincount(y))
    k = max(2, min(n_splits, min_class))
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000, C=1.0))
    try:
        scores = cross_val_score(clf, X, y, cv=skf, scoring='roc_auc')
    except Exception as e:
        print(f'      [separability warn] {e}')
        return float('nan')
    return float(np.mean(scores))


def run_diagnostic(name, data, splits, hks, exact_pi, n_repeats=5):
    """For the TEST segment (and train segment as a within-graph control),
    compute feature-only CV-AUC discriminating pos vs neg, for both:
      - A: node-HKS 3K-dim edge feature
      - exact-PI: the 25-dim vicinity PI from data/TLCGNN/<name>.npy
    Averaged over n_repeats CV seeds.
    """
    (train_edges, train_edges_false, val_edges, val_edges_false,
     test_edges, test_edges_false) = splits
    bounds, _ = segment_bounds(data)

    # --- TEST segment: pos vs neg ---
    test_pos = test_edges
    test_neg = test_edges_false
    y_test = np.concatenate([np.ones(len(test_pos)), np.zeros(len(test_neg))])
    edges_test = np.concatenate([test_pos, test_neg])

    # A feature on test edges
    Xa_test = build_edge_features(hks, edges_test)
    # exact PI on test edges (slice the cache at the test segment)
    lo_p, hi_p = bounds['test_pos']; lo_n, hi_n = bounds['test_neg']
    Xpi_test = np.concatenate([exact_pi[lo_p:hi_p], exact_pi[lo_n:hi_n]], axis=0)

    # --- TRAIN segment control (subsample neg to balance, like pos count) ---
    rng = np.random.RandomState(123)
    n_tr = min(len(train_edges), 2000)
    tp_idx = rng.choice(len(train_edges), n_tr, replace=False)
    tn_idx = rng.choice(len(train_edges_false), n_tr, replace=False)
    train_pos_s = train_edges[tp_idx]
    train_neg_s = train_edges_false[tn_idx]
    y_train = np.concatenate([np.ones(n_tr), np.zeros(n_tr)])
    edges_train = np.concatenate([train_pos_s, train_neg_s])
    Xa_train = build_edge_features(hks, edges_train)
    lo_tp, hi_tp = bounds['train_pos']; lo_tn, hi_tn = bounds['train_neg']
    Xpi_train = np.concatenate([exact_pi[lo_tp:hi_tp][tp_idx],
                                exact_pi[lo_tn:hi_tn][tn_idx]], axis=0)

    out = {}
    for feat_name, (Xtr, Xte) in [
            ('A_nodeHKS', (Xa_train, Xa_test)),
            ('exact_PI', (Xpi_train, Xpi_test))]:
        tr_aucs = [feature_separability_auc(Xtr, y_train, seed=s) for s in range(n_repeats)]
        te_aucs = [feature_separability_auc(Xte, y_test, seed=s) for s in range(n_repeats)]
        out[feat_name] = {
            'train_auc': float(np.nanmean(tr_aucs)),
            'train_auc_std': float(np.nanstd(tr_aucs)),
            'test_auc': float(np.nanmean(te_aucs)),
            'test_auc_std': float(np.nanstd(te_aucs)),
        }
    return out


# ── LP model: GCN encoder + decoder concatenating an edge feature ─────────────

class GCNEncoder(torch.nn.Module):
    def __init__(self, in_feats):
        super().__init__()
        self.conv1 = GCNConv(in_feats, HIDDEN, cached=True)
        self.conv2 = GCNConv(HIDDEN, OUT, cached=True)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=DROPOUT, training=self.training)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=DROPOUT, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        return x


class LPModel(torch.nn.Module):
    """Mirrors baselines/TLCGNN.py Net: encode with GCN, decode with
    |emb_u - emb_v|^2 concatenated with an edge feature vector.

    mode:
      'none'  -> no edge feature (zeros), pure GCN baseline
      'pi'    -> the 25-dim exact vicinity PI (feat_array = exact_pi)
      'hks'   -> the 3K-dim node-HKS diffusion feature (feat_array = per-edge A)

    feat_array is a numpy array aligned with data.total_edges (row i is the
    feature of total_edges[i]); feat_dim is its width.
    """
    def __init__(self, in_feats, feat_array, feat_dim, mode):
        super().__init__()
        self.encoder = GCNEncoder(in_feats)
        self.feat = feat_array
        self.feat_dim = feat_dim
        self.mode = mode
        self.leakyrelu = torch.nn.LeakyReLU(0.2, True)
        self.linear_1 = torch.nn.Linear(feat_dim + OUT, feat_dim, bias=True)
        self.linear = torch.nn.Linear(feat_dim, 1, bias=True)

    def encode(self, data):
        return self.encoder(data.x, data.edge_index)

    def decode(self, data, emb, split='train'):
        tp = data.train_pos; tn = data.train_neg
        vp = data.val_pos;   vn = data.val_neg
        feat_slice = None
        if split == 'train':
            edges_pos = data.total_edges[:tp]
            idx = np.random.randint(0, tn, tp)
            edges_neg = data.total_edges[tp:tp + tn][idx]
            total_e = np.concatenate([edges_pos, edges_neg])
            edges_y = torch.cat([data.total_edges_y[:tp],
                                 data.total_edges_y[tp:tp + tn][idx]])
            if self.mode != 'none':
                feat_slice = np.concatenate([self.feat[:tp], self.feat[tp:tp + tn][idx]])
        elif split == 'val':
            start = tp + tn; end = start + vp + vn
            total_e = data.total_edges[start:end]
            edges_y = data.total_edges_y[start:end]
            if self.mode != 'none':
                feat_slice = self.feat[start:end]
        else:  # test
            start = tp + tn + vp + vn
            total_e = data.total_edges[start:]
            edges_y = data.total_edges_y[start:]
            if self.mode != 'none':
                feat_slice = self.feat[start:]

        emb = emb.renorm(2, 0, 1)

        if self.mode == 'none':
            feat_t = torch.zeros(len(total_e), self.feat_dim, device=emb.device)
        else:
            feat_t = torch.tensor(feat_slice.reshape(len(total_e), -1),
                                  dtype=torch.float32, device=emb.device)

        eu = emb[total_e[:, 0]]; ev = emb[total_e[:, 1]]
        sq = (eu - ev).pow(2)
        h = self.leakyrelu(self.linear_1(torch.cat([sq, feat_t], dim=1)))
        h = torch.abs(self.linear(h)).reshape(-1)
        h = torch.clamp(h, 0, 40)
        prob = 1.0 / (torch.exp((h - 2.0) / 1.0) + 1.0)
        return prob, edges_y.float()


def weights_init(m):
    if isinstance(m, torch.nn.Linear):
        xavier(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def run_one_trial(data_cpu, feat_array, feat_dim, mode, in_feats, seed, dev):
    setup_seed(seed)
    data = copy.deepcopy(data_cpu).to(dev)
    model = LPModel(in_feats, feat_array, feat_dim, mode).to(dev)
    model.apply(weights_init)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0)

    best_val_roc = 0.0
    test_roc = 0.0
    wait = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        opt.zero_grad()
        emb = model.encode(data)
        pred, y = model.decode(data, emb, 'train')
        loss = F.binary_cross_entropy(pred, y)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            emb = model.encode(data)
            vp, vy = model.decode(data, emb, 'val')
            vp = vp.cpu().numpy(); vy = vy.cpu().numpy()
            val_roc = roc_auc_score(vy, vp) if len(np.unique(vy)) >= 2 else 0.5
        if val_roc >= best_val_roc:
            best_val_roc = val_roc
            with torch.no_grad():
                tpp, ty = model.decode(data, emb, 'test')
                tpp = tpp.cpu().numpy(); ty = ty.cpu().numpy()
                test_roc = roc_auc_score(ty, tpp) if len(np.unique(ty)) >= 2 else 0.5
            wait = 0
        else:
            wait += 1
            if wait >= EARLY_STOP:
                break
    del model
    return test_roc


# ── PI cache loader (reuse exact cache; validate shape) ───────────────────────

def load_pi_cache(name, expected_total):
    for cand in (f'./data/TLCGNN/{name}.npy', f'./data/TLCGNN/{name.lower()}.npy'):
        if os.path.exists(cand):
            cached = np.load(cand)
            if cached.shape[0] != expected_total:
                raise ValueError(
                    f'PI cache shape mismatch for {name}: got {cached.shape[0]}, '
                    f'expected {expected_total}')
            return cached
    raise FileNotFoundError(f'PI cache not found for {name}')


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=['Cora', 'Chameleon'])
    parser.add_argument('--trials', type=int, default=50)
    parser.add_argument('--K', type=int, default=K_SCALES)
    parser.add_argument('--outdir', default='results/diffusion_feat_A')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    print(f'Device: {device}  |  datasets={args.datasets}  trials={args.trials}  K={args.K}')

    diagnostic = {}     # name -> {feat -> {train/test auc}}
    lp_raw = {}         # (name, mode) -> [aucs]
    hks_meta = {}

    for name in args.datasets:
        print(f'\n{"="*64}\n{name}\n{"="*64}')
        dataset = load_dataset(name)
        raw = dataset[0]
        in_feats = raw.x.size(1)
        data_prepped, splits = prepare_data(raw)
        bounds, total = segment_bounds(data_prepped)
        print(f'  segments: ' + ', '.join(f'{k}={v[1]-v[0]}' for k, v in bounds.items())
              + f'  total={total}')

        # exact PI cache (reused; no recompute)
        exact_pi = load_pi_cache(name, total)
        print(f'  exact PI cache: {exact_pi.shape}')

        # multi-scale HKS on the leakage-free training graph
        print('  computing multi-scale HKS (GPU eigendecomp)...')
        hks, meta = compute_hks_features(data_prepped, K=args.K)
        hks_meta[name] = meta
        if not meta.get('degenerate'):
            print(f'    K={args.K} diffusion times t = '
                  + ', '.join(f'{t:.3g}' for t in meta['ts']))
        # per-edge 3K feature aligned with total_edges (for LP mode 'hks')
        feat_hks_all = build_edge_features(hks, data_prepped.total_edges)
        feat_dim_hks = feat_hks_all.shape[1]
        print(f'    node-HKS edge feature dim = {feat_dim_hks} (3*K)')

        # ---- §14 diagnostic ----
        print('  §14 diagnostic: feature-only CV-AUC (test_pos vs test_neg)...')
        diag = run_diagnostic(name, data_prepped, splits, hks, exact_pi)
        diagnostic[name] = diag
        for fn, d in diag.items():
            print(f'    {fn:12} train-AUC={d["train_auc"]:.4f}±{d["train_auc_std"]:.4f}  '
                  f'TEST-AUC={d["test_auc"]:.4f}±{d["test_auc_std"]:.4f}')

        # ---- LP: no-PI / exact-PI / A(node-HKS) ----
        pi_dim = exact_pi.shape[1]
        modes = [
            ('none', None, pi_dim),                     # decoder width matches PI baseline
            ('pi',   exact_pi, pi_dim),
            ('hks',  feat_hks_all, feat_dim_hks),
        ]
        for mode, feat_arr, fdim in modes:
            tag = {'none': 'no-PI', 'pi': 'exact-PI', 'hks': 'A(node-HKS)'}[mode]
            print(f'  LP [{tag}] (feat_dim={fdim}) x {args.trials} trials...')
            aucs = []
            t0 = time.time()
            for trial in range(args.trials):
                auc = run_one_trial(data_prepped, feat_arr, fdim, mode,
                                    in_feats, seed=trial, dev=device)
                aucs.append(auc)
                if (trial + 1) % 10 == 0:
                    print(f'    trial {trial+1}/{args.trials}: mean={np.mean(aucs):.4f} '
                          f'[{time.time()-t0:.0f}s]')
            lp_raw[(name, mode)] = aucs
            print(f'    --> {tag}: mean={np.mean(aucs):.4f} std={np.std(aucs):.4f} '
                  f'median={np.median(aucs):.4f}')

    # ── write outputs ─────────────────────────────────────────────────────────
    # raw scores
    raw_out = {
        'lp': {f'{n}|{m}': v for (n, m), v in lp_raw.items()},
        'diagnostic': diagnostic,
        'hks_meta': hks_meta,
        'config': {'trials': args.trials, 'K': args.K, 'datasets': args.datasets},
    }
    with open(os.path.join(args.outdir, 'raw_scores.json'), 'w') as f:
        json.dump(raw_out, f, indent=2)

    # diagnostic CSV
    with open(os.path.join(args.outdir, 'diagnostic_auc.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['dataset', 'feature', 'train_auc', 'train_auc_std',
                    'test_auc', 'test_auc_std'])
        for name in args.datasets:
            for fn, d in diagnostic.get(name, {}).items():
                w.writerow([name, fn, f'{d["train_auc"]:.6f}', f'{d["train_auc_std"]:.6f}',
                            f'{d["test_auc"]:.6f}', f'{d["test_auc_std"]:.6f}'])

    # LP CSV
    with open(os.path.join(args.outdir, 'lp_auc.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['dataset', 'variant', 'mean_auc', 'std_auc', 'median_auc', 'n_trials'])
        for name in args.datasets:
            for mode in ['none', 'pi', 'hks']:
                if (name, mode) in lp_raw:
                    a = np.array(lp_raw[(name, mode)])
                    tag = {'none': 'no-PI', 'pi': 'exact-PI', 'hks': 'A_nodeHKS'}[mode]
                    w.writerow([name, tag, f'{a.mean():.6f}', f'{a.std():.6f}',
                                f'{np.median(a):.6f}', len(a)])

    # summary.txt
    write_summary(os.path.join(args.outdir, 'summary.txt'),
                  args, diagnostic, lp_raw, hks_meta)
    print(f'\nOutputs written to {args.outdir}/')


def write_summary(path, args, diagnostic, lp_raw, hks_meta):
    L = []
    L.append('=' * 72)
    L.append('Diffusion Features — Variant A (node-level multi-scale HKS)')
    L.append('Spec: docs/superpowers/specs/2026-05-28-diffusion-features-lp-design.md')
    L.append(f'Config: trials={args.trials}, K={args.K}, datasets={args.datasets}')
    L.append('=' * 72)
    L.append('')
    L.append('Feature: per candidate edge (u,v): concat[HKS(u), HKS(v), |HKS(u)-HKS(v)|]')
    L.append('  = 3K-dim, from normalized-Laplacian heat kernel on the leakage-free')
    L.append('  training graph (val/test pos removed). HKS_t(i)=sum_k exp(-t*lam_k)phi_k(i)^2.')
    L.append('')

    # --- §14 diagnostic table ---
    L.append('-' * 72)
    L.append('MEASUREMENT 1 — §14 DIAGNOSTIC')
    L.append('Feature-only logistic-regression stratified-CV AUC.')
    L.append('TEST-AUC = can the feature ALONE discriminate test_pos from test_neg?')
    L.append('(train-AUC shown as within-graph control.)')
    L.append('-' * 72)
    L.append(f'{"dataset":10} {"feature":12} {"train-AUC":>16} {"TEST-AUC":>16}')
    for name in args.datasets:
        for fn, d in diagnostic.get(name, {}).items():
            L.append(f'{name:10} {fn:12} '
                     f'{d["train_auc"]:.4f}±{d["train_auc_std"]:.4f}  '
                     f'{d["test_auc"]:.4f}±{d["test_auc_std"]:.4f}')
    L.append('')
    L.append('Interpretation: exact-PI is expected to collapse to TEST-AUC ~ 0.5')
    L.append('(membership artifact: PI=0 at test for both classes). Variant A is a')
    L.append('node-level global feature; if TEST-AUC > 0.5 it supplies genuine test signal.')
    L.append('')

    # --- LP table ---
    L.append('-' * 72)
    L.append('MEASUREMENT 2 — END-TO-END LP AUC (mean ± std over trials)')
    L.append('-' * 72)
    L.append(f'{"dataset":10} {"no-PI":>16} {"exact-PI":>16} {"A(node-HKS)":>16}')
    for name in args.datasets:
        def cell(mode):
            if (name, mode) in lp_raw:
                a = np.array(lp_raw[(name, mode)])
                return f'{a.mean():.4f}±{a.std():.4f}'
            return '—'
        L.append(f'{name:10} {cell("none"):>16} {cell("pi"):>16} {cell("hks"):>16}')
    L.append('')
    L.append('Gaps vs baselines:')
    for name in args.datasets:
        if all((name, m) in lp_raw for m in ['none', 'pi', 'hks']):
            none = np.array(lp_raw[(name, 'none')]).mean()
            pi = np.array(lp_raw[(name, 'pi')]).mean()
            hks = np.array(lp_raw[(name, 'hks')]).mean()
            L.append(f'  {name}: A-vs-noPI = {hks-none:+.4f}   '
                     f'A-vs-exactPI = {hks-pi:+.4f}   '
                     f'(exactPI-vs-noPI = {pi-none:+.4f})')
    L.append('')

    # --- verdict ---
    L.append('=' * 72)
    L.append('VERDICT')
    L.append('=' * 72)
    for name in args.datasets:
        diag = diagnostic.get(name, {})
        a_te = diag.get('A_nodeHKS', {}).get('test_auc', float('nan'))
        pi_te = diag.get('exact_PI', {}).get('test_auc', float('nan'))
        genuine = (a_te > 0.55)
        L.append(f'[{name}]')
        L.append(f'  §14: node-HKS TEST-AUC = {a_te:.4f}  vs  exact-PI TEST-AUC = {pi_te:.4f}')
        L.append(f'       node-HKS gives genuine test signal? '
                 f'{"YES" if genuine else "NO (~chance)"} '
                 f'(>0.55 threshold; exact-PI {"collapses ~0.5" if abs(pi_te-0.5)<0.06 else f"={pi_te:.3f}"})')
        if all((name, m) in lp_raw for m in ['none', 'pi', 'hks']):
            none = np.array(lp_raw[(name, 'none')]).mean()
            pi = np.array(lp_raw[(name, 'pi')]).mean()
            hks = np.array(lp_raw[(name, 'hks')]).mean()
            helps_nopi = hks - none
            helps_pi = hks - pi
            L.append(f'  LP:  A {"HELPS" if helps_nopi>0.002 else ("HURTS" if helps_nopi<-0.002 else "≈")} '
                     f'vs no-PI ({helps_nopi:+.4f}); A {"beats" if helps_pi>0.002 else ("below" if helps_pi<-0.002 else "≈")} '
                     f'exact-PI ({helps_pi:+.4f})')
        L.append('')

    with open(path, 'w') as f:
        f.write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
