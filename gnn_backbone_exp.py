"""
gnn_backbone_exp.py — Experiment S2: GNN backbone sensitivity for LP + PI.

Question: Is the "PI helps homophilic / hurts heterophilic" LP pattern specific
to the GCN encoder, or general across GNN backbones?

Encoders: GCN (baseline), GAT, GraphSAGE
Conditions: PI vs no-PI
Datasets: Cora (homophilic), Chameleon (heterophilic)
Trials: 50 (20 if slow)
PI: reused from existing exact cache (data/TLCGNN/<name>.npy); no recomputation.

Output: results/gnn_backbone/ — AUC table CSV + gap table + JSON raw scores.
"""

import os
import sys
import copy
import json
import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
import torch_geometric.datasets
import scipy.sparse as sp
import networkx as nx

from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch.nn.init import xavier_normal_ as xavier
from sklearn.metrics import roc_auc_score

# ── Change working directory to repo root so relative paths work ──────────────
_REPO = os.path.dirname(os.path.abspath(__file__))
os.chdir(_REPO)

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--trials', type=int, default=50)
parser.add_argument('--datasets', nargs='+', default=['Cora', 'Chameleon'])
parser.add_argument('--encoders', nargs='+', default=['GCN', 'GAT', 'SAGE'])
args = parser.parse_args()

TRIALS = args.trials
DATASETS = args.datasets
ENCODERS = args.encoders

HIDDEN = 100
OUT = 16
DROPOUT = 0.5
LR = 0.005
EPOCHS = 2000
EARLY_STOP = 200
PI_DIM = 5  # 5x5 = 25-dim PI

os.makedirs('results/gnn_backbone', exist_ok=True)

# ── Data loading helpers (mirror loaddatas.py logic) ─────────────────────────

def load_dataset(name):
    if name == 'Cora':
        return torch_geometric.datasets.Planetoid('./data/Cora', 'Cora',
                                                   transform=T.NormalizeFeatures())
    elif name == 'Chameleon':
        return torch_geometric.datasets.WikipediaNetwork('./data/Chameleon', 'chameleon')
    else:
        raise ValueError(f"Unknown dataset: {name}")


def get_edges_split(data, val_prop=0.05, test_prop=0.1, seed=1234):
    np.random.seed(seed)
    g = nx.Graph()
    g.add_nodes_from(range(len(data.y)))
    ei = np.array(data.edge_index)
    edges = [(int(ei[0, i]), int(ei[1, i])) for i in range(ei.shape[1])]
    g.add_edges_from(edges)
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


def load_pi_cache(name, expected_total):
    """Load PI from the existing exact cache; splice stale layout if needed."""
    filename = f'./data/TLCGNN/{name}.npy'
    if not os.path.exists(filename):
        raise FileNotFoundError(f"PI cache not found: {filename}")
    cached = np.load(filename)
    if cached.shape[0] == expected_total:
        return cached
    raise ValueError(
        f"PI cache shape mismatch for {name}: "
        f"got {cached.shape[0]}, expected {expected_total}. "
        f"Edge split may differ from what generated the cache. "
        f"(Cache uses seed=1234; ensure same neg_cap.)"
    )


def prepare_data(raw_data, dataset_name):
    """Run deterministic edge split and mask val/test edges. Returns data copy + splits."""
    data = copy.deepcopy(raw_data)
    (train_edges, train_edges_false,
     val_edges, val_edges_false,
     test_edges, test_edges_false) = get_edges_split(data)

    total_edges = np.concatenate(
        [train_edges, train_edges_false,
         val_edges, val_edges_false,
         test_edges, test_edges_false])
    data.train_pos = len(train_edges)
    data.train_neg = len(train_edges_false)
    data.val_pos   = len(val_edges)
    data.val_neg   = len(val_edges_false)
    data.test_pos  = len(test_edges)
    data.test_neg  = len(test_edges_false)
    data.total_edges = total_edges
    data.total_edges_y = torch.cat([
        torch.ones(len(train_edges)),
        torch.zeros(len(train_edges_false)),
        torch.ones(len(val_edges)),
        torch.zeros(len(val_edges_false)),
        torch.ones(len(test_edges)),
        torch.zeros(len(test_edges_false)),
    ]).long()

    # Remove val/test positive edges from the training graph
    ei = np.array(data.edge_index)
    remove = set()
    for e in val_edges.tolist():
        remove.add((e[0], e[1])); remove.add((e[1], e[0]))
    for e in test_edges.tolist():
        remove.add((e[0], e[1])); remove.add((e[1], e[0]))
    keep = np.array([(int(ei[0, i]), int(ei[1, i])) not in remove
                     for i in range(ei.shape[1])])
    data.edge_index = torch.from_numpy(ei[:, keep]).long()

    return data, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false


# ── Encoder definitions ───────────────────────────────────────────────────────

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


class GATEncoder(torch.nn.Module):
    """2-layer GAT. Layer1: 8 heads × 13 dim = 104 ≈ HIDDEN; Layer2: 1 head × OUT."""
    def __init__(self, in_feats):
        super().__init__()
        # heads=8, each dim=HIDDEN//8=12 → total 96; close to HIDDEN=100
        heads1 = 8
        head_dim1 = HIDDEN // heads1  # 12
        self.conv1 = GATConv(in_feats, head_dim1, heads=heads1, dropout=DROPOUT)
        self.conv2 = GATConv(head_dim1 * heads1, OUT, heads=1, concat=False, dropout=DROPOUT)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=DROPOUT, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=DROPOUT, training=self.training)
        x = F.elu(self.conv2(x, edge_index))
        return x


class SAGEEncoder(torch.nn.Module):
    """2-layer GraphSAGE."""
    def __init__(self, in_feats):
        super().__init__()
        self.conv1 = SAGEConv(in_feats, HIDDEN)
        self.conv2 = SAGEConv(HIDDEN, OUT)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=DROPOUT, training=self.training)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=DROPOUT, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        return x


ENCODER_MAP = {'GCN': GCNEncoder, 'GAT': GATEncoder, 'SAGE': SAGEEncoder}


# ── Full LP model: encoder + PI-augmented decoder ────────────────────────────

class LPModel(torch.nn.Module):
    """Mirrors TLCGNN.Net exactly except the encoder is swappable."""
    def __init__(self, in_feats, pi_cache, use_pi=True, encoder_cls=GCNEncoder):
        super().__init__()
        self.encoder = encoder_cls(in_feats)
        self.PI = pi_cache          # numpy array, (total_edges, 25)
        self.use_pi = use_pi
        self.leakyrelu = torch.nn.LeakyReLU(0.2, True)
        pi_flat = PI_DIM * PI_DIM   # 25
        self.linear   = torch.nn.Linear(pi_flat, 1, bias=True)
        self.linear_1 = torch.nn.Linear(pi_flat + OUT, pi_flat, bias=True)

    def encode(self, data):
        return self.encoder(data.x, data.edge_index)

    def decode(self, data, emb, split='train'):
        tp = data.train_pos; tn = data.train_neg
        vp = data.val_pos;   vn = data.val_neg

        if split == 'train':
            edges_pos = data.total_edges[:tp]
            idx = np.random.randint(0, tn, tp)
            edges_neg = data.total_edges[tp:tp + tn][idx]
            total_e = np.concatenate([edges_pos, edges_neg])
            edges_y = torch.cat([
                data.total_edges_y[:tp],
                data.total_edges_y[tp:tp + tn][idx],
            ])
            pi_slice = np.concatenate([
                self.PI[:tp],
                self.PI[tp:tp + tn][idx],
            ])
        elif split == 'val':
            start = tp + tn; end = start + vp + vn
            total_e  = data.total_edges[start:end]
            edges_y  = data.total_edges_y[start:end]
            pi_slice = self.PI[start:end]
        else:  # test
            start = tp + tn + vp + vn
            total_e  = data.total_edges[start:]
            edges_y  = data.total_edges_y[start:]
            pi_slice = self.PI[start:]

        emb = emb.renorm(2, 0, 1)

        if self.use_pi:
            pi_t = torch.tensor(
                pi_slice.reshape(len(total_e), -1), dtype=torch.float32,
                device=emb.device)
        else:
            pi_t = torch.zeros(len(total_e), self.linear.in_features,
                               device=emb.device)

        eu = emb[total_e[:, 0]]
        ev = emb[total_e[:, 1]]
        sq = (eu - ev).pow(2)
        h  = self.leakyrelu(self.linear_1(torch.cat([sq, pi_t], dim=1)))
        h  = torch.abs(self.linear(h)).reshape(-1)
        h  = torch.clamp(h, 0, 40)
        prob = 1.0 / (torch.exp((h - 2.0) / 1.0) + 1.0)
        return prob, edges_y.float()


# ── Training / evaluation helpers ────────────────────────────────────────────

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


def run_one_trial(data_cpu, pi_cache, in_feats, encoder_cls, use_pi, seed, device):
    setup_seed(seed)
    data = copy.deepcopy(data_cpu).to(device)

    model = LPModel(in_feats, pi_cache, use_pi=use_pi, encoder_cls=encoder_cls).to(device)
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
            if len(np.unique(vy)) < 2:
                val_roc = 0.5
            else:
                val_roc = roc_auc_score(vy, vp)

        if val_roc >= best_val_roc:
            best_val_roc = val_roc
            # evaluate test
            with torch.no_grad():
                tp, ty = model.decode(data, emb, 'test')
                tp = tp.cpu().numpy(); ty = ty.cpu().numpy()
                test_roc = roc_auc_score(ty, tp) if len(np.unique(ty)) >= 2 else 0.5
            wait = 0
        else:
            wait += 1
            if wait >= EARLY_STOP:
                break

    del model
    return test_roc


# ── Main experiment loop ──────────────────────────────────────────────────────

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
print(f"Encoders: {ENCODERS}, Datasets: {DATASETS}, Trials: {TRIALS}")

results = {}   # {(encoder, dataset, use_pi): [list of AUC]}

for d_name in DATASETS:
    print(f"\n{'='*60}")
    print(f"Loading {d_name}...")
    dataset = load_dataset(d_name)
    raw_data = dataset[0]
    in_feats = raw_data.x.size(1)

    # Deterministic edge split (seed=1234 fixed, matches existing cache)
    data_prepped, *splits = prepare_data(raw_data, d_name)
    train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = splits

    expected_total = (len(train_edges) + len(train_edges_false)
                      + len(val_edges) + len(val_edges_false)
                      + len(test_edges) + len(test_edges_false))
    print(f"  Total edges in split: {expected_total}")
    print(f"  train_pos={len(train_edges)}, train_neg={len(train_edges_false)}, "
          f"val_pos={len(val_edges)}, val_neg={len(val_edges_false)}, "
          f"test_pos={len(test_edges)}, test_neg={len(test_edges_false)}")

    pi_cache = load_pi_cache(d_name, expected_total)
    print(f"  PI cache loaded: {pi_cache.shape}")

    for enc_name in ENCODERS:
        enc_cls = ENCODER_MAP[enc_name]
        for use_pi in [True, False]:
            pi_tag = "PI" if use_pi else "noPi"
            key = (enc_name, d_name, pi_tag)
            print(f"\n  [{enc_name} / {d_name} / {pi_tag}]")
            aucs = []
            t0 = time.time()
            for trial in range(TRIALS):
                auc = run_one_trial(data_prepped, pi_cache, in_feats,
                                    enc_cls, use_pi, seed=trial, device=device)
                aucs.append(auc)
                if (trial + 1) % 10 == 0:
                    elapsed = time.time() - t0
                    print(f"    trial {trial+1}/{TRIALS}: mean AUC so far = {np.mean(aucs):.4f}  "
                          f"[{elapsed:.0f}s elapsed]")
            results[key] = aucs
            print(f"  --> {enc_name}/{d_name}/{pi_tag}: "
                  f"mean={np.mean(aucs):.4f}, std={np.std(aucs):.4f}, "
                  f"median={np.median(aucs):.4f}")

# ── Save raw results ──────────────────────────────────────────────────────────
raw_path = 'results/gnn_backbone/raw_scores.json'
serialisable = {str(k): v for k, v in results.items()}
with open(raw_path, 'w') as f:
    json.dump(serialisable, f, indent=2)
print(f"\nRaw scores saved to {raw_path}")

# ── Build summary table ───────────────────────────────────────────────────────
def stats(lst):
    return np.mean(lst), np.std(lst)

print("\n" + "="*72)
print("AUC TABLE  (mean ± std, n={})".format(TRIALS))
print("="*72)

# Header
col_order = [(d, p) for d in DATASETS for p in ['PI', 'noPi']]
header = f"{'Encoder':<8}" + "".join(f"  {d}-{p:<8}" for d, p in col_order)
print(header)
print("-" * len(header))

table_rows = {}
for enc in ENCODERS:
    row = f"{enc:<8}"
    for d_name, pi_tag in col_order:
        k = (enc, d_name, pi_tag)
        if k in results:
            m, s = stats(results[k])
            row += f"  {m:.4f}±{s:.4f}"
        else:
            row += f"  {'N/A':>13}"
    print(row)
    table_rows[enc] = {f"{d}-{p}": stats(results.get((enc, d, p), [float('nan')])) for d, p in col_order}

print("\n" + "="*72)
print("PI GAIN TABLE  (PI_AUC - noPi_AUC, mean ± propagated std)")
print("="*72)
gap_header = f"{'Encoder':<8}" + "".join(f"  {'Gap-'+d:<14}" for d in DATASETS)
print(gap_header)
print("-" * len(gap_header))

gap_results = {}
for enc in ENCODERS:
    row = f"{enc:<8}"
    for d_name in DATASETS:
        ki = (enc, d_name, 'PI')
        kn = (enc, d_name, 'noPi')
        if ki in results and kn in results:
            mi, si = stats(results[ki])
            mn, sn = stats(results[kn])
            gap = mi - mn
            gap_std = np.sqrt(si**2 + sn**2)
            row += f"  {gap:+.4f}±{gap_std:.4f}"
            gap_results[(enc, d_name)] = (gap, gap_std, mi, si, mn, sn)
        else:
            row += f"  {'N/A':>15}"
    print(row)

# ── Save CSV ──────────────────────────────────────────────────────────────────
import csv

auc_csv_path = 'results/gnn_backbone/auc_table.csv'
with open(auc_csv_path, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['encoder'] + [f'{d}_{p}' for d, p in col_order] +
               [f'{d}_{p}_std' for d, p in col_order])
    for enc in ENCODERS:
        means = []
        stds  = []
        for d_name, pi_tag in col_order:
            k = (enc, d_name, pi_tag)
            if k in results:
                m, s = stats(results[k])
            else:
                m, s = float('nan'), float('nan')
            means.append(f'{m:.6f}')
            stds.append(f'{s:.6f}')
        w.writerow([enc] + means + stds)
print(f"\nAUC table saved to {auc_csv_path}")

gap_csv_path = 'results/gnn_backbone/gap_table.csv'
with open(gap_csv_path, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['encoder'] + [f'gap_{d}' for d in DATASETS] +
               [f'gap_{d}_std' for d in DATASETS] +
               [f'pi_{d}' for d in DATASETS] +
               [f'nopi_{d}' for d in DATASETS])
    for enc in ENCODERS:
        gaps = []; gap_stds = []; pi_means = []; nopi_means = []
        for d_name in DATASETS:
            k = (enc, d_name)
            if k in gap_results:
                gap, gstd, mi, si, mn, sn = gap_results[k]
                gaps.append(f'{gap:.6f}')
                gap_stds.append(f'{gstd:.6f}')
                pi_means.append(f'{mi:.6f}')
                nopi_means.append(f'{mn:.6f}')
            else:
                gaps.append('nan'); gap_stds.append('nan')
                pi_means.append('nan'); nopi_means.append('nan')
        w.writerow([enc] + gaps + gap_stds + pi_means + nopi_means)
print(f"Gap table saved to {gap_csv_path}")

# ── Verdict ───────────────────────────────────────────────────────────────────
print("\n" + "="*72)
print("VERDICT")
print("="*72)

for d_name in DATASETS:
    gaps_for_dataset = [gap_results.get((enc, d_name), (float('nan'),)*6)[0]
                        for enc in ENCODERS if (enc, d_name) in gap_results]
    if not gaps_for_dataset:
        continue
    all_same_sign = all(g > 0 for g in gaps_for_dataset) or all(g < 0 for g in gaps_for_dataset)
    direction = "positive (PI helps)" if np.mean(gaps_for_dataset) > 0 else "negative (PI hurts)"
    consistency = "CONSISTENT" if all_same_sign else "INCONSISTENT (encoder-dependent)"
    print(f"  {d_name}: gap direction = {direction}  |  across encoders: {consistency}")
    for enc in ENCODERS:
        k = (enc, d_name)
        if k in gap_results:
            g = gap_results[k][0]
            print(f"    {enc}: gap={g:+.4f}")

print("\nDone.")
