"""Variant B LP experiment: GCN + {no-PI | exact-PI | B (multi-scale HKS-PD)}.

50-trial link-prediction AUC, GCN encoder, on Cora + Chameleon.  Mirrors the
S2 harness (gnn_backbone_exp.py): same GCN encoder, same decoder structure,
same canonical 0.05/0.1 seed-1234 split (which reproduces the exact-PI cache
row layout), same early-stop/optimizer.  The ONLY change vs that harness is the
PI feature width is variable so it can ingest B's stacked (N, K*25) PI as well
as the 25-dim exact PI (and the zero/no-PI condition).

Conditions:
  no-PI     : decoder sees only |emb_u - emb_v|^2 (GCN-only)
  exact-PI  : + exact vicinity PI         (data/TLCGNN/<name>.npy, 25-dim)
  B         : + multi-scale HKS-filtration vicinity PI, K scales stacked
              (data/HKS_MULTI_TLCGNN_<tag>/<name>.npy, K*25-dim)

Output: results/diffusion_feat_B/lp_auc.{csv,txt} (+ raw JSON).
"""
from __future__ import annotations
import os
import sys
import copy
import json
import time
import argparse

_REPO = os.path.dirname(os.path.abspath(__file__))
os.chdir(_REPO)
sys.path.insert(0, _REPO)

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
import torch_geometric.datasets
import scipy.sparse as sp
import networkx as nx
from torch_geometric.nn import GCNConv
from torch.nn.init import xavier_normal_ as xavier
from sklearn.metrics import roc_auc_score

# NOTE: gnn_backbone_exp.py runs its whole experiment at import time (module-level
# argparse + loop), so we replicate its GCN encoder / split / prepare_data /
# constants here VERBATIM rather than importing it.  This keeps the harness
# byte-for-byte identical to the S2 baselines while staying importable.
HIDDEN = 100
OUT = 16
DROPOUT = 0.5
LR = 0.005
EPOCHS = 2000
EARLY_STOP = 200


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


def prepare_data(raw_data, dataset_name):
    data = copy.deepcopy(raw_data)
    (train_edges, train_edges_false, val_edges, val_edges_false,
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
        torch.ones(len(val_edges)), torch.zeros(len(val_edges_false)),
        torch.ones(len(test_edges)), torch.zeros(len(test_edges_false))]).long()
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


class LPModelVar(torch.nn.Module):
    """gnn_backbone_exp.LPModel but with a variable PI width (pi_dim).

    pi_dim == 0  -> no-PI (decoder sees only the squared-diff embedding).
    pi_dim  > 0  -> concatenate the per-edge PI (any width) into the decoder,
    exactly mirroring how the 25-dim PI is concatenated in the reference model.
    """
    def __init__(self, in_feats, pi_cache, pi_dim, encoder_cls=GCNEncoder):
        super().__init__()
        self.encoder = encoder_cls(in_feats)
        self.PI = pi_cache          # numpy (total_edges, pi_dim) or None
        self.pi_dim = pi_dim
        self.leakyrelu = torch.nn.LeakyReLU(0.2, True)
        feat_in = pi_dim + OUT      # squared-diff (OUT) ++ PI (pi_dim)
        hid = max(pi_dim, OUT)      # hidden width of the decoder MLP
        self.linear_1 = torch.nn.Linear(feat_in, hid, bias=True)
        self.linear = torch.nn.Linear(hid, 1, bias=True)

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
            pi_slice = None if self.PI is None else np.concatenate([
                self.PI[:tp], self.PI[tp:tp + tn][idx]])
        elif split == 'val':
            start = tp + tn; end = start + vp + vn
            total_e = data.total_edges[start:end]
            edges_y = data.total_edges_y[start:end]
            pi_slice = None if self.PI is None else self.PI[start:end]
        else:  # test
            start = tp + tn + vp + vn
            total_e = data.total_edges[start:]
            edges_y = data.total_edges_y[start:]
            pi_slice = None if self.PI is None else self.PI[start:]

        emb = emb.renorm(2, 0, 1)

        if self.pi_dim > 0:
            pi_t = torch.tensor(pi_slice.reshape(len(total_e), -1),
                                dtype=torch.float32, device=emb.device)
        else:
            pi_t = torch.zeros(len(total_e), 0, device=emb.device)

        eu = emb[total_e[:, 0]]
        ev = emb[total_e[:, 1]]
        sq = (eu - ev).pow(2)
        h = self.leakyrelu(self.linear_1(torch.cat([sq, pi_t], dim=1)))
        h = torch.abs(self.linear(h)).reshape(-1)
        h = torch.clamp(h, 0, 40)
        prob = 1.0 / (torch.exp((h - 2.0) / 1.0) + 1.0)
        return prob, edges_y.float()


def run_one_trial(data_cpu, pi_cache, pi_dim, in_feats, seed, device):
    setup_seed(seed)
    data = copy.deepcopy(data_cpu).to(device)
    model = LPModelVar(in_feats, pi_cache, pi_dim).to(device)
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
            val_roc = 0.5 if len(np.unique(vy)) < 2 else roc_auc_score(vy, vp)
        if val_roc >= best_val_roc:
            best_val_roc = val_roc
            with torch.no_grad():
                tp_, ty = model.decode(data, emb, 'test')
                tp_ = tp_.cpu().numpy(); ty = ty.cpu().numpy()
                test_roc = roc_auc_score(ty, tp_) if len(np.unique(ty)) >= 2 else 0.5
            wait = 0
        else:
            wait += 1
            if wait >= EARLY_STOP:
                break
    del model
    return test_roc


def load_cache(path, name, expected_total):
    for cand in (f'{path}/{name}.npy', f'{path}/{name.lower()}.npy'):
        if os.path.exists(cand):
            arr = np.load(cand)
            if arr.shape[0] != expected_total:
                raise ValueError(
                    f"{cand}: rows {arr.shape[0]} != expected {expected_total}")
            return arr, cand
    raise FileNotFoundError(f"no cache for {name} under {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trials', type=int, default=50)
    ap.add_argument('--datasets', nargs='+', default=['Cora', 'Chameleon'])
    ap.add_argument('--b_dir', default='data/HKS_MULTI_TLCGNN_0.1x1x10')
    ap.add_argument('--out_dir', default='results/diffusion_feat_B')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}  trials={args.trials}  datasets={args.datasets}")
    print(f"B cache dir: {args.b_dir}")

    results = {}   # {(dataset, cond): [auc,...]}
    meta = {}
    for d_name in args.datasets:
        print(f"\n{'='*64}\nLoading {d_name}...")
        dataset = load_dataset(d_name)
        raw_data = dataset[0]
        in_feats = raw_data.x.size(1)
        data_prepped, *splits = prepare_data(raw_data, d_name)
        (tr, trf, va, vaf, te, tef) = splits
        expected_total = sum(len(x) for x in splits)
        print(f"  split total={expected_total} "
              f"(tp={len(tr)},tn={len(trf)},vp={len(va)},vn={len(vaf)},"
              f"tep={len(te)},ten={len(tef)})")

        # exact-PI cache (25-dim)
        exact_pi, exact_path = load_cache('data/TLCGNN', d_name, expected_total)
        # B multi-scale cache (K*25-dim)
        b_pi, b_path = load_cache(args.b_dir, d_name, expected_total)
        meta[d_name] = {'exact_cache': exact_path, 'exact_dim': int(exact_pi.shape[1]),
                        'b_cache': b_path, 'b_dim': int(b_pi.shape[1]),
                        'expected_total': expected_total}
        print(f"  exact PI: {exact_pi.shape} ({exact_path})")
        print(f"  B  PI   : {b_pi.shape} ({b_path})")

        conditions = [
            ('no-PI', None, 0),
            ('exact-PI', exact_pi, exact_pi.shape[1]),
            ('B-hks-multi', b_pi, b_pi.shape[1]),
        ]
        for cond, cache, pdim in conditions:
            print(f"\n  [{d_name} / {cond}]  pi_dim={pdim}")
            aucs = []
            t0 = time.time()
            for trial in range(args.trials):
                auc = run_one_trial(data_prepped, cache, pdim, in_feats,
                                    seed=trial, device=device)
                aucs.append(auc)
                if (trial + 1) % 10 == 0:
                    print(f"    trial {trial+1}/{args.trials}: "
                          f"mean AUC={np.mean(aucs):.4f} "
                          f"[{time.time()-t0:.0f}s]")
            results[(d_name, cond)] = aucs
            print(f"  --> {d_name}/{cond}: mean={np.mean(aucs):.4f} "
                  f"std={np.std(aucs):.4f}")

    # ── save raw ──
    with open(os.path.join(args.out_dir, 'lp_auc_raw.json'), 'w') as f:
        json.dump({'meta': meta,
                   'scores': {f'{d}|{c}': v for (d, c), v in results.items()}},
                  f, indent=2)

    # ── table ──
    conds = ['no-PI', 'exact-PI', 'B-hks-multi']
    txt = []
    txt.append('=' * 90)
    txt.append(f'VARIANT B — LP AUC (GCN, {args.trials} trials, mean ± std)')
    txt.append('  no-PI | exact-PI (collapses at test, §14) | B = multi-scale HKS-PD (stacked)')
    txt.append('=' * 90)
    txt.append('')
    hdr = f"{'dataset':12} " + "".join(f"{c:>20}" for c in conds) + \
          f"{'B−noPI':>12}{'B−exact':>12}"
    txt.append(hdr)
    txt.append('-' * len(hdr))
    csv_lines = ['dataset,' + ','.join(conds) +
                 ',' + ','.join(f'{c}_std' for c in conds) +
                 ',B_minus_noPI,B_minus_exact']
    for d_name in args.datasets:
        row = f"{d_name:12} "
        means = {}; stds = {}
        for c in conds:
            k = (d_name, c)
            if k in results:
                m, s = np.mean(results[k]), np.std(results[k])
            else:
                m, s = float('nan'), float('nan')
            means[c] = m; stds[c] = s
            row += f"{m:>13.4f}±{s:.4f}"
        b_m = means['B-hks-multi']
        d_nopi = b_m - means['no-PI']
        d_exact = b_m - means['exact-PI']
        row += f"{d_nopi:>+12.4f}{d_exact:>+12.4f}"
        txt.append(row)
        csv_lines.append(
            f"{d_name}," + ','.join(f'{means[c]:.6f}' for c in conds) +
            ',' + ','.join(f'{stds[c]:.6f}' for c in conds) +
            f",{d_nopi:.6f},{d_exact:.6f}")
    txt.append('')
    txt.append('Δ(B−noPI) > 0 ⇒ B helps over GCN-only ; Δ(B−exact) compares B to exact vicinity-PI.')
    out_txt = '\n'.join(txt)
    print('\n' + out_txt)
    with open(os.path.join(args.out_dir, 'lp_auc.txt'), 'w') as f:
        f.write(out_txt + '\n')
    with open(os.path.join(args.out_dir, 'lp_auc.csv'), 'w') as f:
        f.write('\n'.join(csv_lines) + '\n')
    print(f"\n[lp] wrote {args.out_dir}/lp_auc.{{txt,csv,json}}")


if __name__ == '__main__':
    main()
