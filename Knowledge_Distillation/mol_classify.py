# Knowledge_Distillation/mol_classify.py
"""GIN graph classifier for TUDataset molecules, with optional whole-graph PI
feature. 10-fold stratified CV. Compares with-PI vs no-PI (--no_pi)."""
from __future__ import annotations
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_mean_pool
from sklearn.model_selection import StratifiedKFold
from Knowledge_Distillation.mol_data import compute_all_pi


class GINClassifier(nn.Module):
    def __init__(self, in_dim, hidden=64, n_classes=2, use_pi=True, pi_dim=25):
        super().__init__()
        def mlp(i, o):
            return nn.Sequential(nn.Linear(i, o), nn.ReLU(), nn.Linear(o, o))
        self.conv1 = GINConv(mlp(in_dim, hidden))
        self.conv2 = GINConv(mlp(hidden, hidden))
        self.conv3 = GINConv(mlp(hidden, hidden))
        self.use_pi = use_pi
        head_in = hidden + (pi_dim if use_pi else 0)
        self.head = nn.Sequential(nn.Linear(head_in, hidden), nn.ReLU(),
                                   nn.Dropout(0.5), nn.Linear(hidden, n_classes))

    def forward(self, x, edge_index, batch, pi=None):
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        h = F.relu(self.conv3(h, edge_index))
        hg = global_mean_pool(h, batch)
        if self.use_pi:
            hg = torch.cat([hg, pi], dim=-1)
        return self.head(hg)


def _make_dataset(name):
    """Load TUDataset, ensuring node features exist (use degree one-hot if not)."""
    import torch_geometric.transforms as T
    ds = TUDataset(root=f'./data/TU_{name}', name=name)
    if ds.num_node_features == 0:
        # compute max degree across dataset for OneHotDegree
        from torch_geometric.utils import degree
        max_deg = 0
        for d in ds:
            if d.edge_index.numel() > 0:
                deg = degree(d.edge_index[0], num_nodes=d.num_nodes).max().item()
                max_deg = max(max_deg, int(deg))
        ds = TUDataset(root=f'./data/TU_{name}', name=name,
                       transform=T.OneHotDegree(max_degree=max_deg))
    return ds


def _train_eval_fold(ds, pis_t, labels, tr_idx, te_idx, in_dim, use_pi, epochs, dev):
    model = GINClassifier(in_dim, use_pi=use_pi, n_classes=int(labels.max()) + 1).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    tr = [ds[int(i)] for i in tr_idx]
    for j, i in enumerate(tr_idx):
        tr[j].orig_idx = torch.tensor(int(i))
    loader = DataLoader(tr, batch_size=32, shuffle=True)
    for ep in range(epochs):
        model.train()
        for batch in loader:
            batch = batch.to(dev)
            pi = pis_t[batch.orig_idx] if use_pi else None
            opt.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch, pi)
            loss = F.cross_entropy(out, batch.y)
            loss.backward(); opt.step()
    model.eval()
    te = [ds[int(i)] for i in te_idx]
    for j, i in enumerate(te_idx):
        te[j].orig_idx = torch.tensor(int(i))
    te_loader = DataLoader(te, batch_size=64)
    correct = 0
    with torch.no_grad():
        for batch in te_loader:
            batch = batch.to(dev)
            pi = pis_t[batch.orig_idx] if use_pi else None
            pred = model(batch.x, batch.edge_index, batch.batch, pi).argmax(1)
            correct += (pred == batch.y).sum().item()
    return correct / len(te_idx)


def run(name, use_pi, epochs=100, seed=1234, device='cuda'):
    torch.manual_seed(seed); np.random.seed(seed)
    ds = _make_dataset(name)
    pis = compute_all_pi(name)
    labels = np.array([int(ds[i].y.item()) for i in range(len(ds))])
    in_dim = ds.num_node_features
    dev = torch.device(device if torch.cuda.is_available() else 'cpu')
    pis_t = torch.tensor(pis, dtype=torch.float, device=dev)

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    accs = []
    for fold, (tr_idx, te_idx) in enumerate(skf.split(np.zeros(len(ds)), labels)):
        acc = _train_eval_fold(ds, pis_t, labels, tr_idx, te_idx, in_dim, use_pi, epochs, dev)
        accs.append(acc)
        print(f'fold {fold}: acc={acc:.4f}')
    print(f'{name} use_pi={use_pi}: {np.mean(accs):.4f} ± {np.std(accs):.4f}')
    return float(np.mean(accs)), float(np.std(accs))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='MUTAG')
    p.add_argument('--no_pi', action='store_true')
    p.add_argument('--epochs', type=int, default=100)
    args = p.parse_args()
    mean, std = run(args.dataset, use_pi=not args.no_pi, epochs=args.epochs)
    os.makedirs('scores', exist_ok=True)
    tag = 'noPI' if args.no_pi else 'withPI'
    with open(f'scores/mol_{args.dataset}_{tag}.txt', 'w') as f:
        f.write(f'{args.dataset} {tag} acc {mean:.4f} std {std:.4f}\n')
    print(f'saved scores/mol_{args.dataset}_{tag}.txt')
