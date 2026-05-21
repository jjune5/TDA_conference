# Knowledge_Distillation/train_pdgnn_lp.py
"""Train PDGNN on edge-centered LP supervision data."""

from __future__ import annotations
import os
import sys
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Knowledge_Distillation.pdgnn_modern import PDGNN


def _bipartite_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Pred: (E, 2). GT: (K, 2). Match each gt to its closest pred (or vice versa).
    Use the Hungarian algorithm on a small cost matrix; clamp at the smaller side."""
    if pred.numel() == 0 or gt.numel() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    # Cost: pairwise sq distance
    cost = ((pred.unsqueeze(1) - gt.unsqueeze(0)) ** 2).sum(dim=-1)  # (E, K)
    cost_np = cost.detach().cpu().numpy()
    row_idx, col_idx = linear_sum_assignment(cost_np)
    matched = cost[row_idx, col_idx]
    return matched.mean()


def train_one_epoch(model, optimizer, samples, device):
    model.train()
    losses = []
    keys = list(samples.keys())
    np.random.shuffle(keys)
    for k in tqdm(keys, desc='train'):
        filt, ei, pd_gt, _, _ = samples[k]
        if ei.size == 0 or pd_gt.size == 0:
            continue
        filt_t = torch.tensor(filt, dtype=torch.float, device=device).view(-1, 1)
        ei_t = torch.tensor(ei, dtype=torch.long, device=device)
        gt_t = torch.tensor(pd_gt, dtype=torch.float, device=device)
        optimizer.zero_grad()
        pred = model(filt_t, ei_t)  # (E, 2)
        loss = _bipartite_loss(pred, gt_t)
        if not torch.isfinite(loss):
            continue
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else float('nan')


def evaluate(model, samples, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for k, (filt, ei, pd_gt, _, _) in samples.items():
            if ei.size == 0 or pd_gt.size == 0:
                continue
            filt_t = torch.tensor(filt, dtype=torch.float, device=device).view(-1, 1)
            ei_t = torch.tensor(ei, dtype=torch.long, device=device)
            gt_t = torch.tensor(pd_gt, dtype=torch.float, device=device)
            pred = model(filt_t, ei_t)
            loss = _bipartite_loss(pred, gt_t)
            losses.append(loss.item())
    return float(np.mean(losses)) if losses else float('nan')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='./data/PDGNN/PubMed_LP_hop2_n10000_train.pkl')
    parser.add_argument('--out', default='./data/PDGNN/checkpoints/pdgnn_lp.pt')
    parser.add_argument('--hidden', type=int, default=32)
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=1234)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(args.data, 'rb') as f:
        samples = pickle.load(f)
    keys = list(samples.keys())
    np.random.RandomState(1234).shuffle(keys)
    n_train = int(0.9 * len(keys))
    train_keys, val_keys = keys[:n_train], keys[n_train:]
    train_samples = {k: samples[k] for k in train_keys}
    val_samples = {k: samples[k] for k in val_keys}
    print(f'train={len(train_samples)} val={len(val_samples)}')

    model = PDGNN(hidden_dim=args.hidden, num_layers=args.layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    best_val = float('inf')
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    for ep in range(1, args.epochs + 1):
        tr = train_one_epoch(model, optimizer, train_samples, device)
        va = evaluate(model, val_samples, device)
        improved = va < best_val
        if improved:
            best_val = va
            torch.save({'state_dict': model.state_dict(),
                        'config': {'hidden_dim': args.hidden, 'num_layers': args.layers}},
                       args.out)
        print(f'ep {ep:3d}  train_mse={tr:.4f}  val_mse={va:.4f}  '
              f'best_val={best_val:.4f}  saved={improved}')


if __name__ == '__main__':
    main()
