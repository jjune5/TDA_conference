# Knowledge_Distillation/edge_analysis.py
"""Edge-level analysis: where does PI cause prediction errors on a heterophilic graph?"""
from __future__ import annotations
import os, sys, copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn.functional as F
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import loaddatas as lds
from baselines import TLCGNN
from torch.nn.init import xavier_normal_ as xavier


def weights_init(m):
    if isinstance(m, torch.nn.Linear):
        xavier(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


def train_and_get_test_preds(name, use_pi, seed=0, max_epochs=2000, patience=200):
    torch.manual_seed(seed); np.random.seed(seed)
    dataset = lds.loaddatas(name)
    data = copy.deepcopy(dataset[0])
    model, data = TLCGNN.call(data, dataset.name, data.x.size(1),
                              dataset.num_classes, 0, use_pi=use_pi)
    model.apply(weights_init)
    opt = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0)
    best_val = -1; best_state = None; wait = 0
    for ep in range(max_epochs):
        model.train(); opt.zero_grad()
        emb = model.encode(data)
        x, y = model.decode(data, emb)
        F.binary_cross_entropy(x, y).backward(); opt.step()
        # val
        model.eval()
        with torch.no_grad():
            emb = model.encode(data)
            vx, vy = model.decode(data, emb, type='val')
            from sklearn.metrics import roc_auc_score
            vroc = roc_auc_score(vy.cpu(), vx.cpu())
        if vroc > best_val:
            best_val = vroc; wait = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        emb = model.encode(data)
        tx, ty = model.decode(data, emb, type='test')
    return tx.cpu().numpy(), ty.cpu().numpy(), data


def main():
    name = 'Chameleon'
    print(f'Training TLC-GNN (with PI) on {name}...')
    pred_pi, y_pi, data = train_and_get_test_preds(name, use_pi=True)
    print(f'Training no-PI on {name}...')
    pred_no, y_no, _ = train_and_get_test_preds(name, use_pi=False)

    # y_pi and y_no should be identical (same split). Use y_pi.
    assert np.allclose(y_pi, y_no), 'label order mismatch!'
    y = y_pi
    bin_pi = (pred_pi > 0.5).astype(int)
    bin_no = (pred_no > 0.5).astype(int)

    correct_pi = (bin_pi == y)
    correct_no = (bin_no == y)

    # Categorize edges
    both_right = (correct_pi & correct_no).sum()
    both_wrong = (~correct_pi & ~correct_no).sum()
    pi_only_right = (correct_pi & ~correct_no).sum()  # PI helps
    no_only_right = (~correct_pi & correct_no).sum()  # PI HURTS

    print(f'\nEdge-level breakdown (n={len(y)} test edges):')
    print(f'  both correct:     {both_right} ({both_right/len(y)*100:.1f}%)')
    print(f'  both wrong:       {both_wrong} ({both_wrong/len(y)*100:.1f}%)')
    print(f'  PI-only correct:  {pi_only_right} ({pi_only_right/len(y)*100:.1f}%)  [PI helps]')
    print(f'  no-PI-only right: {no_only_right} ({no_only_right/len(y)*100:.1f}%)  [PI HURTS]')

    # Plot prediction scatter
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    ax.scatter(pred_no[y == 1], pred_pi[y == 1], s=15, alpha=0.5, label='true edge', color='C0')
    ax.scatter(pred_no[y == 0], pred_pi[y == 0], s=15, alpha=0.5, label='non-edge', color='C1')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax.axhline(0.5, color='gray', ls=':', alpha=0.5); ax.axvline(0.5, color='gray', ls=':', alpha=0.5)
    ax.set_xlabel('no-PI prediction'); ax.set_ylabel('TLC-GNN (PI) prediction')
    ax.set_title(f'{name}: per-edge predictions')
    ax.legend()

    ax = axes[1]
    cats = ['both\ncorrect', 'both\nwrong', 'PI helps', 'PI HURTS']
    vals = [both_right, both_wrong, pi_only_right, no_only_right]
    colors = ['green', 'gray', 'blue', 'red']
    ax.bar(cats, vals, color=colors)
    ax.set_ylabel('# test edges')
    ax.set_title(f'{name}: PI effect breakdown')
    for i, v in enumerate(vals):
        ax.annotate(str(v), (i, v), ha='center', va='bottom')

    os.makedirs('docs/figures', exist_ok=True)
    plt.tight_layout()
    plt.savefig('docs/figures/edge_disagreement.png', dpi=120, bbox_inches='tight')
    print('saved docs/figures/edge_disagreement.png')


if __name__ == '__main__':
    main()
