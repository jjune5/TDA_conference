# Knowledge_Distillation/mol_lowdata.py
"""EXP-6: Molecular low-data regime.

Question: Does topology (whole-graph PI) help molecular graph classification
MORE when training data is scarce? Tests the "topology = data-efficient
inductive bias" hypothesis.

Protocol (MUTAG, 188 graphs):
  - Fixed 80/20 train/test split (stratified, seed=1234).
  - Subsample the TRAIN set to fractions {0.1, 0.3, 0.5, 1.0} (stratified).
  - Train GIN with-PI and no-PI on each subsample, evaluate on the SAME fixed
    test set.
  - Repeat over multiple seeds for stability (subsampling + model init vary).
  - Report mean +/- std test accuracy per (fraction, with/no-PI).

Key signal: the (with-PI - no-PI) accuracy gap as a function of train fraction.
If the gap GROWS as the fraction shrinks, topology acts as a data-efficient
prior (the finding).

Self-contained: imports GINClassifier from mol_classify and compute_all_pi /
_make_dataset (read-only). Does NOT edit any shared file.
"""
from __future__ import annotations
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split

from Knowledge_Distillation.mol_classify import GINClassifier, _make_dataset
from Knowledge_Distillation.mol_data import compute_all_pi

DATASET = 'MUTAG'
FRACTIONS = [0.1, 0.3, 0.5, 1.0]
SEEDS = [0, 1, 2, 3, 4]
EPOCHS = 100
SPLIT_SEED = 1234           # fixed 80/20 split (test set identical across runs)
TEST_SIZE = 0.2


def _train_eval(ds, pis_t, labels, tr_idx, te_idx, in_dim, use_pi, epochs, dev,
                init_seed):
    """Train on tr_idx, evaluate on te_idx. Mirrors mol_classify._train_eval_fold
    but with a controllable init/shuffle seed for multi-seed stability."""
    torch.manual_seed(init_seed)
    model = GINClassifier(in_dim, use_pi=use_pi,
                          n_classes=int(labels.max()) + 1).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    tr = [ds[int(i)] for i in tr_idx]
    for j, i in enumerate(tr_idx):
        tr[j].orig_idx = torch.tensor(int(i))
    loader = DataLoader(tr, batch_size=32, shuffle=True)

    for _ in range(epochs):
        model.train()
        for batch in loader:
            batch = batch.to(dev)
            pi = pis_t[batch.orig_idx] if use_pi else None
            opt.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch, pi)
            loss = F.cross_entropy(out, batch.y)
            loss.backward()
            opt.step()

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


def _subsample(train_idx, train_labels, frac, seed):
    """Stratified subsample of the train indices to a given fraction."""
    if frac >= 1.0:
        return train_idx
    # stratified subsample; keep at least 1 sample per class
    n_keep = max(int(round(len(train_idx) * frac)),
                 len(np.unique(train_labels)))
    sub_idx, _ = train_test_split(
        np.arange(len(train_idx)), train_size=n_keep,
        stratify=train_labels, random_state=seed)
    return train_idx[sub_idx]


def run(fractions=FRACTIONS, seeds=SEEDS, epochs=EPOCHS, device='cuda'):
    dev = torch.device(device if torch.cuda.is_available() else 'cpu')
    ds = _make_dataset(DATASET)
    pis = compute_all_pi(DATASET)
    labels = np.array([int(ds[i].y.item()) for i in range(len(ds))])
    in_dim = ds.num_node_features
    pis_t = torch.tensor(pis, dtype=torch.float, device=dev)

    # fixed 80/20 stratified split -> identical test set for every config
    all_idx = np.arange(len(ds))
    train_idx, test_idx = train_test_split(
        all_idx, test_size=TEST_SIZE, stratify=labels,
        random_state=SPLIT_SEED)
    train_labels = labels[train_idx]
    print(f'{DATASET}: total={len(ds)} train={len(train_idx)} '
          f'test={len(test_idx)} in_dim={in_dim} device={dev}')

    # results[frac][use_pi] = list of accuracies over seeds
    results = {f: {True: [], False: []} for f in fractions}

    for frac in fractions:
        for seed in seeds:
            sub_idx = _subsample(train_idx, train_labels, frac, seed)
            for use_pi in (True, False):
                acc = _train_eval(ds, pis_t, labels, sub_idx, test_idx,
                                  in_dim, use_pi, epochs, dev, init_seed=seed)
                results[frac][use_pi].append(acc)
                tag = 'PI' if use_pi else 'noPI'
                print(f'frac={frac} seed={seed} n_train={len(sub_idx)} '
                      f'{tag}: acc={acc:.4f}')

    os.makedirs('scores', exist_ok=True)
    out_path = 'scores/mol_lowdata.txt'
    with open(out_path, 'w') as fh:
        for frac in fractions:
            for use_pi in (True, False):
                accs = np.array(results[frac][use_pi])
                tag = 'PI' if use_pi else 'noPI'
                line = (f'{DATASET} frac {frac} pi {tag} acc MEAN '
                        f'{accs.mean():.4f} std STD {accs.std():.4f}')
                fh.write(line + '\n')
                print(line)
        # also write the gap trend as a comment for quick reading
        fh.write('# gap = mean(PI) - mean(noPI) per fraction\n')
        for frac in fractions:
            gap = (np.mean(results[frac][True]) -
                   np.mean(results[frac][False]))
            gline = f'# {DATASET} frac {frac} gap {gap:+.4f}'
            fh.write(gline + '\n')
            print(gline)
    print(f'saved {out_path}')
    return results


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--smoke', action='store_true',
                   help='quick: 1 fraction (0.3), 20 epochs, 1 seed')
    args = p.parse_args()
    if args.smoke:
        run(fractions=[0.3], seeds=[0], epochs=20)
    else:
        run()
