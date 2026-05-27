# Knowledge_Distillation/mol_resolution_sweep.py
"""EXP-2: Molecular PI resolution sweep.

Question: is 5x5 the right PI resolution for molecular graph classification, or
does finer (10x10, 20x20) help / overfit?

Self-contained: builds the degree-filtered whole-graph extended persistence
diagram (same machinery as mol_data.graph_to_pi) but at an arbitrary
PersistenceImager resolution R -> R*R flat PI. Then runs the with-PI GINClassifier
(imported from mol_classify, with pi_dim=R*R) under 10-fold stratified CV.

Does NOT edit any shared file. The fold trainer is copied here (the one in
mol_classify hardcodes pi_dim=25) so that pi_dim can match the resolution.

Output: scores/mol_resolution_sweep.txt, one line per dataset x resolution:
    dataset <name> res <R> acc MEAN <mean> std <STD>
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold

from sg2dgm import PersistenceImager as pimg_mod
from Knowledge_Distillation.accelerated_PD import (perturb_filter_function,
                                                   Union_find, Accelerate_PD)
from Knowledge_Distillation.mol_data import load_tudataset
from Knowledge_Distillation.mol_classify import GINClassifier, _make_dataset


# ---------------------------------------------------------------------------
# PI at an arbitrary resolution
# ---------------------------------------------------------------------------
_IMAGER_CACHE: dict[int, object] = {}


def _imager(resolution: int):
    if resolution not in _IMAGER_CACHE:
        _IMAGER_CACHE[resolution] = pimg_mod.PersistenceImager(resolution=resolution)
    return _IMAGER_CACHE[resolution]


def _degree_filter(g: nx.Graph) -> np.ndarray:
    """Normalized node degree as filtration value (matches mol_data)."""
    g = nx.convert_node_labels_to_integers(g)
    deg = np.array([d for _, d in sorted(g.degree(), key=lambda x: x[0])],
                   dtype=np.float64)
    m = deg.max() + 1e-10
    return deg / m


def graph_to_pi_res(g: nx.Graph, resolution: int) -> np.ndarray:
    """Whole-graph degree-filtered extended PD -> R x R PI flattened to (R*R,).

    Same construction as mol_data.graph_to_pi, parameterized by resolution.
    Degenerate graphs / accelerated_PD failures -> zeros, exactly as the
    shared pipeline does.
    """
    dim = resolution * resolution
    g = nx.convert_node_labels_to_integers(g)
    if g.number_of_edges() == 0 or g.number_of_nodes() < 3:
        return np.zeros(dim, dtype=np.float64)
    filt = _degree_filter(g)
    sf = perturb_filter_function(g, filt)
    try:
        PD_up, ess0, PD_down, Pos, Neg = Union_find(sf)
        PD_one = Accelerate_PD(Pos, Neg, sf)
    except (ValueError, IndexError, KeyError):
        return np.zeros(dim, dtype=np.float64)
    pd = []
    for arr in [PD_up, ess0, PD_down, PD_one]:
        a = np.asarray(arr, dtype=np.float64).reshape(-1, 2) if len(arr) else np.empty((0, 2))
        if a.size:
            pd.append(a)
    pd_all = np.concatenate(pd, axis=0) if pd else np.empty((0, 2))
    if pd_all.size == 0:
        return np.zeros(dim, dtype=np.float64)
    return _imager(resolution).transform(pd_all).reshape(-1)


def compute_all_pi_res(name: str, resolution: int, cache_dir: str = './data/MOL') -> np.ndarray:
    """Compute + cache the (N, R*R) PI array for a dataset at given resolution."""
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, f'{name}_PI_res{resolution}.npy')
    if os.path.exists(cache):
        return np.load(cache)
    graphs, _ = load_tudataset(name)
    from tqdm import tqdm
    pis = np.stack([graph_to_pi_res(g, resolution)
                    for g in tqdm(graphs, desc=f'{name} PI res{resolution}')])
    np.save(cache, pis)
    return pis


# ---------------------------------------------------------------------------
# Minimal 10-fold CV trainer (copied from mol_classify but with pi_dim wired)
# ---------------------------------------------------------------------------
def _train_eval_fold(ds, pis_t, labels, tr_idx, te_idx, in_dim, pi_dim, epochs, dev):
    model = GINClassifier(in_dim, use_pi=True, pi_dim=pi_dim,
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
            pi = pis_t[batch.orig_idx]
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
            pi = pis_t[batch.orig_idx]
            pred = model(batch.x, batch.edge_index, batch.batch, pi).argmax(1)
            correct += (pred == batch.y).sum().item()
    return correct / len(te_idx)


def run(name: str, resolution: int, epochs: int = 50, seed: int = 1234,
        device: str = 'cuda'):
    torch.manual_seed(seed); np.random.seed(seed)
    ds = _make_dataset(name)
    pis = compute_all_pi_res(name, resolution)
    assert pis.shape[1] == resolution * resolution, \
        f'PI dim {pis.shape[1]} != {resolution * resolution}'
    labels = np.array([int(ds[i].y.item()) for i in range(len(ds))])
    in_dim = ds.num_node_features
    dev = torch.device(device if torch.cuda.is_available() else 'cpu')
    pis_t = torch.tensor(pis, dtype=torch.float, device=dev)

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    accs = []
    for fold, (tr_idx, te_idx) in enumerate(skf.split(np.zeros(len(ds)), labels)):
        acc = _train_eval_fold(ds, pis_t, labels, tr_idx, te_idx, in_dim,
                               resolution * resolution, epochs, dev)
        accs.append(acc)
        print(f'  {name} res{resolution} fold {fold}: acc={acc:.4f}', flush=True)
    mean, std = float(np.mean(accs)), float(np.std(accs))
    print(f'{name} res {resolution}: {mean:.4f} +/- {std:.4f}', flush=True)
    return mean, std


def main(datasets=('MUTAG', 'PROTEINS'), resolutions=(5, 10, 20), epochs=50):
    os.makedirs('scores', exist_ok=True)
    out_path = 'scores/mol_resolution_sweep.txt'
    lines = []
    for name in datasets:
        for R in resolutions:
            print(f'=== {name} resolution {R} ===', flush=True)
            mean, std = run(name, R, epochs=epochs)
            line = f'{name} res {R} acc MEAN {mean:.4f} std STD {std:.4f}'
            lines.append(line)
            # write incrementally so partial results survive
            with open(out_path, 'w') as f:
                f.write('\n'.join(lines) + '\n')
            print('saved ->', line, flush=True)
    print(f'done -> {out_path}', flush=True)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--smoke', action='store_true',
                   help='reduced run: MUTAG only, res=10, 5 epochs')
    p.add_argument('--epochs', type=int, default=50)
    args = p.parse_args()
    if args.smoke:
        main(datasets=('MUTAG',), resolutions=(10,), epochs=5)
    else:
        main(epochs=args.epochs)
