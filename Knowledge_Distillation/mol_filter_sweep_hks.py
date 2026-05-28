# Knowledge_Distillation/mol_filter_sweep_hks.py
"""EXP-3b: Molecular filter-function sweep including HKS (Heat Kernel Signature).

Extends mol_filter_sweep.py to add filter='hks' (physical-diffusion-based)
alongside the geometric filters degree/clustering/closeness.

Does NOT modify mol_filter_sweep.py. Writes to scores/mol_filter_sweep_hks.txt.
Reuses the same GIN 10-fold CV and PI machinery.
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold

from sg2dgm import PersistenceImager as pimg_mod
from Knowledge_Distillation.accelerated_PD import (perturb_filter_function,
                                                   Union_find, Accelerate_PD)
from Knowledge_Distillation.mol_data import load_tudataset
from Knowledge_Distillation.mol_classify import GINClassifier, _make_dataset
from Knowledge_Distillation.hks_filtration import compute_hks

_IMAGER = pimg_mod.PersistenceImager(resolution=5)

# All four filters — three geometric + one diffusion
FILTERS = ['degree', 'clustering', 'closeness', 'hks']
DATASETS = ['MUTAG', 'PROTEINS', 'NCI1']
EPOCHS = 50
SEED = 1234


def _node_filtration(g: nx.Graph, filt_name: str) -> np.ndarray:
    """Per-node filtration values normalized to [0,1]. g has integer labels 0..n-1."""
    n = g.number_of_nodes()
    if filt_name == 'degree':
        vals = np.array([d for _, d in sorted(g.degree(), key=lambda x: x[0])],
                        dtype=np.float64)
    elif filt_name == 'clustering':
        c = nx.clustering(g)
        vals = np.array([c[i] for i in range(n)], dtype=np.float64)
    elif filt_name == 'closeness':
        c = nx.closeness_centrality(g)
        vals = np.array([c[i] for i in range(n)], dtype=np.float64)
    elif filt_name == 'hks':
        # compute_hks already returns normalized [0,1] values
        return compute_hks(g)
    else:
        raise ValueError(f'unknown filter {filt_name}')
    rng = vals.max() - vals.min()
    if rng < 1e-12:
        return np.zeros(n, dtype=np.float64)
    return (vals - vals.min()) / rng


def graph_to_pi_filter(g: nx.Graph, filt_name: str) -> np.ndarray:
    """Whole-graph extended PD under chosen filtration -> 5x5 PI flat (25,)."""
    g = nx.convert_node_labels_to_integers(g)
    if g.number_of_edges() == 0 or g.number_of_nodes() < 3:
        return np.zeros(25, dtype=np.float64)
    filt = _node_filtration(g, filt_name)
    # If HKS returned all zeros (degenerate graph), skip PD computation
    if filt.sum() == 0.0:
        return np.zeros(25, dtype=np.float64)
    sf = perturb_filter_function(g, filt)
    try:
        PD_up, ess0, PD_down, Pos, Neg = Union_find(sf)
        PD_one = Accelerate_PD(Pos, Neg, sf)
    except (ValueError, IndexError, KeyError):
        return np.zeros(25, dtype=np.float64)
    pd = []
    for arr in [PD_up, ess0, PD_down, PD_one]:
        a = np.asarray(arr, dtype=np.float64).reshape(-1, 2) if len(arr) else np.empty((0, 2))
        if a.size:
            pd.append(a)
    pd_all = np.concatenate(pd, axis=0) if pd else np.empty((0, 2))
    if pd_all.size == 0:
        return np.zeros(25, dtype=np.float64)
    return _IMAGER.transform(pd_all).reshape(-1)


def compute_filter_pi(name: str, filt_name: str, cache_dir: str = './data/MOL') -> np.ndarray:
    """Compute + cache PI for every graph under a given filter. Returns (N, 25)."""
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, f'{name}_PI_{filt_name}.npy')
    if os.path.exists(cache):
        return np.load(cache)
    graphs, _ = load_tudataset(name)
    from tqdm import tqdm
    pis = np.stack([graph_to_pi_filter(g, filt_name)
                    for g in tqdm(graphs, desc=f'{name}/{filt_name} PI')])
    np.save(cache, pis)
    return pis


def _train_eval_fold(ds, pis_t, labels, tr_idx, te_idx, in_dim, epochs, dev):
    model = GINClassifier(in_dim, use_pi=True, n_classes=int(labels.max()) + 1).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    tr = [ds[int(i)] for i in tr_idx]
    for j, i in enumerate(tr_idx):
        tr[j].orig_idx = torch.tensor(int(i))
    loader = DataLoader(tr, batch_size=32, shuffle=True)
    for ep in range(epochs):
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


def run_filter(name: str, filt_name: str, epochs: int = EPOCHS,
               seed: int = SEED, device: str = 'cuda'):
    torch.manual_seed(seed); np.random.seed(seed)
    ds = _make_dataset(name)
    pis = compute_filter_pi(name, filt_name)
    labels = np.array([int(ds[i].y.item()) for i in range(len(ds))])
    in_dim = ds.num_node_features
    dev = torch.device(device if torch.cuda.is_available() else 'cpu')
    pis_t = torch.tensor(pis, dtype=torch.float, device=dev)

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    accs = []
    for fold, (tr_idx, te_idx) in enumerate(skf.split(np.zeros(len(ds)), labels)):
        acc = _train_eval_fold(ds, pis_t, labels, tr_idx, te_idx, in_dim, epochs, dev)
        accs.append(acc)
        print(f'{name}/{filt_name} fold {fold}: acc={acc:.4f}')
    mean, std = float(np.mean(accs)), float(np.std(accs))
    print(f'{name} filter {filt_name}: {mean:.4f} +/- {std:.4f}')
    return mean, std


def main():
    os.makedirs('scores', exist_ok=True)
    out_path = 'scores/mol_filter_sweep_hks.txt'
    lines = []
    for name in DATASETS:
        for filt_name in FILTERS:
            try:
                mean, std = run_filter(name, filt_name)
                line = f'{name} filter {filt_name} acc {mean:.4f} std {std:.4f}'
            except Exception as e:
                line = f'{name} filter {filt_name} ERROR {e}'
                print(f'ERROR: {name}/{filt_name}: {e}')
            lines.append(line)
            # Incremental write so partial progress survives
            with open(out_path, 'w') as f:
                f.write('\n'.join(lines) + '\n')
    print(f'saved {out_path}')


if __name__ == '__main__':
    main()
