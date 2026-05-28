# Knowledge_Distillation/mol_dim_ablation.py
"""Experiment M2: Homology dimension ablation for molecular graph classification.

Masks persistence diagram by dimension before PI vectorization:
  - H0-only: keep 0-dim pairs (PD_up, ess0, PD_down) → connected components
  - H1-only: keep 1-dim pairs (PD_one) → rings/loops
  - both:    keep all pairs (= existing baseline)

Runs GINClassifier 10-fold stratified CV for each variant + no-PI baseline
on MUTAG, PROTEINS, NCI1.

Output: scores/mol_dim_ablation.txt
"""
from __future__ import annotations
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_mean_pool
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

# ---- absolute path anchors ----
_REPO_ROOT = '/mnt/data/users/junyoungpark/code/TLC-GNN'
_DATA_ROOT = os.path.join(_REPO_ROOT, 'data')

sys.path.insert(0, _REPO_ROOT)
from sg2dgm import PersistenceImager as pimg_mod
from Knowledge_Distillation.accelerated_PD import (
    perturb_filter_function, Union_find, Accelerate_PD
)

_IMAGER = pimg_mod.PersistenceImager(resolution=5)


# ---------------------------------------------------------------------------
# Filtration and PD
# ---------------------------------------------------------------------------

def _degree_filter(g: nx.Graph) -> np.ndarray:
    g = nx.convert_node_labels_to_integers(g)
    deg = np.array([d for _, d in sorted(g.degree(), key=lambda x: x[0])],
                   dtype=np.float64)
    m = deg.max() + 1e-10
    return deg / m


def _graph_to_pd_raw(g: nx.Graph):
    """Compute raw PD arrays, returning H0 and H1 separately.

    Returns:
        h0_pairs: np.ndarray (N0, 2) — union of PD_up, ess0, PD_down (all 0-dim)
        h1_pairs: np.ndarray (N1, 2) — PD_one (1-dim loops)
        None, None on degenerate graphs.
    """
    g = nx.convert_node_labels_to_integers(g)
    if g.number_of_edges() == 0 or g.number_of_nodes() < 3:
        return None, None
    filt = _degree_filter(g)
    sf = perturb_filter_function(g, filt)
    try:
        PD_up, ess0, PD_down, Pos, Neg = Union_find(sf)
        PD_one = Accelerate_PD(Pos, Neg, sf)
    except (ValueError, IndexError, KeyError):
        return None, None

    # H0: all 0-dim pairs
    h0_parts = []
    for arr in [PD_up, ess0, PD_down]:
        a = np.asarray(arr, dtype=np.float64).reshape(-1, 2) if len(arr) else np.empty((0, 2))
        if a.size:
            h0_parts.append(a)
    h0 = np.concatenate(h0_parts, axis=0) if h0_parts else np.empty((0, 2))

    # H1: 1-dim pairs
    h1 = np.asarray(PD_one, dtype=np.float64).reshape(-1, 2) if len(PD_one) else np.empty((0, 2))

    return h0, h1


def _pd_to_pi(pd: np.ndarray) -> np.ndarray:
    """Convert an (N, 2) PD array to a flattened 25-dim PI. Returns zeros if empty."""
    if pd is None or pd.size == 0:
        return np.zeros(25, dtype=np.float64)
    img = _IMAGER.transform(pd)
    return img.reshape(-1).cpu().numpy() if hasattr(img, 'cpu') else img.reshape(-1)


# ---------------------------------------------------------------------------
# Per-graph PI variants
# ---------------------------------------------------------------------------

def graph_to_pi_variants(g: nx.Graph):
    """Compute 3 PI vectors per graph: H0-only, H1-only, both.

    Returns dict with keys 'h0', 'h1', 'both', each (25,).
    """
    h0, h1 = _graph_to_pd_raw(g)
    if h0 is None:
        return {
            'h0': np.zeros(25, dtype=np.float64),
            'h1': np.zeros(25, dtype=np.float64),
            'both': np.zeros(25, dtype=np.float64),
        }
    # combine for 'both'
    parts = [a for a in [h0, h1] if a.size > 0]
    pd_both = np.concatenate(parts, axis=0) if parts else np.empty((0, 2))

    return {
        'h0': _pd_to_pi(h0),
        'h1': _pd_to_pi(h1),
        'both': _pd_to_pi(pd_both),
    }


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

def _cache_path(name: str, dim: str) -> str:
    """dim in {'h0', 'h1', 'both'}"""
    return os.path.join(_DATA_ROOT, 'MOL', f'{name}_PI_dim{dim}.npy')


def ensure_all_dim_caches(name: str) -> dict:
    """Compute all three dim variant caches in a single pass (efficient).

    Returns dict: {'h0': np.ndarray, 'h1': np.ndarray, 'both': np.ndarray}
    """
    dims = ['h0', 'h1', 'both']
    caches = {d: _cache_path(name, d) for d in dims}

    # Check which are already cached
    missing = [d for d in dims if not os.path.exists(caches[d])]
    all_cached = {d: np.load(caches[d]) for d in dims if d not in missing}

    if not missing:
        print(f'{name}: all dim caches loaded from disk')
        return all_cached

    # Load graphs once
    ds_path = os.path.join(_DATA_ROOT, f'TU_{name}')
    ds_raw = TUDataset(root=ds_path, name=name)
    graphs = [to_networkx(data, to_undirected=True) for data in ds_raw]

    # Compute all variants in ONE pass
    accs = {d: [] for d in dims}
    for g in tqdm(graphs, desc=f'{name} PI (h0+h1+both)'):
        variants = graph_to_pi_variants(g)
        for d in dims:
            accs[d].append(variants[d])

    result = {}
    for d in dims:
        arr = np.stack(accs[d])
        np.save(caches[d], arr)
        print(f'  Saved {caches[d]}')
        result[d] = arr

    # Merge with any already-cached
    result.update(all_cached)
    return result


def compute_all_pi_dim(name: str, dim: str) -> np.ndarray:
    """Load cached PI for a given dim variant, computing all if any are missing."""
    cache = _cache_path(name, dim)
    if os.path.exists(cache):
        return np.load(cache)
    all_caches = ensure_all_dim_caches(name)
    return all_caches[dim]


# ---------------------------------------------------------------------------
# GIN Classifier (same as mol_classify.py)
# ---------------------------------------------------------------------------

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


def _make_dataset(name: str):
    import torch_geometric.transforms as T
    from torch_geometric.utils import degree as pyg_degree
    ds_path = os.path.join(_DATA_ROOT, f'TU_{name}')
    ds = TUDataset(root=ds_path, name=name)
    if ds.num_node_features == 0:
        max_deg = 0
        for d in ds:
            if d.edge_index.numel() > 0:
                deg = pyg_degree(d.edge_index[0], num_nodes=d.num_nodes).max().item()
                max_deg = max(max_deg, int(deg))
        ds = TUDataset(root=ds_path, name=name,
                       transform=T.OneHotDegree(max_degree=max_deg))
    return ds


def _make_batch(ds, idx_list, dev):
    """Pre-batch a list of graph indices into a single PyG Batch on device."""
    from torch_geometric.data import Batch
    data_list = [ds[int(i)] for i in idx_list]
    batch = Batch.from_data_list(data_list)
    return batch.to(dev)


def _train_eval_fold(ds, pis_t, labels, tr_idx, te_idx, in_dim, use_pi, epochs, dev):
    model = GINClassifier(in_dim, use_pi=use_pi, n_classes=int(labels.max()) + 1).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Pre-fetch training data in fixed minibatches (avoid repeated DataLoader overhead)
    batch_size = 32
    tr_idx = list(tr_idx)
    import random
    rng = random.Random(42)

    # Pre-compute PI slices for train and test
    if use_pi:
        pi_tr = pis_t[tr_idx]   # (N_tr, 25) already on device
        pi_te = pis_t[list(te_idx)]

    for ep in range(epochs):
        model.train()
        # Shuffle training indices each epoch
        perm = list(range(len(tr_idx)))
        rng.shuffle(perm)
        for start in range(0, len(tr_idx), batch_size):
            batch_perm = perm[start:start + batch_size]
            batch_orig_idx = [tr_idx[k] for k in batch_perm]
            batch = _make_batch(ds, batch_orig_idx, dev)
            pi = pi_tr[batch_perm] if use_pi else None
            opt.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch, pi)
            loss = F.cross_entropy(out, batch.y)
            loss.backward(); opt.step()

    model.eval()
    te_idx = list(te_idx)
    te_batch = _make_batch(ds, te_idx, dev)
    with torch.no_grad():
        pi = pi_te if use_pi else None
        pred = model(te_batch.x, te_batch.edge_index, te_batch.batch, pi).argmax(1)
        correct = (pred == te_batch.y).sum().item()
    return correct / len(te_idx)


def run_variant(name: str, dim: str | None, epochs=100, seed=1234, device='cuda'):
    """Run 10-fold CV for one (dataset, variant) combo.

    dim=None → no-PI baseline.
    dim in {'h0','h1','both'} → use that PI variant.
    """
    torch.manual_seed(seed); np.random.seed(seed)
    ds = _make_dataset(name)
    use_pi = dim is not None
    labels = np.array([int(ds[i].y.item()) for i in range(len(ds))])
    in_dim = ds.num_node_features
    dev = torch.device(device if torch.cuda.is_available() else 'cpu')

    if use_pi:
        pis = compute_all_pi_dim(name, dim)
        pis_t = torch.tensor(pis, dtype=torch.float, device=dev)
    else:
        pis_t = None

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    accs = []
    for fold, (tr_idx, te_idx) in enumerate(skf.split(np.zeros(len(ds)), labels)):
        acc = _train_eval_fold(ds, pis_t, labels, tr_idx, te_idx, in_dim, use_pi, epochs, dev)
        accs.append(acc)
        tag = dim if dim else 'no-PI'
        print(f'  {name}/{tag} fold {fold}: {acc:.4f}')

    mean, std = float(np.mean(accs)), float(np.std(accs))
    tag = dim if dim else 'no-PI'
    print(f'  {name}/{tag}: {mean:.4f} +/- {std:.4f}')
    return mean, std


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=['MUTAG', 'PROTEINS', 'NCI1'],
                    help='Datasets to run (space-separated). NCI1 is slow; omit if needed.')
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    variants = ['h0', 'h1', 'both', None]   # None = no-PI
    variant_names = ['H0-only', 'H1-only', 'both', 'no-PI']

    results = {}  # (variant_name, dataset) -> (mean, std)

    for dataset in args.datasets:
        print(f'\n=== Dataset: {dataset} ===')
        # Pre-compute all dim caches in ONE pass before training
        print(f'  Pre-computing PI caches for {dataset}...')
        ensure_all_dim_caches(dataset)
        print(f'  All caches ready for {dataset}.')

        for dim, vname in zip(variants, variant_names):
            mean, std = run_variant(dataset, dim, epochs=args.epochs,
                                    seed=args.seed, device=args.device)
            results[(vname, dataset)] = (mean, std)

    # ---- Format table ----
    os.makedirs('scores', exist_ok=True)

    header = f"{'Variant':<12}" + ''.join(f"  {d:>20}" for d in args.datasets)
    rows = [header, '-' * len(header)]
    for vname in variant_names:
        row = f"{vname:<12}"
        for dataset in args.datasets:
            if (vname, dataset) in results:
                m, s = results[(vname, dataset)]
                row += f"  {m*100:>8.2f} +/- {s*100:>5.2f}%"
            else:
                row += f"  {'N/A':>20}"
        rows.append(row)

    table_str = '\n'.join(rows)

    # find which dim recovers the 'both' gain
    findings = []
    for dataset in args.datasets:
        if ('both', dataset) not in results or ('no-PI', dataset) not in results:
            continue
        gain_both = results[('both', dataset)][0] - results[('no-PI', dataset)][0]
        gain_h0 = results[('H0-only', dataset)][0] - results[('no-PI', dataset)][0] if ('H0-only', dataset) in results else None
        gain_h1 = results[('H1-only', dataset)][0] - results[('no-PI', dataset)][0] if ('H1-only', dataset) in results else None
        if gain_h0 is not None and gain_h1 is not None:
            driver = 'H1' if gain_h1 > gain_h0 else 'H0'
            findings.append(f'{dataset}: gain_both={gain_both*100:+.2f}%p, '
                            f'gain_H0={gain_h0*100:+.2f}%p, gain_H1={gain_h1*100:+.2f}%p '
                            f'=> driver={driver}')

    output = (
        'Experiment M2: Molecular PI homology dimension ablation\n'
        '========================================================\n\n'
        'Accuracy (%) | 10-fold stratified CV | mean +/- std\n\n'
        + table_str + '\n\n'
        + 'FINDINGS:\n' + '\n'.join(findings) + '\n'
    )

    out_path = 'scores/mol_dim_ablation.txt'
    with open(out_path, 'w') as f:
        f.write(output)
    print('\n' + output)
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()
