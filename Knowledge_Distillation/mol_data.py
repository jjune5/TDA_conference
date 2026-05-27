# Knowledge_Distillation/mol_data.py
"""Load TUDataset molecule graphs and compute a whole-graph persistence image
per molecule. Filter = normalized node degree. Reuses the repo's accelerated_PD
+ PersistenceImager machinery (same as the LP pipeline, but on the full molecule
graph instead of an edge vicinity)."""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import networkx as nx
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx
from sg2dgm import PersistenceImager as pimg_mod
from Knowledge_Distillation.accelerated_PD import (perturb_filter_function,
                                                    Union_find, Accelerate_PD)

_IMAGER = pimg_mod.PersistenceImager(resolution=5)


def load_tudataset(name: str):
    """Return (list_of_networkx_graphs, label_tensor)."""
    ds = TUDataset(root=f'./data/TU_{name}', name=name)
    graphs, labels = [], []
    for data in ds:
        g = to_networkx(data, to_undirected=True)
        graphs.append(g)
        labels.append(int(data.y.item()))
    return graphs, torch.tensor(labels, dtype=torch.long)


def _degree_filter(g: nx.Graph) -> np.ndarray:
    """Normalized node degree as filtration value."""
    g = nx.convert_node_labels_to_integers(g)
    deg = np.array([d for _, d in sorted(g.degree(), key=lambda x: x[0])],
                   dtype=np.float64)
    m = deg.max() + 1e-10
    return deg / m


def graph_to_pi(g: nx.Graph) -> np.ndarray:
    """Whole-graph extended PD → 5x5 PI flattened to (25,)."""
    g = nx.convert_node_labels_to_integers(g)
    if g.number_of_edges() == 0 or g.number_of_nodes() < 3:
        return np.zeros(25, dtype=np.float64)
    filt = _degree_filter(g)
    sf = perturb_filter_function(g, filt)
    try:
        PD_up, ess0, PD_down, Pos, Neg = Union_find(sf)
        PD_one = Accelerate_PD(Pos, Neg, sf)
    except (ValueError, IndexError):
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


def compute_all_pi(name: str, cache_dir: str = './data/MOL'):
    """Compute + cache PI for every graph in a TUDataset. Returns (N, 25) array."""
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, f'{name}_PI.npy')
    if os.path.exists(cache):
        return np.load(cache)
    graphs, _ = load_tudataset(name)
    from tqdm import tqdm
    pis = np.stack([graph_to_pi(g) for g in tqdm(graphs, desc=f'{name} PI')])
    np.save(cache, pis)
    return pis


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='MUTAG')
    args = p.parse_args()
    pis = compute_all_pi(args.dataset)
    print(f'{args.dataset}: PI cache shape {pis.shape}')
