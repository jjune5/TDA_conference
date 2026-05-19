"""Generate PDGNN training data: (filt_value, edge_index, gt_PD0, gt_PD1, gt_PI) tuples
for node-centered vicinity subgraphs of a graph dataset.

Replaces the original `data_utils_NC.py` which depended on missing modules and
hard-coded absolute paths.
"""

from __future__ import annotations
import os
import sys
import argparse
import pickle
import numpy as np
import networkx as nx
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import loaddatas as lds
from sg2dgm import PersistenceImager as pimg_mod
from Knowledge_Distillation.accelerated_PD import (perturb_filter_function,
                                                    Union_find, Accelerate_PD)


def _hks_signature(subgraph: nx.Graph, t: float = 10.0) -> np.ndarray:
    from scipy.sparse import csgraph
    from scipy.linalg import eigh
    A = nx.adjacency_matrix(subgraph)
    L = csgraph.laplacian(A, normed=True).toarray()
    egvals, egvecs = eigh(L)
    return (egvecs ** 2 * np.exp(-t * egvals)).sum(axis=1)


def _node_filtration(subgraph: nx.Graph, filt: str, hks_time: float = 10.0,
                      ricci_curv: dict | None = None) -> np.ndarray:
    """Compute filtration value per node in the subgraph (relabeled to 0..n-1)."""
    n = subgraph.number_of_nodes()
    if filt == 'degree':
        vals = np.array([d for _, d in subgraph.degree()], dtype=np.float64)
    elif filt == 'centrality':
        bc = nx.degree_centrality(subgraph)
        vals = np.array([bc[i] for i in subgraph.nodes()], dtype=np.float64)
    elif filt == 'clustering':
        cc = nx.clustering(subgraph)
        vals = np.array([cc[i] for i in subgraph.nodes()], dtype=np.float64)
    elif filt == 'hks':
        vals = _hks_signature(subgraph, t=hks_time)
    elif filt == 'ricci':
        # Use Ricci-based shortest-path distance to the root node (node 0).
        # ricci_curv: dict mapping (u, v) -> curvature in original node ids
        # but subgraph nodes are relabeled; we use unit weights here.
        vals = np.array([nx.shortest_path_length(subgraph, source=0, target=v)
                         if nx.has_path(subgraph, 0, v) else 100.0
                         for v in subgraph.nodes()], dtype=np.float64)
    else:
        raise ValueError(f'Unknown filter: {filt}')
    m = vals.max() + 1e-10
    return vals / m


def compute_gt_pd_and_pi(subgraph: nx.Graph, filt_values: np.ndarray,
                          imager: pimg_mod.PersistenceImager) -> tuple:
    """Compute ground-truth 0/1-dim EPD using the accelerated_PD algorithm,
    plus the persistence image."""
    g = nx.convert_node_labels_to_integers(subgraph)
    sf = perturb_filter_function(g, filt_values)
    PD_up, ess0, PD_down, Pos_edges, Neg_edges = Union_find(sf)  # 0-dim parts + bookkeeping
    PD_one = Accelerate_PD(Pos_edges, Neg_edges, sf)              # 1-dim Ext

    def _to2d(arr):
        a = np.asarray(arr, dtype=np.float64)
        if a.ndim == 1:
            a = a.reshape(-1, 2) if a.size and a.size % 2 == 0 else np.empty((0, 2))
        if a.size == 0:
            a = np.empty((0, 2))
        return a

    PD0 = np.concatenate([_to2d(PD_up), _to2d(ess0)], axis=0)
    PD1 = np.concatenate([_to2d(PD_down), _to2d(PD_one)], axis=0)
    PD_all = np.concatenate([PD0, PD1], axis=0) if (len(PD0) + len(PD1)) else np.empty((0, 2))
    PI = imager.transform(PD_all).reshape(-1) if len(PD_all) else np.zeros(25)
    return PD0, PD1, PI


def build_vicinity_dataset(name: str, filt: str, hks_time: float = 10.0,
                            hop: int | None = None, max_nodes: int | None = None,
                            cache_dir: str = './data/PDGNN'):
    os.makedirs(cache_dir, exist_ok=True)
    suffix = f'_hks{hks_time}' if filt == 'hks' else ''
    cache = os.path.join(cache_dir, f'{name}_{filt}{suffix}_NC.pkl')
    if os.path.exists(cache):
        with open(cache, 'rb') as f:
            return pickle.load(f), cache

    print(f'[PDGNN data] building dataset for {name} / filt={filt}')
    ds = lds.loaddatas(name)
    data = ds[0]
    if hop is None:
        hop = 2 if name in ('Cora', 'Citeseer', 'PubMed') else 1

    g = nx.Graph()
    g.add_nodes_from(range(data.num_nodes))
    ei = np.array(data.edge_index)
    g.add_edges_from((int(ei[0, i]), int(ei[1, i])) for i in range(ei.shape[1]))

    imager = pimg_mod.PersistenceImager(resolution=5)
    samples = {}

    node_iter = list(g.nodes())
    if max_nodes is not None:
        rng = np.random.RandomState(1234)
        node_iter = rng.choice(node_iter, size=min(max_nodes, len(node_iter)),
                               replace=False).tolist()

    for cnt, u in enumerate(tqdm(node_iter, desc=f'{name}/{filt}')):
        nodes = [u] + [v for _, v in nx.bfs_edges(g, u, depth_limit=hop)]
        sub = g.subgraph(nodes).copy()
        if sub.number_of_edges() == 0:
            continue
        sub_relabel = nx.convert_node_labels_to_integers(sub, label_attribute='orig')
        filt_values = _node_filtration(sub_relabel, filt, hks_time=hks_time)
        PD0, PD1, PI = compute_gt_pd_and_pi(sub_relabel, filt_values, imager)
        if (len(PD0) == 0) and (len(PD1) == 0):
            continue
        ei_sub = np.array(list(sub_relabel.edges()), dtype=np.int64).T  # (2, E)
        # add reverse direction
        ei_sub = np.concatenate([ei_sub, ei_sub[[1, 0]]], axis=1)
        samples[cnt] = (PD0, PD1, PI, filt_values.astype(np.float32),
                        ei_sub.astype(np.int64))

    with open(cache, 'wb') as f:
        pickle.dump(samples, f, pickle.HIGHEST_PROTOCOL)
    print(f'[PDGNN data] saved {len(samples)} samples to {cache}')
    return samples, cache


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='Cora')
    parser.add_argument('--filt', default='degree',
                        choices=['degree', 'centrality', 'clustering', 'hks', 'ricci'])
    parser.add_argument('--hks_time', type=float, default=10.0)
    parser.add_argument('--hop', type=int, default=None)
    parser.add_argument('--max_nodes', type=int, default=None,
                        help='subsample nodes (e.g. 1000 for fast iter)')
    args = parser.parse_args()
    build_vicinity_dataset(args.name, args.filt, hks_time=args.hks_time,
                            hop=args.hop, max_nodes=args.max_nodes)
