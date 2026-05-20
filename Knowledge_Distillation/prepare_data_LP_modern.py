# Knowledge_Distillation/prepare_data_LP_modern.py
"""Generate PDGNN training data for link prediction (edge-centered vicinities).

For each (u, v) edge in the source graph, extracts the intersection of
hop-k neighborhoods, computes the same Ollivier-Ricci-based 'sum' filtration
as loaddatas.compute_persistence_image, then records (filt_values,
edge_index, ground-truth (birth, death) coords).
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
from sg2dgm import riccidist2dgm as sg2dgm
from sg2dgm import PersistenceImager as pimg_mod


def _edge_vicinity(g: nx.Graph, u: int, v: int, hop: int) -> nx.Graph:
    """Edge-centered vicinity: intersection of k-hop neighborhoods of u and v.
    Matches TLC-GNN's V_12 (cf. loaddatas.compute_persistence_image)."""
    Nu = {u} | {w for _, w in nx.bfs_edges(g, u, depth_limit=hop)}
    Nv = {v} | {w for _, w in nx.bfs_edges(g, v, depth_limit=hop)}
    V12 = Nu & Nv
    if u not in V12: V12.add(u)
    if v not in V12: V12.add(v)
    return g.subgraph(V12).copy()


def _ollivier_ricci_filt(sub: nx.Graph, u: int, v: int,
                          ricci_lookup: dict) -> np.ndarray:
    """For each node x in sub, filter value = d(x,u) + d(x,v) under
    Ollivier-Ricci-weighted shortest paths. Matches TLC-GNN's 'sum' filter."""
    nodes = list(sub.nodes())
    vals = []
    for x in nodes:
        if x in (u, v):
            vals.append(0.0)
            continue
        try:
            p1 = nx.dijkstra_path(sub, x, u, weight='weight')
            d1 = sum(ricci_lookup.get((p1[i], p1[i+1]), 1.0) + 1
                     for i in range(len(p1)-1))
        except Exception:
            d1 = 100.0
        try:
            p2 = nx.dijkstra_path(sub, x, v, weight='weight')
            d2 = sum(ricci_lookup.get((p2[i], p2[i+1]), 1.0) + 1
                     for i in range(len(p2)-1))
        except Exception:
            d2 = 100.0
        vals.append(d1 + d2)
    arr = np.array(vals, dtype=np.float64)
    m = arr.max() + 1e-10
    return arr / m


def build_lp_vicinity_dataset(name: str, hop: int | None = None,
                               max_edges: int | None = None,
                               cache_dir: str = './data/PDGNN'):
    """Returns dict of {edge_id: (filt_values, edge_index_local, PD_pairs, u_local, v_local)}.

    PD_pairs is (K, 2) numpy: ground-truth (birth, death) coordinates for
    the K extended-persistence pairs computed via dionysus on this vicinity.
    Stored as the per-edge label for PDGNN supervised training."""
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, f'{name}_LP_train.pkl')
    if os.path.exists(cache):
        with open(cache, 'rb') as f:
            return pickle.load(f), cache

    ds = lds.loaddatas(name)
    data = ds[0]
    if hop is None:
        hop = 2 if name in ('PubMed',) else 1

    # Build full graph + Ricci curvature lookup
    g = nx.Graph()
    g.add_nodes_from(range(data.num_nodes))
    ei = np.array(data.edge_index)
    g.add_edges_from((int(ei[0, i]), int(ei[1, i])) for i in range(ei.shape[1]))
    ricci_list = lds.compute_ricci_curvature(data)
    ricci_lookup = {(int(a), int(b)): float(c) for a, b, c in ricci_list}
    for a, b in g.edges():
        w = ricci_lookup.get((a, b), ricci_lookup.get((b, a), 0.0)) + 1
        g[a][b]['weight'] = max(w, 1e-6)

    # Sample source edges
    edges = list(g.edges())
    if max_edges is not None and len(edges) > max_edges:
        rng = np.random.RandomState(1234)
        edges = [edges[i] for i in rng.choice(len(edges), max_edges, replace=False)]

    samples = {}
    imager = pimg_mod.PersistenceImager(resolution=5)
    g2pi = sg2dgm.graph2pi(g, ricci_curv=ricci_list)  # holds PD compute method
    for idx, (u, v) in enumerate(tqdm(edges, desc=f'{name}/LP')):
        sub = _edge_vicinity(g, u, v, hop)
        if sub.number_of_edges() == 0:
            continue
        filt_vals = _ollivier_ricci_filt(sub, u, v, ricci_lookup)
        # ground-truth PD via dionysus (same path TLC-GNN uses)
        try:
            dgms = g2pi.compute_extended_pd_for_edge(sub, u, v, filt_vals)
        except AttributeError:
            # If sg2dgm doesn't expose a single-edge method, fall back to
            # the local accelerated_PD path:
            from Knowledge_Distillation.accelerated_PD import (
                perturb_filter_function, Union_find, Accelerate_PD)
            sub_re = nx.convert_node_labels_to_integers(sub)
            sf = perturb_filter_function(sub_re, filt_vals)
            PD_up, ess0, PD_down, Pos, Neg = Union_find(sf)
            PD_one = Accelerate_PD(Pos, Neg, sf)
            dgms = np.concatenate([
                np.asarray(PD_up, dtype=np.float64).reshape(-1, 2) if len(PD_up) else np.empty((0,2)),
                np.asarray(ess0, dtype=np.float64).reshape(-1, 2) if len(ess0) else np.empty((0,2)),
                np.asarray(PD_down, dtype=np.float64).reshape(-1, 2) if len(PD_down) else np.empty((0,2)),
                np.asarray(PD_one, dtype=np.float64).reshape(-1, 2) if len(PD_one) else np.empty((0,2)),
            ], axis=0)
        ei_sub = np.array(list(sub.edges()), dtype=np.int64).T  # (2, E)
        if ei_sub.size:
            ei_sub = np.concatenate([ei_sub, ei_sub[[1, 0]]], axis=1)
        else:
            ei_sub = np.zeros((2, 0), dtype=np.int64)
        # Map sub-graph node ids to local 0..n-1
        node_list = list(sub.nodes())
        remap = {n: i for i, n in enumerate(node_list)}
        ei_sub_local = np.array([[remap[ei_sub[0, k]] for k in range(ei_sub.shape[1])],
                                  [remap[ei_sub[1, k]] for k in range(ei_sub.shape[1])]],
                                 dtype=np.int64) if ei_sub.size else np.zeros((2, 0), dtype=np.int64)
        samples[idx] = (filt_vals.astype(np.float32),
                        ei_sub_local,
                        dgms.astype(np.float64),
                        int(remap[u]), int(remap[v]))

    with open(cache, 'wb') as f:
        pickle.dump(samples, f, pickle.HIGHEST_PROTOCOL)
    print(f'[PDGNN-LP data] saved {len(samples)} samples to {cache}')
    return samples, cache


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='PubMed')
    parser.add_argument('--hop', type=int, default=None)
    parser.add_argument('--max_edges', type=int, default=5000)
    args = parser.parse_args()
    build_lp_vicinity_dataset(args.name, hop=args.hop, max_edges=args.max_edges)
