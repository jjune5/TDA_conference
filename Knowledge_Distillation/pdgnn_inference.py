# Knowledge_Distillation/pdgnn_inference.py
"""Generate PDGNN-predicted PI cache for a dataset.

Produces ./data/PDGNN/<name>.npy with shape (N_edges, 25),
where N_edges and the row order match exactly
loaddatas.compute_persistence_image's output (so cached files are
interchangeable via --pi-source flag in pipelines.py)."""

from __future__ import annotations
import os
import sys
import argparse
import numpy as np
import networkx as nx
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import loaddatas as lds
from Knowledge_Distillation.pdgnn_modern import PDGNN
from Knowledge_Distillation.prepare_data_LP_modern import _edge_vicinity, _ollivier_ricci_filt
from sg2dgm import PersistenceImager as pimg_mod


def _pd_to_pi(pd: np.ndarray, imager) -> np.ndarray:
    """Convert (K, 2) PD coords to 5x5 PI flatten = (25,)."""
    if pd.size == 0:
        return np.zeros(25, dtype=np.float64)
    return imager.transform(pd).reshape(-1)


@torch.no_grad()
def run_inference(name: str, ckpt_path: str = './data/PDGNN/checkpoints/pdgnn_lp.pt',
                  out_dir: str = './data/PDGNN', hop: int | None = None):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{name}.npy')
    if os.path.exists(out_path):
        print(f'cache exists: {out_path}; skipping')
        return out_path

    # Load model
    ckpt = torch.load(ckpt_path, map_location='cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = ckpt['config']
    model = PDGNN(hidden_dim=cfg['hidden_dim'], num_layers=cfg['num_layers']).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    # Load dataset + compute the SAME edge ordering as loaddatas
    ds = lds.loaddatas(name)
    data = ds[0]
    val_prop = 0.2 if name in ('PPI',) else 0.05
    test_prop = 0.2 if name in ('PPI',) else 0.1
    tr, trf, va, vaf, te, tef = lds.get_edges_split(data, val_prop=val_prop,
                                                     test_prop=test_prop)
    total_edges = np.concatenate((tr, trf, va, vaf, te, tef))

    if hop is None:
        hop = 2 if name == 'PubMed' else 1

    # Remove val/test positive edges from graph — matches TLC-GNN's compute_persistence_image
    # (loaddatas.py builds graph after these are deleted from data.edge_index in TLCGNN.call).
    # Without this, PDGNN gets to "see" val/test edges, biasing comparison.
    _mask_remove = set()
    for e in va.tolist():
        _mask_remove.add((e[0], e[1])); _mask_remove.add((e[1], e[0]))
    for e in te.tolist():
        _mask_remove.add((e[0], e[1])); _mask_remove.add((e[1], e[0]))
    ei_full = np.array(data.edge_index)
    keep = np.array([(int(u), int(v)) not in _mask_remove
                     for u, v in zip(ei_full[0], ei_full[1])])
    ei = ei_full[:, keep]

    # Build graph + Ricci on the val/test-positive-removed edge set
    g = nx.Graph()
    g.add_nodes_from(range(data.num_nodes))
    g.add_edges_from((int(ei[0, i]), int(ei[1, i])) for i in range(ei.shape[1]))
    # ricci curvature: compute on a Data clone with the leakage-free edge_index
    import copy
    data_for_ricci = copy.copy(data)
    data_for_ricci.edge_index = torch.from_numpy(ei).long()
    ricci_list = lds.compute_ricci_curvature(data_for_ricci)
    ricci_lookup = {(int(a), int(b)): float(c) for a, b, c in ricci_list}
    for a, b in g.edges():
        w = ricci_lookup.get((a, b), ricci_lookup.get((b, a), 0.0)) + 1
        g[a][b]['weight'] = max(w, 1e-6)

    imager = pimg_mod.PersistenceImager(resolution=5)
    PIs = np.zeros((len(total_edges), 25), dtype=np.float64)
    from tqdm import tqdm
    for i, (u, v) in enumerate(tqdm(total_edges, desc=f'PDGNN-PI {name}')):
        u, v = int(u), int(v)
        sub = _edge_vicinity(g, u, v, hop)
        if sub.number_of_edges() == 0:
            continue
        filt_vals = _ollivier_ricci_filt(sub, u, v, ricci_lookup)
        node_list = list(sub.nodes())
        remap = {n: idx for idx, n in enumerate(node_list)}
        ei_sub = np.array([(remap[a], remap[b]) for a, b in sub.edges()],
                          dtype=np.int64).T
        if ei_sub.size:
            ei_sub = np.concatenate([ei_sub, ei_sub[[1, 0]]], axis=1)
        else:
            ei_sub = np.zeros((2, 0), dtype=np.int64)
        filt_t = torch.tensor(filt_vals, dtype=torch.float, device=device).view(-1, 1)
        ei_t = torch.tensor(ei_sub, dtype=torch.long, device=device)
        if ei_t.size(1) == 0:
            continue
        pred = model(filt_t, ei_t).cpu().numpy()  # (E_sub, 2)
        PIs[i] = _pd_to_pi(pred, imager)

    np.save(out_path, PIs)
    print(f'saved PDGNN PI cache: {out_path} shape={PIs.shape}')
    return out_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('--ckpt', default='./data/PDGNN/checkpoints/pdgnn_lp.pt')
    parser.add_argument('--hop', type=int, default=None)
    args = parser.parse_args()
    run_inference(args.name, ckpt_path=args.ckpt, hop=args.hop)
