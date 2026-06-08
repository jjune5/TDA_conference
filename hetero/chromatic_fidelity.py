"""Stage-1 GO/NO-GO: does the NEURAL ChromaticPDGNN approximate the exact 0-dim
image/kernel/cokernel persistence on real data? Compares predicted-PI vs exact-PI per
leg on held-out target nodes. Gate: kernel-leg mean cosine >= 0.80. If below, the
engine cannot represent cross-type mingling and Stage-2 would be uninterpretable.
"""
from __future__ import annotations
import argparse, time
import numpy as np
import torch
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hetero.metapath_graph import load_hgb
from hetero.unified_filter import build_homo
from hetero.pdgnn_metapath import _graph_hks, _ego_filt_edges
from hetero.chromatic_labels import compute_chromatic_0dim
from hetero.chromatic_pdgnn import (gen_chromatic_samples, train_chromatic_pdgnn,
                                    _LEGS, device)
from node_ph_features import PI_RES
from sg2dgm import PersistenceImager as pimg_mod

_PL = PI_RES * PI_RES


def _cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 and nb < 1e-9:
        return 1.0
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(a @ b / (na * nb))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='ACM')
    ap.add_argument('--K', type=int, default=2)
    ap.add_argument('--hop', type=int, default=2)
    ap.add_argument('--max_nodes', type=int, default=60)
    ap.add_argument('--n_train', type=int, default=300)
    ap.add_argument('--n_test', type=int, default=80)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--hidden', type=int, default=32)
    a = ap.parse_args()

    t0 = time.time()
    d = load_hgb(a.dataset)
    g, ntype, n_t, y, masks = build_homo(d, a.dataset)
    print(f"[{a.dataset}] |V|={g.number_of_nodes()} n_target={n_t}; HKS K={a.K} ...", flush=True)
    hks = _graph_hks(g, a.K)
    gmax = [float(hks[:, k].max()) for k in range(a.K)]
    print(f"  HKS done ({time.time()-t0:.0f}s). gmax={np.round(gmax,3).tolist()}", flush=True)

    train = gen_chromatic_samples(g, hks, ntype, 0, a.hop, a.max_nodes, a.n_train, seed=0)
    print(f"  {len(train)} train ego-samples; training ...", flush=True)
    model = train_chromatic_pdgnn(train, hidden=a.hidden, layers=3, epochs=a.epochs)
    imager = pimg_mod.PersistenceImager(resolution=PI_RES)

    rng = np.random.RandomState(7)
    test_nodes = rng.choice(n_t, min(a.n_test, n_t), replace=False)
    cos = {L: [] for L in _LEGS}
    model.eval()
    with torch.no_grad():
        for v in test_nodes:
            for k in range(a.K):
                nf = {nd: float(hks[nd, k]) for nd in g.nodes()}
                res = _ego_filt_edges(g, int(v), a.hop, nf, a.max_nodes)
                if res is None:
                    continue
                filt, ei, nodelist = res
                if ei.shape[1] == 0:
                    continue
                is_L = (np.asarray(ntype)[np.array(nodelist)] == 0)
                lab = compute_chromatic_0dim(filt, ei, is_L, max_f=gmax[k])
                color = is_L.astype(np.float32).reshape(-1, 1)
                pred = model(torch.tensor(filt.reshape(-1, 1).astype(np.float32), device=device),
                             torch.tensor(color, device=device),
                             torch.tensor(ei.astype(np.int64), device=device))
                for L in _LEGS:
                    ex = lab[L]
                    ev = np.asarray(imager.transform(ex)).reshape(-1) if ex.shape[0] else np.zeros(_PL)
                    pts = pred[L].cpu().numpy(); pts = pts[pts[:, 1] > pts[:, 0]]
                    pv = np.asarray(imager.transform(pts.astype(np.float64))).reshape(-1) if pts.size else np.zeros(_PL)
                    cos[L].append(_cos(pv, ev))

    print(f"\n=== fidelity {a.dataset} (n_test={len(test_nodes)}, K={a.K}, {time.time()-t0:.0f}s) ===")
    for L in _LEGS:
        m = float(np.mean(cos[L])) if cos[L] else float('nan')
        print(f"  {L:9s} mean cosine = {m:.3f}  (n={len(cos[L])})")
    kmean = float(np.mean(cos['kernel'])) if cos['kernel'] else 0.0
    print(f"GATE kernel>=0.80: {'GO' if kmean >= 0.80 else 'NO-GO'} ({kmean:.3f})")


if __name__ == '__main__':
    main()
