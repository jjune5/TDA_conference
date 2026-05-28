"""S3 centerpiece: exact PI is a train-graph-membership artifact; PDGNN restores genuine test signal.

For each dataset, segment the per-edge PI cache into
[train_pos | train_neg | val_pos | val_neg | test_pos | test_neg] (the layout
written by loaddatas.compute_persistence_image) and report mean PI L1-mass +
nonzero fraction per segment, for BOTH the exact (dionysus, data/TLCGNN/) and
the neural (PDGNN, data/PDGNN/) caches.

Key result:
- exact PI: train_pos >> everything else; val_pos/test_pos collapse to ~0
  (they are deleted from the graph before PH → no vicinity persistence). So at
  test time PI is ~0 for BOTH classes → no genuine discriminative signal; its
  training signal is the spurious "edge is in the training graph" indicator.
- PDGNN PI: predicts the diagram from structure on the SAME leakage-free
  (val/test-pos-removed) graph, yet assigns discriminative mass to test_pos
  (e.g. Photo test_pos >> test_neg) → genuine, leakage-free test signal.

This unifies: PDGNN>exact (P1/D1), the heterophily PI-hurt, and the shuffle
control recovery (EXP-1).
"""
from __future__ import annotations
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import loaddatas as lds


def segment_bounds(data):
    """Reconstruct the [train_pos|train_neg|val_pos|val_neg|test_pos|test_neg] split
    the same way the PI cache was generated (get_adj_split, seed 1234)."""
    ei = np.array(data.edge_index)
    adj = lds.sp.coo_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])),
                            shape=(data.num_nodes, data.num_nodes))
    tr, trf, va, vaf, te, tef = lds.get_adj_split(adj)
    counts = [len(tr), len(trf), len(va), len(vaf), len(te), len(tef)]
    names = ['train_pos', 'train_neg', 'val_pos', 'val_neg', 'test_pos', 'test_neg']
    bounds, c = {}, 0
    for n, k in zip(names, counts):
        bounds[n] = (c, c + k)
        c += k
    return bounds, c


def seg_stats(pi, lo, hi):
    if hi <= lo or hi > len(pi):
        return None
    rows = np.abs(pi[lo:hi]).sum(1)
    return float(rows.mean()), float((rows > 1e-6).mean()), hi - lo


def analyze(name):
    ds = lds.loaddatas(name)
    data = ds[0]
    bounds, total = segment_bounds(data)
    out = {'name': name, 'expected_rows': total, 'segments': {}}
    for tag, subdir in [('exact', 'data/TLCGNN'), ('PDGNN', 'data/PDGNN')]:
        cache = None
        for cand in (f'{subdir}/{name}.npy', f'{subdir}/{name.lower()}.npy'):
            if os.path.exists(cand):
                cache = cand
                break
        if cache is None:
            out['segments'][tag] = 'cache missing'
            continue
        pi = np.load(cache)
        seg = {}
        for s, (lo, hi) in bounds.items():
            r = seg_stats(pi, lo, hi)
            if r:
                seg[s] = {'mean_L1': round(r[0], 4), 'nonzero': round(r[1], 3), 'n': r[2]}
        out['segments'][tag] = {'cache': cache, 'rows': len(pi), 'per_segment': seg}
    return out


def main():
    names = sys.argv[1:] or ['Chameleon', 'Photo', 'Cora']
    print(f"{'dataset':10} {'src':6} {'train_pos':>16} {'test_pos':>16} {'test_neg':>16}")
    print('-' * 70)
    for name in names:
        res = analyze(name)
        for tag in ('exact', 'PDGNN'):
            seg = res['segments'].get(tag)
            if not isinstance(seg, dict):
                print(f"{name:10} {tag:6} {str(seg):>16}")
                continue
            ps = seg['per_segment']
            def fmt(k):
                d = ps.get(k)
                return f"{d['mean_L1']:.3f}({d['nonzero']:.2f}nz)" if d else "—"
            print(f"{name:10} {tag:6} {fmt('train_pos'):>16} {fmt('test_pos'):>16} {fmt('test_neg'):>16}")
        print()


if __name__ == '__main__':
    main()
