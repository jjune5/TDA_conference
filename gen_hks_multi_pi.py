"""Variant B — generate the multi-scale HKS-filtration vicinity PI cache.

For each dataset, runs the EXACT same edge split (loaddatas.get_edges_split,
seed=1234) that the exact-PI / single-scale-HKS caches use, then calls
loaddatas.compute_persistence_image with TLCGNN_LP_FILTER=hks_multi so the
per-edge PI is computed at K diffusion scales (TLCGNN_HKS_SCALES) and the
per-scale (N,25) PI vectors are horizontally stacked -> (N, K*25).

The result is cached to data/HKS_MULTI_TLCGNN_<tag>/<name>.npy with the
canonical [train_pos|train_neg|val_pos|val_neg|test_pos|test_neg] layout, so
both the §14 diagnostic and the LP runner can consume it directly.

Run (env vars set by the SLURM wrapper or shell):
    TLCGNN_LP_FILTER=hks_multi TLCGNN_HKS_SCALES=0.1,1.0,10.0 \
    TLCGNN_CORES=64 python gen_hks_multi_pi.py Cora Chameleon
"""
from __future__ import annotations
import os
import sys
import time

# Ensure repo root on path and as cwd (relative ./data/... paths in loaddatas).
_REPO = os.path.dirname(os.path.abspath(__file__))
os.chdir(_REPO)
sys.path.insert(0, _REPO)

import numpy as np
import loaddatas as lds


def gen_one(name: str):
    print(f"\n{'='*64}\n[gen] dataset={name}  filter={os.environ.get('TLCGNN_LP_FILTER')}  "
          f"scales={os.environ.get('TLCGNN_HKS_SCALES')}  cores={os.environ.get('TLCGNN_CORES')}",
          flush=True)
    ds = lds.loaddatas(name)
    data = ds[0]
    # IMPORTANT: 0.05/0.1 reproduces the canonical split used to generate the
    # exact-PI / single-scale-HKS caches (28508 rows Cora, 169674 Chameleon).
    # The get_edges_split DEFAULT (0.2/0.2) does NOT match those caches.
    (train_edges, train_edges_false,
     val_edges, val_edges_false,
     test_edges, test_edges_false) = lds.get_edges_split(
        data, val_prop=0.05, test_prop=0.1)
    counts = dict(train_pos=len(train_edges), train_neg=len(train_edges_false),
                  val_pos=len(val_edges), val_neg=len(val_edges_false),
                  test_pos=len(test_edges), test_neg=len(test_edges_false))
    total = sum(counts.values())
    print(f"[gen] split counts={counts}  total={total}", flush=True)
    t0 = time.time()
    pi = lds.compute_persistence_image(
        data, train_edges, train_edges_false,
        val_edges, val_edges_false, test_edges, test_edges_false, name)
    dt = time.time() - t0
    print(f"[gen] {name}: PI shape={pi.shape}  L1mean={np.abs(pi).sum(1).mean():.4f}  "
          f"elapsed={dt/60:.1f} min", flush=True)
    assert pi.shape[0] == total, f"row mismatch {pi.shape[0]} != {total}"
    return pi.shape


if __name__ == '__main__':
    names = sys.argv[1:] or ['Cora', 'Chameleon']
    for nm in names:
        gen_one(nm)
    print("\n[gen] ALL DONE", flush=True)
