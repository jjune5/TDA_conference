"""
smoke_test.py — Thread C Rung 0 smoke test on ONE target.

Confirms, before any full run:
  1. PDB parse + CDR-H3 extraction yields ~10-25 CA residues (with anchors).
  2. Rips PH yields H1 features for native and generated loops.
  3. topo_distance(candidate, native) computes.
  4. DockQ join via reconstructed row<->file mapping works.

Prints diagnostics. If H1 is mostly empty, raise K / max_edge_length; if still
empty, the analysis falls back to H0 + loop-likeness and says so.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pdb_to_cdrh3 as p2c
import cdrh3_ph as ph

RABD_DIR = "/mnt/data/users/junyoungpark/code/antibody_models/FlowDesign/results/codesign_single_H3_rabd"
RABD_CSV = "/mnt/data/users/junyoungpark/code/antibody_models/FlowDesign/results/codesign_single_H3_rabd_comprehensive_eval/per_sample_metrics.csv"


def reconstruct_row_to_file(target_dir, n_rows):
    """Replicate eval_baseline.py ordering: sorted glob of NNNN.pdb."""
    pred_dir = os.path.join(target_dir, "H_CDR3")
    files = sorted(glob.glob(os.path.join(pred_dir, "[0-9][0-9][0-9][0-9].pdb")))
    return files  # files[j] corresponds to CSV row j within this target block


def main(target_idx=5, k_anchor=3):
    # pick a target dir (sorted, to match eval ordering)
    target_dirs = sorted(
        d for d in glob.glob(os.path.join(RABD_DIR, "*")) if os.path.isdir(d)
    )
    tdir = target_dirs[target_idx]
    meta_path = os.path.join(tdir, "metadata.json")
    ref_pdb = os.path.join(tdir, "reference.pdb")
    meta = json.load(open(meta_path))
    pdb_id = meta["identifier"].split("_")[0]
    chain, rf, rl = p2c.load_metadata_range(meta_path)
    print(f"=== SMOKE TEST target={os.path.basename(tdir)} pdb={pdb_id} ===")
    print(f"H3 window: chain={chain} resseq [{rf},{rl}]  K_anchor={k_anchor}")

    # native
    nat_coords = p2c.extract_from_flowdesign(ref_pdb, meta_path, k_anchor=k_anchor)
    print(f"\nNATIVE CA atoms: {nat_coords.shape[0]} (expect ~{(rl-rf+1)+2*k_anchor})")
    nat_ph = ph.compute_persistence(nat_coords)
    print(f"NATIVE diameter={nat_ph['diameter']:.2f}  H0_bars={nat_ph['pd0'].shape[0]}  H1_bars={nat_ph['pd1'].shape[0]}")
    if nat_ph["pd1"].shape[0]:
        lt = nat_ph["pd1"][:, 1] - nat_ph["pd1"][:, 0]
        print(f"NATIVE H1 lifetimes: {np.round(lt,2)}  loop_likeness(total)={ph.loop_likeness(nat_ph['pd1']):.3f}")

    # DockQ join
    df = pd.read_csv(RABD_CSV)
    sub = df[df["pdb"] == pdb_id].reset_index(drop=True)
    files = reconstruct_row_to_file(tdir, len(sub))
    print(f"\nCSV rows for {pdb_id}: {len(sub)}   generated files: {len(files)}")
    print(f"  mapping check: row 0 -> {os.path.basename(files[0])}, row {len(sub)-1} -> {os.path.basename(files[-1])}")

    # process first 5 candidates
    print("\n--- first 5 candidates ---")
    n_h1_nonempty = 0
    for j in range(min(5, len(files))):
        c = p2c.extract_from_flowdesign(files[j], meta_path, k_anchor=k_anchor)
        cph = ph.compute_persistence(c)
        td = ph.topo_distance(cph["pd1"], nat_ph["pd1"], metric="bottleneck")
        ll = ph.loop_likeness(cph["pd1"])
        dockq = sub.loc[j, "DockQ"]
        rmsd = sub.loc[j, "RMSD(CA) CDRH3"]
        if cph["pd1"].shape[0] > 0:
            n_h1_nonempty += 1
        print(f"  {os.path.basename(files[j])}: N={c.shape[0]} H1={cph['pd1'].shape[0]} "
              f"topo_dist_H1={td:.3f} loop_like={ll:.3f} DockQ={dockq} RMSD_H3={rmsd:.3f}")

    # H1 coverage across ALL candidates of this target
    h1_counts = []
    for j in range(len(files)):
        c = p2c.extract_from_flowdesign(files[j], meta_path, k_anchor=k_anchor)
        cph = ph.compute_persistence(c)
        h1_counts.append(cph["pd1"].shape[0])
    h1_counts = np.array(h1_counts)
    print(f"\nH1 coverage over {len(files)} candidates: "
          f"{(h1_counts>0).sum()}/{len(files)} have >=1 H1 bar; "
          f"mean H1 bars={h1_counts.mean():.2f}")
    print("SMOKE TEST OK" if (h1_counts > 0).mean() > 0.5 else
          "WARNING: H1 sparse -> consider larger K / fall back to loop-likeness+H0")


if __name__ == "__main__":
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    main(target_idx=idx, k_anchor=k)
