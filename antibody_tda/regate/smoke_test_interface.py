"""
smoke_test_interface.py — Thread C Rung 0' smoke test on ONE multi_cdr target.

Confirms BEFORE the full re-gate run:
  1. Chain identification (heavy/light/antigen) from the dir identifier AND the
     RabD/SAbDab JSON annotation agree.
  2. Interface extraction yields a sensible #residues (~10-40 expected).
  3. Interface Rips PH yields features (H0 always; H1 hopefully) for native and
     generated complexes; topo_distance(candidate,native) computes.
  4. Metric join: sorted glob '[0-9]{4}.pdb' (excludes REF1.pdb) <-> CSV rows by
     index; verify against (a) per_target DockQ_mean and (b) RMSD-CDRH3 content
     correlation for a high-confidence target.

Prints diagnostics.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pdb_to_cdrh3 as p2c
import cdrh3_ph as ph
import interface_ph as iph

BASE = "/mnt/data/users/junyoungpark/code/antibody_models/FlowDesign/results"
RABD_DIR = f"{BASE}/codesign_multi_cdr_rabd"
RABD_CSV = f"{BASE}/codesign_multi_cdr_rabd_comprehensive_eval/per_sample_metrics.csv"
RABD_PTM = f"{BASE}/codesign_multi_cdr_rabd_comprehensive_eval/per_target_metrics.csv"
RABD_JSON = "/mnt/data/users/junyoungpark/code/antibody_models/RabD_dataset/test.json"


def load_jsonl(path):
    out = {}
    for line in open(path):
        line = line.strip()
        if line:
            r = json.loads(line)
            out[r["pdb"]] = r
    return out


def h3_window_ca(pdb, chain, lo, hi):
    cas = sorted(p2c._parse_ca_atoms(pdb, chain), key=lambda t: (t[0], t[1]))
    sel = [xyz for (rs, ic, xyz) in cas if lo <= rs <= hi]
    return np.array(sel) if sel else np.zeros((0, 3))


def main(target_idx=0, cutoff=10.0):
    recs = load_jsonl(RABD_JSON)
    tdirs = sorted(d for d in glob.glob(os.path.join(RABD_DIR, "*")) if os.path.isdir(d))
    tdir = tdirs[target_idx]
    meta = json.load(open(os.path.join(tdir, "metadata.json")))
    ident = meta["identifier"]
    pid = ident.split("_")[0]
    print(f"=== SMOKE (interface) target={os.path.basename(tdir)} pdb={pid} ===")

    # ---- chain identity: identifier vs JSON ----
    parsed = iph.chains_from_identifier(ident)
    rec = recs.get(pid)
    print(f"identifier='{ident}' -> parsed {parsed}")
    if rec:
        print(f"JSON: heavy={rec['heavy_chain']} light={rec['light_chain']} "
              f"antigen={rec['antigen_chains']} cdrh3_pos={rec.get('cdrh3_pos')}")
        H, L = rec["heavy_chain"], rec["light_chain"]
        Ag = rec["antigen_chains"]
    else:
        H, L, Ag = parsed
        print("(no JSON record; using identifier parse)")
    ab_chains = [H, L]
    print(f"USING ab_chains={ab_chains} ag_chains={Ag}")

    ref_pdb = os.path.join(tdir, "reference.pdb")

    # ---- native interface ----
    nat_if = iph.extract_interface(ref_pdb, ab_chains, Ag, cutoff=cutoff)
    print(f"\nNATIVE interface: ok={nat_if['ok']} n_ab={nat_if['n_ab']} n_ag={nat_if['n_ag']} "
          f"total={nat_if['coords'].shape[0]} edges={nat_if['n_edges']} reason='{nat_if.get('reason','')}'")
    nat_pd = iph.interface_pd(nat_if["coords"])
    print(f"NATIVE interface PD: diam={nat_pd['diameter']:.2f} H0={nat_pd['pd0'].shape[0]} "
          f"H1={nat_pd['pd1'].shape[0]} loop_like={iph.interface_loop_likeness(nat_pd['pd1']):.3f}")

    # ---- metric join check ----
    df = pd.read_csv(RABD_CSV)
    sub = df[df["pdb"] == pid].reset_index(drop=True)
    files = sorted(glob.glob(os.path.join(tdir, "MultipleCDRs", "[0-9][0-9][0-9][0-9].pdb")))
    print(f"\nfiles(4-digit)={len(files)} [excludes REF1.pdb] CSV_rows={len(sub)}")
    print(f"  first={os.path.basename(files[0])} last={os.path.basename(files[-1])}")
    ptm = pd.read_csv(RABD_PTM)
    ptm_row = ptm[ptm["pdb"] == pid]
    if len(ptm_row):
        print(f"  per_target_metrics: n={int(ptm_row['n'].iloc[0])} "
              f"DockQ_mean={ptm_row['DockQ_mean'].iloc[0]:.4f}")
    dockq = pd.to_numeric(sub["DockQ"], errors="coerce").to_numpy()
    print(f"  my-join block DockQ mean(finite)={np.nanmean(dockq):.4f} "
          f"(should match per_target DockQ_mean)")

    # RMSD-CDRH3 content correlation (order check), if we know the H3 window
    if rec and rec.get("cdrh3_pos"):
        lo, hi = rec["cdrh3_pos"]
        nat_h3 = h3_window_ca(ref_pdb, H, lo, hi)
        my = []
        for f in files:
            c = h3_window_ca(f, H, lo, hi)
            my.append(float(np.sqrt(((c - nat_h3) ** 2).sum(1).mean()))
                      if c.shape == nat_h3.shape and c.shape[0] > 0 else np.nan)
        my = np.array(my)
        csvr = pd.to_numeric(sub["RMSD(CA) CDRH3"], errors="coerce").to_numpy()
        n = min(len(my), len(csvr))
        m = np.isfinite(my[:n]) & np.isfinite(csvr[:n])
        if m.sum() > 2 and np.std(my[:n][m]) > 0 and np.std(csvr[:n][m]) > 0:
            rr = np.corrcoef(my[:n][m], csvr[:n][m])[0, 1]
            print(f"  RMSD-CDRH3 order-check corr(my_resseq_window, CSV)={rr:.4f} "
                  f"(native H3 CA={nat_h3.shape[0]}; high corr => correct ordering)")
        else:
            print(f"  RMSD-CDRH3 order-check: insufficient variance (nat H3 CA={nat_h3.shape[0]})")

    # ---- first 5 candidate interfaces ----
    print("\n--- first 5 candidate interfaces ---")
    n_h1_nonempty = 0
    for j in range(min(5, len(files))):
        cif = iph.extract_interface(files[j], ab_chains, Ag, cutoff=cutoff)
        cpd = iph.interface_pd(cif["coords"])
        tdb = iph.interface_topo_distance(cpd["pd1"], nat_pd["pd1"], "bottleneck")
        tdw = iph.interface_topo_distance(cpd["pd1"], nat_pd["pd1"], "wasserstein")
        ll = iph.interface_loop_likeness(cpd["pd1"])
        dq = sub.loc[j, "DockQ"]
        rm = sub.loc[j, "RMSD(CA) CDRH3"]
        if cpd["pd1"].shape[0] > 0:
            n_h1_nonempty += 1
        print(f"  {os.path.basename(files[j])}: ifN={cif['coords'].shape[0]} "
              f"(ab={cif['n_ab']},ag={cif['n_ag']}) H1={cpd['pd1'].shape[0]} "
              f"tdH1_bn={tdb:.3f} tdH1_ws={tdw:.3f} loop={ll:.3f} DockQ={dq} RMSD_H3={rm}")

    # ---- H1 coverage across all candidates ----
    if_sizes, h1_counts = [], []
    for f in files:
        cif = iph.extract_interface(f, ab_chains, Ag, cutoff=cutoff)
        cpd = iph.interface_pd(cif["coords"])
        if_sizes.append(cif["coords"].shape[0])
        h1_counts.append(cpd["pd1"].shape[0])
    if_sizes = np.array(if_sizes); h1_counts = np.array(h1_counts)
    print(f"\nInterface size over {len(files)} candidates: "
          f"mean={if_sizes.mean():.1f} min={if_sizes.min()} max={if_sizes.max()}")
    print(f"H1 coverage: {(h1_counts > 0).sum()}/{len(files)} have >=1 H1 bar; "
          f"mean H1 bars={h1_counts.mean():.2f}")
    ok = (if_sizes.mean() >= 6) and ((h1_counts > 0).mean() > 0.3)
    print("SMOKE TEST OK" if ok else
          "WARNING: small interface or sparse H1 -> rely on H0 + topo-dist; documented")


if __name__ == "__main__":
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    cut = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
    main(target_idx=idx, cutoff=cut)
