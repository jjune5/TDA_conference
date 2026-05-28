"""
regate_analysis.py — Thread C, Rung 0' RE-GATE GO/NO-GO.

Re-runs the Rung-0 gate with TWO deliberate changes designed to give topology its
best shot before we either build PH-guided generation (Rung 1) or close the thread:

  (1) RICHER signal: antibody--antigen INTERFACE contact-graph PH (binding-relevant
      geometry) instead of the CDR-H3 loop's own intrinsic shape.
  (2) HIGHER-variance setting: codesign_multi_cdr (all 6 CDRs regenerated) instead
      of codesign_single_H3 -> ~2.5x more within-target RMSD-CDRH3 variance.

Simpson's-paradox-safe: the Rung-0 NO-GO came from a strong POOLED corr (-0.58)
that was a between-target artifact; the WITHIN-target corr was ~0. Selection is a
within-target decision, so WE TRUST WITHIN-TARGET numbers and report pooled only to
expose the paradox.

Per target (in the comprehensive_eval set, so metrics exist):
  - native interface-PD (reference.pdb) + each candidate interface-PD -> topo_dist
    (bottleneck & wasserstein, H1 & H0); native-free interface loop-likeness.
  - CONTROL: the Rung-0 LOOP-PH signal (CDR-H3 CA Rips PH) on the SAME candidates,
    using the IMGT cdrh3_pos window from the RabD/SAbDab JSON (+ K anchors).
  - join per-candidate DockQ / RMSD(CA) CDRH3 / pmetric by index (sorted glob of
    '[0-9]{4}.pdb', which EXCLUDES the REF1.pdb decoy) <-> the target's CSV block.
    Verified in smoke test: block DockQ mean == per_target_metrics DockQ_mean
    (correct rows) AND RMSD-CDRH3 content corr ~1 on clean targets (correct order).

C1  WITHIN-target Pearson+Spearman of interface-topo-dist vs (a) DockQ, (b) RMSD-H3.
    Report per-target mean AND pooled; flag Simpson's paradox if they disagree.
C2  SELECTION per target by {random, best pmetric, min interface-topo-dist [ORACLE,
    uses native], max interface-loop-likeness [native-FREE]}; compare mean DockQ /
    RMSD-H3 of the picked candidate. + oracle ceiling and mean-of-all references.
C3  CONTROL: repeat C1 with the LOOP-PH signal on the SAME data -> does the richer
    INTERFACE signal predict better than the loop signal (isolates "richer signal"
    from "higher-variance setting")?

HONEST LIMITS: interface PH is Calpha-only (coarse, rigid-motion-invariant shape
lens; ignores side chains / backbone orientation / atomic packing). pmetric is
hardcoded 0.0 in eval -> 'best pmetric' is degenerate (picks first). DockQ has some
NaN/empty entries -> dropped per-candidate.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pdb_to_cdrh3 as p2c
import cdrh3_ph as ph
import interface_ph as iph

warnings.filterwarnings("ignore", category=RuntimeWarning)

BASE = "/mnt/data/users/junyoungpark/code/antibody_models/FlowDesign/results"
DSETS = {
    "rabd": (
        f"{BASE}/codesign_multi_cdr_rabd",
        f"{BASE}/codesign_multi_cdr_rabd_comprehensive_eval/per_sample_metrics.csv",
        f"{BASE}/codesign_multi_cdr_rabd_comprehensive_eval/per_target_metrics.csv",
    ),
    "time_split": (
        f"{BASE}/codesign_multi_cdr_time_split",
        f"{BASE}/codesign_multi_cdr_time_split_comprehensive_eval/per_sample_metrics.csv",
        f"{BASE}/codesign_multi_cdr_time_split_comprehensive_eval/per_target_metrics.csv",
    ),
}
CHAIN_JSONS = [
    "/mnt/data/users/junyoungpark/code/antibody_models/RabD_dataset/test.json",
    "/mnt/data/users/junyoungpark/code/antibody_models/SAbDab_split_time/test.json",
    "/mnt/data/users/junyoungpark/code/antibody_models/SAbDab_split_time/train.json",
    "/mnt/data/users/junyoungpark/code/antibody_models/SAbDab_split_time/valid.json",
]
RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))
INTERFACE_CUTOFF = 10.0
K_ANCHOR = 3


# -----------------------------------------------------------------------------
# Per-target candidate processing
# -----------------------------------------------------------------------------
def _h3_window_coords(pdb_path, chain, lo, hi, k_anchor):
    """CDR-H3 CA coords with K anchors, using the IMGT resseq window (loop-PH
    CONTROL). Mirrors pdb_to_cdrh3.extract_cdrh3_ca but takes explicit window."""
    return p2c.extract_cdrh3_ca(pdb_path, chain, lo, hi, k_anchor=k_anchor)


def process_target(args):
    tdir, csv_rows, chain_rec, cutoff, k_anchor = args
    ref_pdb = os.path.join(tdir, "reference.pdb")
    meta_path = os.path.join(tdir, "metadata.json")
    try:
        meta = json.load(open(meta_path))
    except Exception:
        return [], {"target": os.path.basename(tdir), "skip": "no metadata"}
    ident = meta["identifier"]
    pid = ident.split("_")[0]

    # chain identity: JSON preferred, identifier fallback
    if chain_rec is not None:
        H, L = chain_rec["heavy_chain"], chain_rec["light_chain"]
        Ag = list(chain_rec["antigen_chains"])
        h3 = chain_rec.get("cdrh3_pos")
    else:
        parsed = iph.chains_from_identifier(ident)
        if parsed is None:
            return [], {"target": os.path.basename(tdir), "skip": "no chains"}
        H, L, Ag = parsed
        h3 = None
    ab_chains = [H, L]

    # native interface PD
    nat_if = iph.extract_interface(ref_pdb, ab_chains, Ag, cutoff=cutoff)
    if not nat_if["ok"]:
        return [], {"target": os.path.basename(tdir), "pdb": pid,
                    "skip": f"native interface bad: {nat_if.get('reason')}"}
    nat_pd = iph.interface_pd(nat_if["coords"])

    # native loop PD (control), if we know the H3 window
    nat_loop_pd = None
    if h3 is not None:
        nat_loop_coords = _h3_window_coords(ref_pdb, H, int(h3[0]), int(h3[1]), k_anchor)
        if nat_loop_coords.shape[0] >= 3:
            nat_loop_pd = ph.compute_persistence(nat_loop_coords)

    files = sorted(glob.glob(os.path.join(tdir, "MultipleCDRs", "[0-9][0-9][0-9][0-9].pdb")))
    n = min(len(files), len(csv_rows))

    rows = []
    n_if_ok = 0
    for j in range(n):
        f = files[j]
        crow = csv_rows[j]
        cif = iph.extract_interface(f, ab_chains, Ag, cutoff=cutoff)
        if not cif["ok"]:
            continue
        n_if_ok += 1
        cpd = iph.interface_pd(cif["coords"])
        rec = {
            "pdb": pid,
            "target_dir": os.path.basename(tdir),
            "sample_file": os.path.basename(f),
            "sample_idx": j,
            # interface geometry
            "if_n": int(cif["coords"].shape[0]),
            "if_n_ab": cif["n_ab"], "if_n_ag": cif["n_ag"], "if_edges": cif["n_edges"],
            "if_n_h1": int(cpd["pd1"].shape[0]), "if_n_h0": int(cpd["pd0"].shape[0]),
            # interface topo distance to native (RICHER signal)
            "if_td_h1_bn": iph.interface_topo_distance(cpd["pd1"], nat_pd["pd1"], "bottleneck"),
            "if_td_h1_ws": iph.interface_topo_distance(cpd["pd1"], nat_pd["pd1"], "wasserstein"),
            "if_td_h0_bn": iph.interface_topo_distance(cpd["pd0"], nat_pd["pd0"], "bottleneck"),
            "if_td_h0_ws": iph.interface_topo_distance(cpd["pd0"], nat_pd["pd0"], "wasserstein"),
            # native-free interface loop-likeness
            "if_loop_total": iph.interface_loop_likeness(cpd["pd1"], "total"),
            "if_loop_max": iph.interface_loop_likeness(cpd["pd1"], "max"),
            # joined metrics (cleaned later)
            "DockQ": pd.to_numeric(crow.get("DockQ"), errors="coerce"),
            "RMSD_CDRH3": pd.to_numeric(crow.get("RMSD(CA) CDRH3"), errors="coerce"),
            "RMSD_aligned": pd.to_numeric(crow.get("RMSD(CA) aligned"), errors="coerce"),
            "TMscore": pd.to_numeric(crow.get("TMscore"), errors="coerce"),
            "LDDT": pd.to_numeric(crow.get("LDDT"), errors="coerce"),
            "AAR_H3": pd.to_numeric(crow.get("AAR H3"), errors="coerce"),
            "pmetric": pd.to_numeric(crow.get("pmetric"), errors="coerce"),
        }
        rec["if_td_combined_ws"] = rec["if_td_h1_ws"] + rec["if_td_h0_ws"]

        # ---- LOOP-PH CONTROL (Rung-0 signal) on the same candidate ----
        if h3 is not None and nat_loop_pd is not None:
            loop_coords = _h3_window_coords(f, H, int(h3[0]), int(h3[1]), k_anchor)
            if loop_coords.shape[0] >= 3:
                lpd = ph.compute_persistence(loop_coords)
                rec["loop_n_ca"] = int(loop_coords.shape[0])
                rec["loop_n_h1"] = int(lpd["pd1"].shape[0])
                rec["loop_td_h1_ws"] = ph.topo_distance(lpd["pd1"], nat_loop_pd["pd1"], "wasserstein")
                rec["loop_td_h1_bn"] = ph.topo_distance(lpd["pd1"], nat_loop_pd["pd1"], "bottleneck")
                rec["loop_likeness_total"] = ph.loop_likeness(lpd["pd1"], "total")
            else:
                rec["loop_n_ca"] = int(loop_coords.shape[0])
                rec["loop_td_h1_ws"] = np.nan; rec["loop_td_h1_bn"] = np.nan
                rec["loop_likeness_total"] = np.nan; rec["loop_n_h1"] = 0
        else:
            rec["loop_n_ca"] = 0; rec["loop_td_h1_ws"] = np.nan
            rec["loop_td_h1_bn"] = np.nan; rec["loop_likeness_total"] = np.nan
            rec["loop_n_h1"] = 0
        rows.append(rec)

    diag = {
        "target": os.path.basename(tdir), "pdb": pid,
        "ab_chains": ab_chains, "ag_chains": Ag,
        "native_if_n": int(nat_if["coords"].shape[0]),
        "native_if_h1": int(nat_pd["pd1"].shape[0]),
        "n_candidates": len(rows), "n_files": len(files),
        "n_if_ok": n_if_ok,
        "has_loop_control": bool(h3 is not None and nat_loop_pd is not None),
        "chain_source": "json" if chain_rec is not None else "identifier",
    }
    return rows, diag


# -----------------------------------------------------------------------------
# Correlation helpers
# -----------------------------------------------------------------------------
def _corr(x, y, method="pearson"):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return np.nan
    xs, ys = x[m], y[m]
    if np.std(xs) == 0 or np.std(ys) == 0:
        return np.nan
    if method == "spearman":
        from scipy.stats import spearmanr
        return float(spearmanr(xs, ys).correlation)
    return float(np.corrcoef(xs, ys)[0, 1])


def per_target_corr(df, xcol, ycol, method="pearson"):
    vals = []
    for pdb, g in df.groupby("pdb"):
        c = _corr(g[xcol], g[ycol], method)
        if np.isfinite(c):
            vals.append(c)
    vals = np.array(vals)
    if vals.size == 0:
        return dict(n=0, mean=np.nan, std=np.nan, median=np.nan, frac_negative=np.nan,
                    t_stat=np.nan, p_value=np.nan)
    # one-sample t-test: is the mean within-target corr != 0?
    from scipy.stats import ttest_1samp
    if vals.size >= 2 and np.std(vals) > 0:
        t = ttest_1samp(vals, 0.0)
        tstat, pval = float(t.statistic), float(t.pvalue)
    else:
        tstat, pval = np.nan, np.nan
    return dict(
        n=int(vals.size), mean=float(np.mean(vals)), std=float(np.std(vals)),
        median=float(np.median(vals)), frac_negative=float(np.mean(vals < 0)),
        t_stat=tstat, p_value=pval,
    )


def within_centered_corr(df, xcol, ycol, method="pearson"):
    """Pooled correlation after subtracting each target's mean from BOTH x and y.
    Isolates the within-target signal that selection exploits."""
    wt = df[["pdb", xcol, ycol]].copy()
    wt = wt[np.isfinite(wt[xcol]) & np.isfinite(wt[ycol])]
    if len(wt) < 3:
        return np.nan
    wt["_x"] = wt[xcol] - wt.groupby("pdb")[xcol].transform("mean")
    wt["_y"] = wt[ycol] - wt.groupby("pdb")[ycol].transform("mean")
    return _corr(wt["_x"], wt["_y"], method)


# -----------------------------------------------------------------------------
# Selection (C2)
# -----------------------------------------------------------------------------
def selection_experiment(df, n_random_seeds=500, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    # (column, direction, uses_native)
    strategies = {
        "min_if_td_h1_ws": ("if_td_h1_ws", "min", True),
        "min_if_td_h1_bn": ("if_td_h1_bn", "min", True),
        "min_if_td_combined_ws": ("if_td_combined_ws", "min", True),
        "max_if_loop_likeness": ("if_loop_total", "max", False),
        "min_loop_td_h1_ws": ("loop_td_h1_ws", "min", True),   # CONTROL (loop signal, native)
        "max_loop_likeness": ("loop_likeness_total", "max", False),  # CONTROL (loop, native-free)
        "best_pmetric": ("pmetric", "max", False),             # DEGENERATE
    }

    picked = defaultdict(list)
    random_dockq, random_rmsd = [], []
    oracle_dockq, oracle_rmsd, meanall_dockq, meanall_rmsd = [], [], [], []

    for pdb, g in df.groupby("pdb"):
        gg = g[np.isfinite(g["DockQ"])].copy()
        if len(gg) == 0:
            continue
        oracle_dockq.append(gg["DockQ"].max())
        meanall_dockq.append(gg["DockQ"].mean())
        rfin = gg[np.isfinite(gg["RMSD_CDRH3"])]
        if len(rfin):
            oracle_rmsd.append(rfin["RMSD_CDRH3"].min())
            meanall_rmsd.append(rfin["RMSD_CDRH3"].mean())

        idxs = gg.index.to_numpy()
        rsel = rng.choice(idxs, size=n_random_seeds, replace=True)
        random_dockq.append(gg.loc[rsel, "DockQ"].to_numpy())
        random_rmsd.append(gg.loc[rsel, "RMSD_CDRH3"].to_numpy())

        for sname, (col, direction, _u) in strategies.items():
            if col not in gg.columns:
                continue
            sub = gg[np.isfinite(gg[col])]
            if len(sub) == 0:
                continue
            if sub[col].nunique() == 1:
                pick = sub.iloc[0]
            elif direction == "min":
                pick = sub.loc[sub[col].idxmin()]
            else:
                pick = sub.loc[sub[col].idxmax()]
            picked[sname].append((pick["DockQ"], pick["RMSD_CDRH3"]))

    def _ms(arr):
        arr = np.asarray(arr, dtype=float); arr = arr[np.isfinite(arr)]
        return (float(np.mean(arr)), float(np.std(arr)), int(arr.size)) if arr.size else (np.nan, np.nan, 0)

    out = {}
    rmean = np.array([a.mean() for a in random_dockq])
    rmean_rmsd = np.array([a[np.isfinite(a)].mean() for a in random_rmsd if np.isfinite(a).any()])
    out["random"] = {"uses_native": False, "mean_DockQ": float(np.mean(rmean)),
                     "std_DockQ": float(np.std(rmean)),
                     "mean_RMSD_CDRH3": float(np.mean(rmean_rmsd)), "n_targets": int(rmean.size)}
    out["mean_of_all"] = {"uses_native": False, "mean_DockQ": _ms(meanall_dockq)[0],
                          "std_DockQ": _ms(meanall_dockq)[1],
                          "mean_RMSD_CDRH3": _ms(meanall_rmsd)[0], "n_targets": _ms(meanall_dockq)[2]}
    out["oracle_best_DockQ_ceiling"] = {"uses_native": True, "mean_DockQ": _ms(oracle_dockq)[0],
                                        "std_DockQ": _ms(oracle_dockq)[1],
                                        "mean_RMSD_CDRH3_of_minRMSD_pick": _ms(oracle_rmsd)[0],
                                        "n_targets": _ms(oracle_dockq)[2]}
    uses_native_map = {k: v[2] for k, v in strategies.items()}
    for sname, pairs in picked.items():
        dq = [p[0] for p in pairs]; rm = [p[1] for p in pairs]
        m, s, nn = _ms(dq)
        out[sname] = {"uses_native": uses_native_map[sname], "mean_DockQ": m, "std_DockQ": s,
                      "mean_RMSD_CDRH3": _ms(rm)[0], "n_targets": nn}
    return out


# -----------------------------------------------------------------------------
# Build candidate table
# -----------------------------------------------------------------------------
def build_candidate_table(dataset, cutoff, k_anchor, max_targets, n_workers):
    base_dir, csv_path, _ptm = DSETS[dataset]
    df_csv = pd.read_csv(csv_path)
    chain_map = iph.load_chain_map(*CHAIN_JSONS)
    target_dirs = sorted(d for d in glob.glob(os.path.join(base_dir, "*")) if os.path.isdir(d))

    csv_blocks = {}
    for pdb, g in df_csv.groupby("pdb", sort=False):
        csv_blocks[pdb] = g.reset_index(drop=True)

    tasks = []
    for tdir in target_dirs:
        meta_path = os.path.join(tdir, "metadata.json")
        if not os.path.exists(meta_path):
            continue
        try:
            ident = json.load(open(meta_path))["identifier"]
        except Exception:
            continue
        pid = ident.split("_")[0]
        if pid not in csv_blocks:
            continue
        chain_rec = chain_map.get(pid)
        tasks.append((tdir, csv_blocks[pid].to_dict("records"), chain_rec, cutoff, k_anchor))

    if max_targets and max_targets < len(tasks):
        idx = np.linspace(0, len(tasks) - 1, max_targets).round().astype(int)
        idx = sorted(set(idx.tolist()))
        tasks = [tasks[i] for i in idx]

    print(f"[{dataset}] processing {len(tasks)} targets (cutoff={cutoff} K_anchor={k_anchor}) "
          f"with {n_workers} workers ...")
    all_rows, diags = [], []
    if n_workers > 1:
        import multiprocessing as mp
        with mp.Pool(n_workers) as pool:
            for i, (rows, diag) in enumerate(pool.imap(process_target, tasks, chunksize=1)):
                all_rows.extend(rows); diags.append(diag)
                if (i + 1) % 10 == 0:
                    print(f"  ... {i+1}/{len(tasks)} targets done")
    else:
        for i, t in enumerate(tasks):
            rows, diag = process_target(t)
            all_rows.extend(rows); diags.append(diag)
            if (i + 1) % 10 == 0:
                print(f"  ... {i+1}/{len(tasks)} targets done")
    return pd.DataFrame(all_rows), diags


# -----------------------------------------------------------------------------
# Verdict
# -----------------------------------------------------------------------------
def _verdict(summary):
    c1d = summary["C1_interface_vs_DockQ"]
    c1r = summary["C1_interface_vs_RMSD"]
    c2 = summary["C2_selection"]

    # primary interface signal for verdict
    if_dockq_pt = c1d["if_td_h1_ws"]["per_target_spearman"]
    if_rmsd_pt = c1r["if_td_h1_ws"]["per_target_spearman"]
    rnd = c2["random"]["mean_DockQ"]
    nf = c2.get("max_if_loop_likeness", {}).get("mean_DockQ", np.nan)   # native-free interface
    orc = c2.get("min_if_td_h1_ws", {}).get("mean_DockQ", np.nan)        # native interface oracle
    ceil = c2["oracle_best_DockQ_ceiling"]["mean_DockQ"]
    headroom = ceil - rnd
    nf_gain = (nf - rnd) if np.isfinite(nf) else -np.inf
    orc_gain = (orc - rnd) if np.isfinite(orc) else -np.inf

    lines = []
    lines.append(f"Interface td_h1_ws vs DockQ  per-target Spearman: mean={if_dockq_pt['mean']:+.3f} "
                 f"(median={if_dockq_pt['median']:+.3f}, frac_neg={if_dockq_pt['frac_negative']:.2f}, "
                 f"p={if_dockq_pt['p_value']:.3g}, n={if_dockq_pt['n']})")
    lines.append(f"Interface td_h1_ws vs RMSD-H3 per-target Spearman: mean={if_rmsd_pt['mean']:+.3f} "
                 f"(frac_pos={1-if_rmsd_pt['frac_negative']:.2f}, p={if_rmsd_pt['p_value']:.3g})")
    lines.append(f"Random selection mean DockQ:               {rnd:.4f}")
    if np.isfinite(nf):
        lines.append(f"Native-FREE (max if-loop-likeness) DockQ:  {nf:.4f} (delta={nf-rnd:+.4f})")
    if np.isfinite(orc):
        lines.append(f"Native-oracle (min if-topo-dist) DockQ:    {orc:.4f} (delta={orc-rnd:+.4f})")
    lines.append(f"Oracle best-DockQ ceiling:                 {ceil:.4f} (headroom={headroom:+.4f})")

    # An interface signal "exists for DockQ" if within-target corr is meaningfully
    # non-zero in the expected (negative) direction AND statistically supported.
    dockq_signal = (if_dockq_pt["mean"] < -0.10 and if_dockq_pt["frac_negative"] > 0.60
                    and (not np.isfinite(if_dockq_pt["p_value"]) or if_dockq_pt["p_value"] < 0.05))
    # RMSD-H3 signal: topo-dist should be POSITIVELY related to RMSD (more distortion).
    rmsd_signal = (if_rmsd_pt["mean"] > 0.10 and if_rmsd_pt["frac_negative"] < 0.40
                   and (not np.isfinite(if_rmsd_pt["p_value"]) or if_rmsd_pt["p_value"] < 0.05))
    sel_beats = (np.isfinite(nf) and nf_gain > 0.2 * max(headroom, 1e-6) and nf_gain > 0.005)
    orc_beats = (np.isfinite(orc) and orc_gain > 0.2 * max(headroom, 1e-6) and orc_gain > 0.005)

    if (dockq_signal and sel_beats):
        verdict = "GO"
        rec = ("Interface-PH shows a real WITHIN-target relationship with DockQ AND native-free "
               "interface selection beats random by a meaningful fraction of headroom. "
               "-> Build Rung 1 (PH-guided sampling) on the interface signal (if_td_h1_ws / "
               "if_loop_likeness).")
    elif (dockq_signal or rmsd_signal) and (sel_beats or orc_beats):
        verdict = "WEAK/CONDITIONAL GO"
        rec = ("Interface-PH carries a within-target signal (DockQ and/or RMSD-H3) and selection "
               "improves over random, but mainly via the native-using oracle and/or only one "
               "metric. Rung 1 is plausible but should be scoped narrowly to the metric that "
               "shows signal; verify the native-free score is deployable.")
    else:
        verdict = "NO-GO"
        rec = ("Interface-PH (even this richer signal, even in the higher-variance multi_cdr "
               "setting) shows ~0 WITHIN-target relationship with DockQ/RMSD-H3 and interface "
               "selection does not beat random. Topology does not help antibody binding here. "
               "-> Close topology-guided antibody generation.")
    return verdict, rec, lines


# -----------------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------------
def run(dataset="rabd", cutoff=INTERFACE_CUTOFF, k_anchor=K_ANCHOR, max_targets=0, n_workers=8):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df, diags = build_candidate_table(dataset, cutoff, k_anchor, max_targets, n_workers)
    if df.empty:
        raise RuntimeError("No candidate rows produced.")

    n_targets = df["pdb"].nunique()
    n_cand = len(df)
    dockq_avail = int(np.isfinite(df["DockQ"]).sum())
    loop_avail = int(np.isfinite(df["loop_td_h1_ws"]).sum())
    print(f"\n[{dataset}] candidates={n_cand} targets={n_targets} "
          f"DockQ_finite={dockq_avail} loop_control_finite={loop_avail}")

    csv_out = os.path.join(RESULTS_DIR, f"candidate_table_{dataset}.csv")
    df.to_csv(csv_out, index=False)
    diag_out = os.path.join(RESULTS_DIR, f"diagnostics_{dataset}.json")
    json.dump(diags, open(diag_out, "w"), indent=2, default=float)

    # ---- C1: interface signal vs DockQ and vs RMSD-H3 ----
    if_cols = ["if_td_h1_ws", "if_td_h1_bn", "if_td_h0_ws", "if_td_combined_ws", "if_loop_total"]
    def _c1block(ycol):
        out = {}
        for col in if_cols:
            out[col] = {
                "per_target_pearson": per_target_corr(df, col, ycol, "pearson"),
                "per_target_spearman": per_target_corr(df, col, ycol, "spearman"),
                "within_centered_pearson": within_centered_corr(df, col, ycol, "pearson"),
                "pooled_pearson": _corr(df[col], df[ycol], "pearson"),
                "pooled_spearman": _corr(df[col], df[ycol], "spearman"),
            }
        return out
    c1_dockq = _c1block("DockQ")
    c1_rmsd = _c1block("RMSD_CDRH3")

    # ---- C2: selection ----
    c2 = selection_experiment(df)

    # ---- C3: CONTROL loop-PH signal vs DockQ and RMSD (same data) ----
    loop_cols = ["loop_td_h1_ws", "loop_td_h1_bn", "loop_likeness_total"]
    def _c3block(ycol):
        out = {}
        for col in loop_cols:
            if not np.isfinite(df[col]).any():
                continue
            out[col] = {
                "per_target_pearson": per_target_corr(df, col, ycol, "pearson"),
                "per_target_spearman": per_target_corr(df, col, ycol, "spearman"),
                "within_centered_pearson": within_centered_corr(df, col, ycol, "pearson"),
                "pooled_pearson": _corr(df[col], df[ycol], "pearson"),
                "pooled_spearman": _corr(df[col], df[ycol], "spearman"),
            }
        return out
    c3 = {
        "loop_vs_DockQ": _c3block("DockQ"),
        "loop_vs_RMSD_CDRH3": _c3block("RMSD_CDRH3"),
        "note": ("CONTROL = Rung-0 loop-PH signal computed on the SAME multi_cdr candidates "
                 "via IMGT cdrh3_pos window (+K anchors). Compare per-target means to the "
                 "interface signal in C1 to isolate 'richer signal' from 'higher variance'."),
    }

    # ---- interface vs loop head-to-head (within-target spearman magnitudes) ----
    def _abs_mean_pt_spearman(block, col):
        d = block.get(col, {}).get("per_target_spearman", {})
        return abs(d.get("mean", np.nan))
    if_vs_loop = {
        "interface_DockQ_pt_spearman_absmean": _abs_mean_pt_spearman(c1_dockq, "if_td_h1_ws"),
        "loop_DockQ_pt_spearman_absmean": _abs_mean_pt_spearman(c3["loop_vs_DockQ"], "loop_td_h1_ws"),
        "interface_RMSD_pt_spearman_absmean": _abs_mean_pt_spearman(c1_rmsd, "if_td_h1_ws"),
        "loop_RMSD_pt_spearman_absmean": _abs_mean_pt_spearman(c3["loop_vs_RMSD_CDRH3"], "loop_td_h1_ws"),
    }

    summary = {
        "dataset": dataset,
        "setting": "codesign_multi_cdr (all 6 CDRs regenerated; rest of complex native)",
        "signal": "antibody-antigen INTERFACE contact-graph Vietoris-Rips PH (CA, cutoff=%.1f A)" % cutoff,
        "interface_cutoff": cutoff, "k_anchor": k_anchor,
        "n_targets": n_targets, "n_candidates": n_cand,
        "n_candidates_with_DockQ": dockq_avail, "n_candidates_with_loop_control": loop_avail,
        "interface_if_n_mean": float(df["if_n"].mean()),
        "interface_h1_coverage_frac": float((df["if_n_h1"] > 0).mean()),
        "DockQ_range": [float(np.nanmin(df["DockQ"])), float(np.nanmax(df["DockQ"]))],
        "DockQ_std_pooled": float(np.nanstd(df["DockQ"])),
        "DockQ_within_target_std_mean": float(df.groupby("pdb")["DockQ"].std().mean()),
        "RMSD_CDRH3_within_target_std_mean": float(df.groupby("pdb")["RMSD_CDRH3"].std().mean()),
        "C1_interface_vs_DockQ": c1_dockq,
        "C1_interface_vs_RMSD": c1_rmsd,
        "C2_selection": c2,
        "C3_loop_control": c3,
        "interface_vs_loop_headtohead": if_vs_loop,
        "join_logic": ("Per-sample, by INDEX. Within each target the generated samples are "
                       "sorted(glob 'MultipleCDRs/[0-9]{4}.pdb') = 0000..0039 (the REF1.pdb "
                       "reference copy is EXCLUDED by the 4-digit glob); these map 1:1 to the "
                       "target's CSV rows in order. VERIFIED: per-target DockQ block-mean == "
                       "per_target_metrics DockQ_mean (correct rows) AND RMSD-CDRH3 content "
                       "corr ~1.0 on clean targets (correct order, smoke test)."),
        "chain_id_logic": ("heavy/light/antigen from RabD/SAbDab JSON (preferred) keyed by pdb; "
                           "fallback = parse FlowDesign identifier '<pdb>_<H>_<L>_<Ag...>'. "
                           "Smoke test: JSON and identifier agree."),
        "pmetric_note": "pmetric hardcoded 0.0 in eval_baseline.py -> 'best pmetric' degenerate.",
        "honest_limits": ("Interface PH is CA-only: coarse, rigid-motion-invariant SHAPE of the "
                          "contact patch; ignores side chains, backbone orientation, atomic "
                          "packing, and chemistry. DockQ has some NaN entries (dropped per "
                          "candidate)."),
    }
    verdict, rec, vlines = _verdict(summary)
    summary["verdict"] = verdict
    summary["recommendation"] = rec
    summary["verdict_lines"] = vlines

    json_out = os.path.join(RESULTS_DIR, f"summary_{dataset}.json")
    json.dump(summary, open(json_out, "w"), indent=2, default=float)
    txt_out = os.path.join(RESULTS_DIR, f"summary_{dataset}.txt")
    _write_txt(txt_out, summary)
    png_out = os.path.join(RESULTS_DIR, f"scatter_interface_{dataset}.png")
    _make_plot(df, png_out, dataset)

    print(f"\nWrote:\n  {csv_out}\n  {diag_out}\n  {json_out}\n  {txt_out}\n  {png_out}")
    return summary, df


def _write_txt(path, s):
    L = []
    L.append("=" * 80)
    L.append(f"Thread C / Rung 0' RE-GATE  GO-NO-GO  —  dataset={s['dataset']}")
    L.append(f"setting: {s['setting']}")
    L.append(f"signal:  {s['signal']}")
    L.append("=" * 80)
    L.append(f"targets={s['n_targets']} candidates={s['n_candidates']} "
             f"DockQ_finite={s['n_candidates_with_DockQ']} loop_control={s['n_candidates_with_loop_control']}")
    L.append(f"interface size mean={s['interface_if_n_mean']:.1f} CAs  "
             f"H1 coverage={s['interface_h1_coverage_frac']*100:.1f}%")
    L.append(f"DockQ range={[round(x,3) for x in s['DockQ_range']]} pooled_std={s['DockQ_std_pooled']:.4f} "
             f"WITHIN-target std(DockQ)={s['DockQ_within_target_std_mean']:.4f} "
             f"WITHIN-target std(RMSD-H3)={s['RMSD_CDRH3_within_target_std_mean']:.4f}")
    L.append("")
    L.append("--- VERDICT: " + s["verdict"] + " ---")
    for ln in s["verdict_lines"]:
        L.append("  " + ln)
    L.append("  RECOMMENDATION: " + s["recommendation"])
    L.append("")
    L.append("C1  INTERFACE-PH vs DockQ   [TRUST per-target; expect NEGATIVE]")
    _c1lines(L, s["C1_interface_vs_DockQ"])
    L.append("")
    L.append("C1  INTERFACE-PH vs RMSD-CDRH3   [TRUST per-target; expect POSITIVE]")
    _c1lines(L, s["C1_interface_vs_RMSD"])
    L.append("")
    L.append("C2  SELECTION — mean DockQ / mean RMSD(CA)CDRH3 of picked candidate")
    L.append(f"  {'strategy':28s} {'native?':8s} {'meanDockQ':>10s} {'+-std':>8s} {'meanRMSD':>9s} {'n':>4s}")
    order = ["oracle_best_DockQ_ceiling", "min_if_td_h1_ws", "min_if_td_h1_bn",
             "min_if_td_combined_ws", "max_if_loop_likeness", "min_loop_td_h1_ws",
             "max_loop_likeness", "mean_of_all", "random", "best_pmetric"]
    for k in order:
        if k not in s["C2_selection"]:
            continue
        d = s["C2_selection"][k]
        rmsd = d.get("mean_RMSD_CDRH3", d.get("mean_RMSD_CDRH3_of_minRMSD_pick", np.nan))
        L.append(f"  {k:28s} {str(d.get('uses_native')):8s} {d['mean_DockQ']:>10.4f} "
                 f"{d.get('std_DockQ', float('nan')):>8.4f} {rmsd:>9.3f} {d.get('n_targets',0):>4d}")
    L.append("  (min_loop_*/max_loop_* = Rung-0 LOOP signal CONTROL on same data)")
    L.append("")
    L.append("C3  CONTROL: LOOP-PH signal (Rung-0) on the SAME multi_cdr candidates")
    L.append("  loop vs DockQ:")
    _c1lines(L, s["C3_loop_control"]["loop_vs_DockQ"], indent=4)
    L.append("  loop vs RMSD-CDRH3:")
    _c1lines(L, s["C3_loop_control"]["loop_vs_RMSD_CDRH3"], indent=4)
    L.append("")
    h = s["interface_vs_loop_headtohead"]
    L.append("INTERFACE vs LOOP head-to-head (|per-target Spearman mean| of td_h1_ws):")
    L.append(f"  DockQ : interface={h['interface_DockQ_pt_spearman_absmean']:.3f}  "
             f"loop={h['loop_DockQ_pt_spearman_absmean']:.3f}")
    L.append(f"  RMSD-H3: interface={h['interface_RMSD_pt_spearman_absmean']:.3f}  "
             f"loop={h['loop_RMSD_pt_spearman_absmean']:.3f}")
    L.append("")
    L.append("JOIN LOGIC: " + s["join_logic"])
    L.append("CHAIN ID: " + s["chain_id_logic"])
    L.append("PMETRIC: " + s["pmetric_note"])
    L.append("LIMITS: " + s["honest_limits"])
    L.append("=" * 80)
    open(path, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


def _c1lines(L, block, indent=2):
    pad = " " * indent
    for col, d in block.items():
        pt = d["per_target_pearson"]; ps = d["per_target_spearman"]
        L.append(f"{pad}{col:20s} per-tgt Pearson={pt['mean']:+.3f}(fneg={pt['frac_negative']:.2f},p={pt['p_value']:.2g}) "
                 f"Spearman={ps['mean']:+.3f}(p={ps['p_value']:.2g}) | within-ctr r={d['within_centered_pearson']:+.3f} "
                 f"| pooled r={d['pooled_pearson']:+.3f} rho={d['pooled_spearman']:+.3f}")


def _make_plot(df, path, dataset):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    panels = [
        (axes[0, 0], "if_td_h1_ws", "DockQ", "interface H1 Wasserstein dist to native", "DockQ"),
        (axes[0, 1], "if_td_h1_ws", "RMSD_CDRH3", "interface H1 Wasserstein dist to native", "RMSD(CA) CDRH3"),
        (axes[1, 0], "if_loop_total", "DockQ", "interface loop-likeness (native-free)", "DockQ"),
        (axes[1, 1], "loop_td_h1_ws", "DockQ", "LOOP H1 Wasserstein dist (Rung-0 control)", "DockQ"),
    ]
    # color points by target so the Simpson's-paradox structure is visible
    pdbs = df["pdb"].astype("category")
    codes = pdbs.cat.codes.to_numpy()
    for ax, xcol, ycol, xlab, ylab in panels:
        x = df[xcol].to_numpy(float); y = df[ycol].to_numpy(float)
        m = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[m], y[m], s=9, alpha=0.35, c=codes[m], cmap="nipy_spectral", edgecolors="none")
        r = _corr(x, y, "pearson"); rho = _corr(x, y, "spearman")
        wc = within_centered_corr(df, xcol, ycol, "pearson")
        if m.sum() > 2 and np.std(x[m]) > 0:
            b, a = np.polyfit(x[m], y[m], 1)
            xs = np.linspace(x[m].min(), x[m].max(), 50)
            ax.plot(xs, b * xs + a, "r-", lw=1.5)
        ax.set_xlabel(xlab); ax.set_ylabel(ylab)
        ax.set_title(f"pooled r={r:+.3f} rho={rho:+.3f} | within-ctr r={wc:+.3f}  (n={int(m.sum())})",
                     fontsize=10)
        ax.grid(alpha=0.2)
    fig.suptitle(f"Thread C Rung 0' RE-GATE — {dataset} (multi_cdr): INTERFACE-PH vs binding "
                 f"(color=target)\nWITHIN-target (within-centered) is what selection exploits",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rabd", choices=list(DSETS.keys()))
    ap.add_argument("--cutoff", type=float, default=INTERFACE_CUTOFF)
    ap.add_argument("--k_anchor", type=int, default=K_ANCHOR)
    ap.add_argument("--max_targets", type=int, default=0, help="0 = all")
    ap.add_argument("--n_workers", type=int, default=8)
    args = ap.parse_args()
    run(args.dataset, args.cutoff, args.k_anchor, args.max_targets, args.n_workers)
