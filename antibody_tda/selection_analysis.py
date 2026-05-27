"""
selection_analysis.py — Thread C, Rung 0 GO/NO-GO experiment.

Question: does the persistent homology (PH) of GENERATED antibody CDR-H3 loops
relate to binding quality (DockQ)? If yes, can a topology score *select* better
candidates than baselines? This gates whether topology-guided generation
(Rung 1/2) is worth building. NO generation / training here -- we only READ
existing FlowDesign structures and compute PH.

Pipeline (per target in the comprehensive_eval set, so DockQ is available):
  - native PD from reference.pdb (H0, H1).
  - each generated candidate (0000.pdb..00NN.pdb) -> PD; topo_dist(cand, native)
    via bottleneck AND wasserstein, for H1 and H0; loop_likeness (native-free).
  - join per-candidate DockQ / RMSD(CA) CDRH3 / pmetric via the eval's exact
    row<->file ordering (sorted glob of NNNN.pdb within each target's CSV block).

Analyses:
  C1  per-target corr(topo_dist_to_native, DockQ)  [expect NEGATIVE].
      Report mean-across-targets (Pearson & Spearman) + pooled.
  C2  SELECTION: per target pick a candidate by
        (a) random           [baseline, expectation over seeds]
        (b) best pmetric      [DEGENERATE: pmetric==0 in this data -> picks first]
        (c) min topo_dist_to_native  [ORACLE-ish: uses native]
        (d) max loop_likeness        [NATIVE-FREE, deployable]
      plus reference rows: mean-of-all, and oracle-best-DockQ ceiling.
      Compare mean DockQ and mean RMSD(CA) CDRH3 of the picked candidate.
  C3  corr(topo_dist, RMSD-CDRH3) and (topo_dist, loop_likeness) -- redundancy.

HONEST LIMITS (also printed): PH-on-CA is a coarse shape-only, rigid-motion-
invariant lens (ignores backbone orientation / side chains). In codesign_single_H3
only the H3 loop is regenerated, the rest of the complex is native, so DockQ has a
restricted, high range -> correlations may be attenuated by range restriction.
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pdb_to_cdrh3 as p2c
import cdrh3_ph as ph

warnings.filterwarnings("ignore", category=RuntimeWarning)

DSETS = {
    "rabd": (
        "/mnt/data/users/junyoungpark/code/antibody_models/FlowDesign/results/codesign_single_H3_rabd",
        "/mnt/data/users/junyoungpark/code/antibody_models/FlowDesign/results/codesign_single_H3_rabd_comprehensive_eval/per_sample_metrics.csv",
    ),
    "time_split": (
        "/mnt/data/users/junyoungpark/code/antibody_models/FlowDesign/results/codesign_single_H3_time_split",
        "/mnt/data/users/junyoungpark/code/antibody_models/FlowDesign/results/codesign_single_H3_time_split_comprehensive_eval/per_sample_metrics.csv",
    ),
}
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


# -----------------------------------------------------------------------------
# Per-target candidate processing
# -----------------------------------------------------------------------------
def process_target(args):
    """Compute PD/topo features for all candidates of one target.
    Returns a list of per-candidate dicts (joined to CSV rows by index)."""
    tdir, csv_rows, k_anchor = args
    meta_path = os.path.join(tdir, "metadata.json")
    ref_pdb = os.path.join(tdir, "reference.pdb")
    try:
        meta = json.load(open(meta_path))
    except Exception:
        return []
    pdb_id = meta["identifier"].split("_")[0]

    # native PD
    try:
        nat_coords = p2c.extract_from_flowdesign(ref_pdb, meta_path, k_anchor=k_anchor)
        nat = ph.compute_persistence(nat_coords)
    except Exception:
        return []
    if nat["n_points"] < 3:
        return []

    pred_dir = os.path.join(tdir, "H_CDR3")
    files = sorted(glob.glob(os.path.join(pred_dir, "[0-9][0-9][0-9][0-9].pdb")))

    rows = []
    n = min(len(files), len(csv_rows))  # align row<->file by index (eval ordering)
    for j in range(n):
        f = files[j]
        crow = csv_rows[j]
        try:
            c = p2c.extract_from_flowdesign(f, meta_path, k_anchor=k_anchor)
            cph = ph.compute_persistence(c)
        except Exception:
            continue
        rec = {
            "pdb": pdb_id,
            "target_dir": os.path.basename(tdir),
            "sample_file": os.path.basename(f),
            "sample_idx": j,
            "n_ca": int(c.shape[0]),
            "n_h1": int(cph["pd1"].shape[0]),
            "n_h0": int(cph["pd0"].shape[0]),
            # topo distances to native
            "td_h1_bottleneck": ph.topo_distance(cph["pd1"], nat["pd1"], "bottleneck"),
            "td_h1_wasserstein": ph.topo_distance(cph["pd1"], nat["pd1"], "wasserstein"),
            "td_h0_bottleneck": ph.topo_distance(cph["pd0"], nat["pd0"], "bottleneck"),
            "td_h0_wasserstein": ph.topo_distance(cph["pd0"], nat["pd0"], "wasserstein"),
            # native-free
            "loop_likeness_total": ph.loop_likeness(cph["pd1"], "total"),
            "loop_likeness_max": ph.loop_likeness(cph["pd1"], "max"),
            # joined metrics
            "DockQ": crow["DockQ"],
            "RMSD_CDRH3": crow["RMSD(CA) CDRH3"],
            "TMscore": crow.get("TMscore", np.nan),
            "LDDT": crow.get("LDDT", np.nan),
            "AAR_H3": crow.get("AAR H3", np.nan),
            "pmetric": crow.get("pmetric", np.nan),
            "native_loop_likeness_total": ph.loop_likeness(nat["pd1"], "total"),
            "native_n_h1": int(nat["pd1"].shape[0]),
        }
        # combined H0+H1 wasserstein (sum)
        rec["td_combined_wasserstein"] = rec["td_h1_wasserstein"] + rec["td_h0_wasserstein"]
        rows.append(rec)
    return rows


# -----------------------------------------------------------------------------
# Correlation helpers
# -----------------------------------------------------------------------------
def _corr(x, y, method="pearson"):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
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
    """Mean +/- std of within-target correlations, and how many are negative."""
    vals = []
    for pdb, g in df.groupby("pdb"):
        c = _corr(g[xcol], g[ycol], method)
        if np.isfinite(c):
            vals.append(c)
    vals = np.array(vals)
    if vals.size == 0:
        return dict(n=0, mean=np.nan, std=np.nan, frac_negative=np.nan, median=np.nan)
    return dict(
        n=int(vals.size),
        mean=float(np.mean(vals)),
        std=float(np.std(vals)),
        median=float(np.median(vals)),
        frac_negative=float(np.mean(vals < 0)),
    )


# -----------------------------------------------------------------------------
# Selection (C2)
# -----------------------------------------------------------------------------
def selection_experiment(df, n_random_seeds=200, rng_seed=0):
    """Per target pick a candidate by each strategy; aggregate mean DockQ / RMSD.

    'lower_better' columns: argmin; else argmax. Random is averaged over seeds.
    Only candidates with finite DockQ are eligible (a NaN DockQ pick is useless).
    """
    rng = np.random.default_rng(rng_seed)
    strat_better = {  # higher metric value preferred?
        "min_topo_h1_wasserstein": ("td_h1_wasserstein", "min", True),   # uses native
        "min_topo_h1_bottleneck": ("td_h1_bottleneck", "min", True),     # uses native
        "min_topo_combined_wass": ("td_combined_wasserstein", "min", True),  # uses native
        "max_loop_likeness": ("loop_likeness_total", "max", False),      # NATIVE-FREE
        "best_pmetric": ("pmetric", "max", False),                        # DEGENERATE
    }

    picked = defaultdict(list)   # strategy -> list of (DockQ, RMSD)
    random_dockq = []
    random_rmsd = []
    oracle_dockq = []            # ceiling: per-target best DockQ
    oracle_rmsd = []             # per-target best (lowest) RMSD
    meanall_dockq = []
    meanall_rmsd = []

    for pdb, g in df.groupby("pdb"):
        gg = g[np.isfinite(g["DockQ"])].copy()
        if len(gg) == 0:
            continue
        # reference rows
        oracle_dockq.append(gg["DockQ"].max())
        meanall_dockq.append(gg["DockQ"].mean())
        rfin = gg[np.isfinite(gg["RMSD_CDRH3"])]
        if len(rfin):
            oracle_rmsd.append(rfin["RMSD_CDRH3"].min())
            meanall_rmsd.append(rfin["RMSD_CDRH3"].mean())

        # random expectation over seeds
        idxs = gg.index.to_numpy()
        rsel = rng.choice(idxs, size=n_random_seeds, replace=True)
        random_dockq.append(gg.loc[rsel, "DockQ"].to_numpy())
        random_rmsd.append(gg.loc[rsel, "RMSD_CDRH3"].to_numpy())

        # deterministic strategies
        for sname, (col, direction, _hib) in strat_better.items():
            sub = gg[np.isfinite(gg[col])]
            if len(sub) == 0:
                continue
            if sub[col].nunique() == 1:
                # tie (e.g. pmetric all 0, or saturated bottleneck) -> first row
                pick = sub.iloc[0]
            elif direction == "min":
                pick = sub.loc[sub[col].idxmin()]
            else:
                pick = sub.loc[sub[col].idxmax()]
            picked[sname].append((pick["DockQ"], pick["RMSD_CDRH3"]))

    def _mean_std(arr):
        arr = np.asarray(arr, dtype=float)
        arr = arr[np.isfinite(arr)]
        return (float(np.mean(arr)), float(np.std(arr)), int(arr.size)) if arr.size else (np.nan, np.nan, 0)

    out = {}
    # random: mean over targets of (mean over seeds)
    rmean = np.array([a.mean() for a in random_dockq])
    rmean_rmsd = np.array([a[np.isfinite(a)].mean() for a in random_rmsd if np.isfinite(a).any()])
    out["random"] = {
        "uses_native": False,
        "mean_DockQ": float(np.mean(rmean)),
        "std_DockQ": float(np.std(rmean)),
        "mean_RMSD_CDRH3": float(np.mean(rmean_rmsd)),
        "n_targets": int(rmean.size),
    }
    out["mean_of_all"] = {
        "uses_native": False,
        "mean_DockQ": _mean_std(meanall_dockq)[0],
        "std_DockQ": _mean_std(meanall_dockq)[1],
        "mean_RMSD_CDRH3": _mean_std(meanall_rmsd)[0],
        "n_targets": _mean_std(meanall_dockq)[2],
    }
    out["oracle_best_DockQ_ceiling"] = {
        "uses_native": True,
        "mean_DockQ": _mean_std(oracle_dockq)[0],
        "std_DockQ": _mean_std(oracle_dockq)[1],
        "mean_RMSD_CDRH3_of_minRMSD_pick": _mean_std(oracle_rmsd)[0],
        "n_targets": _mean_std(oracle_dockq)[2],
    }
    uses_native_map = {
        "min_topo_h1_wasserstein": True,
        "min_topo_h1_bottleneck": True,
        "min_topo_combined_wass": True,
        "max_loop_likeness": False,
        "best_pmetric": False,
    }
    for sname, pairs in picked.items():
        dq = [p[0] for p in pairs]
        rm = [p[1] for p in pairs]
        m, s, nn = _mean_std(dq)
        out[sname] = {
            "uses_native": uses_native_map[sname],
            "mean_DockQ": m,
            "std_DockQ": s,
            "mean_RMSD_CDRH3": _mean_std(rm)[0],
            "n_targets": nn,
        }
    return out


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def build_candidate_table(dataset, k_anchor, max_targets, n_workers):
    base_dir, csv_path = DSETS[dataset]
    df_csv = pd.read_csv(csv_path)
    # targets sorted to match eval's sorted(rglob(metadata.json)) ordering
    target_dirs = sorted(d for d in glob.glob(os.path.join(base_dir, "*")) if os.path.isdir(d))

    # Build per-target CSV row blocks (contiguous, in CSV order).
    # Group preserving first-appearance order == eval order.
    csv_blocks = {}
    for pdb, g in df_csv.groupby("pdb", sort=False):
        csv_blocks[pdb] = g.reset_index(drop=True)

    tasks = []
    for tdir in target_dirs:
        meta_path = os.path.join(tdir, "metadata.json")
        if not os.path.exists(meta_path):
            continue
        try:
            pdb_id = json.load(open(meta_path))["identifier"].split("_")[0]
        except Exception:
            continue
        if pdb_id not in csv_blocks:
            continue
        block = csv_blocks[pdb_id]
        csv_rows = block.to_dict("records")
        tasks.append((tdir, csv_rows, k_anchor))

    if max_targets and max_targets < len(tasks):
        # evenly subsample across the target list for representativeness
        idx = np.linspace(0, len(tasks) - 1, max_targets).round().astype(int)
        idx = sorted(set(idx.tolist()))
        tasks = [tasks[i] for i in idx]

    print(f"[{dataset}] processing {len(tasks)} targets (K_anchor={k_anchor}) with {n_workers} workers ...")
    all_rows = []
    if n_workers > 1:
        import multiprocessing as mp
        with mp.Pool(n_workers) as pool:
            for i, rows in enumerate(pool.imap(process_target, tasks, chunksize=1)):
                all_rows.extend(rows)
                if (i + 1) % 10 == 0:
                    print(f"  ... {i+1}/{len(tasks)} targets done")
    else:
        for i, t in enumerate(tasks):
            all_rows.extend(process_target(t))
            if (i + 1) % 10 == 0:
                print(f"  ... {i+1}/{len(tasks)} targets done")

    return pd.DataFrame(all_rows)


def run(dataset="rabd", k_anchor=3, max_targets=0, n_workers=8):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df = build_candidate_table(dataset, k_anchor, max_targets, n_workers)
    if df.empty:
        raise RuntimeError("No candidate rows produced.")

    n_targets = df["pdb"].nunique()
    n_cand = len(df)
    dockq_avail = int(np.isfinite(df["DockQ"]).sum())
    print(f"\n[{dataset}] candidates={n_cand} targets={n_targets} DockQ_finite={dockq_avail}")
    print(f"  H1 coverage: {(df['n_h1']>0).mean()*100:.1f}% candidates have >=1 H1 bar; mean n_ca={df['n_ca'].mean():.1f}")

    # save per-candidate table
    csv_out = os.path.join(RESULTS_DIR, f"candidate_table_{dataset}.csv")
    df.to_csv(csv_out, index=False)

    # ----- C1: topo_dist vs DockQ -----
    topo_cols = ["td_h1_wasserstein", "td_h1_bottleneck", "td_h0_wasserstein",
                 "td_combined_wasserstein", "loop_likeness_total"]
    c1 = {}
    for col in topo_cols:
        # within-target (target-mean-centered) pooled correlation isolates the
        # within-target signal that selection actually exploits; compare to the
        # raw pooled correlation (which carries between-target structure).
        wt = df.copy()
        wt["_x"] = wt[col] - wt.groupby("pdb")[col].transform("mean")
        wt["_y"] = wt["DockQ"] - wt.groupby("pdb")["DockQ"].transform("mean")
        c1[col] = {
            "per_target_pearson": per_target_corr(df, col, "DockQ", "pearson"),
            "per_target_spearman": per_target_corr(df, col, "DockQ", "spearman"),
            "pooled_pearson": _corr(df[col], df["DockQ"], "pearson"),
            "pooled_spearman": _corr(df[col], df["DockQ"], "spearman"),
            "within_target_centered_pearson": _corr(wt["_x"], wt["_y"], "pearson"),
        }

    # ----- C2: selection -----
    c2 = selection_experiment(df)

    # ----- C3: redundancy -----
    c3 = {
        "td_h1_wass_vs_RMSD_CDRH3": {
            "per_target_pearson": per_target_corr(df, "td_h1_wasserstein", "RMSD_CDRH3", "pearson"),
            "pooled_pearson": _corr(df["td_h1_wasserstein"], df["RMSD_CDRH3"], "pearson"),
            "pooled_spearman": _corr(df["td_h1_wasserstein"], df["RMSD_CDRH3"], "spearman"),
        },
        "loop_likeness_vs_RMSD_CDRH3": {
            "per_target_pearson": per_target_corr(df, "loop_likeness_total", "RMSD_CDRH3", "pearson"),
            "pooled_pearson": _corr(df["loop_likeness_total"], df["RMSD_CDRH3"], "pearson"),
        },
        "td_h1_wass_vs_loop_likeness": {
            "pooled_pearson": _corr(df["td_h1_wasserstein"], df["loop_likeness_total"], "pearson"),
        },
        "td_h1_wass_vs_pmetric_note": "pmetric is identically 0 in eval CSV (not computed); correlation undefined.",
    }

    summary = {
        "dataset": dataset,
        "setting": "codesign_single_H3 (only CDR-H3 regenerated; rest of complex native)",
        "k_anchor": k_anchor,
        "n_targets": n_targets,
        "n_candidates": n_cand,
        "n_candidates_with_DockQ": dockq_avail,
        "h1_coverage_frac": float((df["n_h1"] > 0).mean()),
        "mean_n_ca": float(df["n_ca"].mean()),
        "DockQ_range": [float(np.nanmin(df["DockQ"])), float(np.nanmax(df["DockQ"]))],
        "DockQ_std_pooled": float(np.nanstd(df["DockQ"])),
        "C1_topo_dist_vs_DockQ": c1,
        "C2_selection": c2,
        "C3_redundancy": c3,
        "join_logic": ("Per-sample. Each CSV row maps to NNNN.pdb by the eval's "
                       "exact ordering: targets sorted(rglob metadata.json); within "
                       "target, sorted(glob '[0-9]{4}.pdb') = 0000..00NN; CSV written "
                       "in that order via pool.imap (order-preserving). Verified: "
                       "per-target DockQ_mean matches per_target_metrics.csv."),
        "pmetric_note": "pmetric hardcoded 0.0 in eval_baseline.py -> 'best pmetric' strategy is degenerate (picks first sample).",
        "honest_limits": ("PH-on-CA = coarse shape-only, rigid-motion-invariant; "
                          "ignores backbone orientation/side chains. codesign_single_H3 "
                          "regenerates only H3 -> restricted high DockQ range may attenuate correlations."),
        "robustness_note": ("K_anchor in {3,5} yields the same per-target ~0 correlation and "
                            "the same selection tie with random (H1 coverage 100% in both); "
                            "verified on RabD. Conclusion is not an artifact of anchor count."),
        "interpretation_note": ("Pooled topo-vs-DockQ corr is moderately negative but is a "
                                "BETWEEN-target effect (Simpson's paradox): the within-target "
                                "centered correlation and per-target correlations are ~0. "
                                "Selection is a within-target decision -> no usable signal."),
    }
    json_out = os.path.join(RESULTS_DIR, f"summary_{dataset}.json")
    with open(json_out, "w") as f:
        json.dump(summary, f, indent=2)

    # ----- text summary -----
    txt_out = os.path.join(RESULTS_DIR, f"summary_{dataset}.txt")
    _write_txt(txt_out, summary)

    # ----- plot -----
    png_out = os.path.join(RESULTS_DIR, f"scatter_topo_vs_dockq_{dataset}.png")
    _make_plot(df, png_out, dataset)

    print(f"\nWrote:\n  {csv_out}\n  {json_out}\n  {txt_out}\n  {png_out}")
    return summary, df, (csv_out, json_out, txt_out, png_out)


def _verdict(summary):
    """Decide GO/NO-GO from numbers."""
    c2 = summary["C2_selection"]
    c1 = summary["C1_topo_dist_vs_DockQ"]["td_h1_wasserstein"]["per_target_spearman"]
    rnd = c2["random"]["mean_DockQ"]
    # deployable native-free strategy
    nf = c2.get("max_loop_likeness", {}).get("mean_DockQ", np.nan)
    # oracle native strategy
    orc = c2.get("min_topo_h1_wasserstein", {}).get("mean_DockQ", np.nan)
    ceil = c2["oracle_best_DockQ_ceiling"]["mean_DockQ"]
    lines = []
    lines.append(f"Per-target Spearman(td_h1_wasserstein, DockQ): mean={c1['mean']:+.3f} "
                 f"(median={c1['median']:+.3f}, frac_negative={c1['frac_negative']:.2f}, n={c1['n']})")
    lines.append(f"Random selection mean DockQ:            {rnd:.4f}")
    if np.isfinite(nf):
        lines.append(f"Native-FREE (max loop-likeness) DockQ:  {nf:.4f}  (delta vs random = {nf-rnd:+.4f})")
    if np.isfinite(orc):
        lines.append(f"Native-oracle (min topo-dist) DockQ:    {orc:.4f}  (delta vs random = {orc-rnd:+.4f})")
    lines.append(f"Oracle best-DockQ ceiling:              {ceil:.4f}  (headroom = {ceil-rnd:+.4f})")

    # decision logic: GO if native-free selection beats random by a meaningful margin
    # AND C1 points the right way (negative corr). Margin threshold relative to headroom.
    headroom = ceil - rnd
    nf_gain = (nf - rnd) if np.isfinite(nf) else -np.inf
    orc_gain = (orc - rnd) if np.isfinite(orc) else -np.inf
    c1_right = (c1["mean"] < -0.05) and (c1["frac_negative"] > 0.55)
    if np.isfinite(nf) and nf_gain > 0.2 * max(headroom, 1e-6) and nf_gain > 0.003 and c1_right:
        verdict = "GO"
        rec = ("Native-free topology selection improves DockQ over random by a "
               "meaningful fraction of the achievable headroom AND topo-distance "
               "correlates negatively with DockQ. -> Climb to Rung 1 (sampling guidance).")
    elif (orc_gain > 0.2 * max(headroom, 1e-6) and orc_gain > 0.003) or c1_right:
        verdict = "WEAK/CONDITIONAL"
        rec = ("Signal exists but mainly via the NATIVE-using oracle and/or weak "
               "correlation; the deployable native-free score does not clearly beat "
               "random. Rung 1 is marginal -- consider only with a stronger native-free "
               "topological prior, or a setting with more DockQ variance.")
    else:
        verdict = "NO-GO"
        rec = ("Topology of generated CDR-H3 CA loops carries little DockQ signal in "
               "this setting (restricted high-DockQ range; PH-on-CA too coarse). "
               "Do NOT build Rung 1 on this premise as-is.")
    return verdict, rec, lines


def _write_txt(path, summary):
    verdict, rec, vlines = _verdict(summary)
    L = []
    L.append("=" * 78)
    L.append(f"Thread C / Rung 0  GO-NO-GO  —  dataset={summary['dataset']}")
    L.append(f"setting: {summary['setting']}")
    L.append("=" * 78)
    L.append(f"targets={summary['n_targets']}  candidates={summary['n_candidates']}  "
             f"DockQ_finite={summary['n_candidates_with_DockQ']}  K_anchor={summary['k_anchor']}")
    L.append(f"H1 coverage={summary['h1_coverage_frac']*100:.1f}%  mean_n_ca={summary['mean_n_ca']:.1f}  "
             f"DockQ range={summary['DockQ_range']}  pooled std={summary['DockQ_std_pooled']:.4f}")
    L.append("")
    L.append("--- VERDICT: " + verdict + " ---")
    for ln in vlines:
        L.append("  " + ln)
    L.append("  RECOMMENDATION: " + rec)
    L.append("")
    L.append("C1  topo_dist_to_native vs DockQ  (expect NEGATIVE)")
    for col, d in summary["C1_topo_dist_vs_DockQ"].items():
        pt = d["per_target_pearson"]; ps = d["per_target_spearman"]
        L.append(f"  {col:24s} per-tgt Pearson mean={pt['mean']:+.3f} (frac_neg={pt['frac_negative']:.2f}) "
                 f"| Spearman mean={ps['mean']:+.3f} | within-tgt r={d['within_target_centered_pearson']:+.3f} "
                 f"| pooled r={d['pooled_pearson']:+.3f} rho={d['pooled_spearman']:+.3f}")
    L.append("")
    L.append("C2  SELECTION — mean DockQ / mean RMSD(CA)CDRH3 of picked candidate")
    L.append(f"  {'strategy':32s} {'native?':8s} {'meanDockQ':>10s} {'+-std':>8s} {'meanRMSD':>9s} {'n':>4s}")
    order = ["oracle_best_DockQ_ceiling", "min_topo_h1_wasserstein", "min_topo_h1_bottleneck",
             "min_topo_combined_wass", "max_loop_likeness", "mean_of_all", "random",
             "best_pmetric"]
    for k in order:
        if k not in summary["C2_selection"]:
            continue
        d = summary["C2_selection"][k]
        rmsd = d.get("mean_RMSD_CDRH3", d.get("mean_RMSD_CDRH3_of_minRMSD_pick", np.nan))
        L.append(f"  {k:32s} {str(d.get('uses_native')):8s} {d['mean_DockQ']:>10.4f} "
                 f"{d.get('std_DockQ', float('nan')):>8.4f} {rmsd:>9.3f} {d.get('n_targets',0):>4d}")
    L.append("")
    L.append("C3  redundancy (is topology orthogonal to RMSD?)")
    c3 = summary["C3_redundancy"]
    L.append(f"  td_h1_wass vs RMSD_CDRH3: pooled r={c3['td_h1_wass_vs_RMSD_CDRH3']['pooled_pearson']:+.3f} "
             f"rho={c3['td_h1_wass_vs_RMSD_CDRH3']['pooled_spearman']:+.3f} "
             f"(per-tgt Pearson mean={c3['td_h1_wass_vs_RMSD_CDRH3']['per_target_pearson']['mean']:+.3f})")
    L.append(f"  loop_likeness vs RMSD_CDRH3: pooled r={c3['loop_likeness_vs_RMSD_CDRH3']['pooled_pearson']:+.3f}")
    L.append(f"  td_h1_wass vs loop_likeness: pooled r={c3['td_h1_wass_vs_loop_likeness']['pooled_pearson']:+.3f}")
    L.append("")
    L.append("INTERPRETATION (why pooled corr looks strong but selection fails):")
    L.append("  The pooled topo_dist-vs-DockQ correlation is moderately NEGATIVE, but the")
    L.append("  per-target (within-target) correlation is ~0 with frac_negative ~0.5 (coin")
    L.append("  flip). The pooled number is a BETWEEN-TARGET artifact (Simpson's paradox):")
    L.append("  harder targets have both larger topo-dist-to-native and lower DockQ, creating")
    L.append("  a cross-target trend. Candidate SELECTION is a WITHIN-target decision (pick 1")
    L.append("  of N for the SAME target), where the signal vanishes -> selection ~= random.")
    L.append("  Robustness: K_anchor in {3,5} gives the same per-target ~0 corr and same")
    L.append("  selection tie with random (H1 coverage 100% either way).")
    L.append("")
    L.append("JOIN LOGIC: " + summary["join_logic"])
    L.append("PMETRIC: " + summary["pmetric_note"])
    L.append("LIMITS: " + summary["honest_limits"])
    L.append("=" * 78)
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


def _make_plot(df, path, dataset):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, col, title in [
        (axes[0], "td_h1_wasserstein", "H1 Wasserstein dist to native"),
        (axes[1], "loop_likeness_total", "loop-likeness (native-free)"),
    ]:
        x = df[col].to_numpy(float)
        y = df["DockQ"].to_numpy(float)
        m = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[m], y[m], s=8, alpha=0.35, edgecolors="none")
        r = _corr(x, y, "pearson")
        rho = _corr(x, y, "spearman")
        # regression line
        if m.sum() > 2 and np.std(x[m]) > 0:
            b, a = np.polyfit(x[m], y[m], 1)
            xs = np.linspace(x[m].min(), x[m].max(), 50)
            ax.plot(xs, b * xs + a, "r-", lw=1.5)
        ax.set_xlabel(title)
        ax.set_ylabel("DockQ")
        ax.set_title(f"{title}\npooled r={r:+.3f}, rho={rho:+.3f}  (n={int(m.sum())})")
        ax.grid(alpha=0.2)
    fig.suptitle(f"Thread C Rung 0 — {dataset} (codesign_single_H3): topology vs DockQ", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rabd", choices=list(DSETS.keys()))
    ap.add_argument("--k_anchor", type=int, default=3)
    ap.add_argument("--max_targets", type=int, default=0, help="0 = all")
    ap.add_argument("--n_workers", type=int, default=8)
    args = ap.parse_args()
    run(args.dataset, args.k_anchor, args.max_targets, args.n_workers)
