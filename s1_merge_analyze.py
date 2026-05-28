"""s1_merge_analyze.py — Merge per-dataset SLURM job outputs for S1 (Solidify N1).

Reads raw_trials.csv from per-dataset subdirs under --root, merges, then
produces:
    results/noise_robust_solid/
        all_raw_trials.csv
        auc_vs_p.csv         (mean/std/n per dataset x variant x p)
        slopes.csv            (linear slope + seed-level std across graph_seeds)
        chameleon_crossover.png  (error-band plot for Chameleon PI vs no-PI)
        auc_vs_p.png          (multi-panel, all datasets)
        summary.txt

Usage:
    python s1_merge_analyze.py [--root results/noise_robust_solid]
"""

import os
import sys
import csv
import argparse
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def load_raw(root, datasets):
    """Load and merge all per-dataset raw_trials.csv files."""
    rows = []
    for d in datasets:
        path = os.path.join(root, d, 'raw_trials.csv')
        if not os.path.exists(path):
            print(f"[warn] {path} not found — skipping {d}", flush=True)
            continue
        with open(path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append({
                    'dataset': row['dataset'],
                    'variant': row['variant'],
                    'p': float(row['p']),
                    'graph_seed': int(row['graph_seed']),
                    'init_seed': int(row['init_seed']),
                    'auc': float(row['auc']),
                })
    return rows


def agg_by_cell(rows, datasets, variants, ps):
    """Aggregate: (dataset, variant, p) -> list of per-seed mean AUCs, then overall mean/std."""
    # Per-seed means: seed_means[(d,v,p,gseed)] = mean over init_seeds
    seed_means = {}
    for r in rows:
        key = (r['dataset'], r['variant'], r['p'], r['graph_seed'])
        seed_means.setdefault(key, []).append(r['auc'])

    # Overall cell stats
    cell_stats = {}  # (d,v,p) -> {'mean', 'std_across_seeds', 'n', 'seed_vals'}
    for d in datasets:
        for v in variants:
            for p in ps:
                # collect per-seed means
                n_seeds = 0
                per_seed = []
                gseed = 0
                while True:
                    key = (d, v, p, gseed)
                    if key not in seed_means:
                        break
                    per_seed.append(float(np.mean(seed_means[key])))
                    n_seeds += 1
                    gseed += 1
                if not per_seed:
                    continue
                all_aucs = [r['auc'] for r in rows
                            if r['dataset'] == d and r['variant'] == v and r['p'] == p]
                cell_stats[(d, v, p)] = {
                    'mean': float(np.mean(all_aucs)),
                    'std': float(np.std(all_aucs)),
                    'n': len(all_aucs),
                    'seed_vals': per_seed,           # per-seed mean AUC
                    'seed_std': float(np.std(per_seed)),  # variability across seeds
                    'n_seeds': n_seeds,
                }
    return cell_stats


def compute_slopes(cell_stats, datasets, variants, ps):
    """Linear slope (AUC per +10% noise) per (dataset, variant), with std across seeds."""
    slopes = {}
    for d in datasets:
        for v in variants:
            pts = [(p, cell_stats[(d, v, p)]['mean'])
                   for p in ps if (d, v, p) in cell_stats]
            if len(pts) < 2:
                continue
            xs = np.array([pp for pp, _ in pts])
            ys = np.array([aa for _, aa in pts])
            A = np.vstack([xs, np.ones_like(xs)]).T
            coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
            slope10 = coef[0] * 10.0  # per 10%
            pred = A @ coef
            ss_res = float(np.sum((ys - pred) ** 2))
            ss_tot = float(np.sum((ys - ys.mean()) ** 2))
            r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else float('nan')

            # Per-seed slopes for spread
            n_seeds = max(cell_stats[(d, v, p)]['n_seeds']
                          for p in ps if (d, v, p) in cell_stats)
            per_seed_slopes = []
            for gseed in range(n_seeds):
                pts_s = [(p, cell_stats[(d, v, p)]['seed_vals'][gseed])
                         for p in ps
                         if (d, v, p) in cell_stats
                         and gseed < len(cell_stats[(d, v, p)]['seed_vals'])]
                if len(pts_s) < 2:
                    continue
                xs_s = np.array([pp for pp, _ in pts_s])
                ys_s = np.array([aa for _, aa in pts_s])
                A_s = np.vstack([xs_s, np.ones_like(xs_s)]).T
                c_s, *_ = np.linalg.lstsq(A_s, ys_s, rcond=None)
                per_seed_slopes.append(c_s[0] * 10.0)

            slope_std = float(np.std(per_seed_slopes)) if per_seed_slopes else float('nan')
            slopes[(d, v)] = {
                'slope10': slope10,
                'slope_std': slope_std,
                'r2': r2,
                'auc_at_0': cell_stats[(d, v, 0.0)]['mean'] if (d, v, 0.0) in cell_stats else float('nan'),
                'auc_at_max': ys[-1],
                'per_seed_slopes': per_seed_slopes,
            }
    return slopes


def write_auc_csv(cell_stats, datasets, variants, ps, out_dir):
    path = os.path.join(out_dir, 'auc_vs_p.csv')
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['dataset', 'variant', 'p', 'auc_mean', 'auc_std', 'n',
                    'n_seeds', 'seed_std'])
        for d in datasets:
            for v in variants:
                for p in ps:
                    if (d, v, p) not in cell_stats:
                        continue
                    st = cell_stats[(d, v, p)]
                    w.writerow([d, v, p,
                                 f'{st["mean"]:.4f}', f'{st["std"]:.4f}', st['n'],
                                 st['n_seeds'], f'{st["seed_std"]:.4f}'])
    print(f"[write] {path}", flush=True)


def write_slopes_csv(slopes, datasets, variants, out_dir):
    path = os.path.join(out_dir, 'slopes.csv')
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['dataset', 'variant', 'slope_per_10pct', 'slope_std_across_seeds',
                    'auc_at_0', 'auc_at_max_p', 'r2'])
        for d in datasets:
            for v in variants:
                if (d, v) not in slopes:
                    continue
                sl = slopes[(d, v)]
                w.writerow([d, v,
                             f'{sl["slope10"]:.5f}',
                             f'{sl["slope_std"]:.5f}',
                             f'{sl["auc_at_0"]:.4f}',
                             f'{sl["auc_at_max"]:.4f}',
                             f'{sl["r2"]:.3f}'])
    print(f"[write] {path}", flush=True)


def plot_auc_vs_p(cell_stats, slopes, datasets, variants, ps, out_dir):
    if not HAS_MPL:
        print("[plot] matplotlib not available", flush=True)
        return

    colors = {'PI': 'tab:blue', 'no-PI': 'tab:gray', 'GDC-PI': 'tab:red'}
    markers = {'PI': 'o', 'no-PI': 's', 'GDC-PI': '^'}
    nd = len(datasets)
    fig, axes = plt.subplots(1, nd, figsize=(6 * nd, 5), squeeze=False)

    for ax, d in zip(axes[0], datasets):
        for v in variants:
            pts = [(p, cell_stats[(d, v, p)]['mean'], cell_stats[(d, v, p)]['std'])
                   for p in ps if (d, v, p) in cell_stats]
            if not pts:
                continue
            xs = [pp for pp, _, _ in pts]
            ys = [mm for _, mm, _ in pts]
            es = [ss for _, _, ss in pts]
            sl = slopes.get((d, v), {})
            sl10 = sl.get('slope10', float('nan'))
            sl_std = sl.get('slope_std', float('nan'))
            label = (f'{v} (slope {sl10:+.4f}±{sl_std:.4f}/10%)' if not np.isnan(sl_std)
                     else f'{v} (slope {sl10:+.4f}/10%)')
            ax.errorbar(xs, ys, yerr=es, marker=markers.get(v, 'o'),
                        color=colors.get(v), capsize=4, linewidth=2,
                        label=label)
        ax.set_title(f'{d}: LP AUC vs structural noise\n(error bars = ±1σ over all trials)')
        ax.set_xlabel('edges removed & re-added as noise (%)')
        ax.set_ylabel('test AUC (ROC)')
        ax.grid(alpha=0.3)
        ax.legend(loc='best', fontsize=8)

    fig.tight_layout()
    path = os.path.join(out_dir, 'auc_vs_p.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[plot] {path}", flush=True)


def plot_chameleon_crossover(cell_stats, slopes, ps, out_dir):
    """Dedicated Chameleon PI vs no-PI crossover plot with per-seed error bands."""
    if not HAS_MPL:
        return
    d = 'Chameleon'
    vs = ['PI', 'no-PI', 'GDC-PI']
    colors = {'PI': 'tab:blue', 'no-PI': 'tab:gray', 'GDC-PI': 'tab:red'}
    markers = {'PI': 'o', 'no-PI': 's', 'GDC-PI': '^'}

    fig, ax = plt.subplots(figsize=(7, 5))

    for v in vs:
        pts = [(p, cell_stats[(d, v, p)]) for p in ps if (d, v, p) in cell_stats]
        if not pts:
            continue
        xs = np.array([pp for pp, _ in pts])
        ys = np.array([st['mean'] for _, st in pts])
        # Use seed_std as the per-seed spread (CI across perturbation realizations)
        seed_stds = np.array([st['seed_std'] for _, st in pts])
        # Also show total std
        total_stds = np.array([st['std'] for _, st in pts])
        ns = np.array([st['n'] for _, st in pts])
        se = total_stds / np.sqrt(ns)  # standard error

        sl = slopes.get((d, v), {})
        sl10 = sl.get('slope10', float('nan'))
        sl_std = sl.get('slope_std', float('nan'))
        n_seeds = pts[0][1]['n_seeds'] if pts else 0

        # Error band = ±seed_std across perturbation seeds (tightest CI meaningful here)
        ax.fill_between(xs, ys - seed_stds, ys + seed_stds,
                        color=colors.get(v), alpha=0.15)
        ax.errorbar(xs, ys, yerr=se, marker=markers.get(v, 'o'),
                    color=colors.get(v), capsize=5, linewidth=2,
                    label=f'{v} (slope {sl10:+.4f}±{sl_std:.4f}/10%, n_seeds={n_seeds})')

    # mark crossover region if PI > no-PI
    pi_pts = {p: cell_stats.get(('Chameleon', 'PI', p), {}).get('mean', float('nan'))
               for p in ps}
    nopi_pts = {p: cell_stats.get(('Chameleon', 'no-PI', p), {}).get('mean', float('nan'))
                for p in ps}
    cross_ps = [p for p in ps if pi_pts[p] > nopi_pts[p] and not np.isnan(pi_pts[p])]
    if cross_ps:
        ax.axvspan(min(cross_ps) - 2.5, max(cross_ps) + 2.5,
                   alpha=0.07, color='tab:blue', label='PI > no-PI region')

    ax.set_title('Chameleon: PI vs no-PI crossover under edge noise\n'
                 '(shaded band = ±1σ across perturbation seeds; error bars = ±SE)')
    ax.set_xlabel('Structural noise level (%)')
    ax.set_ylabel('Test AUC (ROC)')
    ax.grid(alpha=0.3)
    ax.legend(loc='lower right', fontsize=9)
    fig.tight_layout()
    path = os.path.join(out_dir, 'chameleon_crossover.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[plot] {path}", flush=True)


def write_summary(cell_stats, slopes, datasets, variants, ps, out_dir, n_seeds_target):
    lines = []
    lines.append("Experiment S1 — Solidify N1: topology robustness to graph structural noise")
    lines.append("=" * 70)
    lines.append(f"datasets={datasets}  ps={ps}  variants={variants}")
    lines.append("")
    lines.append("AUC vs Noise Level (mean ± std [n trials | n_seeds]):")
    lines.append("-" * 70)
    for d in datasets:
        lines.append(f"\n{d}:")
        for v in variants:
            parts = []
            for p in ps:
                if (d, v, p) not in cell_stats:
                    parts.append(f"p={p}%: N/A")
                    continue
                st = cell_stats[(d, v, p)]
                parts.append(f"p={int(p)}%: {st['mean']:.4f}±{st['std']:.4f}"
                              f"[n={st['n']}|s={st['n_seeds']}]")
            lines.append(f"  {v:8s}: " + "  ".join(parts))

    lines.append("")
    lines.append("Degradation slope (Δ AUC per +10% noise; slope_std = std across perturbation seeds):")
    lines.append("-" * 70)
    for d in datasets:
        lines.append(f"\n{d}:")
        for v in variants:
            if (d, v) not in slopes:
                continue
            sl = slopes[(d, v)]
            lines.append(f"  {v:8s}: slope={sl['slope10']:+.5f} "
                         f"(±{sl['slope_std']:.5f} across seeds, R²={sl['r2']:.3f})")
        # verdict
        if (d, 'no-PI') in slopes:
            base_sl = slopes[(d, 'no-PI')]['slope10']
            base_std = slopes[(d, 'no-PI')]['slope_std']
            for v in ['PI', 'GDC-PI']:
                if (d, v) not in slopes:
                    continue
                sl_v = slopes[(d, v)]['slope10']
                sl_std_v = slopes[(d, v)]['slope_std']
                diff = sl_v - base_sl
                # conservative test: is the slope difference larger than 1 combined std?
                combined_std = (base_std**2 + sl_std_v**2) ** 0.5
                robust = (diff > 0)
                sig = (abs(diff) > combined_std) if combined_std > 0 else False
                status = "ROBUST" if (robust and sig) else ("POSITIVE" if robust else "NULL")
                lines.append(f"  -> {v} vs no-PI: {status} "
                              f"(Δslope={diff:+.5f}, combined_std={combined_std:.5f}, "
                              f"{'|Δ|>1σ' if sig else '|Δ|<1σ'})")

    lines.append("")
    # Chameleon crossover check
    if 'Chameleon' in datasets:
        lines.append("Chameleon crossover (PI AUC > no-PI AUC):")
        d = 'Chameleon'
        cross_ps = []
        for p in ps:
            pi_m = cell_stats.get((d, 'PI', p), {}).get('mean', float('nan'))
            nopi_m = cell_stats.get((d, 'no-PI', p), {}).get('mean', float('nan'))
            if not np.isnan(pi_m) and not np.isnan(nopi_m) and pi_m > nopi_m:
                diff = pi_m - nopi_m
                # Check if within combined error
                pi_std = cell_stats.get((d, 'PI', p), {}).get('seed_std', 0)
                nopi_std = cell_stats.get((d, 'no-PI', p), {}).get('seed_std', 0)
                cross_ps.append((p, diff, pi_std, nopi_std))
                lines.append(f"  p={int(p)}%: PI={pi_m:.4f} > no-PI={nopi_m:.4f} "
                              f"(Δ={diff:.4f}, PI_seed_std={pi_std:.4f}, "
                              f"no-PI_seed_std={nopi_std:.4f})")
            elif not np.isnan(pi_m) and not np.isnan(nopi_m):
                lines.append(f"  p={int(p)}%: PI={pi_m:.4f} ≤ no-PI={nopi_m:.4f} "
                              f"(no crossover)")
        if cross_ps:
            # Is the crossover robust (Δ > combined seed_std)?
            robust_cross = [(p, d, ps, ns) for p, d, ps, ns in cross_ps
                            if d > (ps + ns)]
            if robust_cross:
                lines.append(f"  Crossover is STATISTICALLY ROBUST (Δ > combined σ) at p={[p for p,*_ in robust_cross]}%")
            else:
                lines.append(f"  Crossover is WEAK (Δ < combined σ) — visible but not significant")
        else:
            lines.append("  No crossover detected at any p level.")

    lines.append("")
    # Overall verdict
    pro = []
    for d in datasets:
        if (d, 'no-PI') not in slopes:
            continue
        for v in ['PI', 'GDC-PI']:
            if (d, v) in slopes:
                pro.append(slopes[(d, v)]['slope10'] > slopes[(d, 'no-PI')]['slope10'])

    lines.append("OVERALL VERDICT:")
    if pro and all(pro):
        lines.append("  PRO-TOPOLOGY: PI/GDC-PI degrade SLOWER than no-PI across all "
                     "tested (dataset, variant) pairs. N1 finding solidified.")
    elif pro and any(pro):
        lines.append("  MIXED: Topology is more robust in some pairs but not all. "
                     "N1 partially solidified.")
    elif pro:
        lines.append("  NULL: Topology does NOT consistently degrade slower than no-PI. "
                     "N1 not supported with more seeds.")
    else:
        lines.append("  INSUFFICIENT DATA: Cannot determine verdict yet.")

    txt = "\n".join(lines)
    path = os.path.join(out_dir, 'summary.txt')
    with open(path, 'w') as f:
        f.write(txt + "\n")
    print(f"[write] {path}", flush=True)
    print("\n" + txt, flush=True)
    return txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='results/noise_robust_solid')
    ap.add_argument('--datasets', nargs='+', default=['Chameleon', 'Texas', 'Cornell'])
    ap.add_argument('--variants', nargs='+', default=['PI', 'no-PI', 'GDC-PI'])
    ap.add_argument('--ps', nargs='+', type=float, default=[0, 5, 10, 20])
    ap.add_argument('--n_seeds', type=int, default=3,
                    help='target graph_seeds (for reporting)')
    args = ap.parse_args()

    out_dir = args.root
    os.makedirs(out_dir, exist_ok=True)

    # Load
    rows = load_raw(args.root, args.datasets)
    if not rows:
        print("[ERROR] No raw_trials.csv found — did SLURM jobs finish?", flush=True)
        sys.exit(1)

    # Save merged raw
    merged_path = os.path.join(out_dir, 'all_raw_trials.csv')
    with open(merged_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['dataset', 'variant', 'p', 'graph_seed', 'init_seed', 'auc'])
        for r in rows:
            w.writerow([r['dataset'], r['variant'], r['p'],
                        r['graph_seed'], r['init_seed'], r['auc']])
    print(f"[write] {merged_path} ({len(rows)} rows)", flush=True)

    # Aggregate
    datasets_found = sorted(set(r['dataset'] for r in rows))
    ps = sorted(set(r['p'] for r in rows))

    cell_stats = agg_by_cell(rows, datasets_found, args.variants, ps)
    slopes = compute_slopes(cell_stats, datasets_found, args.variants, ps)

    # Write outputs
    write_auc_csv(cell_stats, datasets_found, args.variants, ps, out_dir)
    write_slopes_csv(slopes, datasets_found, args.variants, out_dir)
    plot_auc_vs_p(cell_stats, slopes, datasets_found, args.variants, ps, out_dir)
    if 'Chameleon' in datasets_found:
        plot_chameleon_crossover(cell_stats, slopes, ps, out_dir)
    write_summary(cell_stats, slopes, datasets_found, args.variants, ps, out_dir,
                  n_seeds_target=args.n_seeds)


if __name__ == '__main__':
    main()
