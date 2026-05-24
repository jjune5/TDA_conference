# Knowledge_Distillation/sbm_plot.py
"""Read scores/sbm_sweep_summary.csv and produce 3 heatmaps:
  1. TLC-GNN AUC
  2. PDGNN AUC
  3. PI hurt magnitude = no-PI AUC − TLC-GNN AUC

Missing cells render as gray (NaN).
"""
import csv
import os
import numpy as np
import matplotlib.pyplot as plt


def load_csv(path='scores/sbm_sweep_summary.csv'):
    rows = []
    with open(path) as f:
        next(f)
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 6:
                continue
            d, h, tag, mean, std, n = parts
            rows.append((float(d), float(h), tag, float(mean), float(std), int(n)))
    return rows


def grid_by_tag(rows, tag_filter, densities, heterophilies):
    grid = np.full((len(densities), len(heterophilies)), np.nan)
    for d, h, tag, mean, _, _ in rows:
        if tag != tag_filter:
            continue
        try:
            i = densities.index(round(d, 4))
            j = heterophilies.index(round(h, 4))
            grid[i, j] = mean
        except ValueError:
            pass
    return grid


def main():
    rows = load_csv()
    densities = sorted({round(d, 4) for d, *_ in rows})
    heterophilies = sorted({round(h, 4) for _, h, *_ in rows})
    print(f'densities: {densities}')
    print(f'heterophilies: {heterophilies}')

    g_tlc = grid_by_tag(rows, 'sbmTLCGNN', densities, heterophilies)
    g_pdg = grid_by_tag(rows, 'sbmPDGNN', densities, heterophilies)
    g_no = grid_by_tag(rows, 'sbmNoPI', densities, heterophilies)
    hurt = g_no - g_tlc  # 양수 = PI 해로움

    # Symmetric color range for hurt panel
    hurt_max = np.nanmax(np.abs(hurt)) if not np.all(np.isnan(hurt)) else 0.1
    if hurt_max < 0.01:
        hurt_max = 0.1

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    cmap_bad = plt.cm.viridis.copy()
    cmap_bad.set_bad('lightgray')

    for ax, grid, title, cmap, vmin, vmax in [
        (axes[0], g_tlc, 'TLC-GNN AUC (exact PI)', cmap_bad, None, None),
        (axes[1], g_pdg, 'PDGNN AUC (approx PI)', cmap_bad, None, None),
        (axes[2], hurt, 'PI hurt magnitude\n(no-PI − TLC-GNN; +=PI hurts)',
         plt.cm.RdBu_r.copy(), -hurt_max, hurt_max),
    ]:
        cmap.set_bad('lightgray')
        masked = np.ma.masked_invalid(grid)
        im = ax.imshow(masked, origin='lower', aspect='auto', cmap=cmap,
                       vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(heterophilies)))
        ax.set_xticklabels([f'{h:.2f}' for h in heterophilies])
        ax.set_yticks(range(len(densities)))
        ax.set_yticklabels([f'{d:.2f}' for d in densities])
        ax.set_xlabel('Heterophily  (p_out / density)')
        ax.set_ylabel('Density  (p_in + p_out)')
        ax.set_title(title)
        # Cell values
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                val = grid[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                            color='white' if (cmap is cmap_bad and val < 0.6) else 'black',
                            fontsize=8)
        plt.colorbar(im, ax=ax, shrink=0.8)
    plt.suptitle('SBM density × heterophily sweep (500 nodes, 5 blocks)', y=1.02)
    plt.tight_layout()
    out = 'docs/figures/sbm_heatmap.png'
    os.makedirs('docs/figures', exist_ok=True)
    plt.savefig(out, dpi=120, bbox_inches='tight')
    print(f'saved {out}')


if __name__ == '__main__':
    main()
