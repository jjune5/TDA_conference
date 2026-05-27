# Knowledge_Distillation/heterophily_analysis.py
"""Compute node homophily for each dataset, correlate with PI hurt magnitude."""
from __future__ import annotations
import os, sys, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import networkx as nx
import loaddatas as lds


def edge_homophily(data) -> float:
    """Fraction of edges connecting same-label nodes. 1=fully homophilic, 0=fully hetero."""
    y = data.y.cpu().numpy()
    ei = np.array(data.edge_index)
    same = sum(1 for i in range(ei.shape[1]) if y[ei[0, i]] == y[ei[1, i]])
    return same / ei.shape[1]


def read_auc_mean(path):
    if not os.path.exists(path):
        return None
    aucs = []
    with open(path) as f:
        next(f, None)
        for line in f:
            parts = [p.strip() for p in line.split(',') if p.strip()]
            if len(parts) >= 3 and parts[0].isdigit():
                aucs.append(float(parts[2]))
    return float(np.mean(aucs)) if aucs else None


def main():
    # dataset → (tlcgnn_tag, nopi_tag)
    cfg = {
        'Cora':      ('homo', 'homoNoPI'),
        'Citeseer':  ('homo', 'homoNoPI'),
        'Chameleon': ('hetero', 'heteroNoPI'),
        'Squirrel':  ('hetero', 'heteroNoPI'),
        'Texas':     ('hetero', 'heteroNoPI'),
        'Cornell':   ('hetero', 'heteroNoPI'),
        'Wisconsin': ('hetero', 'heteroNoPI'),
    }
    # Photo/PubMed/Computers have no no-PI run, skip (or note homophily only)
    results = []
    for name, (tlc_tag, nopi_tag) in cfg.items():
        ds = lds.loaddatas(name)
        data = ds[0]
        h = edge_homophily(data)
        tlc = read_auc_mean(f'scores/pipe_benchmark_{name}_LP_scores{tlc_tag}.txt')
        nopi = read_auc_mean(f'scores/pipe_benchmark_{name}_LP_scores{nopi_tag}.txt')
        if tlc is not None and nopi is not None:
            hurt = nopi - tlc  # +ve = PI hurts
            results.append((name, h, tlc, nopi, hurt))
            print(f'{name:12s} homophily={h:.3f} TLC-GNN={tlc:.4f} noPI={nopi:.4f} hurt={hurt:+.4f}')

    # Pearson correlation between homophily and hurt
    if len(results) >= 3:
        hs = np.array([r[1] for r in results])
        hurts = np.array([r[4] for r in results])
        r = np.corrcoef(hs, hurts)[0, 1]
        print(f'\nPearson r(homophily, PI_hurt) = {r:.3f}')
        print('(negative r → more homophilic = less PI hurt = PI helps)')

    # Scatter plot
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, h, tlc, nopi, hurt in results:
        ax.scatter(h, hurt, s=80)
        ax.annotate(name, (h, hurt), fontsize=9, xytext=(5, 5), textcoords='offset points')
    ax.axhline(0, color='gray', ls='--', alpha=0.5)
    ax.set_xlabel('Edge homophily (fraction same-label edges)')
    ax.set_ylabel('PI hurt magnitude (no-PI − TLC-GNN AUC)')
    ax.set_title('PI helpfulness vs graph homophily')
    os.makedirs('docs/figures', exist_ok=True)
    plt.tight_layout()
    plt.savefig('docs/figures/heterophily_correlation.png', dpi=120, bbox_inches='tight')
    print('saved docs/figures/heterophily_correlation.png')


if __name__ == '__main__':
    main()
