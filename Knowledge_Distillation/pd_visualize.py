# Knowledge_Distillation/pd_visualize.py
"""Visualize persistence diagrams for edges from homo / hetero / drug graphs."""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import loaddatas as lds
from Knowledge_Distillation.prepare_data_LP_modern import _edge_vicinity, _ollivier_ricci_filt
from Knowledge_Distillation.accelerated_PD import perturb_filter_function, Union_find, Accelerate_PD


def compute_pd_for_edge(g, u, v, ricci_lookup, hop):
    sub = _edge_vicinity(g, u, v, hop)
    if sub.number_of_nodes() < 3:
        return None
    filt = _ollivier_ricci_filt(sub, u, v, ricci_lookup)
    sub_re = nx.convert_node_labels_to_integers(sub)
    sf = perturb_filter_function(sub_re, filt)
    try:
        PD_up, ess0, PD_down, Pos, Neg = Union_find(sf)
        PD_one = Accelerate_PD(Pos, Neg, sf)
    except (ValueError, IndexError):
        # Degenerate vicinity (e.g. empty loop set) — skip this edge.
        return None
    pd = []
    for arr in [PD_up, ess0, PD_down, PD_one]:
        a = np.asarray(arr, dtype=np.float64).reshape(-1, 2) if len(arr) else np.empty((0, 2))
        if a.size:
            pd.append(a)
    return np.concatenate(pd, axis=0) if pd else np.empty((0, 2))


def collect_pds(name, hop, n_edges=8, seed=0):
    ds = lds.loaddatas(name)
    data = ds[0]
    g = nx.Graph()
    g.add_nodes_from(range(data.num_nodes))
    ei = np.array(data.edge_index)
    g.add_edges_from((int(ei[0, i]), int(ei[1, i])) for i in range(ei.shape[1]))
    # NOTE: full-graph Ollivier-Ricci (Sinkhorn) is prohibitively slow on dense
    # graphs (Photo 238K edges → 시간). For PD-shape visualization, unit edge
    # weights suffice — same sum-of-shortest-path filtration structure, just
    # unweighted. Skip the expensive Ricci computation.
    ricci_lookup = {}
    for a, b in g.edges():
        g[a][b]['weight'] = 1.0
    rng = np.random.RandomState(seed)
    edges = list(g.edges())
    rng.shuffle(edges)
    pds = []
    for u, v in edges:
        pd = compute_pd_for_edge(g, u, v, ricci_lookup, hop)
        if pd is not None and pd.size:
            pds.append(pd)
        if len(pds) >= n_edges:
            break
    return pds


def main():
    datasets = [('Photo', 1, 'Homophilic (Amazon)'),
                ('Chameleon', 1, 'Heterophilic (Wiki)'),
                ('ChChMiner', 1, 'Drug (DDI)')]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (name, hop, title) in zip(axes, datasets):
        print(f'Computing PDs for {name}...')
        pds = collect_pds(name, hop, n_edges=8)
        colors = plt.cm.tab10(np.linspace(0, 1, len(pds)))
        for pd, c in zip(pds, colors):
            ax.scatter(pd[:, 0], pd[:, 1], s=30, alpha=0.6, color=c)
        all_pts = np.concatenate(pds, axis=0) if pds else np.empty((0, 2))
        if all_pts.size:
            lo, hi = float(all_pts.min()), float(all_pts.max())
            ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.3)
        ax.set_xlabel('birth'); ax.set_ylabel('death')
        ax.set_title(f'{name}\n{title}\n({len(pds)} edges)')
    plt.suptitle('Persistence Diagrams across domains', y=1.02)
    os.makedirs('docs/figures', exist_ok=True)
    plt.tight_layout()
    plt.savefig('docs/figures/pd_comparison.png', dpi=120, bbox_inches='tight')
    print('saved docs/figures/pd_comparison.png')


if __name__ == '__main__':
    main()
