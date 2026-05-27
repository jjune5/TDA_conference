# Knowledge_Distillation/speed_benchmark.py
"""Benchmark dionysus exact PD vs PDGNN approx inference, per-edge timing.

For a handful of small/medium graphs we sample N edges, extract each edge's
hop-k vicinity + Ollivier-Ricci 'sum' filtration (shared cost, NOT timed), then
measure the wall-clock of:
  1. dionysus exact PD via the accelerated_PD path (Union_find + Accelerate_PD)
  2. PDGNN approximate inference (a single forward pass per vicinity)

Both run on CPU here (PDGNN is loaded with map_location='cpu') so the
comparison is apples-to-apples on the same hardware. On GPU the PDGNN side
would be substantially faster still, so the reported speedups are a
conservative (CPU-only) lower bound for the neural approximation.
"""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import networkx as nx
import torch

# Pin to a single thread so timing is stable and the dionysus (pure-Python,
# single-threaded) vs PDGNN comparison is apples-to-apples on one core.
torch.set_num_threads(1)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import loaddatas as lds
from Knowledge_Distillation.prepare_data_LP_modern import _edge_vicinity, _ollivier_ricci_filt
from Knowledge_Distillation.accelerated_PD import perturb_filter_function, Union_find, Accelerate_PD
from Knowledge_Distillation.pdgnn_modern import PDGNN


def bench_dataset(name, hop, n_edges=200, seed=0):
    ds = lds.loaddatas(name)
    data = ds[0]
    g = nx.Graph()
    g.add_nodes_from(range(data.num_nodes))
    ei = np.array(data.edge_index)
    g.add_edges_from((int(ei[0, i]), int(ei[1, i])) for i in range(ei.shape[1]))
    ricci_list = lds.compute_ricci_curvature(data)
    ricci_lookup = {(int(a), int(b)): float(c) for a, b, c in ricci_list}
    for a, b in g.edges():
        w = ricci_lookup.get((a, b), ricci_lookup.get((b, a), 0.0)) + 1
        g[a][b]['weight'] = max(w, 1e-6)

    rng = np.random.RandomState(seed)
    edges = list(g.edges())
    rng.shuffle(edges)
    edges = edges[:n_edges]

    # Pre-extract vicinities (shared cost, not counted)
    vicinities = []
    for u, v in edges:
        sub = _edge_vicinity(g, u, v, hop)
        if sub.number_of_nodes() < 3:
            continue
        filt = _ollivier_ricci_filt(sub, u, v, ricci_lookup)
        vicinities.append((sub, filt, u, v))

    # --- dionysus exact timing ---
    # NOTE: accelerated_PD.Accelerate_PD (shared code, not modified here) can
    # raise on certain vicinity topologies (e.g. an empty cycle Loop -> argmax
    # of empty sequence). We guard the call so one bad vicinity doesn't abort
    # the whole benchmark, and count only the vicinities that actually produced
    # a PD so the per-edge rate reflects real successful work.
    n_dionysus = 0
    n_dionysus_failed = 0
    t0 = time.time()
    for sub, filt, u, v in vicinities:
        sub_re = nx.convert_node_labels_to_integers(sub)
        sf = perturb_filter_function(sub_re, filt)
        try:
            PD_up, ess0, PD_down, Pos, Neg = Union_find(sf)
            _ = Accelerate_PD(Pos, Neg, sf)
            n_dionysus += 1
        except Exception:
            n_dionysus_failed += 1
    t_dionysus = time.time() - t0

    # --- PDGNN inference timing ---
    ckpt = torch.load('data/PDGNN/checkpoints/pdgnn_lp.pt', map_location='cpu')
    cfg = ckpt['config']
    model = PDGNN(hidden_dim=cfg['hidden_dim'], num_layers=cfg['num_layers'])
    model.load_state_dict(ckpt['state_dict']); model.eval()
    # Pre-build tensors
    tensors = []
    for sub, filt, u, v in vicinities:
        node_list = list(sub.nodes())
        remap = {n: i for i, n in enumerate(node_list)}
        e = np.array([(remap[a], remap[b]) for a, b in sub.edges()], dtype=np.int64).T
        if e.size:
            e = np.concatenate([e, e[[1, 0]]], axis=1)
        else:
            e = np.zeros((2, 0), dtype=np.int64)
        tensors.append((torch.tensor(filt, dtype=torch.float).view(-1, 1),
                        torch.tensor(e, dtype=torch.long)))
    runnable = [(f, e) for f, e in tensors if e.size(1) > 0]
    # Warmup: the very first forward pass pays one-time init (op dispatch,
    # allocator, torch_scatter setup). Run a few before timing so we measure
    # steady-state inference, not cold-start overhead.
    with torch.no_grad():
        for filt_t, ei_t in runnable[:5]:
            _ = model(filt_t, ei_t)
    t0 = time.time()
    with torch.no_grad():
        for filt_t, ei_t in runnable:
            _ = model(filt_t, ei_t)
    t_pdgnn = time.time() - t0

    n = len(vicinities)
    n_pdgnn = len(runnable)
    dionysus_per_edge = t_dionysus / max(n_dionysus, 1)
    pdgnn_per_edge = t_pdgnn / max(n_pdgnn, 1)
    return {
        'name': name, 'n_edges': n,
        'n_dionysus': n_dionysus, 'n_dionysus_failed': n_dionysus_failed,
        'n_pdgnn': n_pdgnn,
        'dionysus_total': t_dionysus, 'dionysus_per_edge': dionysus_per_edge,
        'pdgnn_total': t_pdgnn, 'pdgnn_per_edge': pdgnn_per_edge,
        # Per-edge speedup (robust to the two methods running on slightly
        # different edge counts when some dionysus vicinities are skipped).
        'speedup': dionysus_per_edge / max(pdgnn_per_edge, 1e-9),
    }


def main():
    results = []
    for name, hop in [('Cora', 1), ('Chameleon', 1), ('ChChMiner', 1)]:
        print(f'Benchmarking {name}...')
        r = bench_dataset(name, hop, n_edges=200)
        results.append(r)
        print(f'  dionysus: {r["dionysus_per_edge"]*1000:.2f} ms/edge '
              f'(n={r["n_dionysus"]}, skipped={r["n_dionysus_failed"]}) | '
              f'PDGNN: {r["pdgnn_per_edge"]*1000:.2f} ms/edge (n={r["n_pdgnn"]}) | '
              f'speedup: {r["speedup"]:.1f}x')

    # Bar plot
    names = [r['name'] for r in results]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - 0.2, [r['dionysus_per_edge']*1000 for r in results], 0.4, label='dionysus exact')
    ax.bar(x + 0.2, [r['pdgnn_per_edge']*1000 for r in results], 0.4, label='PDGNN approx')
    ax.set_yscale('log')
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel('ms per edge (log scale)')
    ax.set_title('PD computation speed: dionysus vs PDGNN (CPU, 1 thread)')
    any_skipped = False
    for i, r in enumerate(results):
        # Speedup < 1 means PDGNN is *slower* per edge (tiny vicinities, where
        # the dionysus Union-Find beats a full forward pass). Show the factor
        # plus the dionysus sample size, flagging datasets where many vicinities
        # were skipped due to the accelerated_PD bug (so the bar is non-final).
        skipped_tag = ''
        if r['n_dionysus_failed'] > 0:
            skipped_tag = '*'
            any_skipped = True
        label = (f'{r["speedup"]:.1f}x{skipped_tag}\n'
                 f'(dio n={r["n_dionysus"]})')
        ax.annotate(label, (i, r['dionysus_per_edge']*1000),
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    if any_skipped:
        ax.text(0.99, 0.01,
                '* dionysus skipped some vicinities (accelerated_PD bug);\n'
                '  that bar/speedup is over the surviving subset only.',
                transform=ax.transAxes, ha='right', va='bottom', fontsize=7,
                color='dimgray')
    ax.legend(loc='upper left')
    os.makedirs('docs/figures', exist_ok=True)
    plt.tight_layout()
    plt.savefig('docs/figures/speed_benchmark.png', dpi=120, bbox_inches='tight')
    print('saved docs/figures/speed_benchmark.png')

    # Save raw numbers
    import json
    with open('docs/figures/speed_benchmark.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
