"""Idea 2 driver: unified type-aware filtration EPD (PDGNN) on the whole hetero graph.

Compares, on the SAME backbone (target-type meta-path GCN) as Idea 1:
  none / ph_unified (Idea-2 whole-graph type-calibrated EPD) / shuffled / random.
The ONLY change vs Idea 1 is the topological-feature source. No exact PI as a feature
(PDGNN predicts EPD; exact only as one-time PDGNN training labels).
"""
from __future__ import annotations
import os, sys, json, csv, time, argparse
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hetero.metapath_graph import load_hgb, build_metapath_graph, METAPATHS, TARGET, subsample_connected
from hetero.unified_filter import build_homo, calibrated_filter
from hetero import pdgnn_metapath as PM
from hetero.metapath_ph import random_filter_node_pi
from hetero.hetero_nc_pipeline import GCNNet, run_variant, _znorm, _graph_to_edge_index, device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='ACM')
    ap.add_argument('--backbone_mp', default=None, help='meta-path for backbone GCN (default first)')
    ap.add_argument('--K', type=int, default=3)
    ap.add_argument('--hop', type=int, default=1)
    ap.add_argument('--max_nodes', type=int, default=150)
    ap.add_argument('--pdgnn_samples', type=int, default=300)
    ap.add_argument('--pdgnn_epochs', type=int, default=30)
    ap.add_argument('--homo_cap', type=int, default=20000, help='subsample homo graph for dense-eigh HKS')
    ap.add_argument('--trials', type=int, default=10)
    ap.add_argument('--rand_feat_dim', type=int, default=64)
    ap.add_argument('--outdir', default=None)
    args = ap.parse_args()
    outdir = args.outdir or f'results/unified_{args.dataset}'
    os.makedirs(outdir, exist_ok=True)
    d = load_hgb(args.dataset)
    tgt = TARGET[args.dataset]
    bmp = args.backbone_mp or list(METAPATHS[args.dataset].keys())[0]

    # --- backbone: target-type meta-path GCN graph (same as Idea 1) ---
    g_bb, y, masks = build_metapath_graph(d, bmp)
    n_tgt = g_bb.number_of_nodes()
    ei_bb = _graph_to_edge_index(g_bb, n_tgt)
    multilabel = (y.ndim > 1)
    n_cls = y.shape[1] if multilabel else int(y.max()) + 1
    raw_x = getattr(d[tgt], 'x', None)
    if raw_x is None:
        x_feat = np.random.RandomState(0).randn(n_tgt, args.rand_feat_dim).astype(np.float64)
    else:
        x_feat = raw_x.numpy().astype(np.float64)
    print(f'Idea2 {args.dataset}: target={tgt} n_target={n_tgt} backbone_mp={bmp} '
          f'n_cls={n_cls} multilabel={multilabel}')

    # --- Idea 2: whole multi-type homo graph + type-calibrated filter ---
    g_homo, ntype, n_target_h, y_h, masks_h = build_homo(d, args.dataset)
    assert n_target_h == n_tgt
    print(f'  homo graph: {g_homo.number_of_nodes()} nodes ({len(np.unique(ntype))} types), '
          f'{g_homo.number_of_edges()} edges')
    if g_homo.number_of_nodes() > args.homo_cap:
        # subsample homo graph but KEEP all target nodes (target are globals 0..n_tgt-1)
        import networkx as nx
        rng = np.random.RandomState(0)
        keep = set(range(n_tgt))
        others = [x for x in range(n_tgt, g_homo.number_of_nodes())]
        extra = rng.choice(others, size=max(0, args.homo_cap - n_tgt), replace=False)
        keep.update(extra.tolist())
        keep = sorted(keep)
        remap = {o: i for i, o in enumerate(keep)}
        g_homo = nx.relabel_nodes(g_homo.subgraph(keep).copy(), remap)
        ntype = ntype[np.array(keep)]
        print(f'  homo subsampled to {g_homo.number_of_nodes()} (kept all {n_tgt} target)')

    t0 = time.time()
    calf = calibrated_filter(g_homo, ntype, K=args.K)        # (Nhomo, K) type-calibrated
    print(f'  calibrated filter {calf.shape} in {time.time()-t0:.0f}s')

    # PDGNN: train on homo egos (exact EPD labels), predict per-node, take target nodes (0..n_tgt-1)
    t0 = time.time()
    samples = PM.gen_training_samples(g_homo, calf, hop=args.hop, max_nodes=args.max_nodes,
                                      n_samples=args.pdgnn_samples, seed=0)
    print(f'  PDGNN train samples={len(samples)}')
    model = PM.train_pdgnn_metapath(samples, epochs=args.pdgnn_epochs, verbose=True)
    pi_all = PM.predict_node_pi(model, g_homo, calf, hop=args.hop, max_nodes=args.max_nodes)
    pi_rand_all = random_filter_node_pi(g_homo, K=args.K, hop=args.hop, max_nodes=args.max_nodes)
    PI = _znorm(pi_all[:n_tgt])                              # target nodes only (globals 0..n_tgt-1)
    PI_rand = _znorm(pi_rand_all[:n_tgt])
    print(f'  PI(unified) {PI.shape} distinct={len(np.unique(np.round(PI,4),axis=0))}/{n_tgt} '
          f'in {time.time()-t0:.0f}s')
    rng = np.random.RandomState(0)
    PI_shuf = PI[rng.permutation(n_tgt)]

    def feats(extra):
        if extra is None:
            return torch.tensor(x_feat, dtype=torch.float32, device=device)
        return torch.tensor(np.concatenate([x_feat, extra], 1), dtype=torch.float32, device=device)

    variants = {'none': None, 'ph_unified': PI, 'shuffled': PI_shuf, 'random': PI_rand}
    results = {}
    for name, extra in variants.items():
        x = feats(extra)
        accs = [run_variant(x, ei_bb, y, masks, n_cls, s, multilabel) for s in range(args.trials)]
        results[name] = accs
        print(f'  [{name:11}] {"MacroF1" if multilabel else "acc"} = {np.mean(accs):.4f} ± {np.std(accs):.4f}')

    with open(os.path.join(outdir, 'nc_acc.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['dataset', 'variant', 'mean', 'std', 'n'])
        for name, accs in results.items():
            a = np.array(accs); w.writerow([args.dataset, name, f'{a.mean():.6f}', f'{a.std():.6f}', len(a)])
    print(f'Outputs -> {outdir}/')


if __name__ == '__main__':
    main()
