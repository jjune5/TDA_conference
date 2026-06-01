"""Proper hetero-GNN baselines (HAN / HGT) + meta-path PDGNN-EPD feature.

Faithful to the Notion design: the backbone is a REAL heterogeneous GNN, and we test
whether adding the meta-path PDGNN-EPD topological feature (Idea 1) helps it.
  - HAN  (Wang+2019): target-only graph with meta-path edge types, semantic attention.
  - HGT  (Hu+2020):   full multi-type graph, type-aware transformer attention.
For each backbone: none / +EPD / shuffled-EPD / random-EPD (controls). EPD = meta-path
PDGNN-EPD on the target type (no exact feature; PDGNN predicts EPD). Featureless node
types get fixed random features. Metric: acc (single-label) or Macro-F1 (multilabel).
"""
from __future__ import annotations
import os, sys, json, csv, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HANConv, HGTConv
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hetero.metapath_graph import load_hgb, build_metapath_graph, METAPATHS, TARGET
from hetero import pdgnn_metapath as PM
from hetero.metapath_ph import random_filter_node_pi
from hetero.unified_filter import build_homo, calibrated_filter
from hetero.hetero_nc_pipeline import _znorm


def compute_epd(d, dataset, source, epd_mp, K, hop, max_nodes, samples_n, epochs, n_tgt, homo_cap=20000):
    """Return (EPD, EPD_rand) for target nodes. source: 'metapath'(Idea1) or 'unified'(Idea2).
    Cached to results/epd_cache/ keyed by (dataset, source, mp, K, hop) so GCN/HAN/HGT share it."""
    ck = f'results/epd_cache/{dataset}_{source}_{epd_mp}_K{K}_h{hop}.npz'
    if os.path.exists(ck):
        z = np.load(ck)
        print(f'  EPD cache hit: {ck}')
        return z['epd'], z['rnd']
    if source == 'metapath':
        g = build_metapath_graph(d, epd_mp)[0]
        hks = PM._graph_hks(g, K)
        samples = PM.gen_training_samples(g, hks, hop=hop, max_nodes=max_nodes, n_samples=samples_n, seed=0)
        print(f'  PDGNN(metapath {epd_mp}) train samples={len(samples)}')
        m = PM.train_pdgnn_metapath(samples, epochs=epochs, verbose=True)
        epd = _znorm(PM.predict_node_pi(m, g, hks, hop=hop, max_nodes=max_nodes))
        rnd = _znorm(random_filter_node_pi(g, K=K, hop=hop, max_nodes=max_nodes))
        os.makedirs('results/epd_cache', exist_ok=True); np.savez(ck, epd=epd, rnd=rnd)
        return epd, rnd
    # unified (Idea 2): whole multi-type graph, type-calibrated filter, target nodes 0..n_tgt-1
    import networkx as nx
    g_homo, ntype, n_t, _, _ = build_homo(d, dataset)
    assert n_t == n_tgt
    if g_homo.number_of_nodes() > homo_cap:
        rng = np.random.RandomState(0)
        others = list(range(n_tgt, g_homo.number_of_nodes()))
        extra = rng.choice(others, size=max(0, homo_cap - n_tgt), replace=False)
        keep = sorted(set(range(n_tgt)) | set(extra.tolist()))
        remap = {o: i for i, o in enumerate(keep)}
        g_homo = nx.relabel_nodes(g_homo.subgraph(keep).copy(), remap)
        ntype = ntype[np.array(keep)]
    calf = calibrated_filter(g_homo, ntype, K=K)
    samples = PM.gen_training_samples(g_homo, calf, hop=hop, max_nodes=max_nodes, n_samples=samples_n, seed=0)
    print(f'  PDGNN(unified) train samples={len(samples)}  homo={g_homo.number_of_nodes()}')
    m = PM.train_pdgnn_metapath(samples, epochs=epochs, verbose=True)
    epd = _znorm(PM.predict_node_pi(m, g_homo, calf, hop=hop, max_nodes=max_nodes)[:n_tgt])
    rnd = _znorm(random_filter_node_pi(g_homo, K=K, hop=hop, max_nodes=max_nodes)[:n_tgt])
    os.makedirs('results/epd_cache', exist_ok=True); np.savez(ck, epd=epd, rnd=rnd)
    return epd, rnd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RAND_DIM, HIDDEN, HEADS, DROPOUT, LR, WD, EPOCHS = 64, 128, 4, 0.5, 0.005, 1e-3, 200


def _mp_edge_index(g, n):
    ei = np.array(list(g.edges())).T if g.number_of_edges() else np.zeros((2, 0), int)
    if ei.size:
        ei = np.concatenate([ei, ei[[1, 0]]], axis=1)
    return torch.tensor(ei, dtype=torch.long, device=device)


class HANNet(nn.Module):
    """HAN over meta-path edge types of a single (target) node type."""
    def __init__(self, in_dim, n_cls, metadata):
        super().__init__()
        self.han = HANConv(in_dim, HIDDEN, metadata, heads=HEADS, dropout=DROPOUT)
        self.lin = nn.Linear(HIDDEN, n_cls)

    def forward(self, x_dict, ei_dict):
        h = self.han(x_dict, ei_dict)['T']
        h = F.dropout(F.elu(h), DROPOUT, self.training)
        return self.lin(h)


class HGTNet(nn.Module):
    """HGT over the full multi-type graph; classify the target type."""
    def __init__(self, in_dims, n_cls, metadata, target, layers=2):
        super().__init__()
        self.target = target
        self.lin_in = nn.ModuleDict({t: nn.Linear(in_dims[t], HIDDEN) for t in metadata[0]})
        self.convs = nn.ModuleList([HGTConv(HIDDEN, HIDDEN, metadata, heads=HEADS)
                                    for _ in range(layers)])
        self.lin_out = nn.Linear(HIDDEN, n_cls)

    def forward(self, x_dict, ei_dict):
        h = {t: F.relu(self.lin_in[t](x)) for t, x in x_dict.items()}
        for conv in self.convs:
            h = conv(h, ei_dict)
        return self.lin_out(h[self.target])


def evaluate(logits, y, mask, multilabel):
    from sklearn.metrics import f1_score
    if multilabel:
        p = (torch.sigmoid(logits[mask]) > 0.5).cpu().numpy().astype(int)
        return float(f1_score(y[mask].cpu().numpy().astype(int), p, average='macro', zero_division=0))
    return float((logits.argmax(1)[mask] == y[mask]).float().mean())


def run_once(model, x_in, ei_dict, y, masks, multilabel, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    yt = torch.tensor(y, dtype=torch.float if multilabel else torch.long, device=device)
    tr = torch.tensor(masks['train'], device=device)
    va = torch.tensor(masks['val'], device=device)
    te = torch.tensor(masks['test'], device=device)
    lossfn = nn.BCEWithLogitsLoss() if multilabel else None
    best_va, best_te = -1.0, 0.0
    for _ in range(EPOCHS):
        model.train(); opt.zero_grad()
        out = model(*x_in, ei_dict) if False else model(x_in[0], ei_dict)
        loss = lossfn(out[tr], yt[tr]) if multilabel else F.cross_entropy(out[tr], yt[tr])
        loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            out = model(x_in[0], ei_dict)
            vs = evaluate(out, yt, va, multilabel)
            if vs >= best_va:
                best_va = vs; best_te = evaluate(out, yt, te, multilabel)
    return best_te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='ACM')
    ap.add_argument('--epd_mp', default=None, help='meta-path for the EPD feature (default first)')
    ap.add_argument('--epd_source', choices=['metapath', 'unified'], default='metapath',
                    help='metapath=Idea1 (collapse), unified=Idea2 (whole multi-type type-calibrated)')
    ap.add_argument('--K', type=int, default=3)
    ap.add_argument('--hop', type=int, default=1)
    ap.add_argument('--max_nodes', type=int, default=200)
    ap.add_argument('--pdgnn_samples', type=int, default=300)
    ap.add_argument('--pdgnn_epochs', type=int, default=30)
    ap.add_argument('--trials', type=int, default=10)
    ap.add_argument('--backbones', nargs='+', default=['HAN', 'HGT'])
    ap.add_argument('--outdir', default=None)
    args = ap.parse_args()
    outdir = args.outdir or f'results/hanhgt_{args.dataset}'
    os.makedirs(outdir, exist_ok=True)
    d = load_hgb(args.dataset)
    tgt = TARGET[args.dataset]
    mps = list(METAPATHS[args.dataset].keys())
    epd_mp = args.epd_mp or mps[0]
    rng0 = np.random.RandomState(0)

    # labels/masks from target
    y = d[tgt].y.numpy()
    multilabel = (y.ndim > 1)
    n_cls = y.shape[1] if multilabel else int(y.max()) + 1
    masks = {k: getattr(d[tgt], f'{k}_mask').numpy()
             for k in ('train', 'val', 'test') if hasattr(d[tgt], f'{k}_mask')}
    if 'val' not in masks:
        tr = masks['train'].copy(); idx = np.where(tr)[0]; cut = idx[int(0.85*len(idx)):]
        masks['val'] = np.zeros_like(tr); masks['val'][cut] = True; masks['train'][cut] = False
    n_tgt = int(d[tgt].num_nodes)
    print(f'HAN/HGT {args.dataset}: target={tgt} n_target={n_tgt} n_cls={n_cls} '
          f'multilabel={multilabel} epd_mp={epd_mp} backbones={args.backbones}')

    # per-type features (random for featureless types)
    feats = {}
    for t in d.node_types:
        x = getattr(d[t], 'x', None)
        feats[t] = (x.numpy().astype(np.float32) if x is not None
                    else rng0.randn(int(d[t].num_nodes), RAND_DIM).astype(np.float32))

    # --- PDGNN-EPD on the target type (no exact feature). source: metapath(Idea1)/unified(Idea2) ---
    print(f'  EPD source = {args.epd_source}')
    EPD, EPD_rand = compute_epd(d, args.dataset, args.epd_source, epd_mp, args.K, args.hop,
                                args.max_nodes, args.pdgnn_samples, args.pdgnn_epochs, n_tgt)
    EPD_shuf = EPD[rng0.permutation(n_tgt)]
    print(f'  EPD {EPD.shape} distinct={len(np.unique(np.round(EPD,4),axis=0))}/{n_tgt}')

    def target_feat(variant):
        base = feats[tgt]
        if variant == 'none':
            return base
        extra = {'epd': EPD, 'shuffled': EPD_shuf, 'random': EPD_rand}[variant]
        return np.concatenate([base, extra], axis=1)

    # edge_index dicts
    ei_full = {et: d[et].edge_index.to(device) for et in d.edge_types}    # for HGT
    ei_han = {('T', mp, 'T'): _mp_edge_index(build_metapath_graph(d, mp)[0], n_tgt) for mp in mps}
    meta_han = (['T'], list(ei_han.keys()))
    meta_full = d.metadata()

    variants = ['none', 'epd', 'shuffled', 'random']
    results = {}
    for bb in args.backbones:
        for variant in variants:
            xt = target_feat(variant)
            accs = []
            for s in range(args.trials):
                if bb == 'HAN':
                    x_dict = {'T': torch.tensor(xt, dtype=torch.float32, device=device)}
                    model = HANNet(xt.shape[1], n_cls, meta_han).to(device)
                    acc = run_once(model, (x_dict,), ei_han, y, masks, multilabel, s)
                else:  # HGT
                    x_dict = {t: torch.tensor(feats[t] if t != tgt else xt,
                                              dtype=torch.float32, device=device)
                              for t in d.node_types}
                    in_dims = {t: x_dict[t].shape[1] for t in d.node_types}
                    model = HGTNet(in_dims, n_cls, meta_full, tgt).to(device)
                    acc = run_once(model, (x_dict,), ei_full, y, masks, multilabel, s)
                accs.append(acc)
            results[(bb, variant)] = accs
            print(f'  [{bb} {variant:9}] {"MacroF1" if multilabel else "acc"} = '
                  f'{np.mean(accs):.4f} ± {np.std(accs):.4f}')

    with open(os.path.join(outdir, 'nc_acc.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['dataset', 'backbone', 'variant', 'mean', 'std', 'n'])
        for (bb, v), accs in results.items():
            a = np.array(accs); w.writerow([args.dataset, bb, v, f'{a.mean():.6f}', f'{a.std():.6f}', len(a)])
    print(f'Outputs -> {outdir}/')


if __name__ == '__main__':
    main()
