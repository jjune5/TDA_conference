"""Heterogeneous node classification: does meta-path PH add genuine signal?

Backbone: 2-layer GCN on the meta-path homogeneous graph of target nodes, node
features = target node features. Variants compared on the SAME backbone:
  - 'none'     : GCN only (no PH)
  - 'ph'       : GCN + per-node meta-path PI concatenated to node features
  - 'shuffled' : GCN + PI rows randomly permuted (control: is PH genuine?)
  - 'random'   : GCN + PI from a RANDOM node filter (control: does structure matter?)
Multiple meta-paths -> concat their PIs. Reports test accuracy per variant.
"""
from __future__ import annotations
import os, sys, json, csv, time, argparse
import numpy as np
import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hetero.metapath_graph import load_hgb, build_metapath_graph, METAPATHS, TARGET
from hetero.metapath_ph import metapath_node_pi, random_filter_node_pi
from hetero.leakage_audit import structure_only_label_acc

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
HIDDEN, DROPOUT, LR, EPOCHS = 64, 0.5, 0.01, 200


def _graph_to_edge_index(g, n):
    ei = np.array(list(g.edges())).T if g.number_of_edges() else np.zeros((2, 0), int)
    if ei.size:
        ei = np.concatenate([ei, ei[[1, 0]]], axis=1)   # symmetric
    return torch.tensor(ei, dtype=torch.long, device=device)


def _znorm(a):
    """z-normalize each column so PH features aren't drowned out by x_feat scale."""
    mu = a.mean(0, keepdims=True); sd = a.std(0, keepdims=True)
    return (a - mu) / np.where(sd > 1e-9, sd, 1.0)


class GCNNet(torch.nn.Module):
    def __init__(self, in_dim, n_cls):
        super().__init__()
        self.c1 = GCNConv(in_dim, HIDDEN, cached=True)
        self.c2 = GCNConv(HIDDEN, n_cls, cached=True)

    def forward(self, x, ei):
        x = F.dropout(x, DROPOUT, self.training)
        x = F.relu(self.c1(x, ei))
        x = F.dropout(x, DROPOUT, self.training)
        return self.c2(x, ei)


def run_variant(x, ei, y, masks, n_cls, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = GCNNet(x.size(1), n_cls).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)
    yt = torch.tensor(y, dtype=torch.long, device=device)
    tr = torch.tensor(masks['train'], device=device)
    va = torch.tensor(masks['val'], device=device)
    te = torch.tensor(masks['test'], device=device)
    best_va, best_te = 0.0, 0.0
    for _ in range(EPOCHS):
        model.train(); opt.zero_grad()
        out = model(x, ei)
        loss = F.cross_entropy(out[tr], yt[tr])
        loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(x, ei).argmax(1)
            va_acc = float((pred[va] == yt[va]).float().mean())
            if va_acc >= best_va:
                best_va = va_acc
                best_te = float((pred[te] == yt[te]).float().mean())
    return best_te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='ACM')
    ap.add_argument('--metapaths', nargs='+', default=None,
                    help='default: all non-leaky for the dataset')
    ap.add_argument('--hop', type=int, default=1)
    ap.add_argument('--K', type=int, default=3)
    ap.add_argument('--max_nodes', type=int, default=200)
    ap.add_argument('--trials', type=int, default=10)
    ap.add_argument('--outdir', default=None)
    args = ap.parse_args()
    outdir = args.outdir or f'results/hetero_{args.dataset}'
    os.makedirs(outdir, exist_ok=True)
    d = load_hgb(args.dataset)
    mps = args.metapaths or list(METAPATHS[args.dataset].keys())
    tgt = TARGET[args.dataset]
    x_feat = d[tgt].x.numpy().astype(np.float64)
    print(f'{args.dataset} target={tgt} x={x_feat.shape} metapaths={mps}')

    # build the (shared) backbone graph from the FIRST metapath; PH from all metapaths
    g0, y, masks = build_metapath_graph(d, mps[0])
    n = g0.number_of_nodes()
    ei = _graph_to_edge_index(g0, n)
    n_cls = int(y.max()) + 1

    # leakage audit per metapath (record, don't silently use leaky ones)
    audit = {}
    pis, pis_rand = [], []
    for mp in mps:
        g, _, _ = build_metapath_graph(d, mp)
        audit[mp] = structure_only_label_acc(g, y, masks)
        print(f'  leakage audit {mp}: structure-only acc={audit[mp]:.4f}')
        t0 = time.time()
        pis.append(metapath_node_pi(g, filter='hks', K=args.K, hop=args.hop,
                                    max_nodes=args.max_nodes))
        pis_rand.append(random_filter_node_pi(g, K=args.K, hop=args.hop,
                                              max_nodes=args.max_nodes))
        print(f'    PI({mp}) computed in {time.time()-t0:.0f}s')
    PI = _znorm(np.concatenate(pis, axis=1))            # z-norm so PH isn't drowned out
    PI_rand = _znorm(np.concatenate(pis_rand, axis=1))
    rng = np.random.RandomState(0)
    PI_shuf = PI[rng.permutation(n)]                    # genuine-signal control
    print(f'  PH feature dim={PI.shape[1]}  distinct rows={len(np.unique(np.round(PI,4),axis=0))}/{n}')

    def feats(extra):
        if extra is None:
            return torch.tensor(x_feat, dtype=torch.float32, device=device)
        return torch.tensor(np.concatenate([x_feat, extra], 1),
                            dtype=torch.float32, device=device)

    variants = {'none': None, 'ph': PI, 'shuffled': PI_shuf, 'random': PI_rand}
    results = {}
    for name, extra in variants.items():
        x = feats(extra)
        accs = [run_variant(x, ei, y, masks, n_cls, s) for s in range(args.trials)]
        results[name] = accs
        print(f'  [{name:8}] test acc = {np.mean(accs):.4f} ± {np.std(accs):.4f}')

    with open(os.path.join(outdir, 'nc_acc.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['dataset', 'variant', 'mean_acc', 'std_acc', 'n'])
        for name, accs in results.items():
            a = np.array(accs)
            w.writerow([args.dataset, name, f'{a.mean():.6f}', f'{a.std():.6f}', len(a)])
    with open(os.path.join(outdir, 'audit.json'), 'w') as f:
        json.dump(audit, f, indent=2)
    print(f'Outputs -> {outdir}/')


if __name__ == '__main__':
    main()
