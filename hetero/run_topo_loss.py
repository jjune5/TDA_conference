"""Direction D: topology as a differentiable LOSS/regularizer (not an input feature).

Position-paper claim (2402.08871): "PH-based regularizers improve generalization."
Our prior work only used PH as a concatenated input feature (null after controls).
Here we test the regularizer route on heterogeneous node classification.

Backbone: the SAME 2-layer GCN as hetero/hetero_nc_pipeline.py, on the meta-path
homogeneous graph of the target nodes, node features = target features.

Regularizer (most-defensible, simplest version = differentiable 0-dim PH on the
learned node embedding, Hofer et al. "Connectivity-Optimized Representation Learning
via Persistent Homology", ICML'19):
  For each class c, take the train-node embeddings of class c as a point cloud.
  The 0-dim persistence diagram of the Vietoris-Rips filtration on that cloud has
  finite death times exactly equal to the edge weights of the Euclidean MST
  (Kruskal). Summing those death times = "total connectivity cost". Minimizing it
  pulls same-class embeddings into ONE tight connected component (death->0). This
  is differentiable: each MST edge weight is a real pairwise distance ||z_i - z_j||,
  so gradients flow to the embeddings that realize the critical (MST) edges.
  L_topo = mean over classes of (mean MST-edge length within that class).

Variants compared on the identical backbone + identical CE loss:
  - 'none'   : cross-entropy only (baseline).
  - 'topo'   : CE + lambda * L_topo grouped by the TRUE class labels.
  - 'random' : CE + lambda * L_topo grouped by a RANDOM permutation of the class
               labels (control: same loss FORM and magnitude, but the grouping is
               meaningless. If 'topo' only matches 'random', the *topology of the
               correct grouping* carries no genuine signal -- it's just a generic
               embedding-shrink regularizer.)

Honest reporting: we report test acc mean+/-std over trials for all three, plus the
mean L_topo value, so a reader can see whether topo beats BOTH no-reg and random-reg.
"""
from __future__ import annotations
import os, sys, json, csv, time, argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hetero.metapath_graph import load_hgb, build_metapath_graph, METAPATHS, TARGET

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
HIDDEN, DROPOUT, LR, EPOCHS = 64, 0.5, 0.01, 200


def _graph_to_edge_index(g, n):
    ei = np.array(list(g.edges())).T if g.number_of_edges() else np.zeros((2, 0), int)
    if ei.size:
        ei = np.concatenate([ei, ei[[1, 0]]], axis=1)   # symmetric
    return torch.tensor(ei, dtype=torch.long, device=device)


class GCNNetEmb(torch.nn.Module):
    """Same GCNNet as hetero_nc_pipeline but exposes the hidden embedding."""
    def __init__(self, in_dim, n_cls):
        super().__init__()
        self.c1 = GCNConv(in_dim, HIDDEN, cached=True)
        self.c2 = GCNConv(HIDDEN, n_cls, cached=True)

    def forward(self, x, ei):
        x = F.dropout(x, DROPOUT, self.training)
        h = F.relu(self.c1(x, ei))          # (N, HIDDEN) embedding used for topo loss
        x = F.dropout(h, DROPOUT, self.training)
        return self.c2(x, ei), h


def _mst_edge_lengths(z: torch.Tensor) -> torch.Tensor:
    """Differentiable 0-dim PH death times = Euclidean MST edge weights (Kruskal).
    z: (m, d). Returns a 1-D tensor of m-1 MST edge lengths (empty if m<2).
    The COMBINATORICS (which pairs are MST edges) are computed without grad; the
    returned lengths are the live torch distances so grad flows to z."""
    m = z.size(0)
    if m < 2:
        return z.new_zeros(0)
    # pairwise Euclidean distances (kept differentiable)
    d = torch.cdist(z, z)                                  # (m, m)
    dn = d.detach().cpu().numpy()
    # Kruskal MST on the dense distance matrix via union-find
    iu, ju = np.triu_indices(m, k=1)
    w = dn[iu, ju]
    order = np.argsort(w, kind='stable')
    parent = np.arange(m)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    sel_i, sel_j = [], []
    for idx in order:
        a, b = int(iu[idx]), int(ju[idx])
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
            sel_i.append(a); sel_j.append(b)
            if len(sel_i) == m - 1:
                break
    if not sel_i:
        return z.new_zeros(0)
    ti = torch.tensor(sel_i, device=z.device, dtype=torch.long)
    tj = torch.tensor(sel_j, device=z.device, dtype=torch.long)
    return d[ti, tj]                                       # differentiable lengths


def topo_connectivity_loss(h: torch.Tensor, groups: torch.Tensor,
                           train_mask: torch.Tensor, n_cls: int) -> torch.Tensor:
    """Mean over groups of (mean within-group MST-edge length) on train nodes.
    groups: (N,) long group id per node (true labels for 'topo', permuted for 'random')."""
    losses = []
    idx_all = torch.where(train_mask)[0]
    for c in range(n_cls):
        sel = idx_all[groups[idx_all] == c]
        if sel.numel() < 2:
            continue
        z = h[sel]
        edges = _mst_edge_lengths(z)
        if edges.numel() > 0:
            losses.append(edges.mean())
    if not losses:
        return h.new_zeros(())
    return torch.stack(losses).mean()


def run_variant(x, ei, y, masks, n_cls, seed, mode, lam):
    """mode in {'none','topo','random'}. Returns (best_test_acc, mean_topo_value)."""
    torch.manual_seed(seed); np.random.seed(seed)
    model = GCNNetEmb(x.size(1), n_cls).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)
    tr = torch.tensor(masks['train'], device=device)
    va = torch.tensor(masks['val'], device=device)
    te = torch.tensor(masks['test'], device=device)
    yt = torch.tensor(y, dtype=torch.long, device=device)

    # grouping used by the regularizer
    if mode == 'random':
        rng = np.random.RandomState(1000 + seed)
        groups = torch.tensor(rng.permutation(y.astype(np.int64)), device=device)
    else:
        groups = yt

    def acc(out, mask):
        return float((out.argmax(1)[mask] == yt[mask]).float().mean())

    best_va, best_te, topo_log = -1.0, 0.0, []
    for _ in range(EPOCHS):
        model.train(); opt.zero_grad()
        out, h = model(x, ei)
        ce = F.cross_entropy(out[tr], yt[tr])
        if mode == 'none' or lam == 0.0:
            loss = ce
        else:
            tl = topo_connectivity_loss(h, groups, tr, n_cls)
            topo_log.append(float(tl.detach()))
            loss = ce + lam * tl
        loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            out, _ = model(x, ei)
            va_s = acc(out, va)
            if va_s >= best_va:
                best_va = va_s
                best_te = acc(out, te)
    return best_te, (float(np.mean(topo_log)) if topo_log else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='ACM')
    ap.add_argument('--metapath', default=None, help='default: first non-leaky metapath')
    ap.add_argument('--lam', type=float, default=0.1, help='topo regularizer weight')
    ap.add_argument('--trials', type=int, default=3)
    ap.add_argument('--outdir', default=None)
    args = ap.parse_args()
    outdir = args.outdir or f'results/topo_loss_{args.dataset}'
    os.makedirs(outdir, exist_ok=True)

    d = load_hgb(args.dataset)
    mp = args.metapath or list(METAPATHS[args.dataset].keys())[0]
    tgt = TARGET[args.dataset]
    g, y, masks = build_metapath_graph(d, mp)
    if y.ndim > 1:
        raise SystemExit('topo_connectivity_loss assumes single-label classes; '
                         f'{args.dataset} is multilabel. Use ACM/DBLP/Freebase.')
    n = g.number_of_nodes()
    ei = _graph_to_edge_index(g, n)
    n_cls = int(y.max()) + 1
    raw_x = getattr(d[tgt], 'x', None)
    if raw_x is None:
        rng0 = np.random.RandomState(0)
        x_feat = rng0.randn(n, 64).astype(np.float32)
    else:
        x_feat = raw_x.numpy().astype(np.float32)
    x = torch.tensor(x_feat, device=device)
    print(f'{args.dataset} metapath={mp} target={tgt} x={x_feat.shape} n_cls={n_cls} '
          f'edges={g.number_of_edges()} lam={args.lam}')
    print(f'  train/val/test = {masks["train"].sum()}/{masks["val"].sum()}/{masks["test"].sum()}')

    results, topo_vals = {}, {}
    for mode in ['none', 'topo', 'random']:
        lam = 0.0 if mode == 'none' else args.lam
        t0 = time.time()
        out = [run_variant(x, ei, y, masks, n_cls, s, mode, lam) for s in range(args.trials)]
        accs = [o[0] for o in out]; tvs = [o[1] for o in out]
        results[mode] = accs; topo_vals[mode] = tvs
        print(f'  [{mode:7}] test acc = {np.mean(accs):.4f} +/- {np.std(accs):.4f}  '
              f'(topoL~{np.mean(tvs):.4f}, {time.time()-t0:.0f}s)')

    with open(os.path.join(outdir, 'topo_loss_acc.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['dataset', 'metapath', 'lam', 'variant', 'mean_acc', 'std_acc',
                    'mean_topoL', 'n'])
        for mode, accs in results.items():
            a = np.array(accs)
            w.writerow([args.dataset, mp, args.lam, mode, f'{a.mean():.6f}',
                        f'{a.std():.6f}', f'{np.mean(topo_vals[mode]):.6f}', len(a)])
    with open(os.path.join(outdir, 'topo_loss_raw.json'), 'w') as f:
        json.dump({'acc': results, 'topoL': topo_vals, 'lam': args.lam,
                   'metapath': mp, 'trials': args.trials}, f, indent=2)
    print(f'Outputs -> {outdir}/')


if __name__ == '__main__':
    main()
