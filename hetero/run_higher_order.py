"""Direction A: architecture-level TDL = higher-order message passing.

Our PH-feature-concat results were null (§14). The ICML2024 position paper
(2402.08871) argues that the *modern* form of TDL is to put topology into the
*architecture* -- higher-order message passing on simplicial/cell/hyper graphs --
not to concat a PH vector to the input. This script tests that last claim on ACM
node classification:

    (i)  GCN          : pairwise message passing on the meta-path (PAP+PSP) graph
                        = clique expansion of the hyperedges (the "old" pairwise view)
    (ii) HypergraphConv: higher-order message passing directly on the hypergraph
                        (each author -> one hyperedge over the papers they wrote;
                         each subject -> one hyperedge over its papers)

Same node features (paper features), same train/val/test split, same hidden dim,
same training loop. The ONLY difference is pairwise-vs-higher-order aggregation, so
the comparison isolates the architectural-topology effect.

Honest controls (project rule -- shuffled/random):
  - HyperConv-shuffled : papers are randomly reassigned to hyperedges, preserving
        the hyperedge-size distribution. If genuine co-authorship/subject grouping
        carries signal, real >> shuffled. If the *size* of the aggregation is all
        that matters, real ~ shuffled.
  - GCN-random         : pairwise graph with the same #edges rewired at random.

Reports test acc per (model, variant) over N trials, mean +- std, to CSV.
"""
from __future__ import annotations
import os, sys, csv, time, argparse
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, HypergraphConv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hetero.metapath_graph import load_hgb, build_metapath_graph, METAPATHS, TARGET

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
HIDDEN, DROPOUT, LR, WD, EPOCHS = 64, 0.5, 0.01, 5e-4, 200


# --------------------------------------------------------------------------- #
#  Hypergraph lift: each non-target node (author / subject) -> one hyperedge   #
#  over the target (paper) nodes it connects to.                               #
# --------------------------------------------------------------------------- #
def build_hyperedges(d, dataset, relations):
    """Return (hyperedge_index, num_hyperedges, sizes).

    relations: list of edge types ('paper','to','author') etc., each ('paper', _, X).
    For each such relation, every node of type X with >=2 incident papers becomes
    one hyperedge spanning those papers. hyperedge_index is the PyG bipartite
    incidence: row0 = paper node id, row1 = hyperedge id (contiguous).
    """
    tgt = TARGET[dataset]
    pap_ids, he_ids, sizes = [], [], []
    he_counter = 0
    for et in relations:
        src, _, dst = et
        assert src == tgt, f'relation {et} must start at target {tgt}'
        ei = d[et].edge_index.numpy()             # (2, E): row0 paper, row1 X
        n_dst = int(d[dst].num_nodes)
        # group papers by dst node
        order = np.argsort(ei[1])
        papers_sorted = ei[0][order]
        dst_sorted = ei[1][order]
        boundaries = np.where(np.diff(dst_sorted) != 0)[0] + 1
        groups = np.split(papers_sorted, boundaries)
        for grp in groups:
            grp = np.unique(grp)
            if grp.size < 2:                      # singleton hyperedge = no higher-order info
                continue
            pap_ids.append(grp)
            he_ids.append(np.full(grp.size, he_counter))
            sizes.append(grp.size)
            he_counter += 1
    if not pap_ids:
        raise RuntimeError('no hyperedges built')
    H = np.stack([np.concatenate(pap_ids), np.concatenate(he_ids)])
    return H.astype(np.int64), he_counter, np.array(sizes)


def shuffle_hyperedges(H, num_he, n_nodes, seed=0):
    """Control: keep the hyperedge-size distribution but assign random papers to
    each hyperedge (sampled without replacement within a hyperedge). Destroys the
    genuine co-authorship/subject grouping while matching aggregation 'shape'."""
    rng = np.random.RandomState(seed)
    sizes = np.bincount(H[1], minlength=num_he)
    pap_ids, he_ids = [], []
    for he in range(num_he):
        k = int(sizes[he])
        if k == 0:
            continue
        members = rng.choice(n_nodes, size=k, replace=False)
        pap_ids.append(members)
        he_ids.append(np.full(k, he))
    return np.stack([np.concatenate(pap_ids), np.concatenate(he_ids)]).astype(np.int64)


# --------------------------------------------------------------------------- #
#  Pairwise meta-path graph (clique expansion) -> edge_index                   #
# --------------------------------------------------------------------------- #
def metapath_edge_index(d, dataset, mps):
    """Union of meta-path graphs (PAP, PSP) as an undirected pairwise edge_index.
    This is the clique expansion of the same author/subject groups, i.e. the
    'pairwise' view of the identical relational information."""
    n = int(d[TARGET[dataset]].num_nodes)
    A = sp.csr_matrix((n, n))
    for mp in mps:
        g = build_metapath_graph(d, mp)[0]
        ei = np.array(list(g.edges())).T if g.number_of_edges() else np.zeros((2, 0), int)
        if ei.size:
            data = np.ones(ei.shape[1])
            A = A + sp.csr_matrix((data, (ei[0], ei[1])), shape=(n, n))
    A = (A + A.T).tocoo()
    mask = A.row != A.col
    ei = np.stack([A.row[mask], A.col[mask]]).astype(np.int64)
    return torch.tensor(ei, dtype=torch.long, device=device), ei.shape[1]


def random_edge_index(n, n_edges, seed=0):
    """Control: same #edges, random pairs (symmetrized)."""
    rng = np.random.RandomState(seed)
    half = n_edges // 2
    src = rng.randint(0, n, half)
    dst = rng.randint(0, n, half)
    keep = src != dst
    src, dst = src[keep], dst[keep]
    ei = np.stack([np.concatenate([src, dst]), np.concatenate([dst, src])]).astype(np.int64)
    return torch.tensor(ei, dtype=torch.long, device=device)


# --------------------------------------------------------------------------- #
#  Models                                                                      #
# --------------------------------------------------------------------------- #
class GCNNet(nn.Module):
    def __init__(self, in_dim, n_cls):
        super().__init__()
        self.c1 = GCNConv(in_dim, HIDDEN)
        self.c2 = GCNConv(HIDDEN, n_cls)

    def forward(self, x, ei):
        x = F.dropout(x, DROPOUT, self.training)
        x = F.relu(self.c1(x, ei))
        x = F.dropout(x, DROPOUT, self.training)
        return self.c2(x, ei)


class HyperNet(nn.Module):
    """2-layer hypergraph conv (Feng+2019 HGNN-style, via PyG HypergraphConv)."""
    def __init__(self, in_dim, n_cls):
        super().__init__()
        self.c1 = HypergraphConv(in_dim, HIDDEN)
        self.c2 = HypergraphConv(HIDDEN, n_cls)

    def forward(self, x, H):
        x = F.dropout(x, DROPOUT, self.training)
        x = F.relu(self.c1(x, H))
        x = F.dropout(x, DROPOUT, self.training)
        return self.c2(x, H)


# --------------------------------------------------------------------------- #
#  Train / eval                                                                #
# --------------------------------------------------------------------------- #
def run_once(model_fn, conn, x, y, masks, n_cls, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = model_fn(x.size(1), n_cls).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    yt = torch.tensor(y, dtype=torch.long, device=device)
    tr = torch.tensor(masks['train'], device=device)
    va = torch.tensor(masks['val'], device=device)
    te = torch.tensor(masks['test'], device=device)

    def score(out, m):
        return float((out.argmax(1)[m] == yt[m]).float().mean())

    best_va, best_te = -1.0, 0.0
    for _ in range(EPOCHS):
        model.train(); opt.zero_grad()
        out = model(x, conn)
        loss = F.cross_entropy(out[tr], yt[tr])
        loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            out = model(x, conn)
            vs = score(out, va)
            if vs >= best_va:
                best_va = vs; best_te = score(out, te)
    return best_te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='ACM')
    ap.add_argument('--trials', type=int, default=3)
    ap.add_argument('--groups', nargs='+', default=['author', 'subject'],
                    choices=['author', 'subject'],
                    help='which node types become hyperedges / meta-paths (PAP=author, PSP=subject)')
    ap.add_argument('--verbose_trials', action='store_true')
    ap.add_argument('--outdir', default=None)
    args = ap.parse_args()
    outdir = args.outdir or f'results/higher_order_{args.dataset}'
    os.makedirs(outdir, exist_ok=True)

    d = load_hgb(args.dataset)
    tgt = TARGET[args.dataset]
    n = int(d[tgt].num_nodes)
    y = d[tgt].y.numpy()
    assert y.ndim == 1, 'this smoke targets single-label ACM'
    n_cls = int(y.max()) + 1
    # NOTE: .numpy() aliases the underlying tensor, and build_metapath_graph()
    # (called later for the pairwise edges) mutates d[tgt].train_mask in place during
    # its own val-synthesis. So we .copy() here to own an independent split, and
    # synthesize val ourselves BEFORE any build_metapath_graph call.
    masks = {k: getattr(d[tgt], f'{k}_mask').numpy().copy()
             for k in ('train', 'val', 'test') if hasattr(d[tgt], f'{k}_mask')}
    if 'val' not in masks:
        tr = masks['train'].copy(); idx = np.where(tr)[0]; cut = idx[int(0.85 * len(idx)):]
        masks['val'] = np.zeros_like(tr); masks['val'][cut] = True; masks['train'][cut] = False

    x_feat = d[tgt].x.numpy().astype(np.float32)
    x = torch.tensor(x_feat, dtype=torch.float32, device=device)

    # hyperedges from the chosen relations (paper-centric). author<->PAP, subject<->PSP
    grp2rel = {'author': ('paper', 'to', 'author'), 'subject': ('paper', 'to', 'subject')}
    grp2mp = {'author': 'PAP', 'subject': 'PSP'}
    he_relations = [grp2rel[g] for g in args.groups]
    H_np, n_he, sizes = build_hyperedges(d, args.dataset, he_relations)
    H = torch.tensor(H_np, dtype=torch.long, device=device)
    H_shuf = torch.tensor(shuffle_hyperedges(H_np, n_he, n, seed=0),
                          dtype=torch.long, device=device)

    # pairwise meta-path (clique expansion of the SAME groups used for hyperedges)
    mps = [grp2mp[g] for g in args.groups]        # PAP and/or PSP
    ei_mp, n_mp_edges = metapath_edge_index(d, args.dataset, mps)
    ei_rand = random_edge_index(n, n_mp_edges, seed=0)

    print(f'{args.dataset}: n_papers={n} n_cls={n_cls} feat_dim={x.size(1)}')
    print(f'  hyperedges: {n_he} (author+subject), size mean={sizes.mean():.1f} '
          f'max={sizes.max()} med={np.median(sizes):.0f}, total incidences={H_np.shape[1]}')
    print(f'  pairwise meta-path (PAP+PSP) edges (directed)={n_mp_edges} '
          f'-> clique expansion of the same groups')
    print(f'  train/val/test = {masks["train"].sum()}/{masks["val"].sum()}/{masks["test"].sum()}')

    configs = [
        ('GCN',       'pairwise',          GCNNet,   ei_mp),
        ('GCN',       'random-edges',      GCNNet,   ei_rand),
        ('HyperConv', 'higher-order',      HyperNet, H),
        ('HyperConv', 'shuffled-he',       HyperNet, H_shuf),
    ]
    results = {}
    for model_name, variant, fn, conn in configs:
        t0 = time.time()
        accs = [run_once(fn, conn, x, y, masks, n_cls, s) for s in range(args.trials)]
        results[(model_name, variant)] = accs
        extra = f'  trials={np.round(accs,3).tolist()}' if args.verbose_trials else ''
        print(f'  [{model_name:9} {variant:13}] acc = {np.mean(accs):.4f} '
              f'± {np.std(accs):.4f}  ({time.time()-t0:.0f}s, {args.trials} trials){extra}')

    with open(os.path.join(outdir, 'nc_acc.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['dataset', 'model', 'variant', 'mean_acc', 'std_acc', 'n_trials'])
        for (m, v), accs in results.items():
            a = np.array(accs)
            w.writerow([args.dataset, m, v, f'{a.mean():.6f}', f'{a.std():.6f}', len(a)])
    print(f'Outputs -> {outdir}/nc_acc.csv')


if __name__ == '__main__':
    main()
