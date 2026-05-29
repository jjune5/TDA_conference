"""§14-style leakage audit for meta-path graphs.

If a meta-path routes through a label-defining node type, the meta-path graph's
STRUCTURE ALONE can predict the label -> the topological feature built on it would
be a label-membership artifact, not genuine topology. We probe this with simple
weighted majority-vote label propagation from train labels: test accuracy from
STRUCTURE ALONE (no node features). Suspiciously high acc => leak flag.
"""
from __future__ import annotations
import numpy as np


def structure_only_label_acc(g, y, masks, n_iter: int = 10) -> float:
    """Weighted majority-vote label propagation on the meta-path graph using only
    train labels; return TEST accuracy (structure-only, no features)."""
    if y.ndim > 1:           # multi-label (e.g. IMDB) -> skip structure-only audit
        return float('nan')
    n = g.number_of_nodes()
    C = int(y.max()) + 1
    train, test = masks['train'], masks['test']
    # one-hot label distribution, fixed on train nodes
    P = np.zeros((n, C), dtype=np.float64)
    P[train, y[train]] = 1.0
    fixed = train.copy()
    # neighbor lists with weights
    nbrs = {u: list(g[u].items()) for u in g.nodes()}
    for _ in range(n_iter):
        Pn = P.copy()
        for u in range(n):
            if fixed[u]:
                continue
            acc = np.zeros(C)
            for v, dd in nbrs[u]:
                acc += dd['weight'] * P[v]
            if acc.sum() > 0:
                Pn[u] = acc / acc.sum()
        P = Pn
    pred = P.argmax(1)
    # nodes with no propagated signal -> predict train majority
    nosig = (P.sum(1) == 0)
    if nosig.any():
        pred[nosig] = np.bincount(y[train]).argmax()
    return float((pred[test] == y[test]).mean())
