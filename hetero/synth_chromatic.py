"""Planted-bridge synthetic hetero graph: label = number of same-type clusters that
are held together ONLY by cross-type structure (= chromatic kernel count).

Design rationale (important characterization, learned the hard way):
0-dim chromatic kernel persistence under a node (lower-star) filtration does NOT see
the internal structure/length of a cross-type bridge — any path a..b has max-edge
>= max(f_a,f_b), so a 1-hop vs 2-hop bridge yields the same single essential kernel
class. What 0-dim kernel DOES encode is (i) the COUNT of same-type components that are
cross-type-bridged (= b0(L) - #L-bearing-K-comps) and (ii) the scalar time they join.

So the positive control must vary that COUNT:
  label 0: target leaves form 2 same-type clusters, all bridged via one shared
           non-target hub  -> kernel rank = 2 - 1 = 1
  label 1: target leaves form 3 same-type clusters, all bridged via one shared hub
           -> kernel rank = 3 - 1 = 2
The full graph K is CONNECTED in both classes, so ordinary H0(K) = 1 (one essential
bar) for both -> a type-blind (achromatic) summary cannot separate the classes, and
node filtration values are noise. Only the chromatic kernel (cross-type mingling)
differs by construction. Same number of target leaves both classes (no size confound).
"""
from __future__ import annotations
import numpy as np


def make_synth_chromatic(label: int, seed: int = 0, n_target: int = 6):
    """Return dict fv (m,), ei (2,E), is_L (m,) bool, y. n_clusters = 2 (label 0) or 3
    (label 1) same-type star clusters, all joined through ONE shared non-target hub."""
    rng = np.random.RandomState(seed)
    n_clusters = 2 if label == 0 else 3
    # split n_target leaves into n_clusters contiguous groups (as equal as possible)
    bounds = np.linspace(0, n_target, n_clusters + 1).astype(int)
    clusters = [list(range(bounds[i], bounds[i + 1])) for i in range(n_clusters)]
    edges = []
    for c in clusters:                       # star within each cluster (same-type edges)
        for x in c[1:]:
            edges.append((c[0], x))
    is_L = [True] * n_target
    hub = n_target; is_L.append(False)       # one shared non-target hub bridges clusters
    for c in clusters:
        edges.append((c[0], hub))            # cross-type bridge: cluster-root -- hub
    m = n_target + 1
    fv = rng.rand(m).astype(np.float64)      # noise filtration (same dist both classes)
    ei = np.array(edges, dtype=np.int64).T
    return {'fv': fv, 'ei': ei, 'is_L': np.array(is_L, bool), 'y': int(label)}
