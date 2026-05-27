"""
cdrh3_ph.py — Vietoris-Rips persistent homology of a CDR-H3 Calpha point cloud.

Thread C, Rung 0. Given an (N, 3) coordinate array we build a Rips complex
(GUDHI, max_dimension=2) and read off persistence diagrams for H0 and H1.

Provides:
  * compute_persistence(coords)         -> dict with PD arrays per dim + diagram
  * topo_distance(pd_a, pd_b, ...)       -> bottleneck (default) / wasserstein
  * loop_likeness(pd_h1)                 -> NATIVE-FREE score from H1 lifetimes
  * persistence_image_vector(pd_h1, ...) -> optional fixed-length PI vector

A persistence diagram here is an (m, 2) array of [birth, death] (finite bars).
For Rips on a finite point cloud, the single infinite H0 bar is dropped before
distance computations (bottleneck distance handles only finite-or-matched bars).

PH-on-CA is a coarse, *shape-only*, rigid-motion-invariant lens. It deliberately
ignores backbone orientation / side chains. This is stated as an honest limit in
the analysis.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

import gudhi
from gudhi.bottleneck import bottleneck_distance

try:
    from gudhi.wasserstein import wasserstein_distance as _wd
    _HAS_WASSERSTEIN = True
except Exception:  # pragma: no cover
    _HAS_WASSERSTEIN = False

try:
    from gudhi.representations import PersistenceImage as _PI
    _HAS_PI = True
except Exception:  # pragma: no cover
    _HAS_PI = False


# -----------------------------------------------------------------------------
# Persistence
# -----------------------------------------------------------------------------
def compute_persistence(
    coords: np.ndarray,
    max_edge_length: Optional[float] = None,
    max_dimension: int = 2,
) -> Dict[str, object]:
    """Compute Rips persistence of a point cloud.

    max_edge_length: if None, use 2.0 x the cloud diameter (i.e. effectively
    unbounded for these small loops) so H1 cycles can fully form. A finite cap
    keeps the complex small; CA-CA spacing is ~3.8 A so a loop of ~15 residues
    has diameter ~20-35 A; default cap comfortably exceeds that.

    Returns dict:
      'dgms' : {0: (n0,2) array, 1: (n1,2) array} finite bars only
      'pd1'  : convenience alias for dgms[1]
      'pd0'  : convenience alias for dgms[0] (finite H0 bars)
      'n_points' : N
      'diameter' : max pairwise distance
    """
    coords = np.asarray(coords, dtype=np.float64)
    n = coords.shape[0]
    if n < 2:
        return {
            "dgms": {0: np.zeros((0, 2)), 1: np.zeros((0, 2))},
            "pd0": np.zeros((0, 2)),
            "pd1": np.zeros((0, 2)),
            "n_points": int(n),
            "diameter": 0.0,
        }

    # diameter
    from scipy.spatial.distance import pdist

    dists = pdist(coords)
    diameter = float(dists.max()) if dists.size else 0.0
    if max_edge_length is None:
        max_edge_length = 2.0 * diameter if diameter > 0 else 1.0

    rips = gudhi.RipsComplex(points=coords, max_edge_length=max_edge_length)
    st = rips.create_simplex_tree(max_dimension=max_dimension)
    st.compute_persistence(persistence_dim_max=True)

    dgms = {0: [], 1: []}
    for dim in (0, 1):
        intervals = st.persistence_intervals_in_dimension(dim)
        for b, d in intervals:
            if np.isinf(d):
                continue  # drop the infinite H0 component bar
            dgms[dim].append([b, d])
    dgm0 = np.array(dgms[0], dtype=np.float64) if dgms[0] else np.zeros((0, 2))
    dgm1 = np.array(dgms[1], dtype=np.float64) if dgms[1] else np.zeros((0, 2))

    return {
        "dgms": {0: dgm0, 1: dgm1},
        "pd0": dgm0,
        "pd1": dgm1,
        "n_points": int(n),
        "diameter": diameter,
    }


# -----------------------------------------------------------------------------
# Topological distance
# -----------------------------------------------------------------------------
def topo_distance(
    pd_a: np.ndarray,
    pd_b: np.ndarray,
    metric: str = "bottleneck",
    order: float = 1.0,
) -> float:
    """Distance between two persistence diagrams (single homology dim).

    metric='bottleneck' (default, robust, always available) or 'wasserstein'.
    Empty-vs-empty -> 0. Empty-vs-nonempty -> diagrams compared to the diagonal,
    which both gudhi routines handle natively.
    """
    pd_a = np.asarray(pd_a, dtype=np.float64).reshape(-1, 2)
    pd_b = np.asarray(pd_b, dtype=np.float64).reshape(-1, 2)
    if pd_a.shape[0] == 0 and pd_b.shape[0] == 0:
        return 0.0
    if metric == "wasserstein":
        if not _HAS_WASSERSTEIN:
            raise RuntimeError("gudhi.wasserstein not available")
        return float(_wd(pd_a, pd_b, order=order, internal_p=2.0))
    # bottleneck
    return float(bottleneck_distance(pd_a, pd_b))


# -----------------------------------------------------------------------------
# Native-FREE loop-likeness
# -----------------------------------------------------------------------------
def loop_likeness(pd_h1: np.ndarray, mode: str = "total") -> float:
    """Native-free 'how loop-like is this shape' score from H1 persistence.

    A well-formed closed loop produces one (or few) persistent 1-cycle(s) with a
    long lifetime (death - birth). Higher = more loop-like / more topologically
    structured.

    mode='total': sum of all H1 lifetimes.
    mode='max'  : single most-persistent H1 lifetime.
    Returns 0.0 if there are no H1 features.
    """
    pd_h1 = np.asarray(pd_h1, dtype=np.float64).reshape(-1, 2)
    if pd_h1.shape[0] == 0:
        return 0.0
    lifetimes = pd_h1[:, 1] - pd_h1[:, 0]
    lifetimes = lifetimes[lifetimes > 0]
    if lifetimes.size == 0:
        return 0.0
    if mode == "max":
        return float(lifetimes.max())
    return float(lifetimes.sum())


# -----------------------------------------------------------------------------
# Optional persistence-image vector
# -----------------------------------------------------------------------------
def persistence_image_vector(
    pd_h1: np.ndarray,
    resolution=(10, 10),
    bandwidth: float = 1.0,
    im_range=None,
) -> np.ndarray:
    """Fixed-length persistence-image vector for H1 (optional feature).

    Returns a flat vector of length resolution[0]*resolution[1]. Empty diagram
    -> zeros. Requires gudhi.representations.
    """
    n = resolution[0] * resolution[1]
    pd_h1 = np.asarray(pd_h1, dtype=np.float64).reshape(-1, 2)
    if pd_h1.shape[0] == 0:
        return np.zeros(n, dtype=np.float64)
    if not _HAS_PI:
        raise RuntimeError("gudhi.representations.PersistenceImage not available")
    kw = dict(bandwidth=bandwidth, resolution=list(resolution))
    if im_range is not None:
        kw["im_range"] = im_range
    pim = _PI(**kw)
    vec = pim.fit_transform([pd_h1])[0]
    return np.asarray(vec, dtype=np.float64).ravel()
