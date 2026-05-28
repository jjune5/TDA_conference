# Knowledge_Distillation/hks_filtration.py
"""Heat Kernel Signature (HKS) filtration for molecular graphs.

HKS is a physical-diffusion-based filtration derived from the graph Laplacian:
    HKS_t(i) = sum_k exp(-t * lambda_k) * phi_k(i)^2

where lambda_k, phi_k are the eigenvalues/eigenvectors of the normalized
graph Laplacian L_norm = I - D^{-1/2} A D^{-1/2}.

Scale selection (t):
    The HKS captures multi-scale structure. For a fixed single-scale,
    we choose t = 1 / lambda_median where lambda_median is the median of the
    *positive* eigenvalues of L_norm. This places the heat diffusion at the
    "characteristic" scale of the graph spectrum — long enough to see
    non-trivial structure (past the Fiedler mode) but short enough to retain
    local contrast. On typical small molecular graphs (n ~ 10–30 nodes) with
    average degree ~ 2, the Fiedler value is ~ 0.1–0.5 and lambda_median ~ 1.0,
    giving t ~ 1, which is a standard literature default. When the spectrum
    is degenerate (all-zero, fully disconnected) we fall back to zeros.

Graceful fallback:
    - n < 3 or no edges: return np.zeros(n)
    - Eigendecomp fails: return np.zeros(n)
    - All eigenvalues near zero (isolated nodes): return np.zeros(n)
    - Resulting HKS is constant (no contrast): return np.zeros(n)

This mirrors the (ValueError, IndexError, KeyError)->zeros pattern in mol_data.py.
"""
from __future__ import annotations
import numpy as np
import networkx as nx
import scipy.linalg


def compute_hks(g: nx.Graph, t: float | None = None) -> np.ndarray:
    """Compute per-node HKS values for a networkx graph.

    Parameters
    ----------
    g   : undirected networkx graph with integer node labels 0..n-1
    t   : diffusion time scale. If None, chosen from spectrum (see module doc).

    Returns
    -------
    hks : (n,) float64 array of HKS values, normalized to [0,1].
          Returns np.zeros(n) on degenerate inputs.
    """
    g = nx.convert_node_labels_to_integers(g)
    n = g.number_of_nodes()

    # --- Graceful fallback for tiny/edgeless graphs ---
    if n < 3 or g.number_of_edges() == 0:
        return np.zeros(n, dtype=np.float64)

    # --- Build normalized Laplacian ---
    # L_norm = I - D^{-1/2} A D^{-1/2}  (same convention as networkx)
    try:
        L = nx.normalized_laplacian_matrix(g).toarray().astype(np.float64)
    except Exception:
        return np.zeros(n, dtype=np.float64)

    # --- Eigendecomposition (symmetric, so use eigh) ---
    try:
        # scipy.linalg.eigh returns eigenvalues in ascending order
        lams, phis = scipy.linalg.eigh(L)
    except Exception:
        return np.zeros(n, dtype=np.float64)

    # Clip tiny negatives from numerical noise (L_norm is PSD)
    lams = np.clip(lams, 0.0, None)

    # --- Choose t if not given ---
    if t is None:
        pos_lams = lams[lams > 1e-8]
        if pos_lams.size == 0:
            # Fully disconnected / trivial graph
            return np.zeros(n, dtype=np.float64)
        t = 1.0 / float(np.median(pos_lams))

    # --- Compute HKS_t(i) = sum_k exp(-t * lambda_k) * phi_k(i)^2 ---
    weights = np.exp(-t * lams)          # (n,)
    # phis[:, k] = k-th eigenvector  -> phis^2 @ weights
    hks = (phis ** 2) @ weights           # (n,)

    # --- Normalize to [0, 1] ---
    hks_min, hks_max = hks.min(), hks.max()
    rng = hks_max - hks_min
    if rng < 1e-12:
        # Constant HKS (no spatial contrast) — return zeros (degenerate)
        return np.zeros(n, dtype=np.float64)
    return (hks - hks_min) / rng


# ---------------------------------------------------------------------------
# Convenience: expose the chosen t for diagnostics
# ---------------------------------------------------------------------------

def hks_t_from_spectrum(g: nx.Graph) -> float:
    """Return the t value that would be selected for this graph (for diagnostics)."""
    g = nx.convert_node_labels_to_integers(g)
    if g.number_of_nodes() < 3 or g.number_of_edges() == 0:
        return float('nan')
    try:
        L = nx.normalized_laplacian_matrix(g).toarray().astype(np.float64)
        lams = scipy.linalg.eigh(L, eigvals_only=True)
        lams = np.clip(lams, 0.0, None)
        pos_lams = lams[lams > 1e-8]
        if pos_lams.size == 0:
            return float('nan')
        return 1.0 / float(np.median(pos_lams))
    except Exception:
        return float('nan')


# ---------------------------------------------------------------------------
# Smoke test (run directly)
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from torch_geometric.datasets import TUDataset
    from torch_geometric.utils import to_networkx

    ds = TUDataset(root='./data/TU_MUTAG', name='MUTAG')
    print(f'MUTAG: {len(ds)} graphs')
    nonzero_count = 0
    t_vals = []
    for i, data in enumerate(ds[:10]):
        g = to_networkx(data, to_undirected=True)
        g = nx.convert_node_labels_to_integers(g)
        hks = compute_hks(g)
        t_val = hks_t_from_spectrum(g)
        t_vals.append(t_val)
        is_nonzero = hks.sum() > 0
        if is_nonzero:
            nonzero_count += 1
        print(f'  graph {i:2d}: n={g.number_of_nodes():3d} e={g.number_of_edges():3d} '
              f't={t_val:.3f} hks_range=[{hks.min():.4f},{hks.max():.4f}] '
              f'nonzero={is_nonzero}')
    print(f'Non-degenerate: {nonzero_count}/10')
    finite_t = [v for v in t_vals if not np.isnan(v)]
    print(f't values: min={min(finite_t):.3f} median={np.median(finite_t):.3f} max={max(finite_t):.3f}')
