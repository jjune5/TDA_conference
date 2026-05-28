"""Per-node persistent homology features under a diffusion (HKS) filtration (DNP).

Each node v gets a persistence vector Phi(v) computed on its OWN k-hop ego-graph
(NOT the candidate edge's vicinity). Removing one incident edge barely changes the
ego-graph -> the feature does not collapse at test time (the §14 fix), unlike
exact vicinity-PI. Diffusion enters as the filtration:
  - A: filter = multi-scale HKS values, sublevel (lower-star) persistence.
  - C: filter = diffusion distance, Vietoris-Rips persistence.
  - B: bifiltration (HKS-time x Ollivier-Ricci) via slicing (optional).
"""
from __future__ import annotations
import numpy as np
import networkx as nx
import gudhi

PI_RES = 5                      # 5x5 = 25-dim persistence image per diagram


def _full_graph(data) -> nx.Graph:
    ei = np.asarray(data.edge_index.cpu() if hasattr(data.edge_index, 'cpu')
                    else data.edge_index)
    g = nx.Graph()
    g.add_nodes_from(range(int(data.num_nodes)))
    g.add_edges_from(zip(ei[0].tolist(), ei[1].tolist()))
    return g


def _diagram_points(st, max_filt: float):
    """Extract finite (birth, death) points (H0 + H1) from a computed SimplexTree.
    Essential classes (death = inf) are capped at max_filt; zero-persistence dropped."""
    pts = []
    for _dim, (b, d) in st.persistence(homology_coeff_field=2, min_persistence=0.0):
        if d == float('inf'):
            d = max_filt
        if d > b:
            pts.append((b, d))
    return pts


def _ego_sublevel_pi(g: nx.Graph, center: int, hop: int, node_filt: dict,
                     imager, max_nodes: int = 300) -> np.ndarray:
    """Sublevel (lower-star) persistence of the ego-graph around `center`, filtered
    by `node_filt`, vectorized to a (25,) persistence image.

    Lower-star filtration: vertex i enters at f(i); edge (i,j) enters at max(f(i),f(j))."""
    ego = nx.ego_graph(g, center, radius=hop)
    if ego.number_of_nodes() > max_nodes:                 # cap cost: keep nearest by filter
        keep = sorted(ego.nodes(), key=lambda n: node_filt.get(n, 0.0))[:max_nodes]
        ego = ego.subgraph(keep).copy()
    if ego.number_of_edges() == 0:
        return np.zeros(PI_RES * PI_RES, dtype=np.float64)
    vals = [node_filt.get(n, 0.0) for n in ego.nodes()]
    max_filt = float(max(vals)) if vals else 1.0
    st = gudhi.SimplexTree()
    for n in ego.nodes():
        st.insert([int(n)], filtration=float(node_filt.get(n, 0.0)))
    for u, v in ego.edges():
        st.insert([int(u), int(v)],
                  filtration=float(max(node_filt.get(u, 0.0), node_filt.get(v, 0.0))))
    pts = _diagram_points(st, max_filt)
    if not pts:
        return np.zeros(PI_RES * PI_RES, dtype=np.float64)
    return np.asarray(imager.transform(np.array(pts, dtype=np.float64))).reshape(-1)


def phi_A(data, K: int = 5, hop: int = 2, max_nodes: int = 300,
          verbose: bool = False) -> np.ndarray:
    """(N, 25*K) HKS-filtered sublevel node-PH.

    Reuses diffusion_features.compute_hks_features for the (N,K) global multi-scale
    HKS filter values, then computes per-node ego-graph sublevel persistence."""
    from diffusion_features import compute_hks_features
    from sg2dgm import PersistenceImager as pimg_mod
    imager = pimg_mod.PersistenceImager(resolution=PI_RES)
    g = _full_graph(data)
    hks, _meta = compute_hks_features(data, K=K, verbose=verbose)   # (N, K)
    n = int(data.num_nodes)
    # one filter dict per scale (reused across all nodes — avoids O(N^2 K) rebuilds)
    filts = [{nd: float(hks[nd, k]) for nd in range(n)} for k in range(K)]
    out = np.zeros((n, PI_RES * PI_RES * K), dtype=np.float64)
    for v in range(n):
        for k in range(K):
            pi = _ego_sublevel_pi(g, v, hop, filts[k], imager, max_nodes)
            out[v, k * 25:(k + 1) * 25] = pi
        if verbose and (v + 1) % 500 == 0:
            print(f'    phi_A {v+1}/{n}')
    return out


def _global_laplacian_eig(data, dev=None):
    """Eigendecomposition of the normalized Laplacian on the (leakage-free) graph.
    Returns (lams (n,), phis (n,n)) as numpy. Mirrors compute_hks_features' eig block."""
    import torch
    dev = dev or ('cuda' if torch.cuda.is_available() else 'cpu')
    n = int(data.num_nodes)
    ei = np.asarray(data.edge_index.cpu() if hasattr(data.edge_index, 'cpu')
                    else data.edge_index)
    A = torch.zeros((n, n), dtype=torch.float64, device=dev)
    src = torch.from_numpy(ei[0]).long().to(dev); dst = torch.from_numpy(ei[1]).long().to(dev)
    A[src, dst] = 1.0; A[dst, src] = 1.0; A.fill_diagonal_(0.0)
    deg = A.sum(1); dinv = torch.where(deg > 0, deg.pow(-0.5), torch.zeros_like(deg))
    L = torch.eye(n, dtype=torch.float64, device=dev) - torch.diag(dinv) @ A @ torch.diag(dinv)
    lams, phis = torch.linalg.eigh(L)
    lams = torch.clamp(lams, min=0.0)
    return lams.cpu().numpy(), phis.cpu().numpy()


def phi_C(data, hop: int = 2, max_nodes: int = 300, t: float | None = None,
          verbose: bool = False) -> np.ndarray:
    """(N, 25) diffusion-distance Vietoris-Rips node-PH.

    Diffusion distance at time t: d_t(i,j)^2 = sum_k exp(-2 t lam_k)(phi_k(i)-phi_k(j))^2.
    Per node v: Rips persistence of the ego-graph nodes under d_t -> (25,) PI."""
    from sg2dgm import PersistenceImager as pimg_mod
    imager = pimg_mod.PersistenceImager(resolution=PI_RES)
    g = _full_graph(data)
    lams, phis = _global_laplacian_eig(data)
    pos = lams[lams > 1e-8]
    if t is None:
        t = 1.0 / float(np.median(pos)) if pos.size else 1.0
    w = np.exp(-2.0 * t * lams)                                  # (n_eig,)
    n = int(data.num_nodes)
    out = np.zeros((n, PI_RES * PI_RES), dtype=np.float64)
    for v in range(n):
        nodes = list(nx.ego_graph(g, v, radius=hop).nodes())
        if len(nodes) > max_nodes:
            nodes = nodes[:max_nodes]
        if len(nodes) < 2:
            continue
        P = phis[nodes, :]                                       # (m, n_eig)
        diff = P[:, None, :] - P[None, :, :]                     # (m, m, n_eig)
        D = np.sqrt(np.clip((diff ** 2 * w).sum(-1), 0, None))   # (m, m) diffusion dist
        rips = gudhi.RipsComplex(distance_matrix=D, max_edge_length=float(D.max()))
        st = rips.create_simplex_tree(max_dimension=2)
        pts = _diagram_points(st, float(D.max()))
        if pts:
            out[v] = np.asarray(imager.transform(np.array(pts, dtype=np.float64))).reshape(-1)
        if verbose and (v + 1) % 500 == 0:
            print(f'    phi_C {v+1}/{n}')
    return out
