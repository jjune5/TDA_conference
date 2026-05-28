# Diffusion-filtered Node-level Persistence (DNP) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a *genuine topological* link-prediction feature — per-node persistent homology under a diffusion (HKS) filtration — that, unlike exact vicinity-PI, does not collapse when the candidate edge is removed (§14), and test whether it helps heterophilic LP.

**Architecture:** Maximally reuse the §15 path in `diffusion_features.py` (data split, leakage-free graph, edge-feature assembly `[Φ(u),Φ(v),|Φ(u)−Φ(v)|]`, §14 diagnostic, GCN+decoder LP model, 50-trial loop). The ONLY new thing is the per-node feature: replace `compute_hks_features` (per-node HKS *scalar* vector, shape `(N,K)`) with **per-node PH vectors** `Φ` (shape `(N,D)`), computed on each node's k-hop ego-graph. Because `prepare_data` already removes val/test positive edges from the training graph, per-node ego-PH on that graph is *inherently* the edge-removed (leakage-free) condition — so `run_diagnostic`'s `test_auc` IS the collapse-test gate.

**Tech Stack:** Python (`tlcgnn` conda env), gudhi 3.11 (sublevel + Rips persistence), `sg2dgm.PersistenceImager` (diagram→5×5 PI), networkx (ego-graphs), torch/PyG (reused LP model), scikit-learn (diagnostic). All new code is **additive**; existing files are not modified except two minimal `--pi_source`-style additions are explicitly **NOT** needed (DNP rides the `diffusion_features.py` path, not the PI slot).

**Environment note:** every Python command below must run with the env active:
```bash
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh
conda activate tlcgnn
cd /mnt/data/users/junyoungpark/code/TLC-GNN
```
(If `conda` path differs, find it with `which conda`. `python` alone fails outside the env — confirmed.)

---

## File Structure

All new files live at the **repo root** (next to `diffusion_features.py`) so `from diffusion_features import ...` works directly.

- **Create `node_ph_features.py`** — per-node PH feature computation. One responsibility: given a prepared `data` object (leakage-free graph in `data.edge_index`), return a `(N, D)` per-node feature matrix for a chosen variant.
  - `_full_graph(data)` → networkx graph from `data.edge_index`
  - `_global_laplacian_eig(data, dev)` → `(lams, phis)` of the normalized Laplacian (for variant C diffusion distance; mirrors `diffusion_features.compute_hks_features` eig block)
  - `_ego_sublevel_pi(g, center, hop, node_filt, imager, max_nodes)` → `(25,)` PI of the ego-graph's sublevel (lower-star) persistence under `node_filt`
  - `phi_A(data, K, hop, max_nodes)` → `(N, 25*K)` — HKS-filtered sublevel node-PH (reuses `compute_hks_features` for the `(N,K)` global HKS filter values)
  - `phi_C(data, hop, max_nodes)` → `(N, 25)` — diffusion-distance Vietoris-Rips node-PH
  - `phi_B(data, K, hop, max_nodes)` → `(N, 25*K)` — bifiltration (HKS-time × Ricci) by slicing; **optional, behind a flag**
- **Create `diffusion_node_ph.py`** — driver. One responsibility: wire a DNP variant into the reused `diffusion_features` pipeline; run §14 diagnostic (collapse gate) + 50-trial LP; write CSVs to `results/dnp_<variant>/`.
- **Create `tests/test_node_ph_features.py`** — pytest unit/smoke tests (shape, nonzero, known-topology, non-collapse).

---

## Phase 1 — exact node-PH (A, C) + collapse-test gate on Cora + Chameleon

### Task 1: Ego-graph sublevel-PI helper + variant A

**Files:**
- Create: `node_ph_features.py`
- Test: `tests/test_node_ph_features.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_node_ph_features.py
import numpy as np
import networkx as nx
import pytest


def test_ego_sublevel_pi_triangle_is_finite_25vec():
    """A triangle (3-cycle) with a non-constant node filter yields a finite (25,) PI."""
    from node_ph_features import _ego_sublevel_pi
    from sg2dgm import PersistenceImager as pimg_mod
    imager = pimg_mod.PersistenceImager(resolution=5)
    g = nx.cycle_graph(3)                      # nodes 0,1,2 ; edges (0,1),(1,2),(2,0)
    node_filt = {0: 0.0, 1: 0.5, 2: 1.0}
    pi = _ego_sublevel_pi(g, center=0, hop=2, node_filt=node_filt,
                          imager=imager, max_nodes=200)
    assert pi.shape == (25,)
    assert np.all(np.isfinite(pi))
    assert pi.sum() >= 0.0                     # PI is non-negative


def test_phi_A_shape_and_nonzero_on_synthetic():
    """phi_A returns (N, 25*K) and is not all-zero on a connected graph."""
    import torch
    from types import SimpleNamespace
    from node_ph_features import phi_A
    # 2 triangles joined by an edge -> 5 nodes
    edges = [(0,1),(1,2),(2,0),(2,3),(3,4),(4,2)]
    ei = torch.tensor([[a for a,b in edges]+[b for a,b in edges],
                       [b for a,b in edges]+[a for a,b in edges]], dtype=torch.long)
    data = SimpleNamespace(edge_index=ei, num_nodes=5)
    phi = phi_A(data, K=3, hop=2, max_nodes=200)
    assert phi.shape == (5, 75)                # 25 * K(=3)
    assert np.isfinite(phi).all()
    assert phi.sum() > 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda activate tlcgnn && cd /mnt/data/users/junyoungpark/code/TLC-GNN && python -m pytest tests/test_node_ph_features.py -x -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'node_ph_features'`

- [ ] **Step 3: Write minimal implementation**

```python
# node_ph_features.py
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
    out = np.zeros((n, PI_RES * PI_RES * K), dtype=np.float64)
    for v in range(n):
        for k in range(K):
            node_filt = {nd: float(hks[nd, k]) for nd in g.nodes()}
            pi = _ego_sublevel_pi(g, v, hop, node_filt, imager, max_nodes)
            out[v, k * 25:(k + 1) * 25] = pi
        if verbose and (v + 1) % 500 == 0:
            print(f'    phi_A {v+1}/{n}')
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda activate tlcgnn && python -m pytest tests/test_node_ph_features.py -x -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add node_ph_features.py tests/test_node_ph_features.py
git commit -m "feat(DNP): per-node ego sublevel-PI helper + variant A (HKS filtration)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Variant C (diffusion-distance Vietoris-Rips)

**Files:**
- Modify: `node_ph_features.py` (add `_global_laplacian_eig`, `phi_C`)
- Test: `tests/test_node_ph_features.py` (add `test_phi_C_shape`)

- [ ] **Step 1: Write the failing test**

```python
def test_phi_C_shape_and_finite_on_synthetic():
    import torch
    from types import SimpleNamespace
    from node_ph_features import phi_C
    edges = [(0,1),(1,2),(2,0),(2,3),(3,4),(4,2)]
    ei = torch.tensor([[a for a,b in edges]+[b for a,b in edges],
                       [b for a,b in edges]+[a for a,b in edges]], dtype=torch.long)
    data = SimpleNamespace(edge_index=ei, num_nodes=5)
    phi = phi_C(data, hop=2, max_nodes=200)
    assert phi.shape == (5, 25)
    assert np.isfinite(phi).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_node_ph_features.py::test_phi_C_shape_and_finite_on_synthetic -x -q`
Expected: FAIL — `ImportError: cannot import name 'phi_C'`

- [ ] **Step 3: Write minimal implementation** (append to `node_ph_features.py`)

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_node_ph_features.py -x -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add node_ph_features.py tests/test_node_ph_features.py
git commit -m "feat(DNP): variant C (diffusion-distance Vietoris-Rips node-PH)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Non-collapse property test (the §14 fix, unit-level)

**Files:**
- Test: `tests/test_node_ph_features.py` (add `test_phi_A_does_not_collapse_on_edge_removal`)

- [ ] **Step 1: Write the test** (this is the core scientific claim, tested in miniature)

```python
def test_phi_A_does_not_collapse_on_edge_removal():
    """Removing ONE edge incident to a node changes its phi_A only slightly
    (the §14 fix: per-node ego features are robust to single-edge deletion)."""
    import torch
    from types import SimpleNamespace
    from node_ph_features import phi_A
    # ring of 8 + chords -> node 0 has several incident edges
    edges = [(i, (i+1) % 8) for i in range(8)] + [(0,2),(0,4),(1,5)]
    def mk(es):
        ei = torch.tensor([[a for a,b in es]+[b for a,b in es],
                           [b for a,b in es]+[a for a,b in es]], dtype=torch.long)
        return SimpleNamespace(edge_index=ei, num_nodes=8)
    phi_full = phi_A(mk(edges), K=3, hop=2, max_nodes=200)[0]      # node 0
    phi_drop = phi_A(mk([e for e in edges if e != (0,2)]), K=3, hop=2, max_nodes=200)[0]
    # feature stays nonzero and close (NOT a collapse-to-zero like vicinity-PI)
    assert phi_full.sum() > 0 and phi_drop.sum() > 0
    rel = np.linalg.norm(phi_full - phi_drop) / (np.linalg.norm(phi_full) + 1e-9)
    assert rel < 0.6, f'phi_A changed too much on single-edge removal: rel={rel:.3f}'
```

- [ ] **Step 2: Run test**

Run: `python -m pytest tests/test_node_ph_features.py::test_phi_A_does_not_collapse_on_edge_removal -x -q`
Expected: PASS. (If it FAILS with `rel` near 1.0, the construction is collapsing — STOP and reconsider `hop`/filter; this is the whole point of DNP.)

- [ ] **Step 3: Commit**

```bash
git add tests/test_node_ph_features.py
git commit -m "test(DNP): non-collapse on single-edge removal (the §14 fix)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Driver — diagnostic (collapse gate) + LP, on Cora + Chameleon

**Files:**
- Create: `diffusion_node_ph.py`

- [ ] **Step 1: Write the driver** (reuses the `diffusion_features` pipeline wholesale)

```python
# diffusion_node_ph.py
"""DNP driver: run a per-node-PH variant through the reused diffusion_features LP
pipeline. Reports the §14 diagnostic (collapse gate) and 50-trial LP AUC vs
no-PI / exact-PI baselines. Variant 'A'|'C'|'B'."""
from __future__ import annotations
import os, sys, json, csv, time, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import diffusion_features as DF          # reuse the whole §15 pipeline
import node_ph_features as NPH


def compute_variant(data, variant, K, hop, max_nodes, verbose):
    if variant == 'A':
        return NPH.phi_A(data, K=K, hop=hop, max_nodes=max_nodes, verbose=verbose)
    if variant == 'C':
        return NPH.phi_C(data, hop=hop, max_nodes=max_nodes, verbose=verbose)
    if variant == 'B':
        return NPH.phi_B(data, K=K, hop=hop, max_nodes=max_nodes, verbose=verbose)
    raise ValueError(variant)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=['Cora', 'Chameleon'])
    ap.add_argument('--variant', choices=['A', 'C', 'B'], default='A')
    ap.add_argument('--trials', type=int, default=50)
    ap.add_argument('--K', type=int, default=5)
    ap.add_argument('--hop', type=int, default=2)
    ap.add_argument('--max_nodes', type=int, default=300)
    ap.add_argument('--outdir', default=None)
    args = ap.parse_args()
    outdir = args.outdir or f'results/dnp_{args.variant}'
    os.makedirs(outdir, exist_ok=True)
    print(f'DNP variant={args.variant}  datasets={args.datasets}  '
          f'trials={args.trials}  K={args.K} hop={args.hop}')

    lp_raw, diag_out = {}, {}
    for name in args.datasets:
        print(f'\n{"="*64}\n{name}\n{"="*64}')
        raw = DF.load_dataset(name)[0]
        in_feats = raw.x.size(1)
        data, splits = DF.prepare_data(raw)
        bounds, total = DF.segment_bounds(data)
        print('  segments: ' + ', '.join(f'{k}={v[1]-v[0]}' for k, v in bounds.items()))

        t0 = time.time()
        print(f'  computing per-node PH (variant {args.variant})...')
        phi = compute_variant(data, args.variant, args.K, args.hop, args.max_nodes, True)
        print(f'    Phi shape={phi.shape}  ({time.time()-t0:.0f}s)')
        feat = DF.build_edge_features(phi, data.total_edges)       # 3*D per edge
        fdim = feat.shape[1]

        # ---- §14 diagnostic = collapse gate (reuse feature_separability_auc) ----
        (tr, trf, va, vaf, te, tef) = splits
        y_te = np.concatenate([np.ones(len(te)), np.zeros(len(tef))])
        X_te = DF.build_edge_features(phi, np.concatenate([te, tef]))
        test_auc = np.mean([DF.feature_separability_auc(X_te, y_te, seed=s) for s in range(5)])
        exact_pi = DF.load_pi_cache(name, total)
        lo_p, hi_p = bounds['test_pos']; lo_n, hi_n = bounds['test_neg']
        Xpi_te = np.concatenate([exact_pi[lo_p:hi_p], exact_pi[lo_n:hi_n]])
        pi_test_auc = np.mean([DF.feature_separability_auc(Xpi_te, y_te, seed=s) for s in range(5)])
        gate = 'PASS' if test_auc > 0.55 else 'COLLAPSE'
        diag_out[name] = {'dnp_test_auc': float(test_auc),
                          'exactPI_test_auc': float(pi_test_auc), 'gate': gate}
        print(f'  COLLAPSE GATE: DNP test-AUC={test_auc:.4f}  '
              f'(exact-PI test-AUC={pi_test_auc:.4f})  -> {gate}')

        # ---- LP: no-PI / exact-PI / DNP ----
        pi_dim = exact_pi.shape[1]
        for mode, arr, d, tag in [('none', None, pi_dim, 'no-PI'),
                                  ('pi', exact_pi, pi_dim, 'exact-PI'),
                                  ('hks', feat, fdim, f'DNP_{args.variant}')]:
            aucs = [DF.run_one_trial(data, arr, d, mode, in_feats, seed=s, dev=DF.device)
                    for s in range(args.trials)]
            lp_raw[(name, tag)] = aucs
            print(f'  LP [{tag}] mean={np.mean(aucs):.4f} std={np.std(aucs):.4f}')

    with open(os.path.join(outdir, 'lp_auc.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['dataset', 'variant', 'mean_auc', 'std_auc', 'n'])
        for (name, tag), a in lp_raw.items():
            a = np.array(a)
            w.writerow([name, tag, f'{a.mean():.6f}', f'{a.std():.6f}', len(a)])
    with open(os.path.join(outdir, 'collapse_gate.json'), 'w') as f:
        json.dump(diag_out, f, indent=2)
    print(f'\nOutputs -> {outdir}/')


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Smoke run (2 trials) on Cora to verify wiring**

Run:
```bash
conda activate tlcgnn
python diffusion_node_ph.py --datasets Cora --variant A --trials 2 --K 3 --hop 2 --outdir results/dnp_smoke
```
Expected: prints `Phi shape=(2708, 75)`, a `COLLAPSE GATE: ... -> PASS|COLLAPSE` line, three `LP [...]` lines, and writes `results/dnp_smoke/lp_auc.csv`. No exceptions.

- [ ] **Step 3: Commit**

```bash
git add diffusion_node_ph.py
git commit -m "feat(DNP): driver wiring DNP variants into the §15 LP pipeline + collapse gate

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Phase-1 GATE run (A + C, full trials, Cora + Chameleon)

**Files:** none (run + record)

- [ ] **Step 1: Run variant A and C on the smallest homo + hetero datasets**

Run (background, log to file):
```bash
conda activate tlcgnn
for V in A C; do
  python diffusion_node_ph.py --datasets Cora Chameleon --variant $V \
      --trials 50 --K 5 --hop 2 > results/dnp_${V}_phase1.log 2>&1
done
```
Expected: `results/dnp_A/lp_auc.csv`, `results/dnp_C/lp_auc.csv`, and `collapse_gate.json` for each.

- [ ] **Step 2: Read the gate + LP result**

Run:
```bash
cat results/dnp_A/collapse_gate.json results/dnp_C/collapse_gate.json
column -s, -t results/dnp_A/lp_auc.csv; column -s, -t results/dnp_C/lp_auc.csv
```
**GATE DECISION (record in the commit message):**
- A variant **passes** iff Chameleon `gate == PASS` (DNP test-AUC > 0.55 while exact-PI test-AUC ≈ 0.5). Same for C.
- A variant is **promising** iff its Chameleon LP mean beats exact-PI (≈0.943).
- If BOTH A and C COLLAPSE → STOP and report (the per-node construction did not fix the artifact; that is itself a finding for §16). Do not proceed to Phase 2 with a collapsing variant.

- [ ] **Step 3: Commit the Phase-1 artifacts + verdict**

```bash
git add results/dnp_A/ results/dnp_C/ results/dnp_*_phase1.log
git commit -m "exp(DNP): Phase-1 collapse gate + LP on Cora/Chameleon (A,C)

<one line: which variants PASS the collapse gate and beat exact-PI>

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 6 (optional, parallel): Variant B (bifiltration by slicing)

Only attempt if A or C passed and there is idle compute. **Higher risk** — gate on a quick smoke first.

**Files:** Modify `node_ph_features.py` (add `phi_B`); Test: add `test_phi_B_shape`.

- [ ] **Step 1: Implement `phi_B`** (HKS-time × Ollivier-Ricci bifiltration via linear slices)

```python
def phi_B(data, K: int = 5, hop: int = 2, max_nodes: int = 300,
          n_slices: int = 5, verbose: bool = False) -> np.ndarray:
    """(N, 25*K) bifiltration (HKS-time x Ricci) approximated by linear slicing.

    Combine the two filter axes as f_theta(i) = cos(theta)*HKS_k(i) + sin(theta)*ricci_node(i)
    over n_slices angles theta in [0, pi/2]; sublevel-PI per slice; concat.
    Ricci-per-node = mean incident Ollivier-Ricci curvature (loaddatas.compute_ricci_curvature)."""
    from diffusion_features import compute_hks_features
    from sg2dgm import PersistenceImager as pimg_mod
    import loaddatas as lds
    import copy, torch
    imager = pimg_mod.PersistenceImager(resolution=PI_RES)
    g = _full_graph(data)
    hks, _ = compute_hks_features(data, K=K, verbose=verbose)        # (N,K)
    # node-level ricci = mean incident edge curvature
    d2 = copy.copy(data); d2.edge_index = data.edge_index
    ricci = lds.compute_ricci_curvature(d2)                          # list (a,b,c)
    rnode = np.zeros(int(data.num_nodes)); cnt = np.zeros(int(data.num_nodes))
    for a, b, c in ricci:
        rnode[int(a)] += c; rnode[int(b)] += c; cnt[int(a)] += 1; cnt[int(b)] += 1
    rnode = np.where(cnt > 0, rnode / np.maximum(cnt, 1), 0.0)
    rnode = (rnode - rnode.min()) / (rnode.ptp() + 1e-9)
    thetas = np.linspace(0, np.pi / 2, n_slices)
    n = int(data.num_nodes)
    out = np.zeros((n, PI_RES * PI_RES * K), dtype=np.float64)
    for v in range(n):
        for k in range(K):
            # average the n_slices PIs into the k-th block (keeps dim = 25*K)
            acc = np.zeros(PI_RES * PI_RES)
            for th in thetas:
                filt = {nd: float(np.cos(th) * hks[nd, k] + np.sin(th) * rnode[nd])
                        for nd in g.nodes()}
                acc += _ego_sublevel_pi(g, v, hop, filt, imager, max_nodes)
            out[v, k*25:(k+1)*25] = acc / len(thetas)
        if verbose and (v+1) % 500 == 0:
            print(f'    phi_B {v+1}/{n}')
    return out
```

- [ ] **Step 2: Smoke test** `python -m pytest tests/test_node_ph_features.py -k phi_B -q` (add a shape test mirroring Task 2's). Expected PASS.
- [ ] **Step 3: Run** `python diffusion_node_ph.py --datasets Chameleon --variant B --trials 50 --K 5` ; record gate + LP.
- [ ] **Step 4: Commit** `git add node_ph_features.py tests/ results/dnp_B/ && git commit -m "feat(DNP): variant B bifiltration slicing + Chameleon result"`

---

## Phase 2 — full LP eval (passing variants × 9 datasets) + results §16

Only for variants that PASSED the Phase-1 collapse gate.

### Task 7: Full 9-dataset sweep

**Files:** none (run)

- [ ] **Step 1: Launch** (one SLURM job or background process per passing variant; datasets are the §14 set)

```bash
conda activate tlcgnn
DATASETS="Cora Citeseer PubMed Photo Chameleon Squirrel Texas Cornell Wisconsin ChChMiner"
for V in A C; do   # only those that PASSED
  python diffusion_node_ph.py --datasets $DATASETS --variant $V \
     --trials 50 --K 5 --hop 2 > results/dnp_${V}_full.log 2>&1
done
```
Expected: `results/dnp_A/lp_auc.csv` (and C) extended to all datasets. Squirrel is the slowest (largest); start it last or give it its own job.

- [ ] **Step 2: Verify** every dataset row is present and gate values recorded:
```bash
column -s, -t results/dnp_A/lp_auc.csv
cat results/dnp_A/collapse_gate.json
```
Expected: 10 datasets × 3 variants (no-PI/exact-PI/DNP_A) rows; gate JSON per dataset.

### Task 8: Aggregate vs baselines + write §16

**Files:** Modify `docs/specs/2026-06-21-tda-conference-results.md` (append §16). (This results doc is a deliverable, additive edit is expected.)

- [ ] **Step 1: Build the comparison table** — join DNP results with existing baselines:
  - exact-PI, no-PI, node-HKS(§15): `results/diffusion_feat_A/lp_auc.csv`
  - PDGNN: from the §12/§14 results in `docs/specs/2026-06-21-tda-conference-results.md`
  - DNP_A / DNP_C: `results/dnp_*/lp_auc.csv`

- [ ] **Step 2: Write §16** with: the per-node-PH construction (A/C/B), the collapse-gate table (DNP test-AUC vs exact-PI test-AUC — the proof it does not collapse), the LP comparison (hetero focus), and the honest verdict (genuine-topology win, tie, or null — whichever the numbers show). Cross-reference §14 (problem) and §15 (spectral cousin).

- [ ] **Step 3: Commit + push**

```bash
git add docs/specs/2026-06-21-tda-conference-results.md results/dnp_*/
git commit -m "results(DNP): §16 diffusion-filtered node-level persistence — collapse gate + LP

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
git push tda HEAD:refs/heads/main
```

---

## Phase 3 — multi-filtration neural engine (CONDITIONAL)

Only if Phase 2 shows a genuine signal worth accelerating. Sketch (own mini-plan when reached):
- Extend `Knowledge_Distillation/pdgnn_modern.PDGNN` to predict the per-node diagram(s) under the HKS filtration (input filter = HKS_t per node), trained against the exact `phi_A` diagrams with the existing `train_pdgnn_lp._bipartite_loss` (Wasserstein/Hungarian).
- Validate predicted-vs-exact PI agreement, then re-run LP with the neural features → report speedup (target ~100× per the PDGNN paper) at matched AUC.
- This is deferred; do not start until Phase 2 lands.

---

## Self-Review

**1. Spec coverage:**
- Design "Method A/B/C" → Tasks 1 (A), 2 (C), 6 (B). ✓
- Design "anti-artifact / collapse test" → Task 3 (unit) + Task 4/5 (`collapse_gate.json`, the `run_diagnostic`-style test-AUC). ✓
- Design "exact-first" → Phases 1–2 are exact; neural is Phase 3 conditional. ✓
- Design "LP eval vs exact-PI/PDGNN/node-HKS/no-PI" → Task 4 runs no-PI/exact-PI/DNP; Task 8 joins node-HKS + PDGNN from existing CSV/doc. ✓
- Design "success criterion / null acceptable" → Task 5 GATE DECISION + Task 8 honest verdict. ✓
- Design "reuse, additive only" → all new files; no existing file modified except the results doc (a deliverable). ✓
- Thread GEN (③) → explicitly out of scope. ✓

**2. Placeholder scan:** No TBD/TODO; every code step has complete code; commands have expected output. Phase 3 is intentionally a conditional sketch (gated), not a placeholder in the active path. ✓

**3. Type consistency:** `phi_A/phi_C/phi_B` all return `(N, D)` numpy; `DF.build_edge_features` consumes `(N,D)` → `(n_edges, 3D)`; `DF.run_one_trial(data, feat_array, feat_dim, mode, in_feats, seed, dev)` and `DF.LPModel` mode `'hks'` consume that edge array — matches `diffusion_features.py:435,365`. `_ego_sublevel_pi` returns `(25,)`; `phi_A` concatenates K of them → `25*K`. ✓ `DF.device`, `DF.load_pi_cache`, `DF.segment_bounds`, `DF.prepare_data`, `DF.load_dataset` all exist (verified in `diffusion_features.py`). ✓

---

## Execution Handoff

Plan saved to `docs/superpowers/plans/2026-05-28-diffusion-node-ph.md`. The user requested **executing-plans** (inline). Phase 1 is the critical path with the collapse gate; if both A and C collapse, that is a reportable §16 finding and Phase 2 is skipped.
