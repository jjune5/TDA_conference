# Chromatic-PDGNN Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a neural approximator of the 0-dim image/kernel/cokernel ("chromatic") persistence of the target-type-subgraph inclusion on heterogeneous graphs, and test whether cross-type mingling (kernel persistence) carries genuine node-classification signal under full controls.

**Architecture:** Exact union-find labeler (`chromatic_labels.py`, pure numpy) generates ground-truth 0-dim image/kernel/cokernel/ordinary barcodes per ego-graph. `ChromaticPDGNN` (extends `Knowledge_Distillation.pdgnn_modern.PDGNN`: input dim 2 = [HKS filter ∥ color indicator], three edge-heads) is trained to predict the three diagrams (per-leg Hungarian loss). At inference it emits per-target-node persistence images for the three legs. A Stage-2 driver runs a controls ladder (none / chromatic / achromatic / shuffled / random) across GCN/HAN/HGT backbones, first on a synthetic positive-control where the label is by-construction a cross-type bridging pattern, then on real hetero datasets.

**Tech Stack:** Python, numpy (labeler), PyTorch + PyG 2.x + torch_scatter (model, env `tlcgnn`), gudhi only for cross-checks (not in the train/infer path), existing repo modules (`unified_filter`, `pdgnn_metapath`, `run_han_hgt`, `hetero_nc_pipeline`).

**Env / test command:** all Python runs use the `tlcgnn` conda env. Tests: `conda run -n tlcgnn python -m pytest tests/<file> -v` (if `conda run` is unavailable, `source $(conda info --base)/etc/profile.d/conda.sh && conda activate tlcgnn` first).

**Grounding (cite in code headers):** Chromatic-TDA (Cultrera di Montesano, Draganov, Edelsbrunner, Saghafian, arXiv 2406.04102) for the color framing; Cohen-Steiner, Edelsbrunner, Harer, Morozov "Persistent Homology for Kernels, Images, and Cokernels" (SODA 2009) for the image/kernel/cokernel algorithm. Point-cloud→graph port is ours. Spec: `docs/specs/2026-06-08-chromatic-pdgnn-hetero-design.md`.

---

## File Structure

| File | Responsibility |
|---|---|
| `hetero/chromatic_labels.py` | exact 0-dim ordinary/image/kernel/cokernel union-find barcodes for one ego (pure numpy, no torch/gudhi) |
| `tests/test_chromatic_labels.py` | TDD tests on tiny hand-computed graphs |
| `hetero/chromatic_pdgnn.py` | `ChromaticPDGNN` model + sample-gen + train + per-node 3-leg PI inference |
| `tests/test_chromatic_pdgnn.py` | smoke tests (shapes, forward/backward, overfit-tiny) |
| `hetero/synth_chromatic.py` | planted-bridge synthetic hetero generator (positive control) |
| `tests/test_synth_chromatic.py` | test that chromatic labels separate planted classes by construction |
| `hetero/run_chromatic.py` | Stage-2 driver: controls ladder × backbone × dataset, results table |

---

## Conventions for the labeler

- A graph is `(fv, ei, is_L)`: `fv` = (m,) float vertex filtration (lower-star HKS), `ei` = (2,E) int edge index (may be symmetric / contain duplicates / self-loops), `is_L` = (m,) bool (vertex in target-type subcomplex L).
- Lower-star: vertex enters at `fv[i]`; undirected edge (u,v) enters at `max(fv[u], fv[v])`.
- Processing order: vertices implicitly at their `fv`; edges sorted by `(t, min(u,v), max(u,v))`.
- Elder rule: on a merge, the component with the **larger birth** dies at the edge time `t`; ties → the one with the larger root index dies (deterministic).
- Essential classes (never die): death = `max_f` (= `fv.max()` unless overridden), matching `pdgnn_metapath._exact_epd`.
- Zero-persistence bars (`death <= birth`) are dropped.
- Output of `compute_chromatic_0dim` = dict with keys `ordinary`, `image`, `kernel`, `cokernel`, each an `(P,2)` float64 array (possibly empty `(0,2)`).

---

## Task 1: Union-find helpers + ordinary H0 barcode

**Files:**
- Create: `hetero/chromatic_labels.py`
- Test: `tests/test_chromatic_labels.py`

- [ ] **Step 1: Write failing test for edge canonicalization + ordinary H0**

```python
# tests/test_chromatic_labels.py
import numpy as np
import pytest
from hetero.chromatic_labels import _unique_undirected_edges, compute_chromatic_0dim


def _as_set(arr, nd=6):
    return {(round(float(b), nd), round(float(d), nd)) for b, d in np.asarray(arr).reshape(-1, 2)}


def test_unique_undirected_edges_dedup_and_selfloops():
    ei = np.array([[0, 1, 1, 0, 2], [1, 0, 2, 0, 2]])  # (0,1),(1,0)dup,(1,2),(0,0)self,(2,2)self
    assert _unique_undirected_edges(ei) == [(0, 1), (1, 2)]


def test_ordinary_h0_path_graph():
    # path a-b-c, fv = [0, 1, 2], no types -> is_L all False (ordinary still defined on K)
    fv = np.array([0.0, 1.0, 2.0])
    ei = np.array([[0, 1], [1, 2]])
    is_L = np.array([False, False, False])
    out = compute_chromatic_0dim(fv, ei, is_L, max_f=2.0)
    # vertices born 0,1,2. edge(a,b)@1 merges b(born1) into a(born0): bar (1,1) zero-pers dropped.
    # edge(b,c)@2 merges c(born2) into {a,b}(born0): bar (2,2) zero dropped. one essential (0,2).
    assert _as_set(out['ordinary']) == {(0.0, 2.0)}
```

- [ ] **Step 2: Run to verify it fails**

Run: `conda run -n tlcgnn python -m pytest tests/test_chromatic_labels.py -v`
Expected: FAIL (ImportError: cannot import name `_unique_undirected_edges`).

- [ ] **Step 3: Implement helpers + ordinary barcode**

```python
# hetero/chromatic_labels.py
"""Exact 0-dim image/kernel/cokernel ("chromatic") persistence for one ego-graph.

Inclusion L (target-type subgraph) into K (full multi-type ego), lower-star HKS
filtration. 0-dim only. Used ONLY to generate one-time training labels for
ChromaticPDGNN (exact never used as a downstream feature -> PDGNN-only policy).

Grounding: image/kernel/cokernel persistence = Cohen-Steiner, Edelsbrunner, Harer,
Morozov, "Persistent Homology for Kernels, Images, and Cokernels" (SODA 2009);
color/chromatic framing = Cultrera di Montesano et al., "Chromatic TDA" (arXiv
2406.04102). Point-cloud -> graph port is ours.
"""
from __future__ import annotations
import numpy as np


def _unique_undirected_edges(ei):
    """(2,E) possibly symmetric/dup/self-loop -> sorted unique [(u,v), ...] with u<v."""
    seen, out = set(), []
    a_arr, b_arr = np.asarray(ei[0]).tolist(), np.asarray(ei[1]).tolist()
    for a, b in zip(a_arr, b_arr):
        if a == b:
            continue
        u, v = (a, b) if a < b else (b, a)
        if (u, v) in seen:
            continue
        seen.add((u, v))
        out.append((u, v))
    return out


def _find(p, x):
    while p[x] != x:
        p[x] = p[p[x]]
        x = p[x]
    return x


def _sorted_edges(fv, ei):
    """[(t, u, v), ...] sorted by (t, u, v); t = max endpoint filtration (lower-star)."""
    edges = _unique_undirected_edges(ei)
    et = [(max(float(fv[u]), float(fv[v])), u, v) for (u, v) in edges]
    et.sort()
    return et


def compute_chromatic_0dim(fv, ei, is_L, max_f=None):
    """Return dict ordinary/image/kernel/cokernel, each (P,2) float64.

    Single-pass union-find over K with simultaneous bookkeeping for image/cokernel,
    plus an L union-find and a K-on-L union-find for kernel. See module tests for the
    authoritative hand-computed barcodes that define correct behavior.
    """
    fv = np.asarray(fv, dtype=np.float64)
    m = len(fv)
    is_L = np.asarray(is_L, dtype=bool)
    if max_f is None:
        max_f = float(fv.max()) if m else 1.0
    et = _sorted_edges(fv, ei)

    ordinary = _ordinary_barcode(fv, et, m, max_f)
    return {
        'ordinary': _to_arr(ordinary),
        'image': _to_arr([]),       # filled in Task 2
        'kernel': _to_arr([]),      # filled in Task 3
        'cokernel': _to_arr([]),    # filled in Task 4
    }


def _ordinary_barcode(fv, et, m, max_f):
    p = list(range(m))
    birth = [float(x) for x in fv]
    bars = []
    for (t, u, v) in et:
        ru, rv = _find(p, u), _find(p, v)
        if ru == rv:
            continue
        # elder rule: larger birth dies; tie -> larger root index dies
        if (birth[ru], ru) >= (birth[rv], rv):
            dead, surv = ru, rv
        else:
            dead, surv = rv, ru
        if t > birth[dead]:
            bars.append((birth[dead], t))
        p[dead] = surv
        birth[surv] = min(birth[ru], birth[rv])
    for r in set(_find(p, i) for i in range(m)):
        if max_f > birth[r]:
            bars.append((birth[r], max_f))
    return bars


def _to_arr(bars):
    return np.array(bars, dtype=np.float64).reshape(-1, 2) if bars else np.zeros((0, 2), np.float64)
```

- [ ] **Step 4: Run to verify pass**

Run: `conda run -n tlcgnn python -m pytest tests/test_chromatic_labels.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add hetero/chromatic_labels.py tests/test_chromatic_labels.py
git commit -m "feat(chromatic): union-find helpers + ordinary 0-dim barcode (TDD)"
```

---

## Task 2: Image persistence (im H0(L)→H0(K))

**Files:**
- Modify: `hetero/chromatic_labels.py` (add `_chromatic_barcodes`, wire `image`)
- Test: `tests/test_chromatic_labels.py`

**Reference example (hand-computed, the source of truth).** Vertices a=0,b=1 target-type (L), c=2 non-target. `fv=[0,0,1]`. Edges (a,c),(b,c) both enter at t=1; no (a,b) edge.
- **image** = `{(0,1), (0,2)}`? No — max_f here = 1, essential death = 1. a,b each start an image class born@0; at t=1 they merge in K (via c) → one image bar (0,1), the other essential (0,1). So image = `{(0,1)}` finite + `{(0,1)}` essential = both (0,1). Dedup as multiset → two bars at (0,1). Since both equal and death==max_f for essential, represent as `[(0,1),(0,1)]`.
- For a cleaner discriminating test use `max_f=2`: then essential image bar = (0,2). image = `{(0,1),(0,2)}`.

- [ ] **Step 1: Write failing test for image**

```python
def test_image_two_L_bridged_by_nonL():
    fv = np.array([0.0, 0.0, 1.0])      # a,b target; c non-target
    ei = np.array([[0, 2, 1, 2], [2, 0, 2, 1]])  # (a,c),(b,c) symmetric
    is_L = np.array([True, True, False])
    out = compute_chromatic_0dim(fv, ei, is_L, max_f=2.0)
    # a,b image classes born@0; merge in K at t=1 -> one image dies (0,1), one essential (0,2)
    assert _as_set(out['image']) == {(0.0, 1.0), (0.0, 2.0)}
```

- [ ] **Step 2: Run to verify it fails**

Run: `conda run -n tlcgnn python -m pytest tests/test_chromatic_labels.py::test_image_two_L_bridged_by_nonL -v`
Expected: FAIL (image is empty).

- [ ] **Step 3: Implement joint barcode pass with image leg**

Replace the body of `compute_chromatic_0dim` after `et = _sorted_edges(...)` so it calls a single joint pass that fills ordinary+image (kernel/cokernel added in later tasks):

```python
def compute_chromatic_0dim(fv, ei, is_L, max_f=None):
    fv = np.asarray(fv, dtype=np.float64)
    m = len(fv)
    is_L = np.asarray(is_L, dtype=bool)
    if max_f is None:
        max_f = float(fv.max()) if m else 1.0
    et = _sorted_edges(fv, ei)
    return _chromatic_barcodes(fv, et, is_L, m, max_f)


def _chromatic_barcodes(fv, et, is_L, m, max_f):
    INF = None
    # K union-find: birth (ordinary), hasL flag, image-birth (elder L-birth) per root
    Kp = list(range(m))
    Kbirth = [float(x) for x in fv]
    KhasL = [bool(is_L[i]) for i in range(m)]
    Kimg = [float(fv[i]) if is_L[i] else INF for i in range(m)]
    ordinary, image, kernel, cokernel = [], [], [], []

    for (t, u, v) in et:
        ru, rv = _find(Kp, u), _find(Kp, v)
        if ru == rv:
            continue
        if (Kbirth[ru], ru) >= (Kbirth[rv], rv):
            dead, surv = ru, rv
        else:
            dead, surv = rv, ru
        # ordinary
        if t > Kbirth[dead]:
            ordinary.append((Kbirth[dead], t))
        hu, hv = KhasL[ru], KhasL[rv]
        # image / cokernel legs (kernel handled in Task 3)
        if hu and hv:
            iu, iv = Kimg[ru], Kimg[rv]
            img_dead, img_surv = (iu, iv) if iu >= iv else (iv, iu)
            if t > img_dead:
                image.append((img_dead, t))
            new_img = img_surv
        elif hu or hv:
            new_img = Kimg[ru] if hu else Kimg[rv]
        else:
            new_img = INF
        Kp[dead] = surv
        Kbirth[surv] = min(Kbirth[ru], Kbirth[rv])
        KhasL[surv] = hu or hv
        Kimg[surv] = new_img

    # essential ordinary + essential image
    for r in set(_find(Kp, i) for i in range(m)):
        if max_f > Kbirth[r]:
            ordinary.append((Kbirth[r], max_f))
        if KhasL[r] and Kimg[r] is not None and max_f > Kimg[r]:
            image.append((Kimg[r], max_f))

    return {'ordinary': _to_arr(ordinary), 'image': _to_arr(image),
            'kernel': _to_arr(kernel), 'cokernel': _to_arr(cokernel)}
```

(The standalone `_ordinary_barcode` from Task 1 may stay for unit reuse, but `_chromatic_barcodes` is now the single source. Keep `test_ordinary_h0_path_graph` passing — the path graph has all `is_L=False`, image stays empty, ordinary identical.)

- [ ] **Step 4: Run all label tests**

Run: `conda run -n tlcgnn python -m pytest tests/test_chromatic_labels.py -v`
Expected: PASS (ordinary path test + image test + edge-dedup test).

- [ ] **Step 5: Commit**

```bash
git add hetero/chromatic_labels.py tests/test_chromatic_labels.py
git commit -m "feat(chromatic): 0-dim image persistence leg (TDD)"
```

---

## Task 3: Kernel persistence (the chromatic heart)

**Files:**
- Modify: `hetero/chromatic_labels.py` (add kernel tracking to `_chromatic_barcodes`)
- Test: `tests/test_chromatic_labels.py`

**Definition.** `rank(ker φ_t) = #components(L_t) − #(K-components containing an L-vertex at t)`. A kernel class is born at the time two L-bearing K-components merge while their L-parts are still separate **in L** (cross-type bridge); it dies when those L-parts later merge **in L**. Essential kernel deaths → `max_f`.

**Reference examples (source of truth):**
1. From Task 2 graph (`fv=[0,0,1]`, edges (a,c),(b,c), is_L=[T,T,F], max_f=2): a,b never connect in L (no a-b edge); they connect in K at t=1 → kernel born@1, never dies in L → essential `(1,2)`. **kernel = {(1,2)}**.
2. Add an a-b edge at high filtration: `fv=[0,0,1]`, vertices a=0,b=1 (L), c=2 (non-L); edges (a,c)@1,(b,c)@1,(a,b)@max(0,0)=0. Now a,b merge in L at t=0 (before any cross-type bridge), so they are L-connected before/at the K-bridge → **no kernel class**. **kernel = {}**. (Tie at t=0 vs vertices: edges processed after vertices; (a,b)@0 merges L-comps at t=0; the K-bridge at t=1 connects an already-L-connected pair → no kernel birth.)

- [ ] **Step 1: Write failing tests for kernel**

```python
def test_kernel_born_at_crosstype_bridge_essential():
    fv = np.array([0.0, 0.0, 1.0])
    ei = np.array([[0, 2, 1, 2], [2, 0, 2, 1]])   # (a,c),(b,c); NO (a,b)
    is_L = np.array([True, True, False])
    out = compute_chromatic_0dim(fv, ei, is_L, max_f=2.0)
    assert _as_set(out['kernel']) == {(1.0, 2.0)}


def test_kernel_absent_when_L_connects_first():
    fv = np.array([0.0, 0.0, 1.0])
    ei = np.array([[0, 2, 1, 2, 0, 1],
                   [2, 0, 2, 1, 1, 0]])           # adds (a,b)@0
    is_L = np.array([True, True, False])
    out = compute_chromatic_0dim(fv, ei, is_L, max_f=2.0)
    assert _as_set(out['kernel']) == set()
```

- [ ] **Step 2: Run to verify failure**

Run: `conda run -n tlcgnn python -m pytest tests/test_chromatic_labels.py -k kernel -v`
Expected: FAIL (kernel empty).

- [ ] **Step 3: Add kernel tracking**

Augment `_chromatic_barcodes`: maintain an L union-find `Lp` with `Lbirth`, and per K-root a representative L-vertex `Krep`. Track a list of "open kernel classes" as `(birth_t, Lroot_a, Lroot_b)` keyed by the pair of L-components they bridge; resolve deaths when an L-edge merges those L-components. Implementation:

```python
    # --- add near the top of _chromatic_barcodes, after Kimg init ---
    Lp = list(range(m))
    Lbirth = [float(fv[i]) if is_L[i] else None for i in range(m)]
    Krep = [i if is_L[i] else None for i in range(m)]   # an L-vertex rep of each K-component
    # open kernel births: map frozenset({Lroot_i, Lroot_j}) -> birth_t (stack-free; elder by birth)
    open_ker = {}   # (min(Lr,Lr2), max(...)) -> birth_t

    # --- inside the loop, REPLACE the merge body to also handle kernel ---
    # (compute hu,hv as before)
    # 1) detect kernel BIRTH: both K-comps L-bearing AND their L-reps in different Lp comps
    if hu and hv:
        la, lb = Krep[ru], Krep[rv]
        rla, rlb = _find(Lp, la), _find(Lp, lb)
        if rla != rlb:
            key = (min(rla, rlb), max(rla, rlb))
            if key not in open_ker:        # first cross-type connection of this L-pair
                open_ker[key] = t          # kernel born at t
    # 2) if this edge is also an L-edge (both endpoints in L), it merges L-comps -> possible kernel DEATH
    if is_L[u] and is_L[v]:
        rlu, rlv = _find(Lp, u), _find(Lp, v)
        if rlu != rlv:
            key = (min(rlu, rlv), max(rlu, rlv))
            if key in open_ker:            # these were already K-bridged -> kernel dies now
                b = open_ker.pop(key)
                if t > b:
                    kernel.append((b, t))
            # elder rule on L
            if (Lbirth[rlu], rlu) >= (Lbirth[rlv], rlv):
                dead_l, surv_l = rlu, rlv
            else:
                dead_l, surv_l = rlv, rlu
            Lp[dead_l] = surv_l
            Lbirth[surv_l] = min(Lbirth[rlu], Lbirth[rlv])
    # 3) maintain Krep on K-merge (survivor keeps an L-rep if either side had one)
    #    (do this where Kp[dead]=surv is set:)
    #    new_rep = Krep[ru] if Krep[ru] is not None else Krep[rv]; Krep[surv] = new_rep
```

Wire `new_rep`/`Krep[surv]` into the existing merge-commit block. Essential kernel classes (still in `open_ker` after the loop) die at `max_f`:

```python
    # after the loop, before assembling the dict:
    for key, b in open_ker.items():
        if max_f > b:
            kernel.append((b, max_f))
```

> **Executor note:** kernel is the correctness-critical leg. The two hand-computed tests above are authoritative — if the algorithm above disagrees with them, fix the algorithm to satisfy the tests (re-derive the birth/death pairing), not the tests. Add a third hand example (two separate L-pairs bridged at different times) if the first two underdetermine the pairing.

- [ ] **Step 4: Run kernel tests**

Run: `conda run -n tlcgnn python -m pytest tests/test_chromatic_labels.py -v`
Expected: PASS (all label tests incl. both kernel cases).

- [ ] **Step 5: Commit**

```bash
git add hetero/chromatic_labels.py tests/test_chromatic_labels.py
git commit -m "feat(chromatic): 0-dim kernel persistence leg = cross-type mingling (TDD)"
```

---

## Task 4: Cokernel persistence + gudhi cross-check for ordinary

**Files:**
- Modify: `hetero/chromatic_labels.py` (cokernel leg)
- Test: `tests/test_chromatic_labels.py`

**Cokernel.** `coker φ_t = H0(K_t)/im φ_t` = K-components with no L-vertex. A cokernel class is born when a non-L vertex starts an L-free K-component, and dies when that component merges into an L-bearing component (absorbed) or merges with another L-free component (elder rule).

**Reference example.** `fv=[0,0,1]`, a,b (L), c (non-L), edges (a,c)@1,(b,c)@1: c starts an L-free comp born@1; at t=1 (a,c) absorbs it into an L-bearing comp → cokernel bar (1,1) zero-persistence → dropped. **cokernel = {}**. For a non-trivial case: two non-L vertices c=2,d=3 with `fv=[0,0,2,2]`, edges (c,d)@2 then (a,c)@... — see test.

- [ ] **Step 1: Write failing test for cokernel**

```python
def test_cokernel_two_nonL_merge_then_absorbed():
    # a(L)=0 ; c,d non-L = 2,3 ; fv=[0, _, 1, 1]; b unused -> drop b, use m=3 relabel:
    fv = np.array([0.0, 1.0, 1.0])     # a=0 (L), c=1, d=2 (non-L)
    ei = np.array([[1, 0], [2, 1]])    # (c,d)@1 ; (a,c)@1
    is_L = np.array([True, False, False])
    out = compute_chromatic_0dim(fv, ei, is_L, max_f=3.0)
    # c,d born@1; (c,d)@1 merge -> cokernel (1,1) dropped; (a,c)@1 absorbs {c,d} into a -> (1,1) dropped
    # no surviving L-free comp -> cokernel essential none
    assert _as_set(out['cokernel']) == set()


def test_cokernel_essential_isolated_nonL():
    fv = np.array([0.0, 1.0])          # a=0 (L), e=1 (non-L), NO edge
    ei = np.zeros((2, 0), dtype=int)
    is_L = np.array([True, False])
    out = compute_chromatic_0dim(fv, ei, is_L, max_f=3.0)
    assert _as_set(out['cokernel']) == {(1.0, 3.0)}   # lone non-L comp essential
```

- [ ] **Step 2: Run to verify failure**

Run: `conda run -n tlcgnn python -m pytest tests/test_chromatic_labels.py -k cokernel -v`
Expected: FAIL.

- [ ] **Step 3: Implement cokernel leg**

In the merge body of `_chromatic_barcodes`, add cokernel deaths:

```python
    # cokernel: track L-free comp birth = Kbirth of the L-free root
    if hu and hv:
        pass  # no cokernel event
    elif hu or hv:
        lfree = ru if not hu else rv
        if t > Kbirth[lfree]:
            cokernel.append((Kbirth[lfree], t))   # L-free class absorbed into image
    else:
        # both L-free: younger cokernel dies
        if t > Kbirth[dead]:
            cokernel.append((Kbirth[dead], t))
```

Essential cokernel after loop:

```python
    for r in set(_find(Kp, i) for i in range(m)):
        if not KhasL[r] and max_f > Kbirth[r]:
            cokernel.append((Kbirth[r], max_f))
```

- [ ] **Step 4: Add a gudhi cross-check test for ORDINARY (sanity that union-find matches a library)**

```python
def test_ordinary_matches_gudhi_random():
    import gudhi
    rng = np.random.RandomState(0)
    for _ in range(5):
        m = rng.randint(5, 12)
        fv = rng.rand(m).astype(np.float64)
        # random edges
        E = rng.randint(m, 2 * m)
        ei = rng.randint(0, m, size=(2, E))
        is_L = rng.rand(m) > 0.5
        out = compute_chromatic_0dim(fv, ei, is_L, max_f=float(fv.max()))
        # gudhi ordinary H0 via lower-star on a SimplexTree
        st = gudhi.SimplexTree()
        for i in range(m):
            st.insert([i], filtration=float(fv[i]))
        for (u, v) in _unique_undirected_edges(ei):
            st.insert([int(u), int(v)], filtration=float(max(fv[u], fv[v])))
        st.compute_persistence(persistence_dim_max=False)
        g0 = st.persistence_intervals_in_dimension(0)
        # replace gudhi inf with max_f, drop zero-persistence
        gset = set()
        mx = float(fv.max())
        for b, d in g0:
            d = mx if d == float('inf') else d
            if d > b:
                gset.add((round(b, 6), round(d, 6)))
        assert _as_set(out['ordinary']) == gset
```

- [ ] **Step 5: Run all + commit**

Run: `conda run -n tlcgnn python -m pytest tests/test_chromatic_labels.py -v`
Expected: PASS (all, incl. gudhi cross-check).

```bash
git add hetero/chromatic_labels.py tests/test_chromatic_labels.py
git commit -m "feat(chromatic): 0-dim cokernel leg + gudhi ordinary cross-check (TDD)"
```

---

## Task 5: `ChromaticPDGNN` model (input dim 2, three legs)

**Files:**
- Create: `hetero/chromatic_pdgnn.py`
- Test: `tests/test_chromatic_pdgnn.py`

- [ ] **Step 1: Write failing smoke test (shapes + backward)**

```python
# tests/test_chromatic_pdgnn.py
import numpy as np
import torch
from hetero.chromatic_pdgnn import ChromaticPDGNN


def test_forward_shapes_and_backward():
    torch.manual_seed(0)
    m, E = 6, 8
    filt = torch.rand(m, 1)
    color = torch.randint(0, 2, (m, 1)).float()
    ei = torch.randint(0, m, (2, E))
    model = ChromaticPDGNN(hidden_dim=16, num_layers=2)
    out = model(filt, color, ei)
    assert set(out.keys()) == {'image', 'kernel', 'cokernel'}
    for leg in out.values():
        assert leg.shape == (E, 2)
    loss = sum(v.sum() for v in out.values())
    loss.backward()
    assert next(model.parameters()).grad is not None
```

- [ ] **Step 2: Run to verify failure**

Run: `conda run -n tlcgnn python -m pytest tests/test_chromatic_pdgnn.py::test_forward_shapes_and_backward -v`
Expected: FAIL (ImportError).

- [ ] **Step 3: Implement the model**

```python
# hetero/chromatic_pdgnn.py
"""ChromaticPDGNN: extends PDGNN to predict the 0-dim image/kernel/cokernel
persistence of the target-type-subgraph inclusion. Type enters the MECHANISM via a
per-node color indicator concatenated to the HKS filter (input dim 2), and three
edge-heads emit one (birth,death) diagram per leg.

Reuses Knowledge_Distillation.pdgnn_modern.PDGNNLayer (SUM(+)MIN union-find emulation).
Exact labels from hetero.chromatic_labels (PDGNN-only: exact never a downstream feature).
"""
from __future__ import annotations
import os, sys
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Knowledge_Distillation.pdgnn_modern import PDGNNLayer


class ChromaticPDGNN(nn.Module):
    def __init__(self, hidden_dim: int = 32, num_layers: int = 3, dropout: float = 0.0):
        super().__init__()
        self.dropout = dropout
        dims = [2] + [hidden_dim] * num_layers      # input = [filt, color]
        self.layers = nn.ModuleList(
            PDGNNLayer(dims[i], dims[i + 1]) for i in range(num_layers))
        def _head():
            return nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim), nn.PReLU(hidden_dim),
                nn.Linear(hidden_dim, 2))
        self.head_image = _head()
        self.head_kernel = _head()
        self.head_cokernel = _head()

    def forward(self, filt_value: torch.Tensor, color: torch.Tensor,
                edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor] = None,
                pred_edges: Optional[torch.Tensor] = None):
        x = torch.cat([filt_value, color], dim=-1)   # (N,2)
        for layer in self.layers:
            x = layer(x, edge_index, edge_weight)
            if self.dropout > 0:
                x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        if pred_edges is None:
            pred_edges = edge_index
        h = torch.cat([x[pred_edges[0]], x[pred_edges[1]]], dim=-1)
        return {'image': self.head_image(h),
                'kernel': self.head_kernel(h),
                'cokernel': self.head_cokernel(h)}
```

- [ ] **Step 4: Run to verify pass**

Run: `conda run -n tlcgnn python -m pytest tests/test_chromatic_pdgnn.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add hetero/chromatic_pdgnn.py tests/test_chromatic_pdgnn.py
git commit -m "feat(chromatic): ChromaticPDGNN model (color input + 3 leg heads)"
```

---

## Task 6: Sample generation + training (per-leg Hungarian)

**Files:**
- Modify: `hetero/chromatic_pdgnn.py` (add `gen_chromatic_samples`, `train_chromatic_pdgnn`)
- Test: `tests/test_chromatic_pdgnn.py`

- [ ] **Step 1: Write failing overfit-tiny test**

```python
def test_train_overfits_tiny():
    from hetero.chromatic_pdgnn import gen_chromatic_samples, train_chromatic_pdgnn
    import networkx as nx
    # 2 separate target nodes bridged by a hub (kernel signal present)
    g = nx.Graph(); g.add_edges_from([(0, 2), (1, 2)])     # 0,1 target ; 2 hub
    ntype = np.array([0, 0, 1])                            # type0 = target
    hks = np.random.RandomState(0).rand(3, 1).astype(np.float32)
    samples = gen_chromatic_samples(g, hks, ntype, target_type=0, hop=2,
                                    max_nodes=20, n_samples=3, seed=0)
    assert len(samples) >= 1
    model = train_chromatic_pdgnn(samples, hidden=16, layers=2, epochs=20, verbose=False)
    assert model is not None
```

- [ ] **Step 2: Run to verify failure**

Run: `conda run -n tlcgnn python -m pytest tests/test_chromatic_pdgnn.py::test_train_overfits_tiny -v`
Expected: FAIL (ImportError).

- [ ] **Step 3: Implement gen + train (reuse ego machinery from pdgnn_metapath)**

```python
# append to hetero/chromatic_pdgnn.py
import networkx as nx
from Knowledge_Distillation.train_pdgnn_lp import _bipartite_loss
from hetero.pdgnn_metapath import _ego_filt_edges
from hetero.chromatic_labels import compute_chromatic_0dim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_LEGS = ('image', 'kernel', 'cokernel')


def gen_chromatic_samples(g, hks, ntype, target_type, hop, max_nodes, n_samples, seed=0):
    """Sample (node, scale) egos on the UNIFIED graph -> per-leg exact labels.
    Returns [(filt(m,1), color(m,1), ei(2,E), {leg: gt(P,2)}), ...]."""
    rng = np.random.RandomState(seed)
    n, K = hks.shape
    ntype = np.asarray(ntype)
    nodes = rng.choice(n, size=min(n_samples, n), replace=False)
    samples = []
    for v in nodes:
        for k in range(K):
            node_filt = {nd: float(hks[nd, k]) for nd in g.nodes()}
            res = _ego_filt_edges(g, int(v), hop, node_filt, max_nodes)
            if res is None:
                continue
            filt, ei, nodelist = res
            if ei.shape[1] == 0:
                continue
            is_L = (ntype[np.array(nodelist)] == target_type)
            labels = compute_chromatic_0dim(filt, ei, is_L, max_f=float(filt.max()))
            if all(labels[L].shape[0] == 0 for L in _LEGS):
                continue
            color = is_L.astype(np.float32).reshape(-1, 1)
            samples.append((filt.reshape(-1, 1).astype(np.float32), color,
                            ei.astype(np.int64),
                            {L: labels[L].astype(np.float32) for L in _LEGS}))
    return samples


def train_chromatic_pdgnn(samples, hidden=32, layers=3, epochs=30, lr=1e-3,
                          seed=1234, verbose=True):
    torch.manual_seed(seed); np.random.seed(seed)
    model = ChromaticPDGNN(hidden_dim=hidden, num_layers=layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(1, epochs + 1):
        model.train(); order = np.random.permutation(len(samples)); losses = []
        for idx in order:
            filt, color, ei, gts = samples[idx]
            ft = torch.tensor(filt, device=device)
            ct = torch.tensor(color, device=device)
            et = torch.tensor(ei, device=device)
            opt.zero_grad()
            pred = model(ft, ct, et)
            loss = 0.0
            for L in _LEGS:
                gt = torch.tensor(gts[L], device=device)
                if gt.shape[0] == 0:
                    continue
                loss = loss + _bipartite_loss(pred[L], gt)
            if not torch.is_tensor(loss) or not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); losses.append(float(loss))
        if verbose and (ep % 5 == 0 or ep == 1):
            print(f'    chromatic-pdgnn ep {ep:3d} loss={np.mean(losses) if losses else float("nan"):.4f}')
    return model
```

- [ ] **Step 4: Run to verify pass**

Run: `conda run -n tlcgnn python -m pytest tests/test_chromatic_pdgnn.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add hetero/chromatic_pdgnn.py tests/test_chromatic_pdgnn.py
git commit -m "feat(chromatic): sample-gen + per-leg Hungarian training"
```

---

## Task 7: Per-node 3-leg PI inference

**Files:**
- Modify: `hetero/chromatic_pdgnn.py` (add `predict_node_chromatic_pi`)
- Test: `tests/test_chromatic_pdgnn.py`

- [ ] **Step 1: Write failing shape test**

```python
def test_predict_node_pi_shape():
    from hetero.chromatic_pdgnn import (gen_chromatic_samples, train_chromatic_pdgnn,
                                        predict_node_chromatic_pi)
    import networkx as nx
    g = nx.Graph(); g.add_edges_from([(0, 2), (1, 2), (3, 2)])
    ntype = np.array([0, 0, 1, 0]); K = 2
    hks = np.random.RandomState(0).rand(4, K).astype(np.float32)
    samples = gen_chromatic_samples(g, hks, ntype, 0, hop=2, max_nodes=20, n_samples=4, seed=0)
    model = train_chromatic_pdgnn(samples, hidden=16, layers=2, epochs=5, verbose=False)
    pi = predict_node_chromatic_pi(model, g, hks, ntype, 0, hop=2, max_nodes=20)
    assert pi.shape == (4, 3 * 25 * K)   # 3 legs x 5x5 PI x K scales
```

- [ ] **Step 2: Run to verify failure**

Run: `conda run -n tlcgnn python -m pytest tests/test_chromatic_pdgnn.py::test_predict_node_pi_shape -v`
Expected: FAIL (ImportError).

- [ ] **Step 3: Implement inference (PI per leg, reuse PersistenceImager)**

```python
# append to hetero/chromatic_pdgnn.py
from node_ph_features import PI_RES
from sg2dgm import PersistenceImager as pimg_mod


@torch.no_grad()
def predict_node_chromatic_pi(model, g, hks, ntype, target_type, hop, max_nodes,
                              verbose=False):
    """(N, 3*25*K): per-node image/kernel/cokernel PI under unified-graph ego. No exact compute."""
    model.eval()
    imager = pimg_mod.PersistenceImager(resolution=PI_RES)
    n, K = hks.shape
    ntype = np.asarray(ntype)
    per_leg = PI_RES * PI_RES
    out = np.zeros((n, 3 * per_leg * K), dtype=np.float64)
    filts_by_k = [{nd: float(hks[nd, k]) for nd in g.nodes()} for k in range(K)]
    for v in range(n):
        for k in range(K):
            res = _ego_filt_edges(g, v, hop, filts_by_k[k], max_nodes)
            if res is None:
                continue
            filt, ei, nodelist = res
            if ei.shape[1] == 0:
                continue
            color = (ntype[np.array(nodelist)] == target_type).astype(np.float32).reshape(-1, 1)
            ft = torch.tensor(filt.reshape(-1, 1).astype(np.float32), device=device)
            ct = torch.tensor(color, device=device)
            et = torch.tensor(ei.astype(np.int64), device=device)
            pred = model(ft, ct, et)
            for li, L in enumerate(_LEGS):
                pts = pred[L].cpu().numpy()
                pts = pts[pts[:, 1] > pts[:, 0]]
                base = (li * K + k) * per_leg
                if pts.size:
                    out[v, base:base + per_leg] = np.asarray(
                        imager.transform(pts.astype(np.float64))).reshape(-1)
        if verbose and (v + 1) % 1000 == 0:
            print(f'    chromatic_pi {v+1}/{n}')
    return out
```

- [ ] **Step 4: Run to verify pass**

Run: `conda run -n tlcgnn python -m pytest tests/test_chromatic_pdgnn.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add hetero/chromatic_pdgnn.py tests/test_chromatic_pdgnn.py
git commit -m "feat(chromatic): per-node 3-leg PI inference"
```

---

## Task 8: Fidelity validation (Stage-1 GO/NO-GO gate)

**Files:**
- Create: `hetero/chromatic_fidelity.py`
- Test: none (analysis script); prints a report

- [ ] **Step 1: Implement fidelity report**

```python
# hetero/chromatic_fidelity.py
"""Stage-1 GO/NO-GO: predicted-PI vs exact-PI per leg on held-out egos.
Gate: kernel-leg mean cosine >= 0.8. If below, fix the engine before Stage 2."""
from __future__ import annotations
import argparse, numpy as np, torch
from hetero.unified_filter import build_homo
from hetero.metapath_graph import load_hgb, TARGET
from hetero.pdgnn_metapath import _graph_hks, _ego_filt_edges
from hetero.chromatic_labels import compute_chromatic_0dim
from hetero.chromatic_pdgnn import (gen_chromatic_samples, train_chromatic_pdgnn,
                                    ChromaticPDGNN, _LEGS, device)
from node_ph_features import PI_RES
from sg2dgm import PersistenceImager as pimg_mod


def _exact_pi_for_node(g, hks, ntype, tt, v, k, hop, max_nodes, imager):
    res = _ego_filt_edges(g, v, hop, {nd: float(hks[nd, k]) for nd in g.nodes()}, max_nodes)
    if res is None:
        return None
    filt, ei, nodelist = res
    if ei.shape[1] == 0:
        return None
    is_L = (np.asarray(ntype)[np.array(nodelist)] == tt)
    lab = compute_chromatic_0dim(filt, ei, is_L, max_f=float(filt.max()))
    pis = {}
    for L in _LEGS:
        pts = lab[L]
        pis[L] = (np.asarray(imager.transform(pts)).reshape(-1) if pts.shape[0]
                  else np.zeros(PI_RES * PI_RES))
    return pis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='ACM'); ap.add_argument('--K', type=int, default=2)
    ap.add_argument('--hop', type=int, default=2); ap.add_argument('--max_nodes', type=int, default=60)
    ap.add_argument('--n_train', type=int, default=400); ap.add_argument('--n_test', type=int, default=100)
    ap.add_argument('--cap', type=int, default=4000); a = ap.parse_args()
    d = load_hgb(a.dataset); g, ntype, n_t, y, masks = build_homo(d, a.dataset)
    # subsample for tractable HKS if huge
    hks = _graph_hks(g, a.K)
    tt = 0  # build_homo puts target type first -> target nodes are color/type 0
    train = gen_chromatic_samples(g, hks, ntype, tt, a.hop, a.max_nodes, a.n_train, seed=0)
    model = train_chromatic_pdgnn(train, hidden=32, layers=3, epochs=30)
    imager = pimg_mod.PersistenceImager(resolution=PI_RES)
    rng = np.random.RandomState(7); test_nodes = rng.choice(n_t, min(a.n_test, n_t), replace=False)
    cos = {L: [] for L in _LEGS}
    model.eval()
    with torch.no_grad():
        for v in test_nodes:
            for k in range(a.K):
                ex = _exact_pi_for_node(g, hks, ntype, tt, int(v), k, a.hop, a.max_nodes, imager)
                if ex is None:
                    continue
                res = _ego_filt_edges(g, int(v), a.hop,
                                      {nd: float(hks[nd, k]) for nd in g.nodes()}, a.max_nodes)
                filt, ei, nodelist = res
                color = (np.asarray(ntype)[np.array(nodelist)] == tt).astype(np.float32).reshape(-1, 1)
                pred = model(torch.tensor(filt.reshape(-1, 1).astype(np.float32), device=device),
                             torch.tensor(color, device=device),
                             torch.tensor(ei.astype(np.int64), device=device))
                for L in _LEGS:
                    pts = pred[L].cpu().numpy(); pts = pts[pts[:, 1] > pts[:, 0]]
                    pv = (np.asarray(imager.transform(pts.astype(np.float64))).reshape(-1)
                          if pts.size else np.zeros(PI_RES * PI_RES))
                    ev = ex[L]
                    if np.linalg.norm(pv) < 1e-9 and np.linalg.norm(ev) < 1e-9:
                        cos[L].append(1.0)
                    elif np.linalg.norm(pv) < 1e-9 or np.linalg.norm(ev) < 1e-9:
                        cos[L].append(0.0)
                    else:
                        cos[L].append(float(pv @ ev / (np.linalg.norm(pv) * np.linalg.norm(ev))))
    print(f"=== fidelity {a.dataset} (n_test_nodes={len(test_nodes)}, K={a.K}) ===")
    for L in _LEGS:
        m = float(np.mean(cos[L])) if cos[L] else float('nan')
        print(f"  {L:9s} mean cosine = {m:.3f}  (n={len(cos[L])})")
    kmean = float(np.mean(cos['kernel'])) if cos['kernel'] else 0.0
    print(f"GATE kernel>=0.80: {'GO' if kmean >= 0.80 else 'NO-GO'} ({kmean:.3f})")


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Run on ACM**

Run: `conda run -n tlcgnn python -m hetero.chromatic_fidelity --dataset ACM --K 2 --n_train 400 --n_test 100`
Expected: prints per-leg cosine + `GATE kernel>=0.80: GO/NO-GO`. If NO-GO: increase `--n_train`, epochs, or hidden; if still NO-GO, kernel labels may be too sparse — document and reconsider (Stage-1 finding).

- [ ] **Step 3: Commit**

```bash
git add hetero/chromatic_fidelity.py
git commit -m "feat(chromatic): Stage-1 fidelity gate (kernel-leg cosine vs exact)"
```

---

## Task 9: Synthetic positive-control generator

**Files:**
- Create: `hetero/synth_chromatic.py`
- Test: `tests/test_synth_chromatic.py`

**Construction.** For each target node we build a small graph whose label ∈ {0,1} is set by a cross-type bridging motif: class 0 = two same-type leaf-clusters bridged by a single shared other-type hub; class 1 = two same-type leaf-clusters bridged by a *chain of two* other-type hubs (different cross-type mingling pattern → different kernel persistence). Node features are pure noise; same-type-only topology is identical across classes (the two clusters are internally identical) — so only the cross-type bridge (kernel) distinguishes classes.

- [ ] **Step 1: Write failing test (chromatic labels separate the two classes)**

```python
# tests/test_synth_chromatic.py
import numpy as np
from hetero.synth_chromatic import make_synth_chromatic
from hetero.chromatic_labels import compute_chromatic_0dim


def test_classes_have_distinct_kernel():
    g0 = make_synth_chromatic(label=0, seed=0)
    g1 = make_synth_chromatic(label=1, seed=0)
    k0 = compute_chromatic_0dim(g0['fv'], g0['ei'], g0['is_L'], max_f=float(g0['fv'].max()))['kernel']
    k1 = compute_chromatic_0dim(g1['fv'], g1['ei'], g1['is_L'], max_f=float(g1['fv'].max()))['kernel']
    # different cross-type bridge -> kernel barcodes differ (count or values)
    assert not np.array_equal(np.sort(k0, axis=0), np.sort(k1, axis=0))
```

- [ ] **Step 2: Run to verify failure**

Run: `conda run -n tlcgnn python -m pytest tests/test_synth_chromatic.py -v`
Expected: FAIL (ImportError).

- [ ] **Step 3: Implement generator**

```python
# hetero/synth_chromatic.py
"""Planted-bridge synthetic hetero graph: label = cross-type bridging motif.
Positive control — chromatic (kernel) MUST separate classes by construction,
while node features and same-type-only topology do not. Lets us tell a real-data
null ('chromatic truly null') from a blind pipeline ('engine can't see chromatic')."""
from __future__ import annotations
import numpy as np


def make_synth_chromatic(label: int, seed: int = 0, n_leaf: int = 3):
    """Return dict fv,(2,E) ei, is_L(bool), y. Target-type leaves in two clusters A,B.
    label 0: clusters bridged by ONE shared other-type hub.
    label 1: clusters bridged by a CHAIN of two other-type hubs."""
    rng = np.random.RandomState(seed)
    # cluster A target leaves 0..n_leaf-1 ; cluster B target leaves n_leaf..2n_leaf-1
    nA = list(range(n_leaf)); nB = list(range(n_leaf, 2 * n_leaf))
    nxt = 2 * n_leaf
    edges = []
    # each cluster: a star around its first leaf (same-type internal, identical both classes)
    for c in (nA, nB):
        for x in c[1:]:
            edges.append((c[0], x))
    is_L = [True] * (2 * n_leaf)
    if label == 0:
        hub = nxt; nxt += 1; is_L.append(False)
        edges += [(nA[0], hub), (nB[0], hub)]           # one cross-type hub
    else:
        h1, h2 = nxt, nxt + 1; nxt += 2; is_L += [False, False]
        edges += [(nA[0], h1), (h1, h2), (h2, nB[0])]   # two-hub chain
    m = nxt
    fv = rng.rand(m).astype(np.float64)                  # noise filter (same dist both classes)
    ei = np.array(edges, dtype=np.int64).T
    return {'fv': fv, 'ei': ei, 'is_L': np.array(is_L, bool), 'y': int(label)}
```

- [ ] **Step 4: Run to verify pass**

Run: `conda run -n tlcgnn python -m pytest tests/test_synth_chromatic.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add hetero/synth_chromatic.py tests/test_synth_chromatic.py
git commit -m "feat(chromatic): planted-bridge synthetic positive-control generator"
```

---

## Task 10: Stage-2 driver — controls ladder × backbone, synthetic first

**Files:**
- Create: `hetero/run_chromatic.py`

**Controls.** `none` (backbone only) / `chromatic` (3-leg PI) / `achromatic` (same ChromaticPDGNN trained to predict only the ORDINARY leg of K, type-blind: color column = zeros, single ordinary head — reuse the model but feed `color=0` and supervise on `labels['ordinary']`) / `shuffled` (chromatic PI rows permuted) / `random` (chromatic PI from a random scalar filter). Verdict: chromatic must beat all four.

> **Implementation note for `achromatic`:** the cleanest type-blind baseline that matches capacity is the ORIGINAL `PDGNN` (`pdgnn_metapath.predict_node_pi`) on the unified graph (ordinary EPD, no color). Use that as `achromatic` rather than crippling ChromaticPDGNN — it is the existing, tested ordinary-EPD engine.

- [ ] **Step 1: Implement driver (synthetic positive-control path first)**

```python
# hetero/run_chromatic.py
"""Stage-2: does chromatic (kernel) persistence beat controls on node classification?
First runs the synthetic positive-control (chromatic MUST win), then real datasets.
Controls: none / chromatic / achromatic / shuffled / random across GCN/HAN/HGT."""
from __future__ import annotations
import argparse, numpy as np, torch
from hetero.synth_chromatic import make_synth_chromatic
from hetero.chromatic_labels import compute_chromatic_0dim
from node_ph_features import PI_RES
from sg2dgm import PersistenceImager as pimg_mod

_LEGS = ('image', 'kernel', 'cokernel')


def _graph_pi(gd, imager):
    """One synthetic graph -> concatenated 3-leg PI vector (graph-level feature)."""
    lab = compute_chromatic_0dim(gd['fv'], gd['ei'], gd['is_L'], max_f=float(gd['fv'].max()))
    vec = []
    for L in _LEGS:
        pts = lab[L]
        vec.append(np.asarray(imager.transform(pts)).reshape(-1) if pts.shape[0]
                   else np.zeros(PI_RES * PI_RES))
    return np.concatenate(vec)


def run_synthetic(n_per_class=200, seed=0):
    """Logistic-regression separability of chromatic feature vs controls on the
    synthetic positive control. chromatic should reach ~1.0 AUC; achromatic ~0.5."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    imager = pimg_mod.PersistenceImager(resolution=PI_RES)
    X_chrom, X_achrom, yv = [], [], []
    rng = np.random.RandomState(seed)
    for i in range(n_per_class):
        for lab in (0, 1):
            gd = make_synth_chromatic(label=lab, seed=rng.randint(1 << 30))
            X_chrom.append(_graph_pi(gd, imager))
            # achromatic = ordinary leg only (type-blind)
            o = compute_chromatic_0dim(gd['fv'], gd['ei'], gd['is_L'],
                                       max_f=float(gd['fv'].max()))['ordinary']
            X_achrom.append(np.asarray(imager.transform(o)).reshape(-1) if o.shape[0]
                            else np.zeros(PI_RES * PI_RES))
            yv.append(lab)
    X_chrom, X_achrom, yv = map(np.array, (X_chrom, X_achrom, yv))
    for name, X in [('chromatic', X_chrom), ('achromatic', X_achrom)]:
        sc = cross_val_score(LogisticRegression(max_iter=2000), X, yv, cv=5, scoring='roc_auc')
        print(f"  synthetic {name:10s} AUC = {sc.mean():.3f} +- {sc.std():.3f}")
    print("GATE: chromatic AUC must be >> achromatic (else pipeline blind to chromatic).")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['synthetic', 'real'], default='synthetic')
    a = ap.parse_args()
    if a.mode == 'synthetic':
        run_synthetic()
    else:
        raise SystemExit("real-data path: see Task 11")
```

- [ ] **Step 2: Run the synthetic positive control**

Run: `conda run -n tlcgnn python -m hetero.run_chromatic --mode synthetic`
Expected: `chromatic AUC` ≈ 0.95–1.0, `achromatic AUC` ≈ 0.5. If chromatic does NOT beat achromatic → STOP and debug labels/engine (pipeline is blind to chromatic signal).

- [ ] **Step 3: Commit**

```bash
git add hetero/run_chromatic.py
git commit -m "feat(chromatic): Stage-2 driver + synthetic positive-control (exact-label sanity)"
```

---

## Task 11: Real-data Stage-2 (controls × datasets × backbones)

**Files:**
- Modify: `hetero/run_chromatic.py` (add `run_real`)

**Approach.** Reuse the node-classification backbones and controls already in `hetero/run_han_hgt.py` (GCN/HAN/HGT) / `hetero/hetero_nc_pipeline.py`. The new feature is the per-node 3-leg chromatic PI from `predict_node_chromatic_pi` on the unified graph. Controls: `none` / `chromatic` / `achromatic`=`pdgnn_metapath.predict_node_pi` (ordinary EPD on the same unified ego) / `shuffled`=chromatic rows permuted / `random`=ChromaticPDGNN trained on a random scalar filter. Run each across GCN/HAN/HGT and ACM/DBLP/Freebase; report a table.

- [ ] **Step 1: Implement `run_real`** — wire `predict_node_chromatic_pi` as the feature into the existing `run_han_hgt` variant runner (pass the precomputed PI matrix as `--external_feat`; if that hook does not exist, add a minimal one to `run_han_hgt.py` that concatenates a provided `(N,F)` numpy array to node features — a targeted change, not a refactor). Controls produced by:
  - `chromatic`: `predict_node_chromatic_pi(model_chrom, g, hks, ntype, 0, hop, max_nodes)`
  - `achromatic`: `pdgnn_metapath.predict_node_pi(model_ord, g, hks, hop, max_nodes)` (ordinary)
  - `shuffled`: `chromatic[np.random.permutation(N)]`
  - `random`: chromatic model retrained on `random_filter` (mirror `pdgnn_metapath.random_filter_node_pi`)

```python
# append to hetero/run_chromatic.py
def run_real(dataset='ACM', backbone='GCN', K=2, hop=2, max_nodes=60, cap=4000, seed=0):
    import numpy as np, torch
    from hetero.metapath_graph import load_hgb
    from hetero.unified_filter import build_homo
    from hetero.pdgnn_metapath import _graph_hks
    from hetero.chromatic_pdgnn import (gen_chromatic_samples, train_chromatic_pdgnn,
                                        predict_node_chromatic_pi)
    d = load_hgb(dataset); g, ntype, n_t, y, masks = build_homo(d, dataset)
    hks = _graph_hks(g, K)
    tt = 0
    samples = gen_chromatic_samples(g, hks, ntype, tt, hop, max_nodes, n_samples=500, seed=seed)
    model = train_chromatic_pdgnn(samples, hidden=32, layers=3, epochs=30)
    feat = predict_node_chromatic_pi(model, g, hks, ntype, tt, hop, max_nodes)  # (N,3*25*K)
    feat = feat[:n_t]  # target-type rows (target nodes are global 0..n_t-1 in build_homo)
    # hand off to the existing NC runner with controls; see run_han_hgt for the variant API.
    from hetero.run_han_hgt import run_variant_with_external  # add this thin hook if missing
    for variant in ('none', 'chromatic', 'achromatic', 'shuffled', 'random'):
        acc = run_variant_with_external(d, dataset, backbone, variant, feat, seed=seed)
        print(f"{dataset:8s} {backbone:4s} {variant:10s} acc={acc:.4f}")
```

- [ ] **Step 2: Smoke run ACM/GCN**

Run: `conda run -n tlcgnn python -c "from hetero.run_chromatic import run_real; run_real('ACM','GCN',K=2)"`
Expected: 5 lines (none/chromatic/achromatic/shuffled/random). Verdict requires `chromatic` > all others.

- [ ] **Step 3: Full grid via SLURM** — ACM/DBLP/Freebase × GCN/HAN/HGT, log to `results/chromatic_<ds>_<bb>.log`. Aggregate into a table; **genuine signal only if `chromatic` beats none AND achromatic AND shuffled AND random, consistently across backbones** (Idea-2 fluke lesson). Record verdict in `docs/specs/2026-06-08-chromatic-pdgnn-hetero-design.md` (append a Results section) and update memory `tda-topology-null-finding`.

- [ ] **Step 4: Commit + push**

```bash
git add hetero/run_chromatic.py hetero/run_han_hgt.py results/ docs/specs/2026-06-08-chromatic-pdgnn-hetero-design.md
git commit -m "exp(chromatic): Stage-2 real-data grid (ACM/DBLP/Freebase x GCN/HAN/HGT) + verdict"
git push tda HEAD
```

---

## Self-Review

**Spec coverage:** §1 motivation → header/grounding ✓. §2 staging (Stage1 engine, Stage2 signal) → Tasks 1–8 (engine+fidelity) / 9–11 (signal) ✓. §3 invariant (0-dim image/ker/coker, L=target-type, unified substrate) → Tasks 1–4 + `build_homo` reuse ✓. §4 engine (color input, 3 heads, per-leg Hungarian, fidelity gate) → Tasks 5–8 ✓. §5 controls ladder + synthetic + datasets/backbones → Tasks 9–11 ✓. §6 files → all created ✓. §7 risks (fidelity gate for off-distribution; synthetic for null-interpretability) → Tasks 8–9 ✓. §8 success criteria → fidelity gate (Task 8) + synthetic gate (Task 10) + cross-backbone (Task 11) ✓.

**Placeholder scan:** kernel algorithm (Task 3) flagged as best-derived with authoritative hand-tests — the executor must make tests pass, fixing the algorithm if needed; this is intentional TDD, not a placeholder. `run_variant_with_external` (Task 11) is a thin hook to add to `run_han_hgt.py` if absent — described, not hand-waved. No TBD/TODO elsewhere.

**Type consistency:** `compute_chromatic_0dim(fv, ei, is_L, max_f=None) -> {ordinary,image,kernel,cokernel}` used identically across Tasks 1–10. `ChromaticPDGNN.forward(filt,color,ei) -> {image,kernel,cokernel}` consistent Tasks 5–11. `_LEGS=('image','kernel','cokernel')` shared. PI dim `3*25*K` consistent (Task 7 output ↔ Task 11 consumer). Target nodes = global `0..n_t-1` (from `build_homo` target-first ordering) used in Tasks 8 & 11.

**Gap fixes applied:** added gudhi cross-check (Task 4) for ordinary correctness confidence; clarified `achromatic` = existing ordinary `predict_node_pi` (Task 11 note) to avoid building a crippled model.
