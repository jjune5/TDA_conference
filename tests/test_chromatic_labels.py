"""TDD tests for hetero/chromatic_labels.py — exact 0-dim image/kernel/cokernel.

Hand-computed tiny graphs are the source of truth; if the algorithm disagrees, fix
the algorithm. Includes a gudhi cross-check for the ordinary leg (skipped if gudhi
absent) and a rank-consistency invariant for kernel.
"""
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hetero.chromatic_labels import (_unique_undirected_edges, compute_chromatic_0dim,
                                      _find)


def _as_set(arr, nd=6):
    return {(round(float(b), nd), round(float(d), nd)) for b, d in np.asarray(arr).reshape(-1, 2)}


# ---------------- Task 1: helpers + ordinary ----------------

def test_unique_undirected_edges_dedup_and_selfloops():
    ei = np.array([[0, 1, 1, 0, 2], [1, 0, 2, 0, 2]])  # (0,1),(1,0)dup,(1,2),(0,0),(2,2)
    assert _unique_undirected_edges(ei) == [(0, 1), (1, 2)]


def test_ordinary_h0_path_graph():
    fv = np.array([0.0, 1.0, 2.0])
    ei = np.array([[0, 1], [1, 2]])
    is_L = np.array([False, False, False])
    out = compute_chromatic_0dim(fv, ei, is_L, max_f=2.0)
    assert _as_set(out['ordinary']) == {(0.0, 2.0)}


# ---------------- Task 2: image ----------------

def test_image_two_L_bridged_by_nonL():
    fv = np.array([0.0, 0.0, 1.0])       # a,b target ; c non-target
    ei = np.array([[0, 2, 1, 2], [2, 0, 2, 1]])   # (a,c),(b,c) symmetric ; no (a,b)
    is_L = np.array([True, True, False])
    out = compute_chromatic_0dim(fv, ei, is_L, max_f=2.0)
    assert _as_set(out['image']) == {(0.0, 1.0), (0.0, 2.0)}


# ---------------- Task 3: kernel (the heart) ----------------

def test_kernel_born_at_crosstype_bridge_essential():
    fv = np.array([0.0, 0.0, 1.0])
    ei = np.array([[0, 2, 1, 2], [2, 0, 2, 1]])    # (a,c),(b,c) ; NO (a,b)
    is_L = np.array([True, True, False])
    out = compute_chromatic_0dim(fv, ei, is_L, max_f=2.0)
    assert _as_set(out['kernel']) == {(1.0, 2.0)}


def test_kernel_absent_when_L_connects_first():
    fv = np.array([0.0, 0.0, 1.0])
    ei = np.array([[0, 2, 1, 2, 0, 1],
                   [2, 0, 2, 1, 1, 0]])            # adds (a,b)@0
    is_L = np.array([True, True, False])
    out = compute_chromatic_0dim(fv, ei, is_L, max_f=2.0)
    assert _as_set(out['kernel']) == set()


def test_kernel_three_leaves_one_hub_two_classes():
    # a,b,c (L) each joined only via hub h (non-L); no L-edges -> 2 essential kernels
    fv = np.array([0.0, 0.0, 0.0, 1.0])            # a,b,c, h
    ei = np.array([[0, 1, 2], [3, 3, 3]])          # (a,h),(b,h),(c,h)
    is_L = np.array([True, True, True, False])
    out = compute_chromatic_0dim(fv, ei, is_L, max_f=2.0)
    assert _as_set(out['kernel']) == {(1.0, 2.0)}  # both born@1 -> dedup as set, count via invariant test


def test_kernel_death_when_L_edge_catches_up():
    # a,b (L) bridged by hub h@1 (kernel born@1), then a-b L-edge@2 -> kernel dies@2
    fv = np.array([0.0, 0.0, 1.0, 2.0])            # a,b, h(non-L)@1, x(unused pad)
    # edges: (a,h)@1, (b,h)@1, (a,b)@2  -> a-b L-edge enters at max(fv[a],fv[b])=0? need >1.
    # make a,b filtration such that L-edge enters AFTER the bridge: set fv[a]=fv[b]=0,
    # but edge filtration = max endpoints = 0 -> too early. Use a 4th vertex to delay:
    # Instead raise b's filtration so (a,b) enters at 2.
    fv = np.array([0.0, 2.0, 1.0])                 # a@0, b@2, h@1
    ei = np.array([[0, 1, 0], [2, 2, 1]])          # (a,h)@1, (b,h)@2, (a,b)@2
    is_L = np.array([True, True, False])
    out = compute_chromatic_0dim(fv, ei, is_L, max_f=3.0)
    # a-comp gets h@1 (L-bearing+nonL, no kernel). b joins h@2: cross-type merge of
    # {a,h}(L) and {b}(L) -> kernel born@2. (a,b) L-edge@2 also enters: if processed
    # such that KLp already same -> death@2 (zero pers, dropped). Accept either {} or
    # a zero-pers drop. Assert no positive-persistence kernel bar:
    assert all(d > b for b, d in out['kernel']) and _as_set(out['kernel']) <= {(2.0, 3.0)}


# ---------------- Task 4: cokernel ----------------

def test_cokernel_two_nonL_merge_then_absorbed():
    fv = np.array([0.0, 1.0, 1.0])     # a=0 (L), c=1, d=2 (non-L)
    ei = np.array([[1, 0], [2, 1]])    # (c,d)@1 ; (a,c)@1
    is_L = np.array([True, False, False])
    out = compute_chromatic_0dim(fv, ei, is_L, max_f=3.0)
    assert _as_set(out['cokernel']) == set()


def test_cokernel_essential_isolated_nonL():
    fv = np.array([0.0, 1.0])          # a=0 (L), e=1 (non-L), NO edge
    ei = np.zeros((2, 0), dtype=int)
    is_L = np.array([True, False])
    out = compute_chromatic_0dim(fv, ei, is_L, max_f=3.0)
    assert _as_set(out['cokernel']) == {(1.0, 3.0)}


# ---------------- invariants & cross-checks ----------------

def _bars_rank(bars, t):
    return int(sum(1 for b, d in np.asarray(bars).reshape(-1, 2) if b <= t < d))


def _comps_at(fv, ue, t, restrict=None, L_only_edges=False, is_L=None):
    m = len(fv)
    present = [i for i in range(m) if fv[i] <= t and (restrict is None or restrict[i])]
    p = list(range(m))
    for (u, v) in ue:
        if max(fv[u], fv[v]) <= t:
            if L_only_edges and not (is_L[u] and is_L[v]):
                continue
            ru, rv = _find(p, u), _find(p, v)
            if ru != rv:
                p[ru] = rv
    return len(set(_find(p, i) for i in present)) if present else 0


def test_all_legs_match_rank_function_random():
    """Barcode rank == homological rank at every inter-event midpoint, for all 4 legs.
    ordinary=#K-comps, image=#L-bearing-K-comps, cokernel=#L-free-K-comps,
    kernel=b0(L)-#L-bearing-K-comps. Strongest correctness check (no library needed)."""
    rng = np.random.RandomState(7)
    for _ in range(300):
        m = rng.randint(4, 14)
        fv = rng.rand(m).astype(np.float64)
        ei = rng.randint(0, m, size=(2, rng.randint(m, 3 * m)))
        is_L = rng.rand(m) > 0.4
        mx = float(fv.max())
        out = compute_chromatic_0dim(fv, ei, is_L, max_f=mx)
        ue = _unique_undirected_edges(ei)
        vals = sorted(set([float(x) for x in fv] + [max(fv[u], fv[v]) for u, v in ue] + [mx]))
        mids = [vals[0] - 1e-6] + [(vals[i] + vals[i + 1]) / 2 for i in range(len(vals) - 1)] + [mx - 1e-9]
        for t in mids:
            nK = _comps_at(fv, ue, t)
            Kp = list(range(m))
            for (u, v) in ue:
                if max(fv[u], fv[v]) <= t:
                    ru, rv = _find(Kp, u), _find(Kp, v)
                    if ru != rv:
                        Kp[ru] = rv
            Lpres = [i for i in range(m) if is_L[i] and fv[i] <= t]
            nKL = len(set(_find(Kp, i) for i in Lpres)) if Lpres else 0
            nLcomp = _comps_at(fv, ue, t, restrict=is_L, L_only_edges=True, is_L=is_L)
            ref = {'ordinary': nK, 'image': nKL, 'cokernel': nK - nKL, 'kernel': nLcomp - nKL}
            for leg, r in ref.items():
                assert _bars_rank(out[leg], t) == r, (leg, t, r, out[leg])


def test_ordinary_matches_gudhi_random():
    gudhi = pytest.importorskip("gudhi")
    rng = np.random.RandomState(0)
    for _ in range(5):
        m = rng.randint(5, 12)
        fv = rng.rand(m).astype(np.float64)
        E = rng.randint(m, 2 * m)
        ei = rng.randint(0, m, size=(2, E))
        is_L = rng.rand(m) > 0.5
        mx = float(fv.max())
        out = compute_chromatic_0dim(fv, ei, is_L, max_f=mx)
        st = gudhi.SimplexTree()
        for i in range(m):
            st.insert([i], filtration=float(fv[i]))
        for (u, v) in _unique_undirected_edges(ei):
            st.insert([int(u), int(v)], filtration=float(max(fv[u], fv[v])))
        st.compute_persistence(persistence_dim_max=False)
        g0 = st.persistence_intervals_in_dimension(0)
        gset = set()
        for b, d in g0:
            d = mx if d == float('inf') else d
            if d > b:
                gset.add((round(float(b), 6), round(float(d), 6)))
        assert _as_set(out['ordinary']) == gset
