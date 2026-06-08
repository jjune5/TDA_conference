"""Exact 0-dim image/kernel/cokernel ("chromatic") persistence for one ego-graph.

Inclusion L (target-type subgraph) into K (full multi-type ego), lower-star HKS
filtration. 0-dim only. Used ONLY to generate one-time training labels for
ChromaticPDGNN (exact never used as a downstream feature -> PDGNN-only policy).

Legs (CEHM SODA'09 specialized to dim 0 on a graph, inclusion L into K):
  ordinary : H0(K)                 -- type-blind connectivity (achromatic baseline)
  image    : im(H0(L) -> H0(K))    -- when same-type clusters merge in the full graph
  kernel   : ker(H0(L) -> H0(K))   -- same-type clusters bridged ONLY by cross-type
                                       structure (= cross-type mingling; the heart)
  cokernel : coker(...)            -- pure non-target-type clusters

Kernel = relative persistence between two union-finds on the L-vertex set:
  Lp  : connectivity using L-edges only      (#components = b0(L_t))
  KLp : connectivity of L-vertices via K     (#components = #L-bearing comps of K_t)
Since K ⊇ L, KLp ⊇ Lp; rank(ker)_t = #Lp-comps − #KLp-comps. A kernel class is born
when a cross-type K-merge unites two L-bearing comps whose L-parts are still Lp-separate
(KLp catches a bridge before L does); it dies when an L-edge later unites that Lp-pair
(Lp catches up). Elder rule pairs the youngest open bridge in the KL-component.

Grounding: Cohen-Steiner, Edelsbrunner, Harer, Morozov, "Persistent Homology for
Kernels, Images, and Cokernels" (SODA 2009); chromatic/color framing = Cultrera di
Montesano, Draganov, Edelsbrunner, Saghafian, "Chromatic TDA" (arXiv 2406.04102).
Point-cloud -> graph port is ours.
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


def _to_arr(bars):
    return np.array(bars, dtype=np.float64).reshape(-1, 2) if bars else np.zeros((0, 2), np.float64)


def compute_chromatic_0dim(fv, ei, is_L, max_f=None):
    """Return dict ordinary/image/kernel/cokernel, each (P,2) float64.

    fv (m,) vertex filtration; ei (2,E) edge index (may be symmetric/dup/self-loop);
    is_L (m,) bool (vertex in target-type subcomplex L). Essential deaths -> max_f.
    Zero-persistence bars (death<=birth) dropped.
    """
    fv = np.asarray(fv, dtype=np.float64)
    m = len(fv)
    is_L = np.asarray(is_L, dtype=bool)
    if max_f is None:
        max_f = float(fv.max()) if m else 1.0
    et = _sorted_edges(fv, ei)

    # --- K union-find: ordinary birth + hasL + image-birth + L-rep ---
    Kp = list(range(m))
    Kbirth = [float(x) for x in fv]
    KhasL = [bool(is_L[i]) for i in range(m)]
    Kimg = [float(fv[i]) if is_L[i] else None for i in range(m)]   # elder L-birth
    Krep = [i if is_L[i] else None for i in range(m)]              # one L-vertex of the K-comp

    # --- L union-find (L-edges) and KL union-find (K-connectivity of L-vertices) ---
    Lp = list(range(m))
    KLp = list(range(m))
    # open kernel births per KLp-root: root -> list of birth times (pop max on death)
    ker_births = {}

    ordinary, image, kernel, cokernel = [], [], [], []

    for (t, u, v) in et:
        is_ledge = bool(is_L[u] and is_L[v])

        # ===== L-side first for an L-edge: Lp/KLp + kernel birth/death (pre-state) =====
        if is_ledge:
            rlu, rlv = _find(Lp, u), _find(Lp, v)
            rku, rkv = _find(KLp, u), _find(KLp, v)
            pre_Lp_same = (rlu == rlv)
            pre_KLp_same = (rku == rkv)
            if not pre_Lp_same:
                if pre_KLp_same:
                    # Lp catches up to an existing cross-type bridge -> kernel DEATH
                    r = rku
                    lst = ker_births.get(r, [])
                    if lst:
                        b = max(lst)
                        lst.remove(b)
                        if t > b:
                            kernel.append((b, t))
                else:
                    # first connection (does both) -> free merge, no kernel; merge KL lists
                    la = ker_births.pop(rku, [])
                    lb = ker_births.pop(rkv, [])
                    KLp[rku] = rkv
                    if la or lb:
                        ker_births[_find(KLp, u)] = la + lb
                # merge Lp
                Lp[rlu] = rlv

        # ===== K-side: ordinary / image / cokernel (+ kernel birth for cross-type) =====
        ru, rv = _find(Kp, u), _find(Kp, v)
        if ru == rv:
            continue
        if (Kbirth[ru], ru) >= (Kbirth[rv], rv):
            dead, surv = ru, rv
        else:
            dead, surv = rv, ru
        if t > Kbirth[dead]:
            ordinary.append((Kbirth[dead], t))
        hu, hv = KhasL[ru], KhasL[rv]
        if hu and hv:
            iu, iv = Kimg[ru], Kimg[rv]
            img_dead, img_surv = (iu, iv) if iu >= iv else (iv, iu)
            if t > img_dead:
                image.append((img_dead, t))
            new_img = img_surv
            # kernel BIRTH on cross-type merge of two L-bearing comps (NOT for L-edges,
            # which already handled their own Lp/KLp/kernel above).
            if not is_ledge:
                repx, repy = Krep[ru], Krep[rv]
                rkx, rky = _find(KLp, repx), _find(KLp, repy)
                if rkx != rky:
                    la = ker_births.pop(rkx, [])
                    lb = ker_births.pop(rky, [])
                    KLp[rkx] = rky
                    la2 = la + lb + [t]            # new bridge born at t
                    ker_births[_find(KLp, repx)] = la2
        elif hu or hv:
            lfree = ru if not hu else rv
            if t > Kbirth[lfree]:
                cokernel.append((Kbirth[lfree], t))   # L-free class absorbed into image
            new_img = Kimg[ru] if hu else Kimg[rv]
        else:
            if t > Kbirth[dead]:
                cokernel.append((Kbirth[dead], t))     # younger L-free class dies
            new_img = None
        # commit K-merge
        Kp[dead] = surv
        Kbirth[surv] = min(Kbirth[ru], Kbirth[rv])
        KhasL[surv] = hu or hv
        Kimg[surv] = new_img
        Krep[surv] = Krep[ru] if KhasL[ru] else Krep[rv]
        if Krep[surv] is None:
            Krep[surv] = Krep[rv] if Krep[rv] is not None else Krep[ru]

    # ===== essentials =====
    for r in set(_find(Kp, i) for i in range(m)):
        if max_f > Kbirth[r]:
            ordinary.append((Kbirth[r], max_f))
        if KhasL[r] and Kimg[r] is not None and max_f > Kimg[r]:
            image.append((Kimg[r], max_f))
        if not KhasL[r] and max_f > Kbirth[r]:
            cokernel.append((Kbirth[r], max_f))
    for lst in ker_births.values():       # essential kernel classes
        for b in lst:
            if max_f > b:
                kernel.append((b, max_f))

    return {'ordinary': _to_arr(ordinary), 'image': _to_arr(image),
            'kernel': _to_arr(kernel), 'cokernel': _to_arr(cokernel)}
