# Diffusion-filtered Node-level Persistence (DNP) — Design

**Date:** 2026-05-28
**Status:** design approved (brainstorming) → writing-plans next

## Goal

A *genuine* topological link-prediction feature that, unlike exact-PI, does **not**
collapse when the candidate edge is removed (passes the §14 leave-one-out test) and
genuinely helps **heterophilic** LP — built from **diffusion-based filtrations at the
node level**, validated **exact-first** then accelerated **neurally**.

One sentence: *take the §15 intuition that worked (a per-node signal that survives
edge removal) and realize it as actual persistent homology instead of a spectral
feature.*

## Motivation (Thread ②, §14 — already evidenced)

§14 (5-fold evidence, in `docs/specs/2026-06-21-tda-conference-results.md`): exact-PI's
LP signal is a **train-graph edge-membership artifact**. For a candidate edge (u,v) the
persistence image is computed on the *edge vicinity*; at test the edge is deleted
(anti-leakage) → the vicinity PH collapses → PI≈0 → no genuine test discrimination.
The "signal" is the spurious "is this edge in the training graph" indicator.

§15 showed a per-**node** spectral feature (node-HKS) avoids this (a global node
property barely changes when one incident edge is removed) and beats exact-PI on
heterophilic LP — **but node-HKS is spectral, not topology.** DNP asks: build a feature
with the *same non-collapsing, per-node* property, but from **genuine persistent
homology**, so it belongs in a TDA contribution.

This §14 framing is the **problem section** that DNP (the solution) completes — together
one coherent story: *"topological LP can leak (§14) → fix it with diffusion node-PH."*

## Method (Thread ①) — build all three constructions A/B/C

For each node `v`, work on its **k-hop ego-graph** `G_v` (k=2 default; the node's *own*
neighborhood, **not** the candidate edge's vicinity — this is the anti-artifact key).
Produce a per-node persistence vector `Φ(v)`. Then for a candidate edge (u,v):

```
edge_feature(u,v) = [ Φ(u) , Φ(v) , |Φ(u) − Φ(v)| ]
```

(same assembly that made node-HKS work in §15; symmetric in u,v).

**A — HKS-filter sublevel node-PH (safe core).**
Filter values `f = HKS_t(·)` on `G_v` for `K` log-spaced scales `t` (reuse
`hks_filtration.compute_hks(G_v, t)`). Sublevel-set **extended** persistence of `(G_v, f)`
→ per-node diagram `D_A^t(v)`; vectorize each to a 5×5 persistence image via
`sg2dgm.PersistenceImager`; concatenate over the K scales → `Φ_A(v) ∈ R^{25K}`.

**B — Multi-parameter (bifiltration) persistence (ambitious).**
2-parameter filtration on `G_v` by **(HKS-time axis, Ollivier-Ricci curvature axis)**.
Tractable realization (avoids needing RIVET): **slicing** — evaluate 1-parameter
persistence along a grid of linear slices of the (t, ricci) plane, vectorize each slice,
concatenate → `Φ_B(v)`. (Pure multiparameter via RIVET = stretch goal, only if slicing is
inconclusive.) **Higher risk** — gated behind a feasibility check.

**C — Diffusion-distance Vietoris-Rips (alternative lens).**
Heat-kernel **diffusion distance** `d_diff(i,j)` between nodes of `G_v` (from the same
Laplacian eigendecomposition as HKS). Vietoris-Rips filtration on `(G_v, d_diff)` →
per-node diagram `D_C(v)`; vectorize → `Φ_C(v)`.

All three are independent variants; we run all (user directive "다 A/B/C 전부").

## Anti-artifact rationale (why this should NOT collapse)

Per-node diagrams are computed on each node's **own** ego-graph. Removing the single
candidate edge (u,v) perturbs `G_u`/`G_v` by at most one edge → `Φ(u)`,`Φ(v)` change
negligibly → **no collapse** (contrast: exact-PI's edge-vicinity construction collapses
to 0). This must be *verified*, not assumed (see Validation gate).

## Validation (rigor — the §14 lesson is non-negotiable)

1. **Exact-first.** Compute A/B/C with exact PH (gudhi, `tlcgnn` env). **No neural approx
   yet** — prove the signal is real before optimizing speed.
2. **Collapse-test GATE.** Reusing `pi_artifact_analysis.py` machinery
   (`segment_bounds`, `seg_stats`): build the DNP feature cache in the standard
   `[train_pos|train_neg|val_pos|val_neg|test_pos|test_neg]` layout, and a second cache
   with each test edge's endpoints recomputed on the **edge-removed** graph. **PASS** iff
   test_pos features stay nonzero & stable (L1 mass ≫ 0, not the 300/300→0 collapse exact-PI
   shows). A construction that collapses is another artifact → **dropped**.
3. **LP evaluation.** 50-trial AUC via existing `pipelines.py`, on heterophilic
   (Chameleon, Squirrel, Texas, Cornell, Wisconsin) + homophilic anchors (Cora, Citeseer,
   PubMed, Photo) + drug (ChChMiner). **Baselines:** exact-PI (dionysus), PDGNN, node-HKS
   (§15), no-PI.
4. **Success criterion.** ≥1 of A/B/C **passes the collapse gate AND beats exact-PI on
   heterophilic LP** (ideally ≥ node-HKS) → a *genuine topological* feature that helps
   where exact-PI hurts. (A null result is still a finding: "even genuine node-PH can't
   beat spectral here" — reported honestly.)

## Neural engine (Thread ④) — Phase 3, conditional

Only if exact A/B/C show genuine signal: extend `pdgnn_modern.PDGNN` to a
**multi-filtration** predictor (predict the per-node diagrams for HKS-t × Ricci jointly),
trained against the exact A/B/C diagrams (Wasserstein/Hungarian loss, as in
`train_pdgnn_lp._bipartite_loss`) → ~100× speedup. Mirrors the PDGNN paper's design.

## Thread GEN (③) — parallel, separate plan

Topology-conditioned generation. Antibody capstone (topology-conditioned DiffAb) is
**already training** (`diffab_phase0/`, Phase 3). Molecular topology-conditioned
generation (H1=rings → condition desired ring topology) queues **after** the capstone
lands. Gets its own design+plan; **out of scope for the DNP plan** (noted here only so the
"do all" program is complete).

## Components / files

**Reuse (do not modify — `feedback-explicit-changes-only`):**
- `Knowledge_Distillation/hks_filtration.py` — `compute_hks(g, t)` (call with multiple t).
- `sg2dgm/PersistenceImager` — diagram → 5×5 PI vectorization.
- `pi_artifact_analysis.py` — `segment_bounds`, `seg_stats` (collapse test).
- `pdgnn_modern.py`, `train_pdgnn_lp.py` — Phase 3 neural-engine base.

**New (guarded, additive only):**
- `Knowledge_Distillation/diffusion_node_ph.py` — A/B/C per-node exact constructions
  (`phi_A(G_v, ts)`, `phi_B(G_v, ts, ricci)`, `phi_C(G_v)`); shared ego-graph + Laplacian
  eigendecomp helper.
- `Knowledge_Distillation/node_ph_cache.py` — build per-dataset DNP cache `.npy`
  (`data/DNP/<name>_<A|B|C>.npy`) in the standard edge-split layout; plus an
  `--edge-removed` mode for the collapse test.
- `Knowledge_Distillation/dnp_collapse_test.py` — collapse-test report (wraps
  `pi_artifact_analysis` helpers).

**Integration (minimal additive edits):**
- `pipelines.py:72` — add `dnp_A`,`dnp_B`,`dnp_C` to `--pi_source` choices.
- `loaddatas.py:~163` — add a branch loading the DNP cache (mirror the `pdgnn` branch).

## Risks

- **B (multiparameter)** is research-grade; gudhi multiparameter support is limited. Use
  slicing approximation; gate behind a feasibility check; RIVET only as stretch.
- **Cost:** exact PH per node over all nodes is heavier than edge-vicinity. Mitigate:
  cache per node (reused across all its incident candidate edges), CPU-parallel, start on
  small graphs (Cora/Chameleon) before Squirrel.
- **Null result** is acceptable and reportable (§14 lesson: honesty over hype).

## Scope decomposition (→ writing-plans)

The first plan covers **Thread LP only**, phased:
- **Phase 1:** A/C exact constructions + per-node cache + collapse-test gate on
  Cora+Chameleon (smallest homo+hetero). *(B in parallel behind feasibility check.)*
- **Phase 2:** full 9-dataset 50-trial LP eval vs 4 baselines; results section.
- **Phase 3 (conditional):** multi-filtration neural engine for speed.

Thread GEN (③) = separate plan, after the antibody capstone reports.
