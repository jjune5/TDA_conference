# Experiment Batch 3 — Deepen mechanism + robustness (LOAD-AWARE) — Design Spec

**Date:** 2026-05-28
**Parent:** TDA conference. **SMALL batch (2 experiments)**, deliberately CPU-light.

**Why small/CPU-light:** the shared node is CPU-contended (another user's GROMACS, load ~58 baseline). Batch-2's 5 parallel **PI-heavy** agents (Ollivier-Ricci = CPU-bound) oversubscribed CPU → load ~3900, 4-hr thrash. Batch-3 experiments **reuse existing PI caches** (no new Ollivier-Ricci) → minimal CPU; GPU is nearly free. Deepen the two most interesting batch-2 threads: P1 (PDGNN mechanism) and N1 (robustness, the first pro-topology result).

## D1 — What does PDGNN actually learn? (explains P1, CPU-only analysis)
P1 showed PDGNN ≠ spectral smoothing. Now characterize the learned representation: load PDGNN-PI (`data/PDGNN/<name>.npy`) vs exact-PI (`data/TLCGNN/<name>.npy`) for Photo/Computers/Chameleon. Per-edge PI difference (PDGNN − exact): where in the 5×5 grid does PDGNN amplify/suppress mass? Is the difference a smooth low-pass (→ smoothing) or structured (→ learned)? Correlate per-edge diff magnitude with edge degree / endpoint homophily. **Pure numpy analysis of existing caches — ~0 CPU/GPU load.**
**Deliverable**: difference heatmaps (avg PDGNN−exact 5×5) per dataset + finding on *what* PDGNN learns beyond blur.

## R2 — Feature-noise robustness (deepens N1, reuse PI cache)
N1 showed topology robust to **edge** noise. Now **feature** noise: corrupt node features (Gaussian noise / random masking at level q ∈ {0, 0.25, 0.5, 1.0}), run LP **PI vs no-PI** on Cora + Chameleon. The graph **topology is unchanged** → the **PI cache is reused** (no new Ollivier-Ricci, CPU-light); only the GCN encoder sees corrupted features. Hypothesis: PI is feature-independent, so its relative value should **grow as features degrade** (GCN's feature signal collapses, topological signal intact).
**Deliverable**: AUC vs feature-noise level, PI−noPI gap curve. If the gap widens with q → topology as a feature-robustness prior (complements N1's edge-robustness).

## Execution
2 background worktree agents, **reuse existing PI caches** (assert no new Ollivier-Ricci), commit locally (no push). Controller extracts + integrates into results doc §13. If load climbs, throttle. Each → finding regardless of outcome.
