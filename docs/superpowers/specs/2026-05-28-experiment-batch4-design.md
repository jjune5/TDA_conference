# Experiment Batch 4 — Solidify headline + mechanism/generalization — Design Spec

**Date:** 2026-05-28 (resumed when node load dropped <30)
**Parent:** TDA conference. 3 experiments: **1 CPU-heavy (paced) + 2 CPU-light (cache-reuse)** — applying batch-2's thrash lesson (5 parallel PI-heavy = load 3900). Goal: solidify the project's only clear positive (N1 pro-topology robustness) + 2 mechanistic/generalization probes.

## S1 — Solidify N1 edge-noise robustness (CPU-heavy, priority)
N1's pro-topology crossover (Chameleon PI overtakes no-PI under edge noise) had a weak-stats caveat (n=10, single perturbation seed). Solidify: extend `noise_robust_exp.py` to **Chameleon with ≥3 perturbation seeds × more inits** (tighten the crossover CI) + add **Texas + Cornell** (tiny hetero graphs, fast PI) for generalization. **Skip Squirrel** (too large/CPU-heavy). 3-way PI/no-PI/GDC-PI, p∈{0,5,10,20}%.
**Finding**: is the edge-noise pro-topology effect statistically robust + does it generalize across hetero datasets? (PI must be recomputed per perturbed graph — this is the CPU-heavy part; pace via SLURM, ≤4 concurrent.)

## S2 — GNN backbone sensitivity (CPU-light, reuse PI cache)
The LP encoder is GCN. Does the "PI helps homo / hurts hetero" pattern depend on the encoder? Swap encoder ∈ {GCN, GAT, GraphSAGE} (PyG built-in), keep PI vs no-PI, **reuse existing PI caches** (topology fixed → no new Ollivier-Ricci). Datasets: Cora (homo) + Chameleon (hetero).
**Finding**: is the PI domain-effect GCN-specific or general across GNN encoders? Self-contained script (own encoder variants); no shared-file edits.

## S3 — Why does hetero PI point "the wrong way"? (CPU-light analysis, reuse cache)
The shuffle control (§6 EXP-1) showed hetero PI is a "wrong-direction" edge signal. Mechanism: load the **PI cache** + the train pos/neg edge labels; measure how **separable positive-edge PIs are from negative-edge PIs** (e.g., AUC of a logistic/SVM on PI features alone predicting edge existence, or distributional distance), on **Chameleon (hetero) vs Cora (homo)**. Hypothesis: on hetero, pos-edge and neg-edge PIs are nearly indistinguishable (or anti-correlated with existence) → PI misleads; on homo they're separable → PI helps.
**Finding**: quantified mechanism of the hetero-hurt. Pure analysis of cached PIs + edge labels (CPU-light).

## Execution
3 background worktree agents (worktree isolation = no conflict). S1 CPU-heavy (SLURM, paced); S2/S3 CPU-light (cache reuse). Commit locally, no push. Controller extracts → results doc §14. If node load climbs back >~80, S1 throttles. Each → finding regardless of outcome.
