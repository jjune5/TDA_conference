# Multi-scale Diffusion Features for Link Prediction — Design Spec

**Date:** 2026-05-28
**Parent:** TDA conference project. Direct follow-up to the §14 centerpiece (exact vicinity-PI is a train-graph-membership artifact that collapses to ≈0 at test). Origin: user idea — "diffusion의 multi-scale(여러 t의 heat kernel)도 PD처럼 입력으로 넣어보자."

## Motivation / hypothesis

§14 finding: for a candidate edge (u,v), the exact vicinity-PI is ≫0 only when the (u,v) edge is present in the graph (train_pos); at test the edge is removed → vicinity persistence collapses → PI≈0 → **no genuine test signal** (a membership artifact). PDGNN restores test signal by predicting the PD neurally.

**Hypothesis**: a **node-level multi-scale diffusion signature** (Heat Kernel Signature, HKS, at several diffusion times t) is computed from the global graph Laplacian, so it is **edge-presence-independent** — removing one candidate edge barely changes HKS_t(u)/HKS_t(v). Therefore a diffusion *edge feature* built from node HKS should **not collapse at test** → it can supply genuine test-time topological signal where exact PI is zero. This is a second, spectral route (alongside PDGNN's neural route) to side-step the §14 artifact. (Bonus: HKS = Laplacian eigendecomposition → GPU-friendly, uses the otherwise-idle GPUs.)

## The 5-way comparison

| variant | input to LP decoder (concat with `|emb_u−emb_v|²`) | §14 prediction |
|---|---|---|
| **no-PI** | none (GCN only) | baseline |
| **exact-PI** | existing vicinity PI (`data/TLCGNN/<name>.npy`) | test≈0 (artifact) |
| **A. node multi-scale HKS** | `[HKS_t(u), HKS_t(v), |HKS_t(u)−HKS_t(v)|]` over t∈{t_1..t_K} (3K-dim) | **node-level → no collapse → genuine test signal** |
| **B. multi-scale HKS-filtration PD** | vicinity PI using HKS filtration at K scales, stacked | **vicinity-based → likely inherits artifact (collapse at test)** |
| **C. A + B** | node HKS features **and** the HKS-PD | isolates which route carries test signal |

HKS: `HKS_t(i) = Σ_k exp(−t·λ_k)·φ_k(i)²` from the normalized graph Laplacian eigendecomposition. K≈5 log-spaced scales (local→global; tune). Reuse `Knowledge_Distillation/hks_filtration.compute_hks`. **All features computed on the leakage-free training graph** (val/test pos removed), exactly like the PI.

## Two measurements (the scientific payoff)

1. **§14 diagnostic per variant** (the key test): per-segment (train_pos / test_pos / test_neg) — does each feature **discriminate test_pos from test_neg** (genuine signal) or collapse to indistinguishable (artifact)? Method: a PI/feature-only logistic-regression CV-AUC on test edges (as in S3 `pi_separability`). **Prediction: A discriminates at test, B collapses like exact-PI, C≈A.** (Note: A's discriminative power may be homophily-dependent — |HKS_u−HKS_v| is small for similar-node edges; measure on both Cora and Chameleon.)
2. **End-to-end LP AUC**: 5-way (no-PI / exact-PI / A / B / C), 50 trials.

## Datasets
- **Cora (homophilic) + Chameleon (heterophilic)** — the §14 reference datasets. Add **Photo** if cheap.

## Implementation (files)
- New `diffusion_features.py`: compute multi-scale HKS node features → per-edge feature builder (variant A). GPU eigendecomp.
- Variant B: extend the existing `TLCGNN_LP_FILTER=hks` hook (`loaddatas.py`, from A2) to **multiple t scales** stacked.
- LP model: a variant of `baselines/TLCGNN.py` that concatenates the diffusion edge feature into the decoder (mirror how PI is concatenated). Guarded/new so default behavior is unchanged.
- §14 diagnostic: reuse the `pi_artifact_analysis.py` / `pi_separability` per-segment pattern on the new features.

## Scope
- **In**: variants A/B/C + baselines, the §14 diagnostic, LP AUC, on Cora+Chameleon (+Photo if cheap).
- **Out**: new GNN backbones (done in S2), molecular GC (separate), antibody (closed).

## Parallelization / load
- 2 worktree agents: **Agent-F** (variant A: node HKS features — GPU, light) + **Agent-G** (variant B: multi-scale HKS-PD — CPU-heavy PI recompute, SLURM-paced). Variant C + the diagnostic + final LP integration done by controller after both land (so the 3 variants compose without conflict). Reuse exact-PI cache for the baseline (no recompute).
- Launch **after** CP-A/CP-B finish + node load < ~40. Verify agent outputs against ground truth (phantom-report lesson).

## Success criterion
Each variant → committed result + the §14-diagnostic verdict (does diffusion feature give genuine test signal?). A clear answer to "does multi-scale diffusion sidestep the §14 membership artifact and help LP (esp. heterophilic)?" — finding either way. Integrate into results doc §15.
