# Experiment Batch 2 — Mechanism + Robustness — Design Spec

**Date:** 2026-05-28
**Parent:** TDA conference project. 6 experiments (5 parallel worktree agents). Each yields a **finding regardless of whether performance improves** (null = finding). Goal: explain the two headline surprises (PDGNN>exact, GDC rescue) + test TDA's core robustness claim + deepen.

## The 6 experiments

### P1 — PDGNN-as-smoothing (mechanism of "neural > exact")
**Q**: Is PDGNN's advantage over exact PI just *spectral smoothing*?
**Method**: Gaussian-blur the exact-PI vectors at σ ∈ {0.5,1,2,3} (reshape 5×5, blur, flatten), save as blurred PI caches, run standard 50-trial LP. Datasets where PDGNN beat exact: **Photo, Computers, Chameleon**.
**Finding**: If blurred-exact ≈ PDGNN gain → "PDGNN advantage = smoothing" (explains the surprise). If not → PDGNN learns something beyond blur.
**Files**: new `pi_blur_exp.py`; reuse `TLCGNN_PI_DIR` hook (no loaddatas edit).

### B1 — GDC strength sweep (mechanism of rescue)
**Q**: How does diffusion strength control the hetero-PI rescue?
**Method**: heat `t ∈ {1,3,5,10}` × sparsify `k ∈ {8,16,32}` on **Chameleon** (+Cora control), GDC-PI 50-trial LP. Parametrize `gdc_pi.py` via env `TLCGNN_GDC_T`, `TLCGNN_GDC_K`.
**Finding**: strength→rescue curve; optimal denoising; confirms diffusion-strength is the lever.

### B2 — GDC rescue generalization
**Q**: Does "rescue to no-PI parity" hold beyond Chameleon/Texas/Cornell?
**Method**: GDC-PI 50-trial LP on **Wisconsin, ChChMiner** (+ **Squirrel** if feasible/subsampled). Compare to PI / no-PI.
**Finding**: generality of the GDC rescue. (Same agent as B1.)

### N1 — Topology robustness to graph noise (TDA core claim, NEW angle)
**Q**: Is topology more robust to structural noise than plain GCN?
**Method**: randomly add+remove `p% ∈ {5,10,20}` of edges (seeded), measure AUC degradation: **PI vs no-PI vs GDC-PI** on **Chameleon + Cora**. Self-contained perturbation script.
**Finding**: if PI/GDC-PI degrade slower → first pro-topology LP result. If not → honest null.
**Files**: new `noise_robust_exp.py` (perturb in-script, no loaddatas edit).

### M2 — Molecular PI: H0 vs H1 contribution
**Q**: In molecular GC (where PI helps), is the signal from rings (H1) or components (H0)?
**Method**: mask PD by homology dim (H0-only / H1-only / both) before PI, GC 10-fold on **MUTAG/PROTEINS/NCI1**.
**Finding**: which dim carries class signal (likely H1=rings). Mechanism of "why PI helps GC".
**Files**: new `Knowledge_Distillation/mol_dim_ablation.py` (mol domain, separate).

### A2 — HKS (diffusion) filtration for LP
**Q**: We applied HKS only to molecular GC. Does a diffusion filtration change the hetero-PI-hurt in LP?
**Method**: env `TLCGNN_LP_FILTER=hks` swaps the LP PI filtration from Ollivier-Ricci → HKS (reuse `Knowledge_Distillation/hks_filtration.py`); 50-trial LP on **Cora + Chameleon**, vs Ricci-PI and no-PI. Default unchanged.
**Finding**: does diffusion filtration help/hurt LP differently than geometric (Ricci).

## Agents (worktree-isolated, parallel, no remote push)
1. **GDC** (B1+B2) — owns `gdc_pi.py`
2. **P1** — new `pi_blur_exp.py`
3. **N1** — new `noise_robust_exp.py`
4. **M2** — new `Knowledge_Distillation/mol_dim_ablation.py`
5. **A2** — LP PI filter env hook (touches loaddatas/sg2dgm)

Each in its own git worktree → zero file/git conflict. Controller extracts deliverable files onto main + commits (no destructive branch merge; verify base, sync from main absolute path if worktree is stale). Base data read from main repo absolute path. GPU ≤8 (mostly small LP/GIN; CPU for PI/blur/perturb generation).

## Success criterion
Each experiment → committed result (table/plot) + one-line finding for results doc §12. Performance gains a bonus; nulls documented.
