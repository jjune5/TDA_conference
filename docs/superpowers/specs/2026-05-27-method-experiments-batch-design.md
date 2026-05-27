# Diverse Method Experiments Batch — Design Spec

**Date:** 2026-05-27
**Parent:** TDA topology study (TLC-GNN/PDGNN). This adds a batch of 6 diverse method experiments, each designed to produce an informative finding **whether or not performance improves**.

## Motivation

We have a rich result base (LP across 9 datasets, molecular GC, PDGNN neural vs exact, gating). Now probe the **mechanism and design space** with experiments that each answer a distinct scientific question. Outcomes are valuable either way (a null result is itself a finding).

## The 6 experiments

### EXP-1 — PI shuffle control (mechanistic: signal vs regularizer)
**Question**: Is the persistence-image contribution a real *signal* tied to each edge, or just a *regularizer/noise* the model exploits regardless of which PI goes with which edge?
**Method**: Take an existing dionysus PI cache, randomly permute the PI rows across edges (destroying edge↔PI correspondence), save as a shuffled cache, run the standard 50-trial LP pipeline on it.
**Finding**: If shuffled-PI ≈ real-PI performance → PI is a regularizer, not signal. If shuffled-PI ≈ no-PI → PI signal is real and edge-specific. Run on **Chameleon** (where exact PI hurts) and **Photo** (homophilic).
**Hook**: `TLCGNN_PI_DIR` env var so pipeline reads an arbitrary PI cache directory.

### EXP-2 — Molecular PI resolution sweep (capacity)
**Question**: Is 5×5 PI the right resolution, or does finer detail help / overfit on molecular GC?
**Method**: Compute whole-graph PI at resolution 5/10/20 (25/100/400-dim) for MUTAG+PROTEINS, run GIN classifier with matching head dim, 10-fold CV.
**Finding**: Accuracy vs resolution curve. Likely sweet-spot or overfit at high res.
**Files**: new `mol_resolution_sweep.py` (reuses `graph_to_pi` with a resolution arg + GIN).

### EXP-3 — Molecular filter function sweep (which topology)
**Question**: Does the choice of filtration function change whether/how much topology helps molecular classification?
**Method**: Compute PI with filter ∈ {degree, clustering coefficient, closeness centrality} for MUTAG+PROTEINS, run GIN classifier.
**Finding**: Filter-sensitivity of topological signal. Some filters may carry more class-relevant structure.
**Files**: new `mol_filter_sweep.py`.

### EXP-4 — PD backend comparison (validation / reproduction-gap diagnosis)
**Question**: Do different PD computation backends produce the same PI? Could backend drift explain our PubMed/Computers gap vs paper?
**Method**: For a sample of edge-vicinities (Cora) and molecule graphs (MUTAG), compute extended PD via (a) the repo's `accelerated_PD`, (b) **GUDHI** (industry reference). Compare PIs (MSE) and downstream impact.
**Finding**: If backends disagree → our reproduction gap may be implementation drift. If they agree → gap is elsewhere (cap, hyperparams).
**Files**: new `pd_backend_compare.py`. Requires `pip install gudhi`.

### EXP-5 — Sparsity λ sweep (gating tuning)
**Question**: What sparsity strength best breaks gate saturation, and at what λ does the gate start discriminating domains?
**Method**: Sparsity-gating (already works) with `GATE_REG_LAMBDA` ∈ {0.01, 0.1, 0.5, 1.0} on Chameleon + ChChMiner, 50 trials each.
**Finding**: λ vs (AUC, mean-gate) curve → optimal regularization + saturation threshold.
**Hook**: make `GATE_REG_LAMBDA` an env var (`TLCGNN_GATE_LAMBDA`).

### EXP-6 — Low-data regime (topology as inductive bias)
**Question**: Does topology help more when training data is scarce (i.e., is PI a useful inductive bias under low data)?
**Method**: LP with train fraction ∈ {10%, 30%, 50%, 85%} on Chameleon + Photo, with-PI vs no-PI.
**Finding**: If PI gap grows as data shrinks → topology is a data-efficient prior. If gap constant → not data-dependent.
**Hook**: `TLCGNN_TRAIN_FRAC` env var in `get_adj_split`.

## Conflict management (parallelization)

- **New self-contained scripts (fully parallel)**: EXP-2 (`mol_resolution_sweep.py`), EXP-3 (`mol_filter_sweep.py`), EXP-4 (`pd_backend_compare.py`).
- **Shared-file env-var hooks (controller adds serially, then SLURM)**:
  - `loaddatas.py`: `TLCGNN_PI_DIR` (EXP-1), `TLCGNN_TRAIN_FRAC` (EXP-6)
  - `baselines/TLCGNN_gated_reg.py`: `TLCGNN_GATE_LAMBDA` (EXP-5)
- All env-var defaults preserve current behavior (backward compatible).

## Scope

- **In**: the 6 experiments above.
- **Out**: D1 attention fusion (new architecture, defer), F1 PDGNN transfer matrix (big, defer), A2 random-PI (subsumed by EXP-1 shuffle).

## Success criterion

Each experiment yields a committed result (table/plot) + a one-line finding for the results doc. Performance improvements are a bonus; null/negative results are documented as findings.

## Time estimate
- New-script experiments (EXP-2,3,4): ~1 day (mostly compute, small graphs)
- Env-var experiments (EXP-1,5,6): ~1-2 days SLURM (LP sweeps)
- Total: ~2-3 days parallel
