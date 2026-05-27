# Diffusion meets Topology — Chapter Design Spec

**Date:** 2026-05-27
**Parent:** TDA conference project (TLC-GNN/PDGNN). New chapter extending the "when is topology a useful signal?" thesis into diffusion.

## Unifying thesis: two faces of diffusion

"Diffusion" means two different things; we use both, unified by persistent homology (PH).

- **Physical diffusion** (heat kernel / random walk on a graph): deterministic, parameter = time t = *scale*. Threads A, B.
- **Generative diffusion** (DDPM-style learned denoising): generates new structures. Thread C.

## Thread A — physical diffusion as a topology LENS (HKS)

**Idea:** Heat Kernel Signature from the graph Laplacian (`H_t = exp(-tL)`, `HKS_t(i) = Σ_k exp(-t λ_k) φ_k(i)²`) used as a filtration function for PD/PI. Diffusion-derived filtration vs geometric (degree/Ricci).

**Scope (autonomous run):** MOLECULAR domain only (clean, no shared-file conflict with B).
- New `hks_filtration.py`: Laplacian eigendecomp → HKS at scale(s) t.
- Extend `mol_filter_sweep.py` to add `hks` alongside degree/clustering/closeness.
- Run on MUTAG/PROTEINS/NCI1, 10-fold CV.

**Deliverable:** filter-comparison table incl. HKS; finding on whether diffusion lens beats geometric.
**Risk:** low. Null result = finding.

## Thread B — physical diffusion as a DENOISER (GDC)

**Idea:** Graph Diffusion Convolution (heat/PPR diffuse + sparsify) smooths the graph; compute PI on the diffused graph. Hypothesis (from §EXP-1 shuffle finding): on heterophilic graphs the PI signal points the wrong way — does diffusion-denoising *rescue* PI?

**Scope:** LP domain. New `gdc_pi.py` + a `loaddatas.py` GDC hook (env var `TLCGNN_GDC`). 3-way: PI-on-GDC vs PI vs no-PI, homophilic (Cora/Photo) + heterophilic (Chameleon/Texas/Cornell).
**Deliverable:** does GDC-PI close the hetero PI-hurt gap? table.
**Risk:** low–moderate.

## Thread C — generative diffusion EVALUATION & IMPROVEMENT (antibody CDR-H3)

**Data (all open-source, local, zero generation cost):**
- Native + generated, co-located: `/mnt/data/users/junyoungpark/code/antibody_models/FlowDesign/results/codesign_single_H3_{rabd,time_split}/<target>/` — each target has `reference.pdb` (native), `H_CDR3/0000.pdb…` (generated samples), `metadata.json`.
- **Per-sample metrics already computed**: `…_comprehensive_eval/per_sample_metrics.csv` with `AAR H3, RMSD(CA) CDRH3, TMscore, LDDT, DockQ, pmetric`.
- Second model: `…/IgGM/outputs/sample_pretrained/<pdb>_CDR_H3.pdb` (447 targets).
- CDR annotations (IMGT): `…/SAbDab_split_time/{train,valid,test}.json` (JSONL: `cdrh3_pos`, `cdrh3_seq`).
- DiffAb code: `…/FlowDesign/diffab/` (modules/diffusion + modules/rectified_flow). Light model if re-generation needed.
- Generated structures are full complexes (chains H/L/antigen) → DockQ valid.

**Metric:** CDR-H3 (+ flanking anchor residues so the loop closes → non-trivial H1) Cα coords → Vietoris-Rips PH (GUDHI; point-cloud Rips is the right tool here — distinct from the LP graph-filtration decision) → PD (H0,H1) → topological distance to native (bottleneck/Wasserstein) and/or PI. **Orientation-invariant by construction** (rigid-motion invariant) → complements alignment-based RMSD.

**The 3-rung ladder (climb as results justify):**
- **Rung 0 — selection / re-ranking (GO/NO-GO gate, autonomous run):** per target, score the K candidates by topological fidelity; test (C1) does topo-fidelity correlate with DockQ across candidates? (C2) does topology-based selection yield higher DockQ/lower RMSD-CDRH3 than random or pmetric selection? (C3) is topology orthogonal to RMSD/pmetric (complementary)? **If no DockQ signal → premise dead, stop C, document finding.**
- **Rung 1 — sampling guidance (next, if Rung 0 positive):** add a differentiable-PH guidance term to DiffAb reverse diffusion (guide Cα toward a topological *prior* — e.g., encourage persistent H1 / native-PD-distribution density, since native is unknown at inference). Measure generated DockQ distribution shift.
- **Rung 2 — training loss (stretch):** fine-tune DiffAb with `λ·L_topo` (PD distance to native, available at train time).

**Enabler:** persistence is differentiable a.e. (`torch_topological` / topology-layer). Caveats: sparse/noisy gradients, per-step PH cost, weight tuning.

**Honest limitation:** PH-on-Cα discards backbone orientation/geometry → coarse shape-only lens; complementary metric, not a complete quality measure. State explicitly.

## Conflict management (parallel autonomous execution)

- 3 background agents in **isolated git worktrees** → no file/git conflicts.
- Disjoint ownership: A = molecular files (`hks_filtration.py`, `mol_filter_sweep.py`); B = LP (`gdc_pi.py`, `loaddatas.py` GDC hook); C = new `antibody_tda/` (reads sibling antibody data via absolute path).
- Base datasets/caches read from main repo via absolute path; results written + committed **locally in each worktree** (controller merges).
- **No remote push by agents** (governance review first; controller/user pushes).

## Sequencing & success

- Phase 1 (parallel, tonight): A (mol HKS), B (GDC LP), C-Rung0 (antibody selection GO/NO-GO).
- Phase 2: C-Rung1 guidance (if Rung0 positive); LP-HKS (fold into B's area later).
- Phase 3: C-Rung2 training (stretch); integrate into chapter doc/slides.
- **Success:** each thread yields a committed result + one-line finding. Performance gains a bonus; null results documented.
