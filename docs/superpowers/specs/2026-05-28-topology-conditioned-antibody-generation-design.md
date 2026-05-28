# Topology-Conditioned Antibody Generation (DiffAb fine-tune) — Design Spec

**Date:** 2026-05-28
**Parent:** TDA conference project — the ambitious capstone (user-directed). Distinct from Thread C's NO-GO: that used topology as a *post-hoc evaluation metric* of binding (failed). This **injects multi-scale diffusion/topology information INTO the generative model** as conditioning, then fine-tunes — a different mechanism.

## Goal
Condition DiffAb's CDR-H3 generation on **per-residue multi-scale diffusion features (HKS)** of the antibody–antigen structural context, fine-tune from the pretrained checkpoint, and test whether topology-conditioned generation changes design quality (AAR / RMSD-CDRH3 / DockQ / diversity) vs the unconditioned baseline.

## Honest framing (not buried)
Thread C showed topology does **not predict** binding (DockQ) — so a binding *improvement* is **not guaranteed**. But conditioning is a *different mechanism* than prediction (it shapes what the model generates, can explore regions selection can't, and may help structure/diversity even if not DockQ). We measure **broadly** (AAR, RMSD, DockQ, diversity, topological fidelity) so the result is informative either way. This is an exploration with an honest prior, not a sure win.

## Architecture (verified from `FlowDesign/diffab/`)
- DiffAb denoising net consumes per-residue `res_feat` (N,L,res_feat_dim) built by `diffab/modules/encoders/residue.ResidueEmbedding` from `aa` (residue type) + heavy-atom positions + masks.
- **Injection point**: augment `ResidueEmbedding` to concatenate a per-residue **K-scale HKS** vector into `res_feat` (so the diffusion conditioning is available at every denoising step). Pretrained ckpt: `FlowDesign/trained_models/cdrh3.pt`. Fine-tune path exists (`models/rectflow_finetune.py` + `diffab/configs/train`). Env: `FlowDesign/env.yaml` (separate from `tlcgnn`).

## Topology feature
Per-residue **Heat Kernel Signature** of the antibody–antigen **Cα contact graph** (edge if Cα–Cα < cutoff, e.g. 8–10Å): graph Laplacian → eigendecomposition → `HKS_t(residue)` for t ∈ K log-spaced scales (local→global). K≈5. Computed on the GIVEN context (framework + antigen; the CDR being generated is masked) → conditioning is leakage-safe (uses context, not the answer). GPU-accelerable eigendecomp (uses idle GPUs).

## Phased plan (each phase is a GATE — verify before next; bail early if a phase fails)

**Phase 0 — Env + baseline repro.** Set up DiffAb env (`env.yaml`); load `cdrh3.pt`; generate CDR-H3 for a few SAbDab/RAbD test targets via `tools/runner/design_for_testset.py`; run the eval pipeline → **reproduce baseline DiffAb AAR/RMSD/DockQ**. GATE: can we run + eval DiffAb at all?

**Phase 1 — Topology feature pipeline.** Implement per-residue multi-scale HKS for the context contact graph; integrate into the data featurization (alongside aa/atoms). GATE: features computed, sane (nonzero, right shape, per-target).

**Phase 2 — Conditioning integration.** Augment `ResidueEmbedding` to accept + concat the HKS channels into res_feat; load `cdrh3.pt` with the enlarged input (new channels zero-init so initial behavior ≈ baseline). GATE: forward + a few denoising steps run without error; baseline-equivalent at init.

**Phase 3 — Fine-tune.** Fine-tune from `cdrh3.pt` on SAbDab time-split with augmented featurization (the new HKS channels learn). Monitor train/val loss vs baseline-finetune. GATE: loss converges, not diverged.

**Phase 4 — Generate + evaluate.** Generate CDR-H3 (test set) with the topology-conditioned model; run comprehensive_eval. Compare to baseline DiffAb on **AAR, RMSD-CDRH3, DockQ, diversity, + topological fidelity (PH of generated vs native CDR-H3)**. GATE: the result.

## Scope
- **In**: DiffAb (lightest, local, has ckpt). CDR-H3 codesign. SAbDab time-split (train) + test eval.
- **Out**: FlowDesign/IgGM (DiffAb first); multi-CDR (single-H3 first); other conditioning signals (HKS first; PD/persistence-image conditioning as a later variant if HKS works).

## Risks / mitigations
- DiffAb env/deps break → Phase 0 gate catches early.
- Conditioning destabilizes fine-tune → zero-init new channels (start = baseline); small LR; short fine-tune.
- No quality change → still a finding ("topology conditioning doesn't move antibody design quality"); measure broadly so it's informative.
- Compute: fine-tune is GPU-heavy/multi-day (user authorized). Use idle GPUs; checkpoint frequently.

## Success criterion
A committed comparison: topology-conditioned DiffAb vs baseline DiffAb on AAR/RMSD/DockQ/diversity/topo-fidelity, with each phase-gate documented. Improvement = bonus; null = a clean finding about whether injecting topology into the generative process matters. Integrate into results doc §16.
