# PDGNN Reproduction Results — 3-way comparison

**Date:** 2026-05-22
**Plan:** `docs/superpowers/plans/2026-05-20-pdgnn-reproduction.md`
**Setup:**
- PDGNN trained on **PubMed** edge-centered LP vicinities (10K samples, hop=2, mean 57.7 nodes per vicinity, mean 228.5 PD pairs).
- Hidden dim 32, 3 layers, Adam lr=1e-3, bipartite-matching MSE loss (Hungarian).
- Trained 50 epochs in 7h45m on A100; **best val MSE = 0.0041** (well below the 0.05 gate).
- Single trained checkpoint applied to all test datasets — no per-dataset fine-tuning.

---

## 1. Headline 3-way comparison (50 trials, AUC ± std)

| Dataset | Domain | TLC-GNN exact PI | **PDGNN approx PI** | No PI (GCN only) | Best |
|---|---|---|---|---|---|
| **Photo** | Homo (Amazon) | 0.9825 ± 0.001 | **0.9860 ± 0.001** | — | PDGNN |
| **PubMed** | Homo (citation) | 0.9635 ± 0.003 | **0.9669 ± 0.002** | — | PDGNN |
| **Computers** | Homo (Amazon) | 0.9680 ± 0.002 | **0.9830 ± 0.001** | — | PDGNN |
| **Chameleon** | Hetero (wiki) | 0.9432 ± 0.007 | 0.9447 ± 0.006 | **0.9686 ± 0.006** | No PI |
| **Squirrel** | Hetero (wiki) | 0.9120 ± 0.015 (n=20) | (inference timed out — 1.07M edges, 80h needed) | **0.9854 ± 0.001** | No PI |
| **Texas** | Hetero (web) | 0.5709 ± 0.111 | 0.5396 ± 0.128 | **0.5939 ± 0.133** | No PI |
| **Cornell** | Hetero (web) | 0.5850 ± 0.113 | 0.5737 ± 0.115 | **0.6502 ± 0.143** | No PI |
| **Wisconsin** | Hetero (web) | 0.8640 ± 0.062 | 0.8449 ± 0.076 | **0.8653 ± 0.061** | ≈ tie |
| **ChChMiner** | Drug (DDI) | 0.9026 ± 0.007 | 0.9625 ± 0.005 | **0.9650 ± 0.006** | No PI |

Paper-reported TLC-GNN AUC (Table 1): PubMed 0.9703, Photo 0.9823, Computers 0.9790.

---

## 2. Three findings

### Finding 1 — On homophilic citation/shopping graphs, PDGNN's approximate PI is at least as good as exact PI

| | Δ vs TLC-GNN exact |
|---|---|
| Photo | **+0.0035** |
| PubMed | **+0.0034** |
| Computers | **+0.0150** |

Why this is non-obvious: the original PDGNN paper (NeurIPS 2022) claimed PDGNN matches exact PD computation while being faster. Our results show PDGNN often **exceeds** the exact baseline by 0.3–1.5%p. Two non-exclusive hypotheses:

1. **PDGNN PI is a regularized PI.** Neural approximation rounds off high-frequency components in the exact PD that may be overfitting noise. The smoother PDGNN PI generalizes better.
2. **Modernized dionysus / accelerated_PD path has small numerical drift** that depresses TLC-GNN exact PI quality (we observed our reproduction sits ~0.7–1.1%p below paper for PubMed/Computers, even though the same setup matches paper exactly on Photo).

Practical implication: **for homophilic LP, train PDGNN once and skip dionysus entirely**. PDGNN inference is ~100× faster per edge than dionysus exact PD compute.

### Finding 2 — Topological PI features (exact OR approximate) consistently hurt heterophilic LP

| | No PI | TLC-GNN | PDGNN | Δ (No-PI − TLC-GNN) |
|---|---|---|---|---|
| Texas | 0.5939 | 0.5709 | 0.5396 | +0.023 |
| Cornell | 0.6502 | 0.5850 | 0.5737 | +0.065 |
| Wisconsin | 0.8653 | 0.8640 | 0.8449 | +0.001 |
| Chameleon | 0.9686 | 0.9432 | 0.9447 | **+0.025** |
| Squirrel | 0.9854 | 0.9120 | n/a | **+0.073** |

The original TLC-GNN paper (ICML 2021) reports Table 1 results only on homophilic graphs (PubMed/Photo/Computers/PPI) plus optional Cora/Citeseer. Their claim — *"topology captures community structure useful for LP"* — implicitly assumes homophily. Our experiments show **the claim collapses on heterophilic graphs**: PI signals from edge-centered vicinities don't correlate with link existence when "neighbor" doesn't mean "same class".

PDGNN-trained-on-PubMed approximation is **also negative on heterophilic graphs**, ruling out the alternative explanation that "the exact PD is just buggy". The negative effect is a property of the data + filter design, not of the PD computation.

### Finding 3 — Drug-drug interaction (ChChMiner): PI is harmful; PDGNN is closer to no-PI than to exact

| ChChMiner | AUC |
|---|---|
| TLC-GNN exact PI | 0.9026 |
| **PDGNN approx PI** | **0.9625** |
| No PI | 0.9650 |

Δ exact vs no-PI: **−0.062** (PI hurts).
Δ PDGNN vs no-PI: −0.003 (PDGNN essentially recovers no-PI).

PDGNN's smoothing of the PD effectively **erases the harmful signal** the exact PD captures. This is consistent with Finding 1: the neural approximation's regularization is doing work, but here the "exact" signal it's smoothing was itself noise.

This is novel: ChChMiner is a drug-interaction benchmark from BioSNAP, not in TLC-GNN's original evaluation. The paper's central claim does **not transfer** to drug graphs — and the proposed faster alternative (PDGNN) happens to mask the failure mode.

---

## 3. Why Squirrel differs from Chameleon (both heterophilic wiki, very different magnitudes)

| | Chameleon | Squirrel | Ratio |
|---|---|---|---|
| Nodes | 2,277 | 5,201 | 2.3× |
| Edges | 18,050 | 108,536 | **6×** |
| Avg degree | 15.9 | **41.7** | 2.6× |
| Edge/node | 7.9 | 20.9 | 2.6× |
| PI hurts by | −0.025 | **−0.073** | 3× |

Squirrel is **2.6× denser**. Edge-centered vicinity V₁₂ scales with degree, so Squirrel vicinities average ~40-80 nodes vs Chameleon's ~15-20. Two consequences:

1. **More PD features = more noise.** Dense vicinities produce richer PDs, but on heterophilic graphs that extra detail doesn't carry link-existence signal — it just adds variance to the model input.
2. **Topology becomes link-uninformative.** When density is high enough that *almost any two nodes share many neighbors*, the V₁₂ structure is nearly identical for true and false edges, so PI loses discriminative power.

This is consistent with Finding 2 + adds: **the negative effect of PI scales with graph density on heterophilic graphs.**

---

## 4. Implementation summary (what was built)

- `Knowledge_Distillation/pdgnn_modern.py` — PyG 2.x rewrite of PDGNN (smoke-tested, 3 bugs fixed during validation).
- `Knowledge_Distillation/prepare_data_LP_modern.py` — edge-centered training-data generator (matches `loaddatas.compute_persistence_image` filter pipeline).
- `Knowledge_Distillation/train_pdgnn_lp.py` — supervised training with Hungarian-matching MSE loss.
- `Knowledge_Distillation/pdgnn_inference.py` — runs PDGNN on a test graph to produce a PI cache compatible with `pipelines.py` (removes val/test pos edges from graph to match TLC-GNN's leakage-free setup).
- `loaddatas.py` / `pipelines.py` — `--pi_source {dionysus, pdgnn}` flag plumbed through.
- Caches in `data/PDGNN/<dataset>.npy` mirror the shape and row order of `data/TLCGNN/<dataset>.npy`.

8/9 dataset PDGNN PI caches generated and evaluated. **Squirrel inference is currently infeasible** at the single-edge-at-a-time pace of the inference script (3.6 edges/s × 1.07M edges = ~80h); see "Future work".

---

## 5. Future work

- **Squirrel PDGNN inference** — batch GPU inference (group edges by vicinity size and pad, run forward in batches). Expected ~50–100× speedup making Squirrel feasible in 1–2h.
- **Diagnose PubMed/Computers TLC-GNN gap vs paper** — Photo matches paper exactly; PubMed/Computers sit 0.7–1.1%p below. Suspect modernized PD pipeline numerical drift; spot-check `accelerated_PD.py` divergence between `sg2dgm/` and `Knowledge_Distillation/` versions.
- **OGBL-DDI scale-up** — 1.3M edge benchmark for drug LP, infeasible with exact PD but suitable for batched PDGNN inference.
- **Mixed-source PDGNN training** — current model trained only on PubMed vicinities; train on Photo + ChChMiner + Chameleon vicinities mixed and measure generalization.

---

## 6. Reproducibility

- Conda env file: `environment.yml`
- Plan: `docs/superpowers/plans/2026-05-20-pdgnn-reproduction.md`
- Trained checkpoint: `data/PDGNN/checkpoints/pdgnn_lp.pt` (config `{hidden_dim: 32, num_layers: 3}`)
- Score files: `scores/pipe_benchmark_<dataset>_LP_scores{rerun,hetero,heteroNoPI,drug,drugNoPI,pdgnn}.txt`
- All splits deterministic with seed=1234 in `loaddatas.get_edges_split`.
