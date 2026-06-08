# Chromatic-PDGNN: neural cross-type 0-dim persistence for heterogeneous graphs

**Date:** 2026-06-08
**Status:** design (brainstorming approved §1–§2 + "ㄱㄱ" on rest; user away, autonomous mandate)
**Repo:** TLC-GNN (remote `tda` = github.com/jjune5/TDA_conference)
**Related:** `docs/specs/2026-06-01-when-topology-helps-negative-result.md` (the 6× robust-null this builds on)

---

## 0. One-line

Extend PDGNN from a single-scalar-filter EPD approximator into a **neural approximator of the 0-dim image/kernel/cokernel ("chromatic") persistence of a typed-subcomplex inclusion** on heterogeneous graphs — putting node *type* into the union-find **mechanism** (not as an appended feature), to test whether *cross-type mingling* is the topological quantity that finally carries genuine signal where homogeneous-PD-with-a-type-tag was null 6×.

## 1. Motivation & grounding

Our project established a robust **null**: persistent-homology features, properly controlled (shuffled/random), give no genuine signal on node classification / link prediction across 6 settings — including the hetero attempts (Idea-1 meta-path, Idea-2 unified filter), which computed the *ordinary* PD on a colored/collapsed graph and concatenated the resulting PI. Lesson (S3): a typed PI is the **same homogeneous invariant with a type tag** — the math object never changed, hence null.

A genuinely different fix must change the **quantity** being measured, not add a tag. **Chromatic TDA** (Cultrera di Montesano, Draganov, Edelsbrunner, Saghafian — *Chromatic Topological Data Analysis*, arXiv 2406.04102, 2024) measures how *colors mingle* via the inclusion of a single-color subcomplex into the full complex, yielding the **kernel/image/cokernel "six-pack"** of persistence diagrams. The underlying image/kernel/cokernel persistence algorithm is **Cohen-Steiner, Edelsbrunner, Harer, Morozov — "Persistent Homology for Kernels, Images, and Cokernels"** (SODA 2009).

**Our contribution = combine both onto graphs + PDGNN neural approximation.** Honest boundary: chromatic TDA is defined for Euclidean point clouds (chromatic Delaunay/alpha); porting the typed-subcomplex inclusion to graph filtrations is *our* design. We claim only: (a) first neural (PDGNN-style) approximator of 0-dim image/kernel/cokernel persistence on graphs; (b) the PDGNN-derivation (its SUM⊕MIN union-find emulation extends naturally to the paired-graph union-find that computes image/ker/coker); (c) hetero node-classification application + characterization. We do **not** claim to have invented chromatic persistence or image-persistence.

## 2. Headline & staging (decision: "둘 다 단계적")

- **Stage 1 — method (stands even if downstream is null):** build the Chromatic-PDGNN engine; validate its predicted diagrams against exact union-find labels (fidelity gate).
- **Stage 2 — signal:** controls ladder + synthetic positive-control + real hetero NC. GO/NO-GO.

Either outcome is publishable: a win = chromatic cracks hetero where plain PH failed; a null = negative result #7, now strengthened by a *synthetic sensitivity proof* (we can show the pipeline detects chromatic signal when it exists by construction, so a real-data null is a real null, not a blind pipeline).

## 3. The invariant (decisions: 0-dim image/ker/coker; L = target-type-only; substrate = unified graph)

**Substrate (forced, not a choice):** the *unified multi-type ego-graph* `K` from `unified_filter.build_homo` — all node types present, **color = node type**. (Meta-path-collapsed graphs are monochromatic → chromatic is undefined there.)

**Filtration:** reuse the existing multi-scale **HKS lower-star** filtration `f` (K scales) from `pdgnn_metapath._graph_hks` / `_ego_filt_edges`. Edge filtration value = max of endpoint `f`.

**Inclusion:** `L ↪ K` where **`L` = induced subgraph on target-type nodes only** (e.g. papers), with the restricted filter. Inclusion is filtration-preserving.

**0-dim legs (computed by paired union-find as `f` rises):**

| Leg | Meaning | Signal |
|---|---|---|
| **Image** | birth in `L`, death in `K` | when same-type clusters merge in the full graph |
| **Kernel** ⭐ | classes separate in `L` but already merged in `K` | **same-type clusters bridged *only* by cross-type structure = cross-type mingling** |
| **Cokernel** | classes in `K` outside im(`L`) | pure non-target-type clusters |

**Kernel persistence is the chromatic heart** and the primary signal channel. It is a quantity a scalar filtration on a homogeneous graph provably cannot represent, and is mathematically distinct from the homogeneous-PD-with-tag that was null.

**Why PDGNN-derivation is clean:** PDGNN's SUM⊕MIN emulates union-find (Find-Root / Relax-Edge). 0-dim image/ker/coker are *also* union-find computations on the nested pair `(L, K)`. The engine = "PDGNN's union-find emulation, run on the pair, with a per-event leg readout." Exact labels via a custom ~100-line union-find (SODA'09 0-dim specialization) — **no Oineus dependency; exact used only as one-time training labels (PDGNN-only policy upheld).**

## 4. Engine (`chromatic_pdgnn.py`, mirrors `pdgnn_metapath.py`)

- **Input:** `(filter f (N,1), edge_index (2,E), color_mask (N,) ∈ {target, other})`. The `color_mask` is the new input — type enters the *mechanism*.
- **Output:** per-edge `(birth, death, leg_logits ∈ {image, kernel, cokernel, none})`; assemble 3 diagrams. (Each union-find merge event is attributed to a leg.)
- **Labels:** `chromatic_labels.py` exact 0-dim image/ker/coker per ego.
- **Loss:** per-leg Hungarian/bipartite (`Knowledge_Distillation.train_pdgnn_lp._bipartite_loss`) summed over legs; grad-clip 5.0 (loss-spike guard, as in `pdgnn_metapath`). *(Sliced-Wasserstein loss is a noted L5 drop-in improvement; Stage 1 keeps Hungarian to reuse infra.)*
- **Inference:** per target node, run engine on its unified ego → predicted image/ker/coker → PI each → concat `(N, 3 · 25 · K)`. No exact compute at inference.
- **Fidelity gate (Stage 2 GO/NO-GO):** on held-out egos, predicted-PI vs exact-PI per leg; require **kernel-leg cosine ≳ 0.8** (or 2-Wasserstein below a set bar). If the engine cannot approximate kernel persistence, Stage 2 is uninterpretable → fix the engine first.

## 5. Stage-2 signal test (`run_chromatic.py`, reuse backbones from `run_han_hgt.py` / `hetero_nc_pipeline.py`)

**Controls ladder (decision: achromatic control is the methodological core):**

| Variant | What |
|---|---|
| `none` | backbone only, no topo feature |
| `chromatic` | genuine image/ker/coker PI (candidate) |
| `achromatic` | **same PDGNN predicting ordinary H0 EPD of `K`, type-blind** — isolates whether the *cross-type* content adds anything over plain topology |
| `shuffled` | `chromatic` rows permuted across nodes |
| `random` | PDGNN on a random scalar filter |

**Verdict:** `chromatic` must beat **all four** (none, achromatic, shuffled, random). Beating `achromatic` is the decisive chromatic-specific test.

**Synthetic positive-control (`synth_chromatic.py`):** generate a hetero graph where each target node's label is *by construction* determined by a cross-type bridging pattern (two internally-disconnected same-type clusters joined by one of K distinguishable cross-type hub motifs; label = which motif), with node features and same-type-only topology made uninformative. **Gate:** `chromatic` must beat `achromatic`/`none` here. If not → the engine/pipeline is insensitive to chromatic signal → debug before trusting any real-data null.

**Datasets:** ACM, DBLP, Freebase (headroom); IMDB secondary (degenerate/multilabel — report with caveat).
**Backbones:** GCN, HAN, HGT — **cross-check all three** (the Idea-2 "win" was a single-cell GCN fluke; never trust one cell).

## 6. Files (all new, under `TLC-GNN/hetero/`)

| File | Role |
|---|---|
| `chromatic_labels.py` | exact 0-dim image/ker/coker union-find labeler + **TDD unit tests** on small hand-checkable graphs |
| `chromatic_pdgnn.py` | PDGNN extension (color-mask input, per-edge leg head), train, per-node 3-leg PI |
| `synth_chromatic.py` | planted-bridge synthetic hetero generator (positive control) |
| `run_chromatic.py` | Stage-2 driver: controls × datasets × backbones, results table |

**Reuse:** `unified_filter.build_homo`, `pdgnn_metapath._graph_hks` / `_ego_filt_edges` / training scaffold, `Knowledge_Distillation.pdgnn_modern.PDGNN`, `_bipartite_loss`, GCN/HAN/HGT nets, controls in `hetero_nc_pipeline.py`. **Do not** refactor unrelated code ([[explicit-changes-only]]).

## 7. Honest risks

1. **0-dim may be too weak even on real data** — cross-type *cycles* (H1) may be where the action is. Contained by the staged design + synthetic control; H1 image-persistence is a Stage-3 extension (needs Oineus-style labels).
2. **PDGNN may approximate kernel persistence poorly off-distribution** (kernel is rarer/sparser than ordinary H0) — the fidelity gate catches this before Stage 2.
3. **Could be null #7** — acceptable; with the synthetic sensitivity proof it is a *stronger* negative result than a bare null.
4. **Leakage check (§14 discipline):** features are PDGNN-predicted from label-free HKS topology; no test-node label flows in. Verify no train/test PI inconsistency before trusting any win.

## 8. Success criteria

- **Stage 1:** engine trains stably; kernel-leg fidelity ≥ gate on held-out egos; reproduced on ≥2 datasets.
- **Stage 2:** synthetic positive-control passes (chromatic > achromatic/none); real-data verdict (win or null) is consistent across GCN/HAN/HGT and reported honestly with the full controls ladder.

## Stage-1 Results (2026-06-08, in progress)

**Labeler (`chromatic_labels.py`) — validated.** Exact 0-dim image/kernel/cokernel via
paired union-find (Lp = L-edges, KLp = K-connectivity of L-vertices; kernel = relative
persistence between them). All 4 legs match the homological rank function at every
inter-event midpoint over 300+ random graphs (0 mismatches); 9 hand-computed cases pass;
ordinary leg matches gudhi. 11/11 pytest pass.

**Key characterization (learned during Stage-1).** Under a node (lower-star) filtration,
0-dim chromatic **kernel** persistence does NOT see a cross-type bridge's internal
structure/length: any path a..b has max-edge ≥ max(f_a,f_b), so a 1-hop vs 2-hop bridge
gives the same single essential kernel class. What 0-dim kernel DOES encode is (i) the
**count** of same-type components held together only by cross-type structure
(= b0(L) − #L-bearing-K-comps) and (ii) the scalar **time** they join. ⇒ a global
filtration **cap** (graph-wide max per scale) is needed so top-born essential kernels stay
visible & comparable across egos (per-ego max drops them). This bounds what the idea can
detect: cross-type *mingling count/timing*, not bridge topology (H1 chromatic would be the
Stage-3 extension for structural bridges).

**Engine (`chromatic_pdgnn.py`) — built & smoke-validated.** ChromaticPDGNN = PDGNN with
input [HKS filter ∥ color] and 3 leg-heads; per-leg Hungarian; gudhi-free pipeline
(achromatic baseline reuses the type-blind unified ordinary EPD). Forward/backward/train/
predict smoke pass; per-node PI shape (N, 3·25·K).

**Synthetic positive control — PASS (the sensitivity proof).** Task: label = #same-type
clusters bridged only cross-type (kernel count 1 vs 2); K connected both classes so
ordinary H0 is blind. n=600, global cap: **kernel AUC 0.731 vs achromatic 0.495 (chance)**
→ the exact-label + PI pipeline DOES detect chromatic signal when present by construction.
(kernel < 1.0 because PI smears the count by random birth location — expected.) This lets a
real-data null be read as a true null rather than a blind pipeline.

**Pending:** real-data fidelity gate (does the NEURAL engine approximate exact kernel,
kernel cosine ≥ 0.80 on ACM) → then Stage-2 controls grid.

## 9. Out of scope (YAGNI)

Per-color six-pack concat (Stage-3 richness), H1/higher-dim chromatic, multiparameter (type-as-2nd-axis / Graphcode), learnable filtration (user rejected), Sliced-Wasserstein loss (noted drop-in, not Stage 1).
