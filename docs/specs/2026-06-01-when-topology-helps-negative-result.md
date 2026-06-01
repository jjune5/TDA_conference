# When Does Topology Help? A Controlled Negative Result for PH-as-Feature

**Date:** 2026-06-01
**Scope:** Synthesis only — no new experiments. All numbers read directly from
`results/*/nc_acc.csv` and `results/*/lp_auc.csv` in this repo (file paths cited inline).
**Positioning target:** Papamarkou et al., *Position: Topological Deep Learning is the New
Frontier for Relational Learning* (ICML 2024, arXiv:2402.08871), Open Problems #1, #9, #10.

---

## 0. TL;DR

We took the *oldest and simplest* recipe in topological deep learning — compute persistent
homology (PH), turn it into a fixed feature (a persistence image / extended persistence
vector), and concatenate it onto a GNN's node/edge representation — and asked, across a
controlled grid, **whether that feature carries genuine task signal once you account for
the obvious confounds**. The ICML 2024 position paper itself flags this recipe as the
weakest, "input-level" form of TDL and asks (Open Problem #1) for *compelling applications
where topology shows a competitive edge*, while warning (the "not all higher-order relations
are created equal" caveat) that we still do not know *when* higher-order / topological
structure becomes useful.

Our answer, on a deliberately honest grid, is a **clean negative**: with
`shuffled` / `random` / `none` controls in place, **PH-as-feature shows no genuine signal**
for (a) link prediction on heterophilic graphs and (b) heterogeneous node classification
across 4 HGB datasets × {GCN, HAN, HGT} × {meta-path PDGNN-EPD (Idea 1), unified type-aware
EPD (Idea 2)}. The one apparent win (ACM-GCN, Idea 1) **did not survive a confirmation grid**
and is consistent with control-band noise. This is not a claim that *topology never helps* —
we explicitly did **not** test architecture-level TDL (message passing on simplicial/cell
complexes), and we have separate *positive* results elsewhere in the project
(molecular graph classification; node-level diffusion features). It is the narrower,
defensible claim the field needs: **a controlled data point on where input-level PH does
*not* clear its own controls.**

---

## 1. The full grid we ran

Two task families, both with the **same three-control protocol** (`hetero/hetero_nc_pipeline.py`,
docstring lines 5–8):

- **`none`** — backbone only, no PH. Headroom baseline.
- **`ph` / `ph_pdgnn` / `ph_unified` / `epd`** — backbone + the PH feature under test
  (z-normalized so it is not drowned out; `_znorm`, pipeline L35).
- **`shuffled`** — same PH rows, **randomly permuted across nodes** (destroys node↔PH
  correspondence; isolates "is the *specific* topology of *this* node load-bearing, or just
  the marginal distribution of PH values?").
- **`random`** — PH computed from a **random node filter** function instead of the real
  one (isolates "does the *structure* the filtration reads matter, or only that we added a
  75-d vector of the right shape/scale?").

A genuine topological signal must beat **all three** controls, not just `none`. If `shuffled`
or `random` match the `ph` variant, the apparent lift is a dimensionality / regularization /
scale artifact, not topology.

### 1a. Heterogeneous node classification — HGB datasets

Backbone GNNs on meta-path-induced or type-aware homogeneous graphs
(`hetero/metapath_graph.py`: `METAPATHS`, `TARGET = {ACM:paper, DBLP:author, IMDB:movie,
Freebase:book}`; loaded via `HGBDataset`). PH feature is **PDGNN-predicted EPD** (75-d),
never exact PI (project policy). Idea 1 = per-meta-path PDGNN-EPD; Idea 2 = single PDGNN over
the whole type-aware homogeneous graph ("unified" filter). 10 trials each unless noted.
Metric = accuracy, except IMDB = Macro-F1 (multi-label).

**GCN backbone, Idea 1 (meta-path PDGNN-EPD)** — `results/hetero_pdgnn_{ACM,DBLP,IMDB,Freebase}/nc_acc.csv`

| Dataset (metric) | none | ph_pdgnn | shuffled | random | Verdict |
|---|---|---|---|---|---|
| ACM (acc)        | 0.8188 ±0.069 | **0.8536 ±0.060** | 0.7852 ±0.100 | 0.8481 ±0.078 | ph > none, but `random` 0.848 ≈ ph → **not clean** |
| DBLP (acc)       | 0.8089 ±0.006 | 0.8018 ±0.005 | 0.8050 ±0.003 | 0.8100 ±0.011 | ph ≤ none, ≤ random → **null** |
| IMDB (MacroF1)   | 0.4678 ±0.012 | 0.4502 ±0.007 | 0.4603 ±0.009 | 0.4498 ±0.010 | ph ≤ none → **null/harmful** |
| Freebase (acc)   | 0.6142 ±0.005 | 0.6103 ±0.009 | 0.6155 ±0.010 | 0.6141 ±0.006 | ph ≤ none ≈ controls → **null** |

**GCN backbone, Idea 2 (unified type-aware EPD)** — `results/unified_{ACM,DBLP,IMDB}/nc_acc.csv`

| Dataset (metric) | none | ph_unified | shuffled | random | Verdict |
|---|---|---|---|---|---|
| ACM (acc)        | 0.8188 ±0.069 | **0.8762 ±0.006** | 0.8008 ±0.082 | 0.7635 ±0.091 | ph > all controls — **the one apparent win** |
| DBLP (acc)       | 0.8089 ±0.006 | 0.8122 ±0.006 | 0.8096 ±0.006 | 0.8069 ±0.005 | +0.3%p over none, within band → **null** |
| IMDB (MacroF1)   | 0.4680 ±0.013 | 0.4611 ±0.007 | 0.4602 ±0.011 | 0.4612 ±0.009 | ph ≤ none → **null** |

**HAN / HGT backbones** — `results/hanhgt_{ACM,IMDB}/nc_acc.csv` (Idea 1),
`results/hanhgt_uni_{ACM,DBLP}/nc_acc.csv` (Idea 2). PH = EPD (75-d).

| Dataset | Backbone | EPD src | none | epd | shuffled | random | Verdict |
|---|---|---|---|---|---|---|---|
| ACM (acc)     | HAN | Idea1 | 0.6650 ±0.025 | 0.7368 ±0.091 | **0.7460 ±0.095** | 0.7140 ±0.064 | shuffled ≥ epd → **null** |
| ACM (acc)     | HGT | Idea1 | 0.7203 ±0.059 | 0.7466 ±0.032 | 0.7379 ±0.060 | 0.7370 ±0.097 | within band → **null** |
| IMDB (MacroF1)| HAN | Idea1 | 0.4702 ±0.015 | 0.5000 ±0.017 | 0.4824 ±0.031 | **0.5047 ±0.014** | random ≥ epd → **null** |
| IMDB (MacroF1)| HGT | Idea1 | 0.5861 ±0.002 | 0.5842 ±0.003 | 0.5803 ±0.002 | 0.5730 ±0.004 | epd ≈ none → **null** |
| ACM (acc)     | HAN | Idea2 | 0.6631 ±0.021 | 0.7247 ±0.060 | 0.7196 ±0.086 | **0.7344 ±0.036** | random ≥ epd → **null** |
| ACM (acc)     | HGT | Idea2 | 0.7350 ±0.060 | 0.7503 ±0.050 | 0.7278 ±0.086 | 0.6620 ±0.080 | within band → **weak/null** |
| DBLP (acc)    | HAN | Idea2 | **0.9155 ±0.004** | 0.7845 ±0.164 | 0.8378 ±0.185 | 0.9178 ±0.006 | epd **harmful**, random = none → **null/harmful** |
| DBLP (acc)    | HGT | Idea2 | 0.8474 ±0.031 | 0.8274 ±0.011 | 0.8445 ±0.025 | 0.8342 ±0.017 | epd ≤ none → **null** |

For reference, the **exact-PI** Idea-1 hetero runs (variant `ph`, not PDGNN; included only as
context, project policy is PDGNN-EPD for the final feature) tell the same story:
`results/hetero_ACM/nc_acc.csv` ACM none 0.8188 / ph 0.8254 / shuffled 0.8350 / random 0.8482
(shuffled and random both ≥ ph); `results/hetero_DBLP/nc_acc.csv` DBLP none 0.8089 / ph 0.8040
/ shuffled 0.7960 / random 0.8102 (ph ≤ none).

**Grid summary (hetero NC):** across 4 datasets × 2 backbone families (GCN; HAN/HGT) × 2
designs (Idea 1; Idea 2), **15 of 16 controlled cells are null** (PH ≤ a control, or within
one std of `none`). The lone exception is **ACM under Idea-2/GCN** (0.8762 vs best control
0.8008), discussed next.

### 1b. The ACM-GCN "win" did not reproduce

The Idea-2/GCN ACM cell (`results/unified_ACM/nc_acc.csv`) is the only place where the PH
variant beats `shuffled`, `random`, **and** `none` outside the noise band. We treated it as a
candidate positive and ran a **confirmation grid** swapping the GCN backbone for the
purpose-built heterogeneous backbones (HAN, HGT) on the *same* unified EPD feature
(`results/hanhgt_uni_ACM/nc_acc.csv`). There the effect **collapsed into the control band**:
HAN epd 0.7247 vs random 0.7344 (random *higher*); HGT epd 0.7503 vs shuffled 0.7278 (Δ within
±0.05–0.09 std). Git history records this directly:

> `dd1e397 exp(hetero): Idea-2 unified-EPD 확인 그리드 = ACM-GCN 승리 비재현(null)`

So the single ACM-GCN cell is best read as a **backbone-specific fluke** (GCN on a dense
type-aware graph is the weakest, highest-variance baseline — note `none` std = 0.069 there),
not a genuine topological signal. Honest accounting: **0 of 16 cells survive as a reproducible
controlled positive.**

### 1c. Link prediction context (§14 centerpiece + DNP)

The hetero NC null is the same phenomenon the project's LP work isolated mechanistically.
On link prediction (`docs/specs/2026-06-21-tda-conference-results.md` §14, reproduced via
`results/pi_artifact_all/pi_artifact_all.csv`), **exact PI on LP is a train-graph edge-membership
artifact**: standard anti-leakage deletes val/test positive edges from the training graph, so
train-positive edges have PI ≫ 0 while every test edge (positive *or* negative) has PI ≈ 0 →
**zero test discriminative signal**. Direct evidence: across 8/9 LP datasets, exact `test_pos`
non-zero fraction ≈ 0% (e.g. Photo 0.0%, Computers 0.0%, Chameleon 0.2%, Squirrel 0.1%;
`pi_artifact_all.csv`), and a leave-one-out control finds **100% collapse** — removing an edge
drops its vicinity PI to *exactly* 0 (300/300 edges; `results/consistent_pi/loo_pi_*.csv`).
The heterophilic LP "PI hurts" effect (Chameleon exact-PI 0.9442 vs no-PI 0.9684;
`results/dnp_A/lp_auc.csv`) is overfitting to that spurious membership signal.

This is the **LP analogue of the hetero-NC null**: in both, the input-level PH feature lacks
genuine *test-time* signal once leakage / controls are accounted for. (The project's *positive*
LP result — node-level diffusion / DNP features that are edge-presence-independent, Chameleon
0.9816, `results/dnp_A/lp_auc.csv` — is a different feature class, not PH-as-feature, and is
discussed only as the contrasting positive in §4.)

---

## 2. Methodological contribution: controls that pop the bubble

The reusable contribution here is **not** "PH failed" — it is the **control design that
separates genuine topology from three cheap confounds** that routinely inflate input-level TDL
results:

1. **Saturation / headroom (`none` baseline).** If the backbone already saturates the task
   (DBLP-HAN at 0.9155, IMDB-HGT at 0.5861), there is no room for any feature to help, and a
   PH "lift" over a *weaker* config is meaningless. The `none` control fixes the ceiling.
2. **Dimensionality / regularization (`random` control).** Concatenating *any* well-scaled
   75-d vector can act as a regularizer or break a degenerate optimization. `random` =
   PH from a **random filter** has the right shape and z-norm but no real structure. When
   `random` matches `ph` (ACM-GCN-Idea1 0.848 vs 0.854; IMDB-HAN 0.505 vs 0.500), the lift is
   dimensionality, not topology.
3. **Leakage / membership (`shuffled` control + §14 LOO).** The PH value may encode *graph
   membership* rather than task-relevant shape. `shuffled` destroys the node↔PH alignment; if
   it matches `ph` (DBLP exact-PI shuffled 0.796 ≥ ph 0.804; ACM-HAN shuffled 0.746 ≥ epd
   0.737), the signal was not node-specific topology. On LP, the leave-one-out collapse
   (§14, 100%/300 edges) is the sharpest form of this control.

**A PH feature is only credited when it beats `none`, `shuffled`, AND `random` by more than the
control band.** Applying this rule mechanically, 15/16 hetero cells are null and the 16th does
not reproduce. We argue this three-control protocol is a low-cost, high-value default that the
position paper's call for "compelling applications" (Open Problem #1) implicitly requires:
without it, an apparent edge cannot be distinguished from saturation/leakage/dimensionality.

---

## 3. Connection to the ICML 2024 position paper's open problems

Papamarkou et al. (arXiv:2402.08871) enumerate open problems for topological deep learning.
Our controlled null is a direct, honest data point against three of them:

- **Open Problem #1 — "compelling applications where topology has a competitive edge."**
  The position paper explicitly asks for applications demonstrating that topological features
  yield a *competitive edge* over strong non-topological baselines. We provide the
  complementary negative half of that evidence ledger: on heterogeneous node classification
  (the canonical "relational learning" setting the paper foregrounds), the simplest TDL recipe
  (PH-as-feature) shows **no competitive edge over a `none` baseline once controlled**. A field
  searching for where topology *does* win benefits from a clearly-bounded map of where the
  cheapest version *does not* — so effort is not wasted re-discovering input-level nulls.

- **Open Problem #9 / the "not all higher-order relations are created equal" caveat —
  *when* do higher-order/topological relations become useful?** The paper warns that adding
  higher-order or topological structure is not automatically beneficial and that the field
  lacks principled answers for *when* it helps. Our grid is a small empirical instance of that
  warning made concrete: meta-path PH (Idea 1) and whole-graph type-aware PH (Idea 2) — two
  reasonable "higher-order relation" encodings — are **both null on the same four datasets**,
  and the apparent ACM win is backbone-dependent. The *contrast* within the broader project
  (PH helps molecular graph classification: MUTAG +1.6%p, the signal living in H1/rings; vs PH
  null on hetero NC and LP) is itself a "when" data point: **task structure (graph-level shape
  vs node/edge-local prediction), not the topology computation, decides the sign of the
  effect.**

- **Open Problem #10 — benchmarking / honest evaluation of TDL.** The paper calls for rigorous
  benchmarking practices. Our three-control protocol (`none`/`shuffled`/`random`) and the
  leave-one-out leakage control are exactly the kind of evaluation hygiene that distinguishes
  a real TDL gain from an artifact, and we show (ACM-GCN) that *without* a confirmation grid
  one would have reported a spurious positive. This is a concrete benchmarking recommendation:
  **input-level PH results should be reported against shuffled/random controls and a
  backbone-swap confirmation, by default.**

---

## 4. Honest scope and limitations

We are deliberately narrow about what this negative result does and does not cover.

- **We tested *PH-as-feature* (input-level TDL) only.** This is exactly the form the position
  paper characterizes as the older, weakest rung of TDL: compute PH, vectorize (persistence
  image / extended-persistence vector), concatenate to a GNN. We did **not** test
  *architecture-level TDL* — message passing on simplicial complexes, cell complexes,
  combinatorial complexes, or hypergraphs, which is what the position paper actually advocates
  as "the new frontier." Our null says nothing about those; if topology has a competitive edge,
  the paper's thesis is that it will come from *architecture*, not from a concatenated feature,
  and our result is consistent with (does not test) that thesis.

- **PH feature = PDGNN-predicted EPD, with exact PI used only as training labels** (project
  policy). The differentiable EPD predictor (`Knowledge_Distillation/pdgnn_modern.py`,
  `hetero/pdgnn_metapath.py`) is itself the project's improvement over exact PI; that the
  *learned, leakage-free* PH feature is also null on hetero NC strengthens the claim (it is not
  merely the membership artifact of exact PI). But we have not swept PH hyperparameters
  (resolution, filtration family, hop radius) exhaustively per dataset.

- **Scale and number of trials are mini, by design.** 10 trials/cell, single PDGNN training
  per dataset, HGB standard splits. This is a feasibility-grade controlled grid, not a
  full benchmark sweep. The consistency of the null across 4 datasets × 3 backbones × 2 designs
  (and the LP §14 mechanism) is what carries the claim, not any single cell's significance.

- **We do have positive topology results elsewhere** — and report them to avoid overclaiming:
  molecular graph classification (PH helps, H1/ring signal), and node-level diffusion features
  for LP (DNP_A, Chameleon 0.9816 > no-PI 0.9684; `results/dnp_A/lp_auc.csv`). The negative
  result is specific to **input-level PH-as-feature on node/edge-local relational tasks**
  (LP, hetero NC), not to topology writ large.

---

## 5. Related work / citations to position against

- **Papamarkou, Hajij, et al. (2024)**, *Position: Topological Deep Learning is the New Frontier
  for Relational Learning*, ICML 2024, arXiv:2402.08871 — primary positioning target
  (Open Problems #1, #9, #10; "not all higher-order relations are created equal").
- **Zhang & Chen (2018)**, *Link Prediction Based on Graph Neural Networks* (SEAL), NeurIPS —
  the subgraph/labeling-based LP framework whose anti-leakage edge-removal protocol is exactly
  what turns LP's exact PI into the membership artifact we isolate in §1c/§14.
- **Zhang, Li, Xia, Wang, Jin (2021)**, *Labeling Trick: A Theory of Using Graph Neural Networks
  for Multi-Node Representation Learning*, NeurIPS — formalizes why node-set-conditioned
  (edge-aware) features can leak structural-membership information; our `shuffled`/LOO controls
  are the empirical counterpart for PH features.
- **Pitfalls of GNN evaluation for link prediction (WSDM 2024)** — argues that LP benchmark
  results are frequently confounded by evaluation protocol / leakage; our controlled null and
  the §14 membership-artifact diagnosis are a TDA-specific instance of that critique.
- **Platonov, Kuznedelev, Diskin, Babenko, Prokhorenkova (2023)**, *A Critical Look at the
  Evaluation of GNNs Under Heterophily: Are We Really Making Progress?*, ICLR — shows
  heterophilic-graph benchmarks (incl. WebKB/Wiki families used in our LP grid) are saturated,
  leaky, or trivially-solvable, so reported gains are often artifacts; directly motivates our
  `none`-headroom control and our caution that the heterophilic "PI hurts" effect is about the
  benchmark + leakage, not a topological law.

---

## 6. One-paragraph abstract (for the writeup)

We stress-test the oldest recipe in topological deep learning — persistent homology as a
concatenated input feature — on heterogeneous node classification (4 HGB datasets ×
{GCN, HAN, HGT} × {meta-path PDGNN-EPD, unified type-aware EPD}) and link prediction, under a
three-control protocol (no-feature headroom, node-shuffled PH, random-filter PH) plus a
leave-one-out leakage control. With controls applied, **15/16 hetero cells are null and the
single apparent win (ACM-GCN) does not survive a backbone-swap confirmation grid**; on link
prediction we show exact PH is a train-graph edge-membership artifact with zero test signal
(100% leave-one-out collapse, 8/9 datasets). We position this as the honest negative half of
the ICML 2024 position paper's Open Problem #1 ("find applications where topology has a
competitive edge") and a concrete instance of its "*when* do higher-order relations help?"
caveat (Open Problem #9), with an explicit scope boundary: we test *input-level* PH only and do
**not** evaluate architecture-level TDL, which the position paper advocates and which our result
neither supports nor refutes.

---

## Appendix A — source files for every number

| Section | CSV / source |
|---|---|
| Idea1 GCN hetero NC | `results/hetero_pdgnn_{ACM,DBLP,IMDB,Freebase}/nc_acc.csv` |
| Idea2 GCN hetero NC | `results/unified_{ACM,DBLP,IMDB}/nc_acc.csv` |
| HAN/HGT Idea1 | `results/hanhgt_{ACM,IMDB}/nc_acc.csv` |
| HAN/HGT Idea2 | `results/hanhgt_uni_{ACM,DBLP}/nc_acc.csv` |
| Exact-PI hetero (context) | `results/hetero_{ACM,DBLP}/nc_acc.csv` |
| ACM-GCN non-repro | git `dd1e397`; `results/hanhgt_uni_ACM/nc_acc.csv` |
| §14 LP membership artifact | `results/pi_artifact_all/pi_artifact_all.csv`, `results/consistent_pi/loo_pi_*.csv` |
| LP "PI hurts" hetero | `results/dnp_A/lp_auc.csv`, `results/dnp_A_g2/lp_auc.csv` |
| DNP positive (contrast) | `results/dnp_A/lp_auc.csv` (DNP_A variant) |
| Controls definition | `hetero/hetero_nc_pipeline.py` L5–8, L35 (`_znorm`), L176 (variants dict) |
| Meta-paths / targets | `hetero/metapath_graph.py` L18 (`METAPATHS`), L42 (`TARGET`) |
| PDGNN-EPD predictor | `Knowledge_Distillation/pdgnn_modern.py`, `hetero/pdgnn_metapath.py` |
