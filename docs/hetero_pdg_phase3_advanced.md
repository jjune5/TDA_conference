# hetero_pdg — Phase 3: Advanced Heterogeneous Topology Features

Experimental, **research-inspired** extensions that make topology features more
heterogeneity-aware, beyond plain meta-path projection. These are NOT reproductions of
any single paper; they combine ideas from PDGNN/TLC-GNN, heterogeneous GNNs (HAN/HGT/
GTN/MAGNN), enclosing-subgraph LP (SEAL), persistence vectorization (Persistence
Images), and multi-parameter/sliced persistence.

> ⚠️ **Toy/simple results are sanity checks, NOT performance claims.** Phase 1/2 behaviour
> is unchanged (defaults reproduce them exactly). Fallback descriptors are never called
> PDGNN/EPD. Each advanced feature is labelled exact / approximate / proxy / prepared.

---

## 1. Files changed

**New modules (built TDD, each self-contained, no edits to existing files):**
- `hetero_pdg/pair_filtration.py` (+ `tests/test_pair_filtration.py`, 12 tests)
- `hetero_pdg/edge_filtration.py` (+ `tests/test_edge_filtration.py`, 16 tests)
- `hetero_pdg/typed_cycle.py` (+ `tests/test_typed_cycle.py`, 9 tests)
- `hetero_pdg/multi_slice.py` (+ `tests/test_multi_slice.py`, 8 tests)
- `tests/test_phase3_integration.py` (12 tests)
- `scripts/run_hetero_ablation.sh`, `docs/hetero_pdg_phase3_advanced.md` (this file)

**Updated:** `hetero_pdg/train_hetero_lp.py` — new CLI flags + a generalized topology
dispatch that maps the advanced modes onto the model's base modes; provenance recorded.
`hetero_pdg/README.md` — Phase-3 section. **Original TLC-GNN/PDGNN code: untouched.**
**Model `HeteroTopoLinkPredictor.MODES` unchanged** (advanced modes are train-level and
map onto `collapsed_topology` / `metapath_*`), so Phase-1/2 model tests stay valid.

## 2. Advanced features implemented

| # | feature | train flag | what it produces |
|---|---|---|---|
| 1 | Pair-conditioned filtration | `--filtration-mode pair_conditioned` | per-pair vicinity stats of a learnable filter conditioned on (u,v,r) |
| 2 | Relation-aware edge filtration | `--edge-filtration-mode relation_delay` | edge filtration g(i,j,ρ)=max(f_i,f_j)+softplus(δ_ρ) (prepared interface) |
| 3 | Typed cycle signature | `--topo-mode typed_cycle_topology` | typed node/edge histograms + per-meta-path common-neighbor / short-cycle proxies |
| 4 | Multi-slice filtration | `--topo-mode multi_slice_topology --num-slices K` | K convex-combination scalar slices of a 4-component vector filtration, attention-fused |

## 3. Mathematical intuition

- **Pair-conditioned filtration.** A node's topological role for a *link* should depend on
  the queried pair. We learn `f_θ(x | u,v,r) = MLP_type(x)([h_x, h_u, h_v, e_r,
  reldeg_x, d(x,u), d(x,v)])` (typed hop-distances on the projected graph), then summarise
  the filter over the (u,v) k-hop enclosing subgraph. Inspired by TLC-GNN pairwise
  topology + SEAL enclosing subgraphs + HGT relation typing. **Not** persistence-diagram
  coordinates — vicinity statistics of a learnable scalar field.
- **Relation-aware edge filtration.** Standard `g(e)=max(f(i),f(j))` ignores edge type.
  We add a per-relation non-negative delay: `g=max(f_i,f_j)+softplus(δ_ρ)`. `softplus≥0`
  *structurally* guarantees the filtration-validity condition `g ≥ max(f_i,f_j)` (an edge
  never appears before its endpoints), while letting edge type shift event timing.
- **Typed cycle signature.** Two cycles with identical birth/death can be semantically
  different (Author-Paper-Author vs Paper-Field-Paper). An untyped persistence diagram
  collapses them. We keep typed counting statistics (node/edge-type histograms, per-meta-
  path common-neighbour and short-cycle proxies) as interpretable side-features.
- **Multi-slice filtration.** A single scalar filtration is restrictive. We build a vector
  filtration `F(x)∈R^4` (type-aware score, relation-degree, centrality/pair-distance,
  type-priority) and view it through `K` slices `f_k=λ_k^T F`, `λ_k≥0, Σλ_k=1` (row-softmax),
  fusing per-slice descriptors with attention. A cheap stand-in for multi-parameter
  persistence — **not** the exact object.

## 4. Reference papers & GitHub

| feature | papers | repos |
|---|---|---|
| pair-conditioned | TLC-GNN (arXiv:2102.10255); SEAL (arXiv:1802.09691); HGT (arXiv:2003.01332) | muhanzhang/SEAL; acbull/pyHGT |
| edge filtration | PDGNN (arXiv:2201.12032); HGT; PH stability (Cohen-Steiner et al. 2007) | pkuyzy/TLC-GNN; acbull/pyHGT |
| typed cycle | GTN (NeurIPS'19); HAN (arXiv:1903.07293); MAGNN (arXiv:2002.01680); Persistence Images (JMLR v18) | seongjunyun/Graph_Transformer_Networks; Jhy1993/HAN; cynricfu/MAGNN; CSU-TDA/PersistenceImages |
| multi-slice | Multiparameter path reps (arXiv:2507.23762); Signed-barcode vectorization (arXiv:2306.03801) | — |
| base engine | PDGNN (NeurIPS'22); TLC-GNN (ICML'21) | pkuyzy/TLC-GNN |

## 5. New CLI options (`hetero_pdg/train_hetero_lp.py`)

```
--topo-mode {... , typed_cycle_topology, multi_slice_topology}
--filtration-mode {type_aware(default), pair_conditioned}
--edge-filtration-mode {max(default), relation_delay}
--num-slices K            # default 3, for multi_slice_topology
```
Defaults (`type_aware`, `max`, base topo-mode) reproduce Phase-1/2 exactly.

## 6. Commands run

```bash
# unit + integration tests (standalone)
env -u PYTHONPATH python -m pytest tests/ -q                 # 113 passed
# ablation (toy)
bash scripts/run_hetero_ablation.sh             # random control
bash scripts/run_hetero_ablation.sh --planted   # positive control
```

## 7. Tests run — **113 passed / 0 failed** (standalone, CPU)

Phase-3 additions: pair_filtration 12 · edge_filtration 16 · typed_cycle 9 · multi_slice 8
· integration 12. Phase-1/2/real-data suites unchanged and still green. (Built via a
parallel multi-agent workflow, each module TDD'd in isolation; then integrated + a full
joint run.)

### Ablation (toy, test AUC mean±std over seeds 0–2, epochs 60) — SANITY ONLY

| mode | RANDOM (neg. control) | PLANTED (pos. control) |
|---|---|---|
| no_topology | 0.59 | 0.48 (chance; features random) |
| metapath_attention (fallback) | 0.60 | 0.75 |
| pair_conditioned (collapsed) | 0.50 | **0.40** |
| relation_delay (metapath; recorded only) | 0.63 | 0.72 |
| typed_cycle_topology | 0.52 | **0.81** |
| multi_slice_topology | 0.48 | **0.40** |

**Honest reading (NOT a performance claim):** on this toy positive control, `typed_cycle`
and the simple fallback/`relation_delay` paths track the planted signal, while the
learnable `pair_conditioned` and `multi_slice` modes do **not** (they sit at/below
chance here). This is expected variance of small experimental modules on a tiny synthetic
graph; it is **not** evidence for or against any method. No real-data evaluation was run.

## 8. Limitations

- Toy graph only; advanced modes smoke-tested for correctness, not evaluated on real data.
- `pair_conditioned` / `multi_slice` filtrations are vicinity statistics of learnable scalar
  fields, **not** persistence-diagram coordinates.
- `relation_delay` edge filtration is a **prepared interface**: the exact-EPD path
  (`_exact_epd`) takes a node filtration and computes edge values internally as max; it does
  not consume explicit per-edge filtration. `edge_filtration.zero_dim_sublevel_persistence`
  shows a union-find that *would* consume them, but it is not wired into the default path.
- `typed_cycle` are proxy counting descriptors, not exact persistent cycles; reverse
  relations are counted as distinct channels by design.
- Advanced modes assume same-type pairs and (for filtration-based ones) featured target
  nodes; they target the toy schema, not arbitrary real schemas.

## 9. Exact / approximate / fallback / prepared

| feature | status |
|---|---|
| Pair-conditioned filtration | **approximate** (learnable filtration + vicinity stats; not PH) |
| Relation-aware edge filtration | **prepared interface** (function + validity constraint + tests; not wired into exact EPD) |
| Typed cycle signature | **proxy** (typed counting descriptors; not persistent homology) |
| Multi-slice filtration | **approximate** (scalar slicing; not exact multi-parameter PH) |
| fallback descriptors (Phase 1) | **fallback** (deterministic graph stats; not PDGNN/EPD) |
| real_tlc / real_pdgnn (Phase 2) | exact EPD / neural-EPD approximation (via TLC-GNN engine) |

## 10. No performance claim

Every toy/simple number here exists only to show the pipeline runs and that provenance is
recorded honestly. Nothing here claims any topology method improves link prediction;
that requires real-data experiments not run in Phase 3.
