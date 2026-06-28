# hetero_pdg — Phase 1 MVP + Phase 2 real backend

Heterogeneous-graph **link prediction** with **PDGNN-style topological features**,
built as a clean, self-contained package. The original homogeneous TLC-GNN/PDGNN
code is untouched.

> ⚠️ **Default backend is `fallback_topology_descriptors` — NOT PDGNN, NOT EPD.**
> No performance improvement is claimed; toy metrics only validate the pipeline.
> **Phase 2** adds optional real backends (`real_tlc` exact EPD, `real_pdgnn` neural
> EPD) that reuse the TLC-GNN engine via `PYTHONPATH` — see "How to run" / "Phase 2".

## Motivation

PDGNN/TLC-GNN compute topological (extended-persistence) features on *homogeneous*
graphs. Heterogeneous graphs have multiple node/edge types whose filtration values
are not directly comparable. This package explores two hetero-aware routes and a
clean LP pipeline around them, with **leakage control** and an **honest split**
between cheap fallback descriptors (Phase 1) and real PDGNN/EPD features (Phase 2).

## Two topological routes

**MetaPath-PDGNN path.** Manually specified meta-paths (APA, PFP, PCP) are projected
to homogeneous graphs over one node type by **adjacency multiplication**
(`A_e1 @ A_e2 @ ...`). Per candidate pair we read topology descriptors on each
projected graph, then combine across meta-paths (`concat` or masked `attention`).
A `collapsed` baseline merges all meta-path graphs into one type-blind graph.

**UnifiedFilter-PDGNN path.** A type-aware filtration over a single homogeneous
view: each node type has its **own MLP** taking `[raw feature ‖ relation-degree
vector]` → one scalar; scalars are mapped to a common range by **type-wise quantile
calibration**; nodes are ordered **lexicographically** by `(calibrated value,
node-type priority)` (no large-constant tie-break). *Calibration is a scale-alignment
heuristic, not a persistent-homology stability guarantee.*

## Fallback descriptors vs. real PDGNN/TLC EPD

| | `fallback` (default) | `real_tlc` (Phase 2) | `real_pdgnn` (Phase 2) |
|---|---|---|---|
| feature_kind | `fallback_topology_descriptors` | `real_tlc_exact_epd_persistence_image` | `real_pdgnn_approx_epd_persistence_image` |
| what | common neighbours, shortest-path, src/dst degree, local edge count/density | exact extended persistence (gudhi lower-star) → persistence image | PDGNN-predicted EPD → persistence image |
| dependency | none (self-contained) | TLC-GNN engine + gudhi on PYTHONPATH | TLC-GNN engine (+ checkpoint or gudhi labels) on PYTHONPATH |
| PDGNN/EPD? | **no** | yes (exact EPD) | yes (neural EPD approx) |

**Phase 2 (implemented):** `topology_adapter.compute_topology_features(..., backend=...)`
provides `fallback` / `real_tlc` / `real_pdgnn`. Real backends reuse the TLC-GNN engine
(`Knowledge_Distillation.pdgnn_modern`, `hetero.pdgnn_metapath`, `sg2dgm.PersistenceImager`)
via `PYTHONPATH`; if unavailable they raise a clear error (no silent fallback unless
`--allow-fallback true`). **Fallback descriptors must never be reported as PDGNN/EPD.**

## Leakage prevention

1. Target val/test edges are removed from the observation graph
   (`remove_target_edges_for_split`, which never mutates the original HeteroData),
   so they appear in no projected/filtration graph.
2. Each candidate pair's own edge is removed before its descriptors are read, then
   restored (`fallback_topology_descriptors(..., remove_target=True)`).

## Supported modes

`no_topology` · `collapsed_topology` · `metapath_topology_concat` ·
`metapath_topology_attention` · `unified_filter_topology`

Scoring: `MLP(concat[z_src, z_dst, z_src*z_dst, |z_src - z_dst|, topology_features])`.

## How to run

Fallback backend is self-contained (no PYTHONPATH); real backends need the TLC-GNN
engine on `PYTHONPATH`.

```bash
conda activate tlcgnn
python -m pytest tests/ -q                          # 52 tests (fallback 46 + adapter 6)

# fallback (default) -> config.json + metrics.json
python -m hetero_pdg.train_hetero_lp --dataset toy --target-rel paper,cites,paper \
    --topo-mode metapath_topology_concat --topo-backend fallback --epochs 30 \
    --hidden-dim 32 --topo-hidden-dim 32 --lr 0.001 --neg-ratio 1.0 --seed 0 \
    --device cpu --output-dir runs/hetero_pdg_toy

# real backends (need TLC-GNN on PYTHONPATH)
export PYTHONPATH=/mnt/data/users/junyoungpark/code/TLC-GNN:$PYTHONPATH
python -m hetero_pdg.train_hetero_lp --topo-mode metapath_topology_concat \
    --topo-backend real_tlc --epochs 10 --output-dir runs/real_tlc          # exact gudhi EPD -> PI
python -m hetero_pdg.train_hetero_lp --topo-mode metapath_topology_concat \
    --topo-backend real_pdgnn \
    --pdgnn-checkpoint $PYTHONPATH/data/PDGNN/checkpoints/pdgnn_lp.pt \
    --epochs 10 --output-dir runs/real_pdgnn                                  # neural EPD approx -> PI

# SLURM (scripts/)
sbatch scripts/run_hetero_toy.slurm      # all modes, fallback, CPU
sbatch scripts/run_hetero_simple.slurm   # planted control + real_tlc smoke
```

CLI: `--dataset toy --target-rel SRC,REL,DST --topo-mode {5 modes}
--topo-backend {fallback,real_pdgnn,real_tlc} --pdgnn-checkpoint PATH
--allow-fallback {true,false} --epochs --hidden-dim --topo-hidden-dim --lr
--neg-ratio --seed --device --output-dir [--planted]`. A requested real backend that
is unavailable errors clearly unless `--allow-fallback true` (then
`metrics.json.fallback_triggered=true`).

## Known limitations

- Synthetic toy graph only; metrics validate correctness, not quality.
- Default backend is fallback descriptors (NOT PDGNN/EPD). Real backends
  (`real_tlc`/`real_pdgnn`) exist but require the TLC-GNN engine on PYTHONPATH and
  were smoke-tested on the toy graph only.
- `unified_filter_topology`: the calibration + lexicographic ordering are
  implemented & tested and define the filtration order, but the differentiable LP
  feature uses **raw** filter-value vicinity stats (so gradients flow); a
  differentiable soft-rank calibration is future work.
- Single train/val/test split (no k-fold); one batch per split for the toy.
- `APA` is implemented but applies to author–author LP (not the default paper–paper target).

## Phase 2 status & remaining work

**Implemented & smoke-tested (toy):** `real_tlc` (exact gudhi extended persistence → PI)
and `real_pdgnn` (neural EPD approximation → PI; train-on-the-fly or `--pdgnn-checkpoint`,
the bundled `data/PDGNN/checkpoints/pdgnn_lp.pt` loads via its `config`). Both reuse the
TLC-GNN engine via `PYTHONPATH`, raise a clear error when unavailable, and respect opt-in
`--allow-fallback`. Provenance (`topo_backend_used`, `feature_kind`, `fallback_triggered`)
is written to every `metrics.json`. The fallback stays the default labelled control.

**Remaining:**
1. Evaluate real backends on real hetero datasets (ACM/DBLP/IMDB/Freebase), not just toy.
2. Per-dataset checkpoint registry (current `--pdgnn-checkpoint` needs arch match, auto-read from `config`).
3. Cache the trained PDGNN across train/val/test (currently retrained per split).
4. Differentiable soft-rank calibration for the unified filtration.

## Phase 3 — Advanced Heterogeneous Topology Features (experimental)

Research-inspired extensions beyond plain meta-path projection. **Not** paper
reproductions; defaults reproduce Phase-1/2 exactly. Full report:
[`docs/hetero_pdg_phase3_advanced.md`](../docs/hetero_pdg_phase3_advanced.md).

| feature | flag | status | inspired by | caveat |
|---|---|---|---|---|
| Pair-conditioned filtration | `--filtration-mode pair_conditioned` | approximate | TLC-GNN, SEAL, HGT | vicinity stats of a learnable filter, **not** PH coordinates; extension, not a reproduction |
| Relation-aware edge filtration | `--edge-filtration-mode relation_delay` | **prepared interface** | PDGNN/TLC-GNN, HGT, PH stability | guarantees g(i,j,ρ) ≥ max(f_i,f_j); exact-EPD path takes node filtration so this is not wired in by default |
| Typed cycle signature | `--topo-mode typed_cycle_topology` | proxy | GTN, HAN, MAGNN, Persistence Images | `typed_cycle_proxy_descriptors` (typed counts), **not** exact persistent homology |
| Multi-slice filtration | `--topo-mode multi_slice_topology --num-slices K` | approximate | multiparameter / sliced persistence | scalar slicing approximation, **not** exact multi-parameter PH |

Caveats: fallback descriptors are not real PDGNN/EPD; proxies/approximations are labelled
as such; **no performance improvement is claimed** — toy/simple metrics are sanity checks
only. Ablation: `bash scripts/run_hetero_ablation.sh [--planted]`.

```bash
# examples (toy, standalone)
python -m hetero_pdg.train_hetero_lp --topo-mode typed_cycle_topology --epochs 60 --planted --output-dir runs/p3_typed
python -m hetero_pdg.train_hetero_lp --topo-mode multi_slice_topology --num-slices 3 --output-dir runs/p3_slice
python -m hetero_pdg.train_hetero_lp --topo-mode collapsed_topology --filtration-mode pair_conditioned --output-dir runs/p3_pair
python -m hetero_pdg.train_hetero_lp --topo-mode metapath_topology_concat --edge-filtration-mode relation_delay --output-dir runs/p3_edge
```
