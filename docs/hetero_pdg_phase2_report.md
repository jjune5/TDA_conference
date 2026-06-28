# hetero_pdg — Phase 2 report (real PDGNN/TLC adapter)

**Goal:** connect the existing TLC-GNN repo's real PDGNN/TLC topology code as an
*optional* backend, keep `fallback_topology_descriptors` as the default, do not break
the Phase-1 fallback MVP. **Status: done and smoke-tested on the toy graph.**

> No performance improvement is claimed anywhere. Toy/simple runs are for pipeline
> correctness only.

---

## 0. Inspection of existing PDGNN/TLC code (exact files/functions, no guessing)

| Need | Found at | Symbols used |
|---|---|---|
| PDGNN model | `Knowledge_Distillation/pdgnn_modern.py` | `PDGNN`, `PDGNNLayer` |
| PDGNN checkpoint | `data/PDGNN/checkpoints/pdgnn_lp.pt` | dict `{"state_dict", "config": {hidden_dim:32, num_layers:3}}` |
| EPD / PI engine | `hetero/pdgnn_metapath.py` | `_graph_hks`, `_ego_filt_edges`, `_exact_epd`, `gen_training_samples`, `train_pdgnn_metapath`, `device` |
| exact EPD (TLC) | `hetero/pdgnn_metapath._exact_epd` | gudhi lower-star extended persistence |
| persistence image | `sg2dgm/` (compiled) | `from sg2dgm import PersistenceImager` → `PersistenceImager.PersistenceImager(resolution=PI_RES)` |
| PI resolution | `node_ph_features.py` | `PI_RES = 5` (→ 25 dims/scale) |
| HKS filter | `diffusion_features.py` | `compute_hks_features` (**note:** does `os.chdir(repo)` at import — handled, see §7) |

Dependencies in the `tlcgnn` env: gudhi ✅, torch_scatter ✅, sg2dgm ✅. **Feasible.**

## 1. Files changed

**New (in `hetero_pdg_lp/`):**
- `hetero_pdg/topology_adapter.py` — backend abstraction (`fallback`/`real_tlc`/`real_pdgnn`), `RealBackendUnavailable`, `AdapterResult`, `compute_topology_features(...)`.
- `tests/test_topology_adapter.py` — 6 tests (fallback deterministic; real = smoke-or-clear-error; allow-fallback).
- `scripts/run_hetero_toy.slurm`, `scripts/run_hetero_simple.slurm`.
- `docs/hetero_pdg_phase2_report.md` (this file).

**Updated:**
- `hetero_pdg/train_hetero_lp.py` — `--topo-backend` / `--pdgnn-checkpoint` / `--allow-fallback`; routes collapsed/metapath topology through the adapter; records provenance; pins `output_dir` to an absolute path (§7).
- `hetero_pdg/README.md` — backends, CLI, provenance, Phase-2 status.

**Not touched:** the original homogeneous TLC-GNN/PDGNN code (the real backends *import* it read-only via `PYTHONPATH`; nothing in that repo was modified). `hetero_pdg/topology_features.py` (Phase-1 fallback) is unchanged.

## 2. Commands run

```bash
# tests (both environments)
env -u PYTHONPATH python -m pytest tests/ -q                               # 52 passed (self-contained)
PYTHONPATH=$TLC        python -m pytest tests/ -q                           # 52 passed (real paths)
# toy, fallback (default)
python -m hetero_pdg.train_hetero_lp --topo-mode metapath_topology_concat --topo-backend fallback --epochs 30 --output-dir runs/p2_fallback_concat
python -m hetero_pdg.train_hetero_lp --topo-mode no_topology --epochs 30 --output-dir runs/p2_no_topology
python -m hetero_pdg.train_hetero_lp --topo-mode metapath_topology_attention --topo-backend fallback --epochs 30 --output-dir runs/p2_attn
# real backend requested, engine ABSENT, allow-fallback false  -> clear error (no silent fallback)
env -u PYTHONPATH python -m hetero_pdg.train_hetero_lp --topo-backend real_pdgnn --allow-fallback false ...   # RealBackendUnavailable
# real backends (PYTHONPATH=$TLC)
python -m hetero_pdg.train_hetero_lp --topo-backend real_tlc   --output-dir runs/p2_real_tlc
python -m hetero_pdg.train_hetero_lp --topo-backend real_pdgnn --output-dir runs/p2_real_pdgnn
python -m hetero_pdg.train_hetero_lp --topo-backend real_pdgnn --pdgnn-checkpoint $TLC/data/PDGNN/checkpoints/pdgnn_lp.pt --output-dir runs/p2_real_pdgnn_ckpt
# allow-fallback true, engine absent -> falls back, flagged
env -u PYTHONPATH python -m hetero_pdg.train_hetero_lp --topo-backend real_pdgnn --allow-fallback true --output-dir runs/p2_allow_fallback
# SLURM
sbatch scripts/run_hetero_toy.slurm    # job 8471 COMPLETED (27s, exit 0)
```

## 3. Tests run — **52 passed / 0 failed** (each env)

`test_hetero_data` 7 · `test_metapath` 8 · `test_filtration` 8 · `test_topology_features` 6 ·
`test_hetero_model` 9 · `test_hetero_train_smoke` 8 · `test_topology_adapter` 6. TDD throughout.
Phase-1 suite unchanged and still green (fallback MVP not broken).

## 4. Metrics file paths (`runs/<name>/{config,metrics}.json`)

`p2_fallback_concat`, `p2_no_topology`, `p2_attn`, `p2_real_tlc`, `p2_real_pdgnn`,
`p2_real_pdgnn_ckpt`, `p2_allow_fallback`, plus SLURM `toy_{no_topology,metapath_topology_concat,metapath_topology_attention,unified_filter_topology}`.

## 5. Real vs fallback status (provenance recorded in every metrics.json)

| run | topo_backend_used | feature_kind | fallback_triggered | topo_dim | test_auc* |
|---|---|---|---|---|---|
| p2_fallback_concat | fallback | fallback_topology_descriptors | false | 6 | 0.557 |
| p2_real_tlc | real_tlc | real_tlc_exact_epd_persistence_image | false | 50 | 0.682 |
| p2_real_pdgnn | real_pdgnn | real_pdgnn_approx_epd_persistence_image | false | 50 | 0.529 |
| p2_real_pdgnn_ckpt | real_pdgnn | real_pdgnn_approx_epd_persistence_image | false | 50 | 0.505 |
| p2_allow_fallback | fallback | fallback_topology_descriptors | **true** | 6 | 0.564 |

\* toy numbers, correctness only — **not** a performance comparison.

## 6. Was real PDGNN/TLC actually used? — **Yes.**

- `real_tlc`: exact gudhi extended persistence → 50-dim persistence image, run end-to-end.
- `real_pdgnn`: PDGNN-predicted EPD → 50-dim PI, both train-on-the-fly **and** loaded from the bundled checkpoint (`pdgnn_lp.pt`, arch read from its `config`).
- Both reuse the existing TLC-GNN engine via `PYTHONPATH`; `fallback_triggered=false`.
- When the engine is absent and `--allow-fallback false`, the backend raises a clear
  `RealBackendUnavailable` naming the fix (PYTHONPATH + gudhi/torch_scatter/sg2dgm) — **no silent fallback**.

## 7. Required / missing pieces & notes

- Real backends require: TLC-GNN repo on `PYTHONPATH`, `gudhi`, `torch_scatter`, `sg2dgm`, the checkpoint (only for `--pdgnn-checkpoint`). All present in the `tlcgnn` env.
- **cwd gotcha (fixed):** `diffusion_features.py` calls `os.chdir(repo)` at import, so a relative `--output-dir` from a real run would land under TLC-GNN. `run_experiment` now `os.path.abspath`s `output_dir` up front.
- `--pdgnn-checkpoint` needs an arch-compatible checkpoint; the bundled one's `config` is auto-read. No per-(meta-path)dataset checkpoints ship, so `real_pdgnn` defaults to train-on-the-fly (exact EPD labels as supervision — intrinsic to PDGNN, not used as a downstream feature).
- PDGNN is retrained per train/val/test split (no cross-split cache) — fine for toy, wasteful at scale.

## 8. Limitations

- Synthetic toy graph only; real backends smoke-tested for *correctness*, not evaluated on real hetero datasets (ACM/DBLP/IMDB/Freebase).
- Real-PDGNN here is the meta-path-graph retrain path; the checkpoint is a homogeneous-LP PDGNN applied transfer-style (smoke only).
- `unified_filter_topology` is the Phase-1 type-aware filtration (raw-value vicinity stats); it is independent of `--topo-backend`.
- Single split + one batch per split on the toy.

## 9. No performance-improvement claim

All toy/simple numbers validate that the **pipeline runs correctly and provenance is
honest**. They are **not** evidence that topology helps; no improvement is claimed.
Fallback descriptors are deterministic graph statistics and are never reported as PDGNN
or EPD features.
