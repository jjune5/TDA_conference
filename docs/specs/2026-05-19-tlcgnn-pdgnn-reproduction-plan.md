# TLC-GNN / PDGNN Reproduction Ultraplan

**Date:** 2026-05-19
**Target paper(s):**
- TLC-GNN — Yan et al., *Link Prediction with Persistent Homology: An Interactive View*, ICML 2021 (`http://proceedings.mlr.press/v139/yan21b/yan21b.pdf`)
- PDGNN — Yan et al., *Neural Approximation of Graph Topological Features*, NeurIPS 2022

**Repo:** https://github.com/pkuyzy/TLC-GNN (cloned at `/mnt/data/users/junyoungpark/code/TLC-GNN`)

---

## 1. Current state assessment

### What we have
| Item | Status |
|---|---|
| Repo cloned | ✅ `code/TLC-GNN` (branch `main`, commit `36e70e5`) |
| Conda env `tlcgnn` | ✅ exists at `miniforge3/envs/tlcgnn` |
| Persistence Image cache | ✅ `data/TLCGNN/{PubMed,Photo,Computers}.npy` (6 GB total — the expensive part is done) |
| GPU | ✅ 5+ × A100-80GB |
| SLURM | ✅ `slurm_run.sh` ready; `gpu` partition available |
| User code modifications | ✅ `loaddatas.py`, `pipelines.py`, `sg2dgm/*` adapted to modern scipy/networkx/multiprocessing; `Knowledge_Distillation/pdgnn_modern.py` is a PyG-2.x rewrite of PDGNN |
| Smoke runs / 50-trial runs | ❌ All crash; score files contain only headers |

### Env vs. paper's `requirements.txt`
| Package | Paper requires | Installed | Compatible? |
|---|---|---|---|
| Python | 3.7 | 3.9.23 | ⚠ functional |
| torch | 1.7.0+cu110 | 2.1.0+cu118 | ⚠ functional with one bugfix |
| torch-geometric | 1.6.1 | 2.5.3 | ⚠ API drift — loaddatas already adapted |
| torch-{scatter,sparse,cluster} | 1.5/0.6/1.5 | matching 2.1+cu118 wheels | ✅ |
| dionysus | 2.0.7 | 2.1.8 | ✅ compatible API |
| numpy | 1.18.5 | 1.26.4 | ✅ |
| networkx | 2.5 | 2.8.8 | ⚠ adjacency_matrix returns sparse array (already handled) |
| scipy | 1.5.4 | 1.10.1 | ✅ |
| scikit-learn | 0.23.2 | 1.2.2 | ✅ |

### Root cause of the crash
All four most recent runs (PubMed, Photo, Computers, smoke) fail identically at `loss.backward()`:

```
RuntimeError: one of the variables needed for gradient computation has been modified
by an inplace operation: [torch.cuda.FloatTensor [N, 16]], which is output 0 of
ReluBackward0, is at version 1; expected version 0 instead.
```

**The offender:** `baselines/TLCGNN.py:48`

```python
emb = emb.renorm_(2,0,1)   # trailing underscore → inplace; corrupts ReLU output's autograd graph
```

PyTorch 1.7 (paper) did not flag this; PyTorch 1.10+ does. **Fix:** drop the underscore (`emb = emb.renorm(2,0,1)`). This is the only blocker — the rest of the pipeline (including the multi-hour Ricci-curvature + persistence-image computation) has already succeeded.

---

## 2. Approach decision: modernize, don't downgrade

Two options were considered:

**Option A — Downgrade to Python 3.7 + torch 1.7.0+cu110 (paper-faithful).**
- ✗ Requires CUDA 11.0; cluster has CUDA 11.8+/12.x drivers. Old CUDA-11.0 binaries may run via forward-compat but PyG ext wheels for cu110 are no longer hosted.
- ✗ Throws away the user's already-completed modernization (loaddatas, multiprocessing, persistence cache, `pdgnn_modern.py`).
- ✗ Re-running persistence-image computation for Photo took several hours and produced a 5.9 GB cache that's keyed to whatever torch/networkx happen to be installed at compute time.

**Option B — Keep the existing `tlcgnn` env, fix the one-line autograd bug, validate numbers match. ← chosen**
- ✓ One-line code change.
- ✓ Reuses the 6 GB of cached PIs (no re-computation).
- ✓ Numerics: GCNConv math and Ollivier-Ricci computation are deterministic-equivalent across torch 1.7→2.1 in fp32.
- ✓ Same architecture, same loss, same hyperparameters, same data splits (`seed=1234` in `get_adj_split`).
- ⚠ Risk: a downstream numerical drift we don't anticipate. *Mitigation:* gated 1-trial smoke before full 50-trial sweep.

**Decision:** Option B. If the smoke trial on PubMed does not land in the [96.5, 97.5] AUC band the paper reports, fall back to a stricter env (still modern, but pin to a torch 1.13 + PyG 2.0 combo before considering a full downgrade).

---

## 3. Reproduction targets (Table 1 / Table 3 of the paper)

These are the numbers we are trying to match. "Goal band" is mean ± 3σ from the paper's reported std.

### TLC-GNN (Ricci) — main configuration
| Dataset | Paper AUC | Goal band (mean ± 3σ) | Trials | Hop | Dropout |
|---|---|---|---|---|---|
| PubMed | 97.03 ± 0.001 | 97.00 – 97.06 | 50 | 2 | 0.5 |
| Photo | 98.23 ± 0.001 | 98.20 – 98.26 | 50 | 1 | 0.5 |
| Computers | 97.90 ± 0.001 | 97.87 – 97.93 | 50 | 1 | 0.5 |
| PPI | 81–84 (Table 3, 5 graphs) | per-graph 80–86 | 20 | 1 | 0.5 |
| Cora* | not in Table 1 | — | 50 | 1 | 0.8 |
| Citeseer* | not in Table 1 | — | 50 | 1 | 0.8 |

\* The paper does not report Cora/Citeseer in Table 1. The README mentions running them with `dropout=0.8` "for slightly higher results". We will run them for completeness but the success criterion is on PubMed/Photo/Computers/PPI.

*Note on the reported std:* `0.001` is suspiciously small for 50 trials with random edge sampling; the paper likely reports std of the mean across trials, not the per-trial std. We will compute and report **both** per-trial std and bootstrap mean-CI.

### Hyperparameters (paper §5.2, code defaults — already in `pipelines.py`)
- Edge splits: train/val/test = 85% / 5% / 10% (PubMed/Photo/Computers); 60% / 20% / 20% (PPI). Seed = 1234 deterministic.
- Optimizer: Adam, lr=0.005, weight_decay=0.
- Encoder: 2-layer GCNConv, hidden=100→16.
- Topological feature: 5×5 persistence image of pairwise extended PD using Ollivier-Ricci filtration.
- Decoder: concat [dist² + PI(5×5)] → Linear(41→25) → LeakyReLU → Linear(25→1) → modified Fermi-Dirac: prob = 1/(exp((sqdist−2)/1)+1).
- Loss: BCE on (prob, edge_label).
- Negative sampling: per-epoch random index over `train_neg` pool of size = 5 × train_pos.
- Early stopping: 200-epoch patience on val ROC, max 2000 epochs.

---

## 4. Execution plan

### Phase 0 — Fix the blocker (≈1 min)
**File:** `baselines/TLCGNN.py:48`
**Change:** `emb = emb.renorm_(2,0,1)` → `emb = emb.renorm(2,0,1)`

Why this is safe: `renorm` returns a new tensor with the same values, so downstream consumers see the same numerics. The only effect is that the original `emb` (output of the second ReLU) is preserved unmodified — which is what autograd's ReLU backward needs.

### Phase 1 — Smoke validation (≈30 min, single trial)
1. Activate env, run a single-trial PubMed pass to verify backward succeeds end-to-end:
   ```bash
   conda activate tlcgnn
   cd /mnt/data/users/junyoungpark/code/TLC-GNN
   python pipelines.py --datasets PubMed --trials 1 --tag smoke
   ```
2. **Pass criterion:** training reaches an epoch where val ROC ≥ 0.95 within 500 epochs (one trial is enough; a fully broken model would never cross 0.95 since random ≈ 0.5).
3. **Fail action:** debug before scaling. Likely candidates if smoke fails:
   - Another inplace op (grep `renorm_\|relu_\|add_\|mul_` across baselines/).
   - PI normalization expecting numpy float64 (PyTorch 2 stricter on float32/64 mixing).
   - LeakyReLU `inplace=True` — already present at TLCGNN.py:14, `torch.nn.LeakyReLU(0.2, True)`. Should not cause an issue here because the input to LeakyReLU is freshly produced by `linear_1` (no autograd-dependent re-use), but worth knowing if a second crash appears.

### Phase 2 — Full 50-trial reproduction (≈12 h on 3 GPUs)
Submit one SLURM job per dataset, each running 50 trials sequentially on one GPU (the persistence-image computation is cached, so each trial is just model training):

```bash
sbatch --job-name=tlcgnn-pubmed     slurm_run.sh --datasets PubMed     --trials 50 --tag rerun
sbatch --job-name=tlcgnn-photo      slurm_run.sh --datasets Photo      --trials 50 --tag rerun
sbatch --job-name=tlcgnn-computers  slurm_run.sh --datasets Computers  --trials 50 --tag rerun
```

**Output:** `scores/pipe_benchmark_<dataset>_LP_scores_rerun.txt` per dataset, each containing 50 (AP, AUC) rows + std + mean.

**Wall-clock estimate per dataset:**
- PubMed: ~2.5 min/trial × 50 ≈ 2 h.
- Photo: ~3 min/trial × 50 ≈ 2.5 h.
- Computers: ~4 min/trial × 50 ≈ 3.5 h.
- All three in parallel on separate GPUs: ≈3.5 h wall-clock.

### Phase 3 — PPI (separate sweep)
PPI uses 5 graphs; the paper runs 50 trials per graph (Table 3 reports one number per graph, presumably mean over those trials). Code allows trials up to 20 for PPI. Plan:
```bash
sbatch --job-name=tlcgnn-ppi slurm_run.sh --datasets PPI --trials 20 --tag rerun
```
PI cache for PPI does not yet exist → first run will compute it. Allocate ≥6 h.

### Phase 4 — Validation report
Aggregate the 50 trials per dataset and compare against the paper. Report:
- Mean ± per-trial std AUC and AP.
- 95% bootstrap CI of the mean (clarifies the "0.001 std" mystery in the paper).
- Pass/fail vs goal band.
- Per-dataset training curves (one example trial) to confirm convergence pattern matches.

Output: `docs/specs/2026-05-19-tlcgnn-pdgnn-reproduction-results.md` with one summary table.

### Phase 5 — PDGNN (NeurIPS 2022)
PDGNN's original code (`Knowledge_Distillation/Teacher_model.py`, `gat_conv.py`, `message_passing.py`) imports from `torch_geometric.nn.conv.message_passing` private API that was removed in PyG 2.x. The user has already written a modernized port at `Knowledge_Distillation/pdgnn_modern.py`. Plan:
1. Generate vicinity-graph training data + ground-truth extended PDs via `data_utils_NC.py` (this also depends on PyG 1.6 internals — may need a thin shim).
2. Train `pdgnn_modern.py` to predict (birth, death) coordinates; sanity-check against ground-truth PDs (MSE ≤ 1e-2 on val per the paper).
3. Plug PDGNN-predicted PIs into `pipelines_GIN.py` / `pipelines_LP_GIN.py` for downstream node classification (NC) and link prediction (LP).
4. Compare against the paper's NeurIPS 2022 Tables. (Defer to a follow-up plan after TLC-GNN reproduction lands — PDGNN reuses TLC-GNN's PI computation pipeline as ground truth, so it depends on Phase 2 being correct.)

---

## 5. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| A second hidden inplace op after fixing line 48 | Medium | Phase-1 smoke catches it; grep for inplace patterns proactively. |
| Numerical drift PyTorch 1.7 vs 2.1 pushes AUC slightly below paper band | Medium | Acceptable if within 0.5 AUC%; flag and discuss. |
| PI cache for Photo (5.9 GB) was computed before user adapted multiprocessing; could be stale | Low | The cache file is consumed via `np.load`; we'll spot-check shape and statistics before trusting it. |
| Photo PI cache is ~5.9 GB; RAM pressure on smaller GPU nodes | Low | A100-80GB has ample host RAM (≥256 GB typical). |
| PPI PI computation hits SLURM 24h wall-clock | Medium | Subgraphs are smaller per graph than Photo; should finish in ≤6 h, but request --time=24:00:00 explicitly. |
| Cora/Citeseer differ from paper (paper omits) | N/A | Not on critical path; informational only. |

---

## 6. Definition of done

- [ ] `pipelines.py` runs end-to-end without crashes on PubMed/Photo/Computers (Phase 1).
- [ ] 50-trial AUC for **PubMed** ≥ 96.5 (paper 97.03).
- [ ] 50-trial AUC for **Photo** ≥ 97.7 (paper 98.23).
- [ ] 50-trial AUC for **Computers** ≥ 97.4 (paper 97.90).
- [ ] 20-trial AUC for **PPI** within [78, 88] for all 5 sample graphs (paper 81.21–83.95).
- [ ] Results table written to `docs/specs/2026-05-19-tlcgnn-pdgnn-reproduction-results.md` and committed.
- [ ] **PDGNN (Phase 5)**: deferred — captured as a follow-up plan only after TLC-GNN passes.
