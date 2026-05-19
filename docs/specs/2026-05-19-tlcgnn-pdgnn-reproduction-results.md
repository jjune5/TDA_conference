# TLC-GNN Reproduction Results (Phase 2 — 50 trials × 3 datasets)

**Date:** 2026-05-19
**Plan:** `docs/specs/2026-05-19-tlcgnn-pdgnn-reproduction-plan.md`
**Environment:** Python 3.9.23 / torch 2.1.0+cu118 / PyG 2.5.3 / dionysus 2.1.8 — Option B (modernize, don't downgrade).
**Hardware:** 3× NVIDIA A100-80GB on SLURM (one job per dataset).

---

## 1. Headline numbers

| Dataset | Trials | Elapsed | **Our AUC** | **Our AP** | AUC range | Paper AUC (Yan et al. Table 1) | Δ vs paper | Goal band (±3σ) |
|---|---|---|---|---|---|---|---|---|
| **Photo** | 50 / 50 | 1 h 40 m | **0.9825 ± 0.0008** | 0.9776 ± 0.0012 | [0.9799, 0.9838] | 0.9823 ± 0.001 | **+0.0002** | 0.9820–0.9826 ✅ |
| **PubMed** | 50 / 50 | 2 h 41 m | **0.9635 ± 0.0025** | 0.9612 ± 0.0028 | [0.9550, 0.9679] | 0.9703 ± 0.001 | **−0.0068** | 0.9700–0.9706 ❌ |
| **Computers** | 50 / 50 | 2 h 25 m | **0.9680 ± 0.0023** | 0.9616 ± 0.0028 | [0.9629, 0.9727] | 0.9790 ± 0.001 | **−0.0110** | 0.9787–0.9793 ❌ |

**Pass/fail summary:** 1 / 3 in the goal band (Photo). PubMed and Computers reproduce ~0.7 %p and ~1.1 %p below the paper. Both gaps are large compared to our per-trial std (0.0023–0.0025), so they are systematic — not statistical noise.

The wider per-trial std we observe (0.0023–0.0025) vs the paper's reported 0.001 is consistent with the suspicion noted in the plan: the paper's "0.001" is almost certainly the standard error of the mean across 50 trials, not the per-trial std. Our SE of the mean is 0.0025 / √50 ≈ 0.0004 — much closer to 0.001.

---

## 2. Why Photo matches but PubMed/Computers don't

This is the central puzzle. All three runs use the same code path post-fix, on the same `tlcgnn` env, with the same seed (1234) and the same 5×train_pos negative-cap. The only structural difference is **who generated the persistence-image cache**.

| Dataset | Cache origin | Cache shape | Row layout |
|---|---|---|---|
| Photo | Generated **before** user added `train_neg_cap` — full 29.3 M edges. Our `loaddatas.py` splice keeps the first 506 K train_neg rows + val/test from their original offset. | (29,300,799, 25) → spliced to (643,038, 25) | Matches paper's original "all negatives" computation, then sliced. |
| PubMed | Generated **after** user added the cap, by user's modernized `sg2dgm/parallel_pi.py` + `accelerated_PD.py` pipeline. | (239,352, 25) | Matches current code's expected layout. |
| Computers | Same as PubMed. | (1,327,650, 25) | Matches current code's expected layout. |

**Working hypothesis:** Photo's cached PI values come from a code path closer to the paper's original `sg2dgm/riccidist2dgm.py` and `dgformat.py`. PubMed/Computers caches come from the user's modernized PI pipeline (added `accelerated_PD.py`, `parallel_pi.py`). A small numerical drift in those modules — e.g., a different filtration tiebreaker, a different normalization in `PersistenceImager.pyx`, or a different summation order during multiprocessing — would produce PI values that look almost-the-same but train to a different optimum.

This is consistent with the observation that the gap is similar in magnitude (~1%p) on both PubMed and Computers (both modernized-PI caches) and ~zero on Photo (legacy-PI cache).

A secondary hypothesis is that the `train_neg_cap` itself shrinks the per-epoch negative diversity from "thousands of unseen negatives every epoch" to "the same 506 K seen 400× over 2000 epochs", and that PubMed/Computers are more sensitive to this than Photo. But this should affect Photo equally (after splice, Photo's effective pool is also 506 K), so this hypothesis alone does not explain the asymmetry.

---

## 3. What we changed in the code (audit)

For reference and reproducibility, the **only** mid-pipeline behavior changes vs. the upstream repo at commit `36e70e5`:

### Mine (added today)
1. `baselines/TLCGNN.py:48` — `emb = emb.renorm_(2,0,1)` → `emb = emb.renorm(2,0,1)`.
   Pure autograd-compatibility fix for PyTorch ≥1.10. Returns a tensor with bitwise-identical values; only difference is that the input `emb` (a ReLU output) is preserved unmodified, which is what backward needs.
2. `loaddatas.py:69–92` — added a cache-layout splice path that detects stale `Photo.npy` (built without `train_neg_cap`) and rearranges rows to match the current loader's offsets. **Does not modify any PI value**; only reorders rows of the cached array.

### Pre-existing (user's modernization before today)
- `loaddatas.py:53–57` — added `train_neg_cap = max(5 × train_pos, 1024)`. Limits train negative-edge pool size. **The most likely driver of the systematic gap.**
- `loaddatas.py:36–58` — replaced `adj.todense()` with bit-mask negative-edge construction. Numerically equivalent given identical seed, but introduces a possibility of mismatch if the upper-triangular ordering changed.
- `sg2dgm/accelerated_PD.py`, `sg2dgm/parallel_pi.py` — modernized PI pipeline. Used to generate the PubMed and Computers caches.
- `Knowledge_Distillation/pdgnn_modern.py`, `Knowledge_Distillation/prepare_data_modern.py` — modernized PDGNN port for Phase 5; not used in this report.

Nothing in the model architecture (`baselines/TLCGNN.py:Net`), loss (BCE), optimizer (Adam, lr=0.005, wd=0), or training loop was changed.

---

## 4. Diagnosis options to close the gap

### Option A — Sanity-check the cache contents (cheap, ~1 h)
Recompute the persistence image for a single edge using the *current* code, and compare the result against the same row in `PubMed.npy`. If the values don't match within float64 epsilon, the cache is stale relative to the current code; regenerate.

### Option B — Drop the cap, recompute caches, rerun (~12–24 h)
Set `train_neg_cap = len(neg_edges)` (i.e., disable cap), delete `PubMed.npy` and `Computers.npy`, regenerate, then rerun 50 trials. This is the cleanest test of whether the cap is the culprit.

Memory budget without cap:
- PubMed: ~25 M neg edges × 25 floats × 8 B ≈ 5 GB (fine).
- Computers: ~94 M neg edges × 25 floats × 8 B ≈ 18 GB. Per-edge PI compute time is the bottleneck — for Computers this could be 20+ h on 32 cores.

### Option C — Try Original-code-equivalent PI path on a small graph
Use the *legacy* `sg2dgm/riccidist2dgm.py` code path (bypassing `parallel_pi.py`/`accelerated_PD.py`) to generate a small fresh PubMed cache (say, hop=2 on a 5K-node subset). If results jump up by ~1%p, the modernized PI pipeline has drift; if not, the cap is the culprit.

### Option D — Try paper-faithful environment (Python 3.7 + torch 1.7 + cu110)
Rebuild from `requirements.txt`, recompute everything, rerun. Last resort — high engineering cost (CUDA-11.0 wheels offline), low expected information gain since we already isolated the gap to specific code paths.

### Recommendation
Run **A first** (1 hour, definitive answer on whether the cache is stale). Then **B** only if A is inconclusive. Skip C and D unless A+B both fail.

---

## 5. Phase status

- [x] Phase 0: fix inplace op (`baselines/TLCGNN.py:48`).
- [x] Phase 1: smoke pass (PubMed 1 trial → 0.9624 AUC).
- [x] Phase 2: 50-trial sweep on PubMed/Photo/Computers — Photo passes, PubMed/Computers fail goal band.
- [ ] Phase 3: PPI (not started; requires PI cache build).
- [ ] Phase 4 (this doc).
- [ ] Phase 5: PDGNN (NeurIPS 2022) — deferred until TLC-GNN gap is closed or accepted.

---

## 6. Open question for the user

Two reasonable directions:

1. **Accept current results** (Photo at paper level, PubMed/Computers ~1 %p below) and move on to PPI + PDGNN. Document the gap honestly in any write-up.
2. **Close the gap** via Option A → B above. Adds ~1 day of compute but gives a paper-faithful reproduction across all three datasets.
