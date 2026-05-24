# TDA Conference Results

**Date:** 2026-06-21
**Plan:** `docs/superpowers/plans/2026-05-24-tda-conference.md`
**Spec:** `docs/superpowers/specs/2026-05-24-pdgnn-tda-conference-design.md`

## 1. Real-world dataset baseline (from PDGNN reproduction)

50 trials, AUC ± std. From earlier work, copied verbatim:

| Dataset | Domain | TLC-GNN exact | PDGNN approx | No PI | Best |
|---|---|---|---|---|---|
| Photo | Homo Amazon | 0.9825 ± 0.001 | **0.9860 ± 0.001** | — | PDGNN |
| PubMed | Homo citation | 0.9635 ± 0.003 | **0.9669 ± 0.002** | — | PDGNN |
| Computers | Homo Amazon | 0.9680 ± 0.002 | **0.9830 ± 0.001** | — | PDGNN |
| Chameleon | Hetero wiki | 0.9432 ± 0.007 | 0.9447 ± 0.006 | **0.9686 ± 0.006** | No PI |
| Squirrel | Hetero wiki | 0.9120 ± 0.015 (n=20) | (inference TBD) | **0.9854 ± 0.001** | No PI |
| Texas | Hetero web | 0.5709 ± 0.111 | 0.5396 ± 0.128 | **0.5939 ± 0.133** | No PI |
| Cornell | Hetero web | 0.5850 ± 0.113 | 0.5737 ± 0.115 | **0.6502 ± 0.143** | No PI |
| Wisconsin | Hetero web | 0.8640 ± 0.062 | 0.8449 ± 0.076 | 0.8653 ± 0.061 | tie |
| ChChMiner | Drug DDI | 0.9026 ± 0.007 | 0.9625 ± 0.005 | **0.9650 ± 0.006** | No PI |

## 2. SBM Sweep (B)

**Setup**: 5×5 grid of synthetic SBM graphs. N=500 nodes, K=5 blocks.
- Density axis: p_in + p_out ∈ {0.05, 0.10, 0.20, 0.30, 0.50}
- Heterophily axis: p_out/(p_in+p_out) ∈ {0.10, 0.30, 0.50, 0.70, 0.90}
- Each config: 3 variants (TLC-GNN exact / PDGNN approx / No PI) × 50 trials

### Heatmap

![SBM heatmap](../figures/sbm_heatmap.png)

3 panels: TLC-GNN AUC, PDGNN AUC, PI hurt magnitude (no-PI − TLC-GNN).

### Quantitative findings (TBD — filled after Phase 1 completes)

- Max hurt: density=___, heterophily=___, hurt=___
- Hurt threshold: density × heterophily > ___
- PDGNN approximation quality across grid: avg |PDGNN − TLC-GNN| = ___

## 3. Adaptive Gating (C)

**Setup**: GatingNet (3-layer MLP, hidden=16) inputs [clustering_u, clustering_v, |emb_u−emb_v|] → sigmoid gate ∈ [0, 1]. PI contribution multiplied by gate. End-to-end training with LP loss.

### 4-dataset comparison (TBD — filled after Phase 2 completes)

| Dataset | TLC-GNN AUC | Gated AUC | No-PI AUC | Best of three | Mean gate value |
|---|---|---|---|---|---|
| Photo | 0.9825 | TBD | — | TBD | TBD |
| Chameleon | 0.9432 | TBD | 0.9686 | TBD | TBD |
| Texas | 0.5709 | TBD | 0.5939 | TBD | TBD |
| ChChMiner | 0.9026 | TBD | 0.9650 | TBD | TBD |

### Gate behavior (TBD)

Expected:
- Photo (homophilic): mean gate > 0.5 → PI used
- Chameleon, Texas, ChChMiner (heterophilic/drug): mean gate < 0.5 → PI suppressed

If pattern holds: Adaptive gating recovers best-of-both-worlds performance.

## 4. 핵심 발견

1. **Homophilic 큰 그래프** (Photo / PubMed / Computers): PI 도움. PDGNN > TLC-GNN exact (의외).
2. **Heterophilic 그래프** (Chameleon / Squirrel / WebKB): PI 무용 또는 유해.
3. **Drug interaction** (ChChMiner): PI 명확히 유해 (−6.2%p), PDGNN approximation은 noise를 smoothing해서 no-PI 수준 회복.
4. **(B) Density × heterophily 정량 관계**: SBM sweep으로 hurt magnitude 측정. (구체 수치 TBD)
5. **(C) Adaptive gating**: 자동 의사결정. (작동 여부 TBD)

## 5. 전망

- **Drug discovery**: OGBL-DDI, BIOSNAP scale-up with batched PDGNN inference
- **Social network analysis**: heterophily strong domain → adaptive gating 적합
- **Brain connectivity**: TDA의 sweet spot, multi-scale topology
- **Heterogeneous KG**: drug-protein-disease 같은 multi-relation에 확장

## 6. Reproducibility

- Env: `environment.yml`
- Spec/plan: 위 헤더 참조
- Trained PDGNN checkpoint: `data/PDGNN/checkpoints/pdgnn_lp.pt`
- SBM caches: `data/TLCGNN/SBM_*.npy`, `data/PDGNN/SBM_*.npy`
- All splits deterministic with seed=1234 in `loaddatas.get_edges_split`
- GitHub: github.com/jjune5/TDA_conference
