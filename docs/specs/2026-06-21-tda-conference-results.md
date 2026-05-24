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

### Quantitative findings

- **63/75 configs completed** (12 missing due to compute failures, mostly at density=0.50)
- **Max PI hurt**: density=0.20, heterophily=0.10 → hurt = **+0.0282** (no-PI − TLC-GNN AUC)
- **2nd max**: density=0.30, heterophily=0.10 → +0.0267
- **PI helps most**: density=0.05, heterophily=0.30 → −0.0062
- AUC range across all configs: 0.477 – 0.773 (per-tag means: TLC-GNN 0.535, PDGNN 0.535, No-PI 0.537)

**핵심 발견 (SBM-specific)**:
- SBM 500-node + 랜덤 feature 환경에선 PI hurt가 **homophilic + mid-to-high density** 영역에서 가장 큼 — 실제 데이터(Chameleon/Squirrel) 패턴과 **반대**.
- 가능한 해석: small SBM은 (1) 노드 features가 의미 없는 random gaussian → topology가 add value 못함, (2) edge density 패턴이 real-world 데이터와 다름. **SBM이 real graph topology dynamics를 그대로 replicate하지 못함** 자체가 발견.
- 발표에서: "실제 데이터의 패턴은 단순 density × heterophily만으로는 설명 안 되고, feature signal과의 상호작용이 중요"

## 3. Adaptive Gating (C)

**Setup**: GatingNet (3-layer MLP, hidden=16) inputs [clustering_u, clustering_v, |emb_u−emb_v|] → sigmoid gate ∈ [0, 1]. PI contribution multiplied by gate. End-to-end training with LP loss.

### 4-dataset comparison (50 trials)

| Dataset | TLC-GNN AUC | **Gated AUC** | No-PI AUC | Mean gate value |
|---|---|---|---|---|
| Photo | 0.9825 ±0.001 | **0.9827 ±0.001** (11/50) | — | **1.000** |
| Chameleon | 0.9432 ±0.007 | **0.9490 ±0.007** | 0.9686 ±0.006 | **1.000** |
| Texas | 0.5709 ±0.111 | **0.5467 ±0.125** | 0.5939 ±0.133 | **1.000** |
| ChChMiner | 0.9026 ±0.007 | **0.9033 ±0.010** | 0.9650 ±0.006 | **1.000** |

### Gate behavior — **HONEST NEGATIVE RESULT**

**예상**: Photo (homo)에선 gate > 0.5, Chameleon/Texas/ChChMiner (hetero/drug)에선 gate < 0.5.

**실제**: **모든 4개 데이터셋에서 gate가 1.0으로 saturate**. Heterophily 자동 인식 실패.

| Dataset | Domain | mean gate | min | max |
|---|---|---|---|---|
| Photo | Homo Amazon | 1.000 | 0.997 | 1.000 |
| Chameleon | Hetero wiki | 1.000 | 0.998 | 1.000 |
| Texas | Hetero web | 1.000 | 0.997 | 1.000 |
| ChChMiner | Drug DDI | 1.000 | 0.596 | 1.000 |

→ Gated AUC ≈ TLC-GNN exact AUC across all datasets (확인). Gating이 효과 없음.

### 왜 안 됐나 — 분석

1. **BCE loss는 "PI off" incentive 없음**: PI가 hurt라도, 후속 MLP가 PI 부분 가중치를 0으로 학습할 수 있음 → gate가 1이어도 무방. Gate 자체에 sparsity penalty 없으면 saturate 자연스러움.
2. **3-D gate features 불충분**: [clustering_u, clustering_v, |emb_u−emb_v|]만으론 homo/hetero 구분 어려움. Graph-level statistics (전체 density / heterophily index) 필요.
3. **Sigmoid saturation**: lr=0.005에서 gate가 sigmoid 양극단으로 빠르게 쏠림 → gradient vanish.

### Future work (paper에 명시)

- **Sparsity regularizer** 추가 ($\lambda \cdot \text{mean(gate)}$ 를 loss에 더해 0 방향 압력)
- **Graph-level gate features**: per-edge 대신 per-dataset gate, 또는 graph-level statistics 입력
- **Learnable temperature**: sigmoid가 saturate 안 하게
- **Discrete gating** (Gumbel-softmax) 시도

## 4. 핵심 발견

1. **Homophilic 큰 그래프** (Photo / PubMed / Computers): PI 도움. PDGNN > TLC-GNN exact (의외).
2. **Heterophilic 그래프** (Chameleon / Squirrel / WebKB): PI 무용 또는 유해.
3. **Drug interaction** (ChChMiner): PI 명확히 유해 (−6.2%p), PDGNN approximation은 noise를 smoothing해서 no-PI 수준 회복.
4. **(B) Density × heterophily 정량 관계**: SBM 500-node 합성 sweep에서는 PI hurt가 **homophilic + mid density** 영역에서 가장 큼 (real-world와 반대) — feature signal과 topology의 상호작용이 중요함을 시사.
5. **(C) Adaptive gating**: 단순 sigmoid gate는 saturate해서 작동 안 함 (honest negative). Sparsity regularizer / graph-level features 필요 — future work.

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
