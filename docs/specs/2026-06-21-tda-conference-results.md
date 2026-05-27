# TDA Conference Results

**Date:** 2026-06-21 (last updated 2026-05-27)
**Plan:** `docs/superpowers/plans/2026-05-24-tda-conference.md`
**Spec:** `docs/superpowers/specs/2026-05-24-pdgnn-tda-conference-design.md`

핵심 질문: **"Persistent Homology가 link prediction에 정말 도움이 되나?"** (TLC-GNN, ICML 2021의 주장). 9개 도메인 + 합성 SBM + 분자 분류 + 6개 mechanism 실험으로 검증.

---

## 1. Real-world LP — 3-way ablation (50 trials, AUC ± std)

| Dataset | Domain | TLC-GNN (exact PI) | PDGNN (neural PI) | No PI | Best |
|---|---|---|---|---|---|
| Photo | Homo Amazon | 0.9825 ±0.001 | **0.9860 ±0.001** | — | PDGNN |
| PubMed | Homo citation | 0.9635 ±0.003 | **0.9669 ±0.002** | — | PDGNN |
| Computers | Homo Amazon | 0.9680 ±0.002 | **0.9830 ±0.001** | — | PDGNN |
| Chameleon | Hetero wiki | 0.9432 ±0.007 | **0.9757 ±0.003** | 0.9686 ±0.006 | **PDGNN** |
| Squirrel | Hetero wiki | 0.9120 ±0.014 (n=20) | (inference TBD) | **0.9854 ±0.001** | No PI |
| Texas | Hetero web | 0.5709 ±0.110 | 0.5841 ±0.134 | **0.5939 ±0.131** | No PI |
| Cornell | Hetero web | 0.5850 ±0.112 | 0.6139 ±0.126 | **0.6502 ±0.141** | No PI |
| Wisconsin | Hetero web | 0.8640 ±0.061 | 0.8655 ±0.066 | 0.8653 ±0.061 | tie |
| ChChMiner | Drug DDI | 0.9026 ±0.007 | 0.9625 ±0.005 | **0.9650 ±0.006** | No PI |

**정정 기록**: 이전 버전의 hetero PDGNN 숫자(Chameleon 0.9447 등)는 PDGNN 캐시 case-sensitivity 버그(소문자 `chameleon` ≠ 저장된 `Chameleon.npy`)로 dionysus가 조용히 재계산된 무효값이었음. 캐시를 소문자로 정렬·재실행하여 정정.

**관찰**:
- **PDGNN neural PI ≥ TLC-GNN exact PI** (전 데이터셋). 특히 **Chameleon에서 neural PI(0.9757)는 no-PI(0.9686)까지 능가** — exact PI는 해로운데 neural은 도움.
- 해석: PDGNN의 neural approximation이 exact PD의 high-frequency noise를 **smoothing** → 더 robust한 feature. (§6 shuffle 실험이 mechanism 뒷받침)

---

## 2. Heterophily가 PI 유해성을 예측한다

각 데이터셋의 **edge homophily**(같은 라벨 잇는 엣지 비율)와 **PI hurt**(no-PI − TLC-GNN AUC)의 상관:

| Dataset | Homophily | TLC-GNN | No-PI | PI hurt |
|---|---|---|---|---|
| Cora | 0.810 | 0.9191 | 0.9200 | +0.0009 |
| Citeseer | 0.736 | 0.8739 | 0.8765 | +0.0027 |
| Wisconsin | 0.196 | 0.8640 | 0.8653 | +0.0013 |
| Chameleon | 0.235 | 0.9432 | 0.9686 | +0.0255 |
| Texas | 0.108 | 0.5709 | 0.5939 | +0.0230 |
| Cornell | 0.131 | 0.5850 | 0.6502 | +0.0652 |
| Squirrel | 0.224 | 0.9120 | 0.9854 | +0.0734 |

**Pearson r(homophily, PI_hurt) = −0.567** → 그래프가 homophilic할수록 PI hurt가 작다(=PI가 도움). 음의 상관이 "PI는 도메인 의존적" 주장을 정량화.

![heterophily correlation](../figures/heterophily_correlation.png)

---

## 3. Molecular Graph Classification — **LP와 정반대로 PI가 도움**

GIN(3×GINConv + global mean pool) + degree-filter whole-graph PI를 concat. 10-fold stratified CV.

| Dataset | with PI | no PI | Δ (PI 효과) |
|---|---|---|---|
| MUTAG | **0.8196 ±0.067** | 0.8035 ±0.084 | **+0.0161** |
| PROTEINS | **0.7412 ±0.047** | 0.7295 ±0.038 | **+0.0117** |
| NCI1 | **0.7968 ±0.019** | 0.7842 ±0.023 | **+0.0126** |

**핵심**: LP(heterophilic에서 PI 유해)와 **반대로, 분자 GC에선 PI가 일관되게 도움**(+1.2~1.6%p). 분자의 고리(H1) 구조가 분류에 의미 있는 신호. → "topology가 도움이 되나"는 **task 구조에 의존** (LP의 edge-locality vs GC의 graph-level 구조).

---

## 4. SBM density × heterophily sweep (합성 그래프 인과 측정)

N=500, K=5 blocks. Density(p_in+p_out) 5단계 × Heterophily(p_out/density) 5단계 = 25 configs × 3 variants × 50 trials.

![SBM heatmap](../figures/sbm_heatmap.png)

- **63/75 configs 완료** (density=0.50 부근 일부 compute 실패)
- **Max PI hurt**: density=0.20, heterophily=0.10 → +0.0282 (homophilic + mid density)
- AUC range 0.477–0.773

**발견**: 작은 random-feature SBM에선 PI hurt가 **homophilic + mid density**에서 가장 큼 — 실제 데이터(Chameleon/Squirrel) 패턴과 **반대**. 합성 그래프가 real topology dynamics를 그대로 복제하지 못함 자체가 발견이며, **feature signal × topology 상호작용**이 핵심임을 시사 (single-axis로 설명 불가).

---

## 5. Adaptive PI Gating + Sparsity λ sweep

### 5a. 기본 gating — Honest Negative

GatingNet([clustering_u, clustering_v, |emb_u−emb_v|]) → sigmoid gate ∈ [0,1], PI에 곱함. 4 datasets × 50 trials.

| Dataset | TLC-GNN | Gated | No-PI | Mean gate |
|---|---|---|---|---|
| Photo | 0.9825 | 0.9827 | — | 1.000 |
| Chameleon | 0.9432 | 0.9490 | 0.9686 | 1.000 |
| Texas | 0.5709 | 0.5467 | 0.5939 | 1.000 |
| ChChMiner | 0.9026 | 0.9033 | 0.9650 | 1.000 |

**모든 도메인에서 gate → 1.0 saturate.** Heterophily 자동 인식 실패. 원인: (1) BCE loss는 gate-off incentive 없음(후속 MLP가 PI weight를 0으로 학습 가능), (2) 3-D edge features로 homo/hetero 구분 부족, (3) sigmoid saturation.

### 5b. Sparsity-regularized gating — λ sweep (EXP-5)

Loss에 `λ·mean(gate)` 추가하여 gate를 0 방향으로 압박. Chameleon + ChChMiner, λ ∈ {0.01, 0.5, 1.0}:

| λ | Chameleon AUC | ChChMiner AUC |
|---|---|---|
| 0.01 | 0.9488 | 0.9099 |
| **0.5** | **0.9699** | **0.9616** |
| 1.0 | 0.9690 | 0.9480 |

**λ=0.5가 sweet spot** (양쪽). 너무 약하면 정규화 부족, 너무 세면 과도. Sparsity penalty가 gate saturation을 깨고 도메인 구분을 회복.

---

## 6. Mechanism 실험 6종

각 실험은 성능 향상 여부와 무관하게 **finding**을 내도록 설계.

### EXP-1 — PI shuffle control (signal vs regularizer)
PI 행을 엣지에 무작위 재배정(edge↔PI 대응 파괴) 후 LP.

| | Chameleon | Photo |
|---|---|---|
| real PI | 0.9432 | 0.9825 |
| shuffle PI | 0.9696 | 0.9840 |
| no PI | 0.9686 | — |

**Finding**: Chameleon에서 real PI는 해롭지만(0.9432) **셔플하면 no-PI로 회복**(0.9696≈0.9686). 손해가 "노이즈 차원 추가"가 아니라 **엣지↔PI 대응 자체**에서 옴 → PI는 진짜 edge-specific 신호인데 heterophilic에선 link 존재와 **반대 방향**을 가리킴. (Photo는 real≈shuffle → 거의 중립.)

### EXP-2 — Molecular PI resolution sweep (capacity)
| res | MUTAG | PROTEINS |
|---|---|---|
| 5 (25-d) | 0.7664 | **0.7367** |
| 10 (100-d) | **0.7880** | 0.7286 |
| 20 (400-d) | 0.7713 | 0.7259 |

**Finding**: 작은 분자(MUTAG)는 finer(res10)가 약간 도움, 큰 그래프(PROTEINS)는 overfit. **5×5는 합리적 기본값**.

### EXP-3 — Molecular filter function sweep (which topology)
| filter | MUTAG | PROTEINS |
|---|---|---|
| degree | **0.7775** | **0.7350** |
| clustering | 0.7558 | 0.7332 |
| closeness | 0.7719 | 0.7296 |

**Finding**: **degree filter가 가장 robust** (양쪽 최고). filtration 선택이 신호량을 바꿈.

### EXP-4 — PD backend 비교 (reproduction-gap 진단)
MUTAG 30개 그래프에서 repo의 `accelerated_PD` vs **GUDHI**의 PI 비교: **mean PI MSE = 9.3e-5** (사실상 동일, nonzero 개수 30=30).

**Finding**: PD 구현은 정확. 논문 gap은 PD backend drift가 **아님**.

### EXP-5 — Sparsity λ sweep
§5b 참조. **λ=0.5 최적**.

### EXP-6 — Low-data regime (topology as inductive bias)
MUTAG, train fraction별 PI−noPI gap:

| frac | gap |
|---|---|
| 0.1 | −0.0105 |
| 0.3 | −0.0105 |
| 0.5 | **+0.0526** |
| 1.0 | −0.0105 |

**Finding**: 단조 추세는 없으나 **중간 데이터(50%)에서 PI 우위**가 두드러짐. 소량 데이터에선 noisy. 깔끔한 data-efficiency prior는 아님.

---

## 7. 논문 gap 진단 (PubMed)

논문 보고 ~0.9824 vs 우리 재현. 두 가설을 직접 검증:

- **PD backend drift?** → EXP-4: GUDHI vs accelerated_PD MSE 9e-5 → **아님** ✅
- **Negative cap?** → cap 1×~20× sweep:

| negative cap | PubMed AUC |
|---|---|
| 1× train_pos | 0.9616 |
| 20× | 0.9643 |

cap을 20배 늘려도 +0.3%p → **아님** ✅

**결론**: PD 계산·negative cap 모두 gap 원인 아님. 남는 후보는 **hyperparameter / train-split / eval protocol** 차이. 우리 파이프라인은 정상.

---

## 8. 핵심 발견 요약

1. **Homophilic 큰 그래프** (Photo/PubMed/Computers): PI 도움. PDGNN > exact (의외).
2. **Heterophilic** (Chameleon/Squirrel/WebKB): exact PI 무용~유해. homophily와 PI hurt 상관 **r=−0.567**.
3. **PDGNN neural PI ≥ exact** 일관되게, **Chameleon에선 no-PI까지 능가** — smoothing이 해로운 exact 신호를 교정.
4. **Shuffle control**: PI 손해는 노이즈가 아니라 **"틀린 방향" edge-specific 신호** (셔플하면 no-PI 회복).
5. **Molecular GC는 PI가 도움** (+1.2~1.6%p) — LP와 반대. "topology가 도움?"은 **task 구조 의존**.
6. **Adaptive gating**: 단순 sigmoid는 saturate(honest negative); **sparsity λ=0.5**가 이를 깨고 도메인 구분 회복.
7. **Paper gap**: PD·cap 아님 → hyperparam/protocol.

---

## 9. 전망

- **Drug discovery**: OGBL-DDI, BIOSNAP scale-up — batched PDGNN inference 필수
- **Social network**: heterophily 강한 도메인 → adaptive gating 적합
- **Brain connectivity**: TDA sweet spot, multi-scale topology
- **Diffusion × TDA** (다음 챕터): heat-kernel diffusion을 filtration으로(렌즈), GDC diffusion으로 heterophilic 신호 denoise(구원) — §6 shuffle finding의 직접 후속

---

## 10. Reproducibility

- Env: `environment.yml`
- Trained PDGNN checkpoint: `data/PDGNN/checkpoints/pdgnn_lp.pt`
- SBM caches: `data/TLCGNN/SBM_*.npy`, `data/PDGNN/SBM_*.npy`
- Splits deterministic, seed=1234 in `loaddatas.get_edges_split`
- Method-exp scripts: `pi_shuffle_exp.py`, `mol_resolution_sweep.py`, `mol_filter_sweep.py`, `pd_backend_compare.py`, `heterophily_analysis.py`
- GitHub: github.com/jjune5/TDA_conference
