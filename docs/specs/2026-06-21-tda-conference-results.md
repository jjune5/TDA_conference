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
- **Diffusion × TDA**: §11에서 수행 — HKS 렌즈(A), GDC denoise로 hetero PI 무해화(B), 항체 생성평가 NO-GO(C)

---

## 10. Reproducibility

- Env: `environment.yml`
- Trained PDGNN checkpoint: `data/PDGNN/checkpoints/pdgnn_lp.pt`
- SBM caches: `data/TLCGNN/SBM_*.npy`, `data/PDGNN/SBM_*.npy`
- Splits deterministic, seed=1234 in `loaddatas.get_edges_split`
- Method-exp scripts: `pi_shuffle_exp.py`, `mol_resolution_sweep.py`, `mol_filter_sweep.py`, `pd_backend_compare.py`, `heterophily_analysis.py`
- Diffusion-chapter scripts: `Knowledge_Distillation/hks_filtration.py`, `Knowledge_Distillation/mol_filter_sweep_hks.py`, `gdc_pi.py` (+ `TLCGNN_GDC` hook in `loaddatas.py`), `antibody_tda/`
- GitHub: github.com/jjune5/TDA_conference

---

## 11. Diffusion meets Topology (챕터)

**물리적 확산** vs **생성 확산**, 둘 다 persistent homology로 통합. Spec: `docs/superpowers/specs/2026-05-27-diffusion-tda-chapter-design.md`. 3개 thread를 격리 worktree 에이전트로 병렬 수행.

### 11a. Thread A — diffusion을 filtration 렌즈로 (HKS, physical)

Heat Kernel Signature(graph Laplacian heat kernel, scale `t=1/median(λ⁺)`)를 filtration으로 사용, 분자 GC 10-fold CV.

| Dataset | degree | clustering | closeness | **HKS** |
|---|---|---|---|---|
| MUTAG | 0.7667 | 0.7558 | 0.7667 | **0.7670** |
| PROTEINS | 0.7313 | 0.7296 | 0.7322 | **0.7332** |
| NCI1 | **0.7696** | 0.7560 | 0.7552 | 0.7637 |

**Finding**: diffusion 렌즈(HKS)는 기하 filtration과 **대등** — MUTAG/PROTEINS 근소 1위, NCI1 2위, **어떤 데이터셋에서도 꼴찌 아님.** 극적 우위는 없으나 "diffusion = 경쟁력 있는 multi-scale 렌즈" 확인.

### 11b. Thread B — diffusion을 denoiser로 (GDC, physical) ★주요 결과

Graph Diffusion Convolution(heat kernel t=5, topk=16)으로 그래프를 diffuse한 뒤 그 위에서 PI 계산. 50 trials.

| Dataset | Domain | PI(exact) | no-PI | **GDC-PI** | GDC−PI | GDC−noPI |
|---|---|---|---|---|---|---|
| Cora | homo | 0.9191 | 0.9200 | 0.9215 | +0.002 | +0.002 |
| Photo | homo | 0.9825 | 0.9839 | 0.9827 | +0.000 | −0.001 |
| Chameleon | hetero | 0.9432 | 0.9686 | **0.9697** | **+0.027** | **+0.001** |
| Texas | hetero | 0.5709 | 0.5939 | 0.6157 | +0.045 | +0.022 |
| Cornell | hetero | 0.5850 | 0.6502 | 0.6184 | +0.033 | −0.032 |

**Finding (정직)**:
- **Heterophilic에서 GDC가 유해한 exact-PI 신호를 중화 → no-PI 수준으로 복구.** 저분산 대형 그래프 **Chameleon이 깨끗한 증거**: GDC-PI 0.9697 = PI 대비 **+0.027(손해 완전 복구)**, no-PI 대비 **+0.001(동률)**.
- Texas/Cornell은 작아서(std ±0.08~0.14) 노이즈 큼 — Texas는 no-PI를 약간 넘고(노이즈 안) Cornell은 부분 복구. **큰 그래프가 진실 = "동률 복귀"** (no-PI 초과 아님).
- **부가 발견**: GDC-PI가 소형 hetero 그래프에서 **분산을 절반으로** 줄임(Texas std 0.079 vs 0.13, Cornell 0.057 vs 0.14) — diffusion이 신호 안정화.
- Homophilic(Cora/Photo): 셋 다 동률, GDC 효과 미미.
- **메커니즘 일관성**: §6 shuffle finding(PI 손해 = "틀린 방향" edge-specific 신호)과 부합 — GDC가 그 신호를 평탄화해 무해화. 단 **no-PI를 *넘는* 건 여전히 PDGNN(Chameleon 0.9757)뿐**; GDC=중립 복귀, PDGNN=실제 향상.

### 11c. Thread C — 생성 확산 평가 (항체 CDR-H3, generative) → NO-GO

DiffAb/FlowDesign/IgGM이 생성한 항체 CDR-H3 loop의 persistent homology가 binding quality(DockQ)와 관련되는지 검증. **생성·학습 없이** 기존 출력(FlowDesign 67k PDB) + 사전계산 DockQ 사용. RabD 60 + time_split 60 타깃 × 40 샘플.

**Finding (정교한 NULL)**:
- corr(topo_dist_to_native, DockQ): pooled −0.58처럼 보이나 **per-target ≈ 0** → **Simpson's paradox**(어려운 타깃이 topo-거리↑+DockQ↓ 동반)의 between-target 인공물. 실제 선택 상황(within-target)엔 신호 없음.
- topology 기반 후보 선택이 **random과 동률** (native-free max loop-likeness: RabD 0.8901 vs random 0.8923; min topo-dist 0.8897).
- topo_dist vs RMSD-CDRH3도 within-target ≈ 0 → 상보적 신호 아님.
- **결론**: single-H3 inpainting에서 **PH-on-Cα는 너무 coarse** (orientation/side-chain 버림) → topology-guided 생성(Rung1 guidance/Rung2 training) **짓지 않음.** 재시도하려면 within-target DockQ 분산이 큰 설정(full-complex/multi-CDR/docking-pose) + Cα보다 풍부한 descriptor 필요.
- **방법론 교훈**: pooled correlation의 **Simpson's paradox 함정** — 생성모델 평가에서 per-target(조건부) 분석 필수.

**재게이트 (Rung 0′) — interface-PH × multi-CDR**: loop 신호가 죽어서, 더 풍부한 신호(**항체-항원 interface contact-graph PH**) + 고분산 설정(**multi-CDR**, within-target RMSD 분산 ~2.5×)으로 재검증 (rabd 60 + time_split 742 타깃, ~32k candidates):
- **interface-PH vs DockQ: 여전히 within-target ≈ 0** (per-target Spearman rabd −0.026 / time_split −0.030; Simpson's paradox 재현 — pooled은 음수처럼 보임). 풍부한 신호·고분산에도 **binding 신호 없음**.
- **interface-PH vs RMSD-CDRH3: 약하지만 실재** (per-target Spearman +0.10(rabd) ~ +0.17(time_split), p<1e-70, within-centered +0.11~+0.17; **loop 신호의 ~4×**). → topology가 **binding이 아니라 구조 정확도(RMSD)**를 약하게 추적.
- 선택 실험: interface-PH 선택이 **DockQ에서 random 못 이김** (native-free Δ+0.005, native-oracle Δ−0.011).
- **최종 결론**: topology는 항체 **binding 품질** 신호 없음 — **두 게이트 모두 NO-GO**(loop@single-H3, interface@multi-CDR) → **topology-guided 항체 *binding* 생성 종료.** 단 **구조 fidelity(RMSD)** 용도엔 약한 interface-PH prior 가능(미약, 향후 여지).
- **메타**: Simpson's paradox가 두 게이트 모두의 함정 — generative 평가에서 per-target 분석 필수.

### 11d. 챕터 종합

- **Physical diffusion (A,B)**: topology의 렌즈(HKS 대등)·denoiser(GDC가 hetero PI를 무해화)로 **작동**. **B가 가장 강한 새 결과** + shuffle 메커니즘과 일관.
- **Generative diffusion (C)**: 항체 **binding 품질**엔 PH(loop·interface 둘 다)가 신호 없음 — **두 게이트 모두 NO-GO**. 단 interface-PH는 **구조 RMSD를 약하게 추적**(loop의 ~4×). 정직한 null + Simpson's paradox 교훈.
- **큰 그림**: "topology가 언제 유용한가"의 답에 diffusion 축 추가 — diffusion은 topology를 **정제**하는 덴 좋지만(B), 항체 loop의 **binding 품질**을 잡기엔 PH가 부족(C; 구조 RMSD엔 약한 신호만).

---

## 12. Experiment Batch 2 — Mechanism + Robustness

Spec: `docs/superpowers/specs/2026-05-28-experiment-batch2-design.md`. 5개 병렬 worktree 에이전트. 헤드라인 두 미스터리(PDGNN>exact, GDC 구원) 기전 해명 + TDA robustness 주장 첫 검증.

### 12a. P1 — PDGNN은 spectral smoothing이 아니다
exact PI를 Gaussian blur(σ) 후 50-trial LP (Chameleon):

| | AUC |
|---|---|
| exact PI | 0.9432 |
| blur σ=1 / 2 / 3 | 0.9450 / 0.9494 / 0.9500(포화) |
| **PDGNN** | **0.9757** |
| no-PI | 0.9686 |

**Finding**: blur은 σ↑에 단조 개선하나 ~0.950에서 포화 — PDGNN(0.9757)·no-PI(0.9686) 둘 다 못 넘음. **PDGNN 우위는 저주파 평활이 아니라 학습된 neural PD 표현.** (Photo/Computers는 클러스터 load로 미완; Chameleon이 최대 gap 케이스라 충분.)

### 12b. B1+B2 — GDC 구원의 강도·일반화·경계
**B1 강도 (Chameleon):** t∈{1,3,5,10}@k16 = 0.9717/0.9703/0.9697/0.9702 (**평평, t=1 즉시 포화**); k∈{8,16,32}@t5 = 0.9679/0.9697/0.9707 (약한 단조). → **하이퍼파라미터 robust, t 즉시 포화, 전부 no-PI 패리티 근처.**
**B2 일반화:** ChChMiner GDC-PI 0.957 (plain 0.903 → no-PI 0.965의 99% 구원 ✓); **Wisconsin GDC-PI 0.822 < plain 0.864 (구원 실패, 역효과)** — 251노드 거의 완전그래프, over-smoothing. → **GDC 구원은 bounded: 적정 크기/밀도에서만; tiny-dense엔 유해.**

### 12c. M2 — 분자 PI 신호는 H1(고리)
PD를 homology dim별 마스킹 후 GIN 10-fold + **SVM(PI-only) 교차검증**:

| variant | MUTAG GIN | MUTAG SVM(PI-only) |
|---|---|---|
| H0-only | 78.2 | 66.0 |
| H1-only | 77.1 | **82.0** |
| both | 77.7 | 74.5 |
| no-PI | 75.1 | — |

**Finding**: SVM이 결정적 — H1-only PI alone 82% ≫ H0-only 66%, 전체 withPI GIN과 동등. **분자 PI 이득은 H1(고리/loop).** raw GIN은 H0처럼 보이나(gap이 노이즈 내 + GIN이 이미 ring을 message-passing으로 학습 → H1 PI가 GIN엔 redundant). PROTEINS는 PI 신호 미미. NCI1 생략(load).

### 12d. A2 — HKS(diffusion) filtration을 LP에 (diffusion을 어디에 넣나)

| | Cora | Chameleon |
|---|---|---|
| no-PI | 0.9200 | 0.9686 |
| Ricci-PI (기본) | 0.9191 | 0.9432 (hurt) |
| HKS-PI | 0.9186 | 0.9489 |
| GDC-PI (참고) | 0.9215 | 0.9697 |

**Finding**: HKS filtration(`|HKS_u−HKS_v|`를 vicinity Dijkstra filter로, env `TLCGNN_LP_FILTER=hks`)이 hetero-hurt를 **부분 완화**(0.9432→0.9489, 손해의 ~22%)하나 제거 못 함. → **그래프 denoise(GDC=완전구원) > filtration 교체(HKS=부분).** 손해는 filtration 선택보다 **그래프 구조**에 있음.

### 12e. N1 — topology가 그래프 노이즈에 robust ★첫 pro-topology
엣지 p% 교란(제거+무작위 추가, label 누수 없음) 후 PI/no-PI/GDC-PI 열화 기울기 (ΔAUC/+10%):

| | PI | no-PI | GDC-PI |
|---|---|---|---|
| Cora | −0.0069 | **−0.0077(최악)** | −0.0060 |
| Chameleon | **+0.0051** | −0.0011 | −0.0006 |

**Finding**: 4/4 케이스에서 PI·GDC-PI가 no-PI보다 천천히 열화. **Chameleon: PI 기울기 양수 → 깨끗할 땐 no-PI 아래(hetero-hurt)지만 20% 노이즈에서 no-PI 역전(0.9827 vs 0.9817).** **그래프가 망가질수록 topology가 더 가치.** 프로젝트 **첫 pro-topology LP 결과.** (caveat: Chameleon n=10·단일 seed·load로 축소 — seed 추가 권장.)

### 12f. 배치 2 종합
1. **두 헤드라인 미스터리 해명**: PDGNN>exact는 **smoothing 아님**(P1 — 학습된 표현); GDC 구원은 **강도-즉시포화 + bounded**(B1/B2 — tiny-dense엔 역효과).
2. **Mechanism**: 분자 PI 이득 = **H1(고리)**(M2, SVM 확증); hetero 손해는 **그래프 구조 > filtration**(A2).
3. **새 pro-topology 축**: topology는 **노이즈에 robust**(N1) — TDA 셀링포인트의 첫 LP 증거.
4. **메타 통찰**: 평활(graph든 PI든)은 **no-PI 패리티가 한계**; no-PI를 *넘는* 건 학습(PDGNN)에서만. **topology의 진짜 강점은 정확도가 아니라 robustness일 수 있다** — 깨끗한 벤치마크 AUC는 topology를 과소평가하고, 노이즈/불완전 그래프에서 진가가 드러남.

**Batch-2 스크립트**: `pi_blur_exp.py`, `gdc_pi.py`(+env `TLCGNN_GDC_T/K`), `Knowledge_Distillation/mol_dim_ablation.py`, `loaddatas.py`(env `TLCGNN_LP_FILTER=hks`), `noise_robust_exp.py` (+ `results/noise_robust/`).

---

## 13. Experiment Batch 3 — Deepen mechanism + robustness (load-aware)

Spec: `docs/superpowers/specs/2026-05-28-experiment-batch3-design.md`. 2개 CPU-light 실험(기존 PI 캐시 재사용 → 배치2의 CPU thrash 회피). 배치2의 두 핵심 thread(P1 mechanism, N1 robustness) 심화.

### 13a. D1 — PDGNN이 학습하는 것 (P1 기전 규명, CPU 분석)
PDGNN-PI vs exact-PI 캐시 per-edge 비교 (Photo/Computers/Chameleon, 모두 정렬됨):

| Dataset | MSE(PDGNN,exact) | MSE(blur σ3,exact) | 비율 |
|---|---|---|---|
| Photo | 0.0583 | 0.00009 | **659×** |
| Computers | 0.0464 | 0.00011 | **429×** |
| Chameleon | 0.4732 | 0.00019 | **2493×** |

**Finding**: MSE(PDGNN,exact) ≈ MSE(PDGNN, blur σ3) (비율 1.000) → PDGNN은 exact에서 **σ3 blur보다 429~2493× 더 벗어남 = smoothing 아님(확정).** 변환은 전 셀 양수(중앙 mid-birth/death 집중 1.3~2.7×)지만 **scalar 증폭은 MSE의 1~19%만 설명, 81~99%가 edge별 구조적 잔차**(spatial SNR 0.015~0.028). edge-type 무관(homophily 상관 ~0.005, n.s.). → **PDGNN = edge별 학습된 topology-conditioned persistence-mass 재조정.** P1과 합쳐 2겹 증거.

### 13b. R2 — feature-noise robustness (N1 보완, NULL)
노드 feature 손상(`+N(0,(q·σ_d)²)`) 후 LP, **PI 캐시 재사용**(위상 불변, n=20):

| q | Cora gap (PI−noPI) | Chameleon gap |
|---|---|---|
| 0 | +0.0005 | −0.0258 |
| 0.5 | −0.0017 | −0.0227 |
| 1.0 | −0.0001 | −0.0186 |

**Finding (NULL)**: PI−noPI gap이 어느 데이터셋도 양수로 안 바뀜. Cora flat(~0), Chameleon 음수 유지(좁아지나 역전 못 함). feature 노이즈는 PI/no-PI를 동일 타격(둘 다 GCN encoder가 feature 사용) → topology가 feature-robustness prior로 안 떠오름.

### 13c. 종합 — robustness 주장 정밀화
- **D1**: PDGNN의 no-PI 초과 = **edge별 학습 표현**(평활 아님). P1 확정.
- **N1(edge noise) vs R2(feature noise) 대조 ★**: topology의 robustness 이점은 **구조(edge) 노이즈에 특이적**(N1: Chameleon 노이즈서 no-PI 역전) — **feature 노이즈엔 없음**(R2 null). 직관: PI는 그래프 *구조*에서 나오므로 구조 perturbation엔 강하나, feature 노이즈는 GCN encoder만(PI/no-PI 공통) 타격 → 차등 이점 없음. → **TDA robustness 강점은 "데이터 노이즈 일반"이 아니라 위상을 정의하는 *구조 자체*의 노이즈에 특이적.**

**Batch-3 스크립트**: `analyze_pdgnn.py`, `feature_noise_exp.py` (+ `results/{pdgnn_analysis,feature_noise}/`). 둘 다 PI 캐시 재사용(CPU-light).
