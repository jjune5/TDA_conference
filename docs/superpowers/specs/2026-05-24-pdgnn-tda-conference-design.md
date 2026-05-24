# TDA 학회 발표 프로젝트 — Design Spec

**Date:** 2026-05-24
**Deadline:** 2026-06-21 (4주)
**Deliverables:** 10-15분 발표 + GitHub repo (`jjune5/TDA_conference`) demo
**Audience:** TDA 비전문가 (학부생/대학원생) 학회 멤버

## 목표

TDA(Topological Data Analysis)가 link prediction에 어떻게 쓰이는지 보여주고, **재현 + 비판적 발견 + 새 방법 제안**까지 묶어 발표. "이런 topology 정보를 사용해봤고 이런 전망이 있다"를 알리는 것이 핵심 메시지.

## Story arc (발표 흐름)

1. **TDA / Persistent Homology intro** (3분) — Betti number, birth/death, PD/PI
2. **TLC-GNN + PDGNN 소개** (2분) — PI를 GNN에 어떻게 합치는지
3. **재현 결과** (2분) — 3-way ablation 표 (TLC-GNN / PDGNN / no-PI) 8 datasets
4. **핵심 발견** (3분):
   - PI 도움 = homophilic
   - PI 무용/유해 = heterophilic / drug
   - PDGNN의 의외성 (exact보다 더 잘 됨)
5. **(B) Density × Heterophily 정량 sweep** (2분) — SBM heatmap
6. **(C) Adaptive PI Gating** (2분) — 우리가 제안하는 자동 의사결정 방법
7. **전망** (1분) — bio/drug discovery, social network, neuroscience 응용 가능성

## Scope

### In scope
- ✅ 현재 8개 데이터셋 결과 (homo 3 + hetero 4 + drug 1)
- ✅ **(B) SBM density × heterophily sweep** — 합성 그래프 2D grid
- ✅ **(C) Adaptive PI Gating prototype** — gate network로 PI on/off 자동 학습
- ✅ 발표 슬라이드 (10-15분)
- ✅ GitHub demo (notebook 또는 워크스루)

### Out of scope
- ⊘ Squirrel PDGNN inference (백그라운드 진행 중, 끝나면 보너스)
- ⊘ OGBL-DDI, OGBL-BioKG 같은 대규모 새 데이터셋
- ⊘ Conference / journal paper draft (학회 발표만)
- ⊘ PubMed/Computers TLC-GNN paper gap 진단 (별개 작업)
- ⊘ 다른 topological method 비교 (PEGN, PersLay 등)

## 컴포넌트

### B. SBM density × heterophily sweep

**합성 그래프 생성**:
- N = 500 노드, K = 5 communities
- Density 축: p_in + p_out ∈ {0.05, 0.1, 0.2, 0.3, 0.5}
- Heterophily 축: p_out/(p_in+p_out) ∈ {0.1, 0.3, 0.5, 0.7, 0.9}
- 5×5 = **25 합성 그래프**

**측정**:
- 각 그래프에서 TLC-GNN / PDGNN / no-PI 50-trial 학습
- Metric: PI hurt magnitude = no-PI AUC − TLC-GNN AUC

**Output**: 2D heatmap (density × heterophily → hurt magnitude). 빨강 = PI 해로움, 파랑 = PI 도움.

**Why**: 실제 데이터셋들은 여러 변수가 동시에 다름 → 노이즈. 합성 그래프로 한 축만 변화시키면 **인과관계 정량 측정**. 발표에서 quantitative finding으로 무게감 ↑.

### C. Adaptive PI Gating

**현재 모델**:
```
features = concat[(emb_u − emb_v)², PI(u,v)]
```
PI 항상 사용.

**제안 모델**:
```
gate = sigmoid(GatingNet(edge_features))    ∈ [0, 1]
features = concat[(emb_u − emb_v)², gate × PI(u,v)]
```

**GatingNet 입력 (per-edge)**:
- Local clustering coefficient (endpoint 주변 community 강도)
- Endpoint feature cosine similarity (heterophily indicator)
- Vicinity 크기

**학습**: 기존 LP loss에 그대로 추가, gate는 end-to-end로 학습.

**예상 동작**:
- Cora / Photo / PubMed → gate ~1 (PI 켜기)
- Chameleon / Texas / ChChMiner → gate ~0 (PI 끄기)

**Why**: "발견 → 활용" 흐름. 사람이 도메인 알기 전에 모델이 자동 결정. 발표에 method contribution.

## Risk

| Risk | 가능성 | Mitigation |
|---|---|---|
| C의 gate가 학습 잘 안 됨 | 중간 | B만으로도 발표 가능. C는 honest negative result로 보고 |
| SBM 학습이 너무 빠르거나 너무 느림 | 낮음 | N=500 노드로 사이즈 통제 |
| 슬라이드 시간 부족 | 중간 | B/C 중 하나만 발표하고 다른 거 백업 슬라이드 |
| Demo 환경 문제 (학회 노트북) | 낮음 | Notebook + 미리 결과 캐싱 |

## 일정

- **Week 1 (5/24–5/30)**: B 코드 + 실험 시작
- **Week 2 (5/31–6/6)**: B 완성 + C 코드 시작
- **Week 3 (6/7–6/13)**: C 실험 + 슬라이드 1차
- **Week 4 (6/14–6/20)**: 슬라이드 다듬기 + 데모 + 리허설
- **6/21**: 발표

## Success criteria

- 발표 끝까지 진행 + Q&A 응답 가능
- GitHub repo에 B의 heatmap + C의 코드/결과 추가
- 청중 중 1명이라도 "topology가 흥미롭다"라고 반응
- B의 정량적 패턴 (density 또는 heterophily가 hurt magnitude와 상관 있음) 보임
- C는 작동하면 bonus, 안 되어도 honest 보고
