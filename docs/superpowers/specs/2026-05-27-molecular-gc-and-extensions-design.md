# Molecular Graph Classification + Project Extensions — Design Spec

**Date:** 2026-05-27
**Parent project:** TDA 학회 발표 (TLC-GNN/PDGNN topology study)
**This spec adds:** (1) a new task type — molecular **graph classification** (GC) — and (2) four targeted extensions to strengthen the existing link-prediction (LP) story.

## Motivation

지금까지의 모든 실험은 **link prediction (LP)** 한 task. 핵심 발견 "topology(PI)가 도움되는지는 도메인에 따라 다르다"를 더 단단하게 하려면:
1. **다른 task type (graph classification)** 에서도 같은 질문을 검증 → 일반성 ↑
2. 기존 LP 결과의 빈틈을 메우는 4개 확장

## Part M — Molecular Graph Classification

### 질문
"분자 property 예측(graph classification)에 persistent-homology feature(PI)가 도움되는가?" — LP에서 본 도메인 의존성이 GC에서도 나타나는가?

### 데이터 (PyG TUDataset)
- **MUTAG** (188 분자, mutagenicity 이진분류) — 작아서 먼저
- **PROTEINS** (1,113 그래프, enzyme 여부)
- **NCI1** (4,110 분자, 항암 활성)

### 방법
각 그래프(분자)마다:
1. Whole-graph extended PD 계산 — filter = node degree (정규화). 기존 `accelerated_PD` (`perturb_filter_function`, `Union_find`, `Accelerate_PD`) 재사용.
2. 5×5 PI 변환 — 기존 `sg2dgm.PersistenceImager` 재사용.
3. 분류기 (PyG `GINConv` 기반):
   - **With-PI**: GIN(3 layer) → mean pool → graph_emb(64) → concat[graph_emb, PI(25)] → MLP → class
   - **No-PI**: GIN → mean pool → graph_emb → MLP → class
4. 평가: **10-fold stratified CV**, accuracy mean ± std (표준 TU 프로토콜). seed=1234.

### 새 파일
- `Knowledge_Distillation/mol_data.py` — TUDataset 로드 + per-graph PI 계산 + 캐시 (`data/MOL/<name>_PI.npy`)
- `Knowledge_Distillation/mol_classify.py` — GIN 분류기 (with/no-PI), 10-fold CV, argparse `--dataset --no_pi`

### Scope (Part M)
- **In**: MUTAG/PROTEINS/NCI1, exact PI, with/no-PI ablation, 10-fold CV
- **Out**: PDGNN neural 근사 (GC용, 별도 future work), 대규모 OGBG-MolHIV/PCBA, regression(ZINC), 다양한 filter functions

### Success criterion (Part M)
3개 데이터셋에서 with-PI vs no-PI accuracy 비교 표. "분자 분류에서 topology 효과" 명확한 답 (도움/무용/유해 어느 쪽이든 valid finding).

## Part I — LP story 확장 (4개)

### I.1 — Homophilic anchor (idea #6: Cora + Citeseer)
**왜**: idea #1 (heterophily correlation)이 null (r≈0)이었는데, 모든 데이터가 heterophilic이라 anchor가 없었음. Homophilic 작은 데이터(Cora, Citeseer)에 no-PI ablation 추가하면 correlation plot의 homophilic 끝이 채워짐.
**작업**: Cora/Citeseer에 TLC-GNN + no-PI + PDGNN 50-trial. (loaddatas에 이미 있음, dropout=0.8 권장)
**산출**: heterophily correlation plot 갱신 (homophilic 점 추가 → r 음수 기대)

### I.2 — Sparsity-regularized gating (idea #2)
**왜**: 기존 adaptive gating이 gate→1.0 saturate (honest negative). Sparsity penalty 추가로 살릴 수 있나.
**작업**: `baselines/TLCGNN_gated_reg.py` (gate에 `λ·mean(gate)` penalty + graph-level features). 4 datasets 재학습.
**산출**: gate가 도메인별로 갈리는지 (homo→1, hetero→0) 재확인

### I.3 — Negative sampling cap sweep (idea #8)
**왜**: PubMed/Computers가 paper보다 0.7~1.1%p 낮은 게 5× cap 때문인지 진단.
**작업**: PubMed에서 cap ∈ {1×, 5×, 20×, all} 비교 (env var `TLCGNN_NEG_CAP`).
**산출**: cap이 paper gap 설명하는지

### I.4 — OGBL-DDI subsample (idea #7) [optional, 가장 큼]
**왜**: ChChMiner보다 잘 알려진 drug LP benchmark로 drug claim 강화.
**작업**: OGBL-DDI에서 ~3K 노드 subgraph + TLC-GNN/PDGNN/no-PI.
**산출**: 더 큰 drug benchmark 결과. **시간 부족하면 drop.**

## 전체 우선순위

1. **이미 진행 중**: 4 hetero PDGNN 재실행 (5715-5718, neural cache 수정 후)
2. **Part M** (분자 GC) — 새 task type, 가장 임팩트
3. **I.1** (Cora/Citeseer) — #1 null 해결, 빠름
4. **I.3** (cap sweep) — 빠름, paper gap 진단
5. **I.2** (sparsity gating) — 중간
6. **I.4** (OGBL-DDI) — optional, 시간 남으면

## 파일 충돌 관리
- Part M: 전부 새 파일 (mol_data, mol_classify) — 충돌 없음
- I.1: 코드 수정 없음 (Cora/Citeseer 이미 loaddatas에 있음), SLURM만
- I.2: 새 파일 `TLCGNN_gated_reg.py` — 충돌 없음
- I.3: `loaddatas.py` get_adj_split에 env-var cap (controller가 직접 편집)
- I.4: `loaddatas.py` OGBL loader (controller가 직접 편집, I.3과 순서 조정)

→ 분석/실험 agent는 새 파일만, loaddatas.py 공유 편집은 controller가 직렬 처리.

## Success criterion (전체)
- 분자 GC 3-dataset with/no-PI 표
- heterophily correlation plot에 homophilic anchor 추가 (r 갱신)
- cap sweep로 paper gap 진단
- (되면) sparsity gating positive 또는 honest negative 확정
- 모든 결과 results doc + slides + README + GitHub push
