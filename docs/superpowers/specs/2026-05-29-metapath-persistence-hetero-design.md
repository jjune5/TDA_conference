# Meta-path-induced Persistence for Heterogeneous Graph Node Classification — Design

**Date:** 2026-05-29
**Author idea:** 박준영 (Notion: "stanford CS224W lec09 + PDGNN 기반 아이디어 제안")
**Status:** design (brainstorming) → writing-plans next
**Env:** `tlcgnn` (PyG 2.5.3 — DBLP/IMDB/HGBDataset/OGB_MAG 내장, gudhi 3.11, `Knowledge_Distillation/pdgnn_modern.py` 재사용 가능 — 모두 검증됨)

## Goal

이종(heterogeneous) 그래프의 node classification에서, **meta-path로 유도한 동종(homogeneous) 서브그래프의 persistent homology(EPD) feature**가 hetero-GNN 성능을 높이는지 검증한다. 핵심: 위상 신호가 **genuine**한지(누수·아티팩트 아님) exact-first로 증명한 뒤, PDGNN으로 대형 그래프(ogbn-mag)에 확장.

## Motivation (prior-art로 확정된 gap)

백그라운드 문헌 검증 결과(high confidence):
- **"meta-path 유도 그래프에 PH → HIN node classification" 조합은 문헌에 없음.** PH-for-GNN 문헌(PDGNN NeurIPS'22, PEGN AISTATS'20, TOGL ICLR'22, RePHINE'23)은 **전부 homogeneous**; hetero-GNN 문헌(HAN, HGT, Simple-HGN)은 meta-path는 쓰되 **위상 feature 없음.** 두 문헌이 한 번도 결합된 적 없음.
- **가장 가까운 선행연구 = HTGNN** (Liu & Kok, WITS 2024, arXiv 2506.06293): PH ⊕ 다중관계 GCN으로 node classification을 하지만 **단일 노드타입(은행)**, node-feature point cloud의 **Rips PH**(meta-path 아님), 비공개 금융데이터. → **반드시 인용·구분**(다른 PH 구성·단일타입·non-benchmark).
- **PDGNN** (NeurIPS'22): neural EPD 근사(~100×), 그러나 **homogeneous 전용**, HAN/HGT와 결합된 적 없음. → 우리 기여 = "neural PH를 meta-path filtration + hetero-GNN에 처음 통합."
- **Disambiguation 필수**: HL-HGAT(2024)의 "heterogeneous"는 *k-simplex(Hodge-Laplacian)* 뜻이지 HIN이 아니고 PH도 아님. 논문에 명시 구분.

**왜 위상이 도울 여지가 있나**: meta-path 그래프(예: APA 공저망)는 의미 있는 고차 구조(공저 클릭, 인용 루프, 커뮤니티)를 가짐. HAN은 meta-path 1-hop 이웃을 attention으로 aggregate하지만 **고차 사이클엔 blind** — PH가 정확히 그걸 잡음. 이게 articulable한 gap.

**왜 §14 함정이 없나**: 이건 node classification이라 LP의 "타겟 엣지 제거 → PI=0" 누수 비대칭이 **구조적으로 없음**. (§14는 LP 전용 함정)

## 정직한 risk (설계가 맞서야 할 것)

1. **HGB 역풍** (Lv et al., KDD'21): "대부분 hetero 데이터셋에서 meta-path는 불필요" — 잘 튜닝된 Simple-HGN이 meta-path HGNN을 따라잡음. → **PH가 Simple-HGN의 edge-type embedding을 *넘어서* 신호를 주는지** ablation으로 증명해야 기여 인정.
2. **DBLP/ACM 포화** (~93–94%, SOTA Simple-HGN DBLP 94.01/94.46, ACM 93.42/93.35). 헤드룸이 거의 없어 위상 이득 보이기 어려움. → **작은 데이터셋은 "신호 genuine + 파이프라인 맞음" 검증용; 진짜 기여 입증은 ogbn-mag(~58%, 헤드룸 큼).**
3. **위상 = modest gain 전력** (우리 §14/§16): 강한 GNN이 이미 구조를 잡을 수 있음. honest ablation + null도 finding으로 보고.

## 새 누수 함정 — §14 교훈을 이 설계에 적용 (비협상)

**label을 정의하는 relation을 경유하는 meta-path는 라벨을 누수시킨다.**
- ACM: target=Paper(3 classes ≈ conference 영역). meta-path **PSP**(paper–subject–paper)의 subject가 라벨과 동치에 가까우면 → meta-path 그래프가 정답 인코딩 = §14와 동형 누수.
- DBLP: target=Author(4 research areas). **APCPA**(저자–논문–학회–논문–저자)의 conference가 research-area 라벨과 강상관 → 누수 위험. **APA**(공저)는 상대적으로 안전.
- **규칙**: 각 meta-path를 라벨-인접 relation 경유 여부로 **감사**하고, 누수 의심 meta-path는 (a) 제외하거나 (b) 별도 표기해 "leaky vs clean" 대조로만 사용. 감사 결과를 결과 문서에 명시.

## Method — Phase 1 (Idea 1: meta-path induced subgraph EPD), exact-first

각 데이터셋에서 **target 노드 타입으로 끝나는** meta-path P 집합을 정한다(처음엔 canonical: ACM=PAP[, PSP는 누수감사], DBLP=APA[, APCPA는 누수감사]).

1. **Meta-path 그래프 구성**: relation adjacency 곱 `A_P = A₁ A₂ … A_k` (PyG `metapath` / 직접 sparse matmul). 결과의 **off-diagonal nonzero = 새 엣지, 그 값 = meta-path 인스턴스 수(= 관계 강도)**.
2. **Weighted filtration** (Notion의 degree 비교보다 풍부): 엣지 가중치 = 인스턴스 수(또는 1/count로 거리화)로 **sublevel/Rips filtration**. 노드 filter = degree 또는 HKS(확산) — 둘 다 ablation.
3. **EPD(exact)**: gudhi로 각 target 노드의 ego 또는 전체 meta-path 그래프에서 extended persistence → persistence image(PI, 5×5=25d) 벡터화. (exact-first: 신호 real인지 먼저 증명, PDGNN은 Phase 2.)
4. **여러 meta-path 결합**: 노드별 `[PI_P1, PI_P2, …]` concat (또는 HAN식 semantic attention). → per-node 위상 feature.
5. **분류기**: hetero-GNN baseline(아래)의 노드 임베딩에 위상 feature를 concat → classifier. PDGNN 논문 방식(EPD feature를 GNN 임베딩에 결합) 미러.

**3-way 비교** (Notion대로): (i) plain hetero-GNN, (ii) +PH feature, (iii) baseline Simple-HGN. + **controls**(§14 교훈): (iv) **random/shuffled filter** PH(구조가 진짜 기여하나?), (v) **shuffled-PH**(노드↔PI 대응 끊기 → genuine인지).

## Validation gates (exact-first, 비협상)

- **G1 파이프라인 smoke**: ACM/DBLP에서 meta-path 그래프 구성 → exact PH → PI 캐시. shape/nonzero 정상.
- **G2 신호 genuine?**: ACM/DBLP 3-way + controls. **PASS 조건**: +PH가 plain 대비 유의 향상 **AND** shuffled-PH 대비 유의 우위(genuine). null이면 "왜 안 되나" 분석(HGB 역풍/포화)으로 보고.
- **G3 기여 입증**: **ogbn-mag**(헤드룸 큼)에서 PDGNN으로 확장(Phase 2) 후 **Simple-HGN 능가** 여부. 이게 진짜 contribution gate.

## Method — Phase 2 (PDGNN으로 확장)

meta-path 그래프는 dense(공저 클릭 → 대형 클릭)해서 exact PH 폭발 → **PDGNN(`pdgnn_modern.py`)으로 EPD 근사**. ogbn-mag(1.9M 노드) 대상. PDGNN을 meta-path 그래프 분포에 재학습 필요할 수 있음(risk). 이 단계에서 PDGNN의 ~100× 속도 + 미분가능성이 실제로 필요한 자리 = 동기 정렬.

## Method — Phase 3 (Idea 2: unified type-aware filter), 조건부·후순위

전체 혼합타입 그래프에 unified filtration:
- **type별 learnable filter MLP**(CS224W type-specific pre-MLP): 타입마다 다른 차원/의미의 feature → scalar filter value.
- **structural augmentation**: relation별 degree concat.
- **calibration**: type별 quantile-norm으로 cross-type 비교가능. tie-break = node-type priority.

**⚠️ 정직한 caveat (§14 교훈, 문서에 반드시 명시)**:
- quantile-norm의 **within-type 순서 보존**은 stability theorem 맞지만, "Paper 중앙값 = Author 중앙값 = 같은 등장시점"은 **정리가 아니라 모델링 가정** — "정리라서 안전"으로 과대주장 금지.
- learnable filter를 라벨로 end-to-end 학습 시 **라벨을 filtration에 심은 학습 feature**(PH 병목 통과한 가짜 위상)가 될 위험 → random-filter control + filter 안정성 검증 필수.
- quantile-rank의 미분가능성(soft-rank) 이슈.
→ fragile하므로 Phase 1이 genuine 신호를 보인 뒤에만 착수.

## Datasets & Baselines (prior-art 표)

| Dataset | nodes | target | classes | meta-paths | SOTA(Simple-HGN, Ma/Mi) |
|---|---|---|---|---|---|
| ACM (HGBDataset) | 10,942 | Paper | 3 | PAP, PSP* | 93.42 / 93.35 |
| DBLP (HGBDataset) | 26,128 | Author | 4 | APA, APCPA* | 94.01 / 94.46 |
| ogbn-mag (OGB_MAG) | 1.9M | Paper(venue) | 349 | PAP, PP(cites) | ~0.58 (acc) |

(*누수 감사 대상 meta-path) Baselines: RGCN, HAN, HGT, **Simple-HGN**(핵심 비교상대). 작은 셋=검증, ogbn-mag=기여.

## Components / files (additive, 기존코드 수정 금지)

신규 모듈 (TLC-GNN repo 내 `hetero/` 디렉토리 제안):
- `hetero/metapath_graph.py` — HeteroData → meta-path 그래프(sparse matmul) + 누수 감사 유틸
- `hetero/metapath_ph.py` — meta-path 그래프 → exact EPD/PI (gudhi; `node_ph_features.py` 패턴 재사용)
- `hetero/hetero_lp_pipeline.py` — 3-way + controls 학습/평가 드라이버 (Simple-HGN baseline 포함)
- 재사용: `Knowledge_Distillation/pdgnn_modern.py`(Phase 2), `sg2dgm.PersistenceImager`

## Scope decomposition (→ writing-plans)

- **Phase 1**: meta-path 그래프 + exact PI + 누수 감사 + 3-way/controls on ACM/DBLP. (검증)
- **Phase 2**: PDGNN 확장 → ogbn-mag, Simple-HGN 능가 시도. (기여)
- **Phase 3 (조건부)**: unified type-aware filter (Idea 2). Phase 1 genuine 신호 후.

## Naming / 논문 framing

- 이름: "topological heterogeneous GNN"은 HTGNN·HL-HGAT가 선점 → **구분되는 이름** + Hodge-Laplacian "heterogeneity"와 명시 구분.
- framing: "first to bring (neural) persistence to **meta-path-induced filtrations** for HINs" (neural PH 자체가 아님 — PDGNN 존재).
- 인용 필수: HTGNN(구분), PDGNN/PEGN/TOGL/RePHINE(homogeneous 계보), HGB(역풍·baseline).
