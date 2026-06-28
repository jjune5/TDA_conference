# Hetero-PDGNN — heterogeneous graph의 semantic topology 보존 (proposal + 실험)

> 상태: **proposal(가설) + MVP 진행 보고.** 아래 claim은 작업가설이며, 지금까지 구현·검증된 것은
> 정직한 파이프라인과 제안 메커니즘이다. **성능 향상 주장 없음** — toy는 정합성 점검, 실데이터
> 결과는 나온 그대로 보고한다. fallback descriptor는 PDGNN/EPD가 아니다.

## Claim (가설)

기존 PDGNN은 EPD 계산을 효율적으로 근사하지만, heterogeneous graph에서는 filtration ordering과
Union-Find event가 node/edge type을 반영하지 못한다. 이로 인해 서로 다른 의미의 topological event가
동일한 birth/death point로 collapse된다. 우리는 **pair-conditioned relation-aware filtration,
multi-slice persistence, typed cycle signature**를 통해 heterogeneous graph의 semantic topology를
보존하는 **Hetero-PDGNN**을 제안한다.

## Background

- **PDGNN**: 스칼라 filtration의 Union-Find merge/relax event를 GNN으로 모방해 EPD를 근사.
- **TLC-GNN**: 링크예측에 pairwise persistent homology feature 사용.
- 둘 다 homogeneous + 단일 스칼라 filtration 가정. hetero에선 타입 간 filter value가 비교 불가하고,
  단일 filtration + untyped Union-Find가 의미가 다른 event(예: author-paper-author vs paper-field-paper
  cycle)를 같은 점으로 합쳐버린다. → **타입 간 비교를 어떻게 하느냐가 핵심 문제.**

## Baseline (우리가 만든 것)

self-contained hetero 링크예측 MVP (`hetero_pdg`):

- toy HeteroData(author/paper/field) + 소형 실데이터(ACM, IMDB)
- meta-path projection (APA / PFP / PCP, adjacency 행렬 곱)
- 5개 LP mode: `no_topology` / `collapsed` / `metapath concat·attention` /
  `unified`(type-aware filtration: 타입별 MLP + quantile calibration + lexicographic 정렬)
- topology backend: **fallback**(결정론적 descriptor, 기본; PDGNN/EPD 아님),
  **real_tlc**(exact EPD), **real_pdgnn**(neural EPD) — 뒤 둘은 기존 PDGNN/TLC 엔진 재사용
- leakage 차단: 관측 그래프에서 val/test 엣지 제거 + 후보 쌍별 target 엣지 제거
- 참고: 원본 아이디어의 tie-break는 `filter*10000 + priority` 큰상수 방식이었으나, 수치 안전성을 위해
  **lexicographic (calibrated value, type priority)** 으로 구현(작은 값 차이가 priority에 묻히지 않음).

## 아이디어 추가 (3 메커니즘)

| 메커니즘 | 아이디어 | 정직 상태 |
|---|---|---|
| Pair-conditioned filtration | 노드 filter가 노드뿐 아니라 쿼리 쌍 (u,v)·relation r에 의존 | approximate (학습형 스칼라장 + vicinity 통계; PH 좌표 아님) |
| Relation-aware edge filtration | g(i,j,ρ)=max(f_i,f_j)+softplus(δ_ρ), 항상 g≥max 보장 | prepared interface (제약·테스트만; exact EPD엔 미연결) |
| Multi-slice persistence | 벡터 filtration F(x)를 K개 convex-combination 슬라이스로 보고 attention 융합 | approximate (스칼라 슬라이싱; exact multi-parameter PH 아님) |
| Typed cycle signature | typed node/edge 히스토그램 + meta-path별 cycle/공통이웃 proxy | proxy (typed counting; PH 아님) |

## 실험

**Toy (통제).** planted 양성대조(인용을 field에 동질화) vs random 음성대조. 파이프라인은 신호가 있으면
잡고(planted: typed_cycle 0.81 / metapath 0.75) 없으면 chance 근처. 단 학습형(pair/multi_slice)은
toy에선 효과 없음(≤chance). → 정합성 점검일 뿐, 성능주장 아님.

**실데이터 (ACM, IMDB).** target = paper-cite-paper(ACM) / 파생 movie-codir-movie(IMDB),
타깃 엣지 1000 cap. test-AUC (seed 3개 평균):

| config | ACM | IMDB |
|---|---|---|
| **no_topology (baseline)** | **0.701** | **0.722** |
| metapath_concat (fallback) | 0.691 | 0.709 |
| metapath_attention (fallback) | 0.723 | 0.720 |
| metapath_concat (real_tlc, exact EPD) | 0.690 | 0.712 |
| metapath_concat (real_pdgnn, neural EPD) | 0.713 | 0.704 |

## 결과 (정직)

- ACM·IMDB 둘 다 **어떤 위상 변형(fallback / exact EPD / neural EPD)도 no_topology baseline을
  유의하게 못 이긴다** (전부 ~0.69–0.72, baseline이 상위권). → **위상-as-feature가 hetero 링크예측을
  돕는다는 증거 없음.**
- Phase-3 고급 3종(pair / multi_slice / typed_cycle)은 **toy 스키마 전용**이라 실데이터의 featureless
  타입에서 실행 실패 — 현재 한계(결과 아님).
- 즉 claim은 **실데이터로 미입증**. 동기/가설로서는 유효하며, 이 정직한 null이 다음 설계를 가리킨다.

## 한계

- 소형 그래프만. 성능주장 없음. fallback은 PDGNN/EPD가 아님.
- pair/multi_slice는 학습형 스칼라장 vicinity 통계(PH 좌표 아님), typed_cycle은 counting proxy,
  edge filtration은 prepared interface(exact EPD는 여전히 node→edge max).
- 고급 3종은 실데이터 스키마(타입/feature)에 미일반화.

## claim 확장 방법

1. 더 크고 많은 실데이터(DBLP, Freebase) + 표준 split + 다중 seed.
2. relation-aware edge filtration을 **exact EPD에 통합**(explicit edge filtration을 소비하는
   Union-Find)해 edge type이 실제로 event 타이밍을 바꾸게.
3. 스칼라 슬라이싱 대신 **진짜 multi-parameter persistence**(예: signed-barcode vectorization).
4. **cross-type(chromatic / H1) persistence**로 교차타입 mingling을 직접 포착.
5. typed topology가 untyped topology·no-topology baseline을 이기는 지점을 분리하는 통제 ablation.
