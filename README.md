# hetero_pdg — heterogeneous graph 링크예측 + 위상(EPD) feature 실험

heterogeneous graph **링크예측**에서, PDGNN/TLC-GNN의 위상(extended persistence, EPD)
feature를 **node/edge 타입을 반영**하도록 확장해보는 실험 프로젝트입니다. 원본 TLC-GNN/PDGNN
코드는 건드리지 않고 별도 패키지(`hetero_pdg/`)로 구성했습니다.

> ⚠️ **성능 향상을 주장하지 않습니다.** toy 결과는 파이프라인 정합성 점검용이고, 실데이터 결과는
> 나온 그대로 보고합니다. `fallback` descriptor는 PDGNN/EPD가 **아닙니다**.

## 이게 뭘 하는 프로젝트?

- hetero graph(예: author–paper–field)를 meta-path로 homogeneous하게 투영하거나, 타입별
  filtration을 학습해서 **위상 feature를 링크예측에 쓸 수 있는지** 실험합니다.
- 비교 축: **위상 안 씀(baseline) vs 위상 씀(여러 방식)**.
- 가설(claim): *기존 PDGNN은 hetero에서 타입을 무시해 의미가 다른 위상 event가 한 점으로 뭉친다 →
  타입을 반영한 filtration으로 semantic topology를 보존하면 더 낫지 않을까?*

## 핵심 결과 (정직)

- 실데이터 **ACM·IMDB**에서 **어떤 위상 방식(fallback / exact EPD / neural EPD)도 no_topology
  baseline을 유의하게 못 이김** (test-AUC 전부 ~0.69–0.72).
- 즉 **현재로선 claim 미입증** — 위상 feature가 hetero 링크예측을 돕는다는 증거가 없습니다.
- 자세한 보고서: [`docs/REPORT.md`](docs/REPORT.md)

| config | ACM | IMDB |
|---|---|---|
| **no_topology (baseline)** | **0.701** | **0.722** |
| metapath_attention (fallback) | 0.723 | 0.720 |
| metapath_concat (real_tlc, exact EPD) | 0.690 | 0.712 |
| metapath_concat (real_pdgnn, neural EPD) | 0.713 | 0.704 |

## 폴더 구조

- `hetero_pdg/` — 패키지
    - `data.py` toy 그래프 + 타입제약 negative sampling
    - `metapath.py` meta-path 투영(APA/PFP/PCP)
    - `filtration.py` 타입별 MLP filtration + quantile calibration
    - `topology_features.py` fallback descriptor / `topology_adapter.py` real backend(real_tlc/real_pdgnn)
    - `real_data.py` 실데이터(ACM, IMDB) 로더
    - `models.py` 링크예측기(5 mode) / `train_hetero_lp.py` 학습·평가
    - (Phase-3 실험) `pair_filtration.py` · `edge_filtration.py` · `typed_cycle.py` · `multi_slice.py`
- `tests/` — 테스트 (standalone 113개 통과)
- `scripts/` — SLURM 실행 스크립트 · `runs/` — 실험 결과(JSON) · `docs/` — 보고서

## 실행

```bash
conda activate tlcgnn
# 테스트 (외부 의존성 없이)
python -m pytest tests/ -q
# toy 학습 (config.json + metrics.json 생성)
python -m hetero_pdg.train_hetero_lp --dataset toy --topo-mode metapath_topology_concat \
    --epochs 30 --output-dir runs/toy
# 실데이터 (TLC-GNN 엔진 필요 → PYTHONPATH 지정)
export PYTHONPATH=/path/to/TLC-GNN:$PYTHONPATH
python -m hetero_pdg.train_hetero_lp --dataset acm --topo-mode metapath_topology_concat \
    --topo-backend real_tlc --max-target-edges 1000 --epochs 100 --output-dir runs/acm
```

- mode: `no_topology` · `collapsed_topology` · `metapath_topology_concat` ·
  `metapath_topology_attention` · `unified_filter_topology` · (실험) `typed_cycle_topology` · `multi_slice_topology`
- backend: `fallback`(기본) · `real_tlc`(exact EPD) · `real_pdgnn`(neural EPD)

## 정직성 / 한계

- 소형 그래프만 평가, 성능주장 없음. `fallback`은 PDGNN/EPD가 아닌 결정론적 graph 통계.
- Phase-3 고급 3종(pair-conditioned / multi-slice / typed-cycle)은 toy 스키마 전용 →
  실데이터의 featureless 타입에서는 아직 동작하지 않음.
- 위상 feature가 hetero 링크예측을 돕는다는 증거는 (이 실험 범위에서) 없음.

## 문서

- [`docs/REPORT.md`](docs/REPORT.md) — 한글 종합 보고서 (claim · 실험 · 결과 · 확장)
- `docs/hetero_pdg_phase2_report.md`, `docs/hetero_pdg_phase3_advanced.md` — 단계별 상세
