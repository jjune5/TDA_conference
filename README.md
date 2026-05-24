# TLC-GNN / PDGNN — modernized

원본 [pkuyzy/TLC-GNN](https://github.com/pkuyzy/TLC-GNN) (ICML 2021 + NeurIPS 2022) 을 **PyTorch 2.1 / PyG 2.5 / Python 3.9 / CUDA 11.8** 에서 동작하도록 수정.

## 핵심 수정

- `baselines/TLCGNN.py:48` — `emb.renorm_()` → `emb.renorm()` (PyTorch 2.x autograd inplace 거부 문제)
- `loaddatas.py` — stale PI 캐시 layout splice, networkx/scipy 새 API 대응
- `sg2dgm/`, `Knowledge_Distillation/pdgnn_modern.py` — PyG 2.x 호환

## 결과 (50 trials, AUC ± std) — 3-way ablation across 9 datasets

| Dataset | 도메인 | TLC-GNN (exact PI) | PDGNN (neural PI) | No PI |
|---|---|---|---|---|
| Photo | Homo Amazon | 0.9825 ±0.001 | **0.9860** ±0.001 | — |
| PubMed | Homo citation | 0.9635 ±0.003 | **0.9669** ±0.002 | — |
| Computers | Homo Amazon | 0.9680 ±0.002 | **0.9830** ±0.001 | — |
| Chameleon | Hetero wiki | 0.9432 ±0.007 | 0.9447 ±0.006 | **0.9686** ±0.006 |
| Squirrel | Hetero wiki | 0.9120 ±0.015 (n=20) | (TBD) | **0.9854** ±0.001 |
| Texas | Hetero web | 0.5709 ±0.111 | 0.5396 ±0.128 | **0.5939** ±0.133 |
| Cornell | Hetero web | 0.5850 ±0.113 | 0.5737 ±0.115 | **0.6502** ±0.143 |
| Wisconsin | Hetero web | 0.8640 ±0.062 | 0.8449 ±0.076 | 0.8653 ±0.061 |
| ChChMiner | Drug DDI | 0.9026 ±0.007 | 0.9625 ±0.005 | **0.9650** ±0.006 |

**3개 핵심 발견** (자세한 분석은 [results doc](docs/specs/2026-05-22-pdgnn-reproduction-results.md)):

1. **Homophilic 큰 그래프**에서 PDGNN의 neural approximation이 dionysus exact PI를 능가 (+0.3 ~ +1.5%p). Smoothing 효과로 일반화↑.
2. **모든 heterophilic 그래프**에서 PI (exact or approx)가 모델을 해침. Squirrel에서 −7.3%p로 가장 크게.
3. **Drug (ChChMiner)**: exact PI가 −6%p로 해롭지만 PDGNN 근사는 no-PI와 거의 동등 — 노이즈를 평탄화함.

→ **paper(ICML 2021)의 "topology helps LP" claim은 homophilic 가정에 의존**. Heterophilic/drug에선 무너짐.

## 확장 실험 — TDA 학회 발표용

### (B) SBM density × heterophily sweep — 25 configs, 합성 그래프

`docs/figures/sbm_heatmap.png` 3-panel heatmap. Density(p_in+p_out) × Heterophily(p_out/density)에서 PI hurt magnitude 측정. 63/75 configs 완료.

→ **SBM 환경에선** PI hurt가 homophilic + mid density 영역에서 큼 (실제 데이터와 다름) — feature signal과 topology의 상호작용 시사.

### (C) Adaptive PI Gating — Honest Negative Result

`baselines/TLCGNN_gated.py`. 4 datasets (Photo/Chameleon/Texas/ChChMiner) 50 trials:

| Dataset | TLC-GNN | **Gated** | No-PI | Mean gate |
|---|---|---|---|---|
| Photo | 0.9825 | **0.9827** | — | **1.000** |
| Chameleon | 0.9432 | **0.9490** | 0.9686 | **1.000** |
| Texas | 0.5709 | **0.5467** | 0.5939 | **1.000** |
| ChChMiner | 0.9026 | **0.9033** | 0.9650 | **1.000** |

**Gate가 saturate해서 사실상 TLC-GNN exact와 동일.** 단순 sigmoid + 3-D edge features로는 heterophily 자동 인식 안 됨. Future work: sparsity regularizer, graph-level features, discrete gating.

자세한 분석: [`docs/specs/2026-06-21-tda-conference-results.md`](docs/specs/2026-06-21-tda-conference-results.md), [발표 슬라이드](slides/tda-conference.md), [demo notebook](notebooks/demo.ipynb).

## 실행

```bash
conda env create -f environment.yml
conda activate tlcgnn
# TLC-GNN exact
python pipelines.py --datasets PubMed Photo Computers --trials 50 --tag rerun
# Ablation (no PI)
python pipelines.py --datasets Chameleon --trials 50 --tag heteroNoPI --no_pi
# PDGNN approx PI (requires trained checkpoint at data/PDGNN/checkpoints/pdgnn_lp.pt)
python pipelines.py --datasets PubMed --trials 50 --tag pdgnn --pi_source pdgnn
```

PDGNN end-to-end 재현 (data prep → train → inference → eval):
```bash
python -m Knowledge_Distillation.prepare_data_LP_modern --name PubMed --max_edges 10000
python -m Knowledge_Distillation.train_pdgnn_lp --data data/PDGNN/PubMed_LP_hop2_n10000_train.pkl
for D in Photo PubMed Computers Texas Cornell Wisconsin Chameleon ChChMiner; do
  python -m Knowledge_Distillation.pdgnn_inference --name $D
done
```

자세한 내용: [`docs/specs/`](docs/specs/) (재현 plan + 결과 doc)
