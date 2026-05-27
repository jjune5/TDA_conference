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
| Chameleon | Hetero wiki | 0.9432 ±0.007 | **0.9757 ±0.003** | 0.9686 ±0.006 |
| Squirrel | Hetero wiki | 0.9120 ±0.014 (n=20) | (TBD) | **0.9854** ±0.001 |
| Texas | Hetero web | 0.5709 ±0.110 | 0.5841 ±0.134 | **0.5939** ±0.131 |
| Cornell | Hetero web | 0.5850 ±0.112 | 0.6139 ±0.126 | **0.6502** ±0.141 |
| Wisconsin | Hetero web | 0.8640 ±0.061 | 0.8655 ±0.066 | 0.8653 ±0.061 |
| ChChMiner | Drug DDI | 0.9026 ±0.007 | 0.9625 ±0.005 | **0.9650** ±0.006 |

**핵심 발견** (자세한 분석은 [results doc](docs/specs/2026-06-21-tda-conference-results.md)):

1. **PDGNN neural PI ≥ exact PI** (전 데이터셋). homophilic에선 +0.3~1.5%p, 심지어 **Chameleon에선 no-PI까지 능가**(0.9757). neural approximation의 smoothing이 일반화↑.
2. **Heterophilic에서 exact PI는 무용~유해.** homophily와 PI hurt의 상관 **r=−0.567** — "topology가 도움?"은 도메인 의존적.
3. **Shuffle control**: PI 손해는 노이즈가 아니라 "틀린 방향" edge-specific 신호 (셔플하면 no-PI 회복).
4. **Molecular GC는 PI가 도움** (MUTAG +1.6%p, PROTEINS +1.2%p, NCI1 +1.3%p) — LP와 반대. task 구조에 의존.

→ **paper(ICML 2021)의 "topology helps LP" claim은 homophilic 가정에 의존**. Heterophilic/drug LP에선 무너지고, 분자 GC에선 성립.

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
