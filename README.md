# TLC-GNN / PDGNN — modernized

원본 [pkuyzy/TLC-GNN](https://github.com/pkuyzy/TLC-GNN) (ICML 2021 + NeurIPS 2022) 을 **PyTorch 2.1 / PyG 2.5 / Python 3.9 / CUDA 11.8** 에서 동작하도록 수정.

## 핵심 수정

- `baselines/TLCGNN.py:48` — `emb.renorm_()` → `emb.renorm()` (PyTorch 2.x autograd inplace 거부 문제)
- `loaddatas.py` — stale PI 캐시 layout splice, networkx/scipy 새 API 대응
- `sg2dgm/`, `Knowledge_Distillation/pdgnn_modern.py` — PyG 2.x 호환

## 결과 (50 trials)

| | Our | Paper |
|---|---|---|
| Photo | 0.9825 | 0.9823 ✓ |
| PubMed | 0.9635 | 0.9703 |
| Computers | 0.9680 | 0.9790 |

## 실행

```bash
conda activate tlcgnn
python pipelines.py --datasets PubMed Photo Computers --trials 50 --tag rerun
```

자세한 내용: [`docs/specs/`](docs/specs/)
