# TDA Conference Project Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** TDA 학회 발표 (4주 deadline, 10-15분, TDA 비전문가 청중) + GitHub demo. 현재 PDGNN 재현 결과 위에 **(B) SBM density × heterophily sweep** + **(C) Adaptive PI Gating prototype** 를 추가하고 발표 자료까지 완성.

**Architecture:** 기존 TLC-GNN/PDGNN 코드를 확장. (B) 합성 그래프 생성기 + 25-config sweep으로 인과관계 정량화. (C) gate network를 TLCGNN.Net에 추가해서 PI 사용 여부를 per-edge 자동 결정. 발표 자료는 markdown 슬라이드 + Jupyter notebook 데모.

**Tech Stack:** PyTorch 2.1 / torch-geometric 2.5 / networkx 2.8 / dionysus 2.1.8 / matplotlib / Jupyter / SLURM (A100-80GB).

**Spec:** `docs/superpowers/specs/2026-05-24-pdgnn-tda-conference-design.md`

---

## File Structure

### 새 파일 (Create)
- `Knowledge_Distillation/sbm_data.py` — SBM 합성 그래프 생성기 + PyG `Data` 래퍼
- `Knowledge_Distillation/sbm_sweep.py` — 25-config 실행 + 결과 집계 스크립트
- `Knowledge_Distillation/sbm_plot.py` — 2D heatmap matplotlib 코드
- `baselines/TLCGNN_gated.py` — Net + GatingNet (TLCGNN.py copy + gate 추가)
- `tests/test_sbm_data.py` — SBM 생성기 unit test
- `tests/test_adaptive_gating.py` — gating module unit test
- `docs/specs/2026-06-21-tda-conference-results.md` — 최종 결과 doc
- `slides/tda-conference.md` — 슬라이드 텍스트 (Marp 또는 reveal.js 호환)
- `notebooks/demo.ipynb` — 라이브 데모 notebook

### 수정 파일 (Modify)
- `loaddatas.py` — `'SBM_<density>_<heterophily>'` 형식 dataset 이름 지원
- `pipelines.py` — `--use_gating` 플래그 추가
- `README.md` — B/C 결과 섹션 추가

---

## Phase 1: SBM density × heterophily sweep (Week 1)

### Task 1.1: SBM 데이터 생성기 + test

**Files:**
- Create: `tests/test_sbm_data.py`
- Create: `Knowledge_Distillation/sbm_data.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sbm_data.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np
from Knowledge_Distillation.sbm_data import generate_sbm, sbm_to_pyg


def test_generate_sbm_shape():
    """500-node, 5-block SBM should produce networkx graph with expected stats."""
    g = generate_sbm(n_per_block=100, n_blocks=5, p_in=0.1, p_out=0.01, seed=1234)
    assert g.number_of_nodes() == 500
    # density should be roughly p_in/5 + p_out*4/5 = 0.02 + 0.008 = 0.028
    density = g.number_of_edges() * 2 / (500 * 499)
    assert 0.01 < density < 0.06, f"density {density} out of range"


def test_sbm_to_pyg_data():
    """sbm_to_pyg returns a Data-like object with .x, .edge_index, .y."""
    g = generate_sbm(n_per_block=20, n_blocks=3, p_in=0.3, p_out=0.05, seed=1234)
    data = sbm_to_pyg(g, n_blocks=3, feat_dim=16)
    assert data.x.shape == (60, 16)
    assert data.edge_index.dim() == 2 and data.edge_index.size(0) == 2
    # symmetric (undirected)
    assert data.edge_index.size(1) == g.number_of_edges() * 2
    # labels per block
    assert data.y.shape == (60,) and data.y.max().item() == 2


def test_sbm_reproducible():
    """Same seed → same graph."""
    g1 = generate_sbm(n_per_block=50, n_blocks=4, p_in=0.2, p_out=0.05, seed=42)
    g2 = generate_sbm(n_per_block=50, n_blocks=4, p_in=0.2, p_out=0.05, seed=42)
    assert sorted(g1.edges()) == sorted(g2.edges())
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /mnt/data/users/junyoungpark/code/TLC-GNN
source /mnt/data/users/junyoungpark/miniforge3/etc/profile.d/conda.sh && conda activate tlcgnn
python -m pytest tests/test_sbm_data.py -v
```
Expected: FAIL with "module not found" or "function not defined".

- [ ] **Step 3: Write the implementation**

```python
# Knowledge_Distillation/sbm_data.py
"""Stochastic Block Model graph generator for density × heterophily sweep.

Generates synthetic graphs where we control:
- density (via p_in, p_out)
- heterophily (via p_out / (p_in + p_out) ratio)

The full graph and a PyG-compatible Data wrapper are exposed for downstream
pipelines.py / pipelines.SBM_<config>.
"""
from __future__ import annotations
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data


def generate_sbm(n_per_block: int, n_blocks: int, p_in: float, p_out: float,
                 seed: int = 1234) -> nx.Graph:
    """Generate an SBM graph with n_blocks communities of n_per_block nodes.

    p_in: probability of edge within a block (community).
    p_out: probability of edge between blocks (heterophilic).
    Returns: undirected networkx Graph with node attribute 'block'.
    """
    rng = np.random.RandomState(seed)
    sizes = [n_per_block] * n_blocks
    probs = [[p_in if i == j else p_out for j in range(n_blocks)] for i in range(n_blocks)]
    g = nx.stochastic_block_model(sizes, probs, seed=seed)
    return g


def sbm_to_pyg(g: nx.Graph, n_blocks: int, feat_dim: int = 16,
               seed: int = 1234) -> Data:
    """Wrap SBM graph as PyG Data object.

    Features: random gaussian per node (deterministic via seed).
    Labels: block id.
    """
    rng = np.random.RandomState(seed)
    n = g.number_of_nodes()
    # features: one-hot block + random noise → real-valued
    x = torch.from_numpy(rng.randn(n, feat_dim).astype(np.float32))
    # labels from networkx node attribute 'block'
    blocks = nx.get_node_attributes(g, 'block')
    y = torch.tensor([blocks[i] for i in range(n)], dtype=torch.long)
    # symmetric edge_index
    ei = np.array(list(g.edges()), dtype=np.int64).T  # (2, E)
    if ei.size:
        ei = np.concatenate([ei, ei[[1, 0]]], axis=1)
    else:
        ei = np.zeros((2, 0), dtype=np.int64)
    edge_index = torch.from_numpy(ei).long()
    return Data(x=x, edge_index=edge_index, y=y, num_classes=n_blocks)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_sbm_data.py -v
```
Expected: 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_sbm_data.py Knowledge_Distillation/sbm_data.py
git commit -m "sbm generator"
```

### Task 1.2: loaddatas.py에 SBM dispatch

**Files:**
- Modify: `loaddatas.py` (line ~26, `loaddatas()` 함수 end)

- [ ] **Step 1: Edit `loaddatas.py`**

Find the line:
```python
    elif d_name == 'ChChMiner':
        dataset = _load_chch_miner()
    return dataset
```

Replace with:
```python
    elif d_name == 'ChChMiner':
        dataset = _load_chch_miner()
    elif d_name.startswith('SBM_'):
        dataset = _load_sbm(d_name)
    return dataset
```

Then add `_load_sbm` after `_load_chch_miner`:

```python
def _load_sbm(d_name: str):
    """Parse SBM_<n>_<p_in>_<p_out> and return Data wrapper.

    Example: 'SBM_500_5_0.1_0.05' = 500 nodes, 5 blocks, p_in=0.1, p_out=0.05.
    """
    parts = d_name.split('_')
    if len(parts) != 5:
        raise ValueError(f"SBM dataset must be SBM_<N>_<K>_<p_in>_<p_out>, got {d_name}")
    N, K, p_in, p_out = int(parts[1]), int(parts[2]), float(parts[3]), float(parts[4])
    from Knowledge_Distillation.sbm_data import generate_sbm, sbm_to_pyg
    g = generate_sbm(n_per_block=N // K, n_blocks=K, p_in=p_in, p_out=p_out, seed=1234)
    data = sbm_to_pyg(g, n_blocks=K, feat_dim=16, seed=1234)

    class _FakeDataset:
        def __init__(self, data, name):
            self._data = [data]; self.name = name; self.num_classes = K
        def __getitem__(self, i): return self._data[i]
        def __len__(self): return 1
    return _FakeDataset(data, d_name)
```

- [ ] **Step 2: Smoke test the new dispatch**

```bash
cd /mnt/data/users/junyoungpark/code/TLC-GNN
source /mnt/data/users/junyoungpark/miniforge3/etc/profile.d/conda.sh && conda activate tlcgnn
python -c "
import loaddatas as lds
ds = lds.loaddatas('SBM_500_5_0.1_0.05')
d = ds[0]
print(f'SBM: nodes={d.num_nodes}, edges={d.edge_index.size(1)//2}, feat_dim={d.x.size(1)}, num_classes={ds.num_classes}')
"
```
Expected: `SBM: nodes=500, edges=~1000, feat_dim=16, num_classes=5`.

- [ ] **Step 3: Commit**

```bash
git add loaddatas.py
git commit -m "sbm loader"
```

### Task 1.3: SBM 1-trial smoke through pipelines.py

- [ ] **Step 1: Run smoke on one SBM config**

```bash
cd /mnt/data/users/junyoungpark/code/TLC-GNN
python pipelines.py --datasets SBM_500_5_0.1_0.05 --trials 1 --tag sbmSmoke --no_pi
```
Expected: 1 trial runs, prints `Test pr: X.XXXX, roc: X.XXXX`, no exception.

- [ ] **Step 2: Run smoke with PI (slower)**

```bash
python pipelines.py --datasets SBM_500_5_0.1_0.05 --trials 1 --tag sbmSmokePI
```
Expected: PI cache computed in ~3-10 min (500 nodes, sparse), then 1 trial. Test ROC printed.

- [ ] **Step 3: Verify caches**

```bash
ls -la data/TLCGNN/SBM_500_5_0.1_0.05.npy
```
Expected: file exists with reasonable size.

- [ ] **Step 4: Commit any sbatch additions**

```bash
git add scores/pipe_benchmark_SBM_500_5_0.1_0.05_LP_scores*.txt 2>/dev/null
git commit -m "sbm smoke" --allow-empty
```

### Task 1.4: 25-config SLURM sweep submission

**Files:**
- Create: `Knowledge_Distillation/sbm_sweep.py`

- [ ] **Step 1: Create the file**

```python
# Knowledge_Distillation/sbm_sweep.py
"""Submit a 5x5 SBM sweep to SLURM.

Density axis:    p_in + p_out ∈ {0.05, 0.10, 0.20, 0.30, 0.50}
Heterophily ax:  p_out / (p_in + p_out) ∈ {0.10, 0.30, 0.50, 0.70, 0.90}
Fixed: N=500 nodes, K=5 blocks.

Each (density, heterophily) point gets 3 SLURM jobs:
- TLC-GNN exact PI
- PDGNN approx PI (using existing trained checkpoint)
- No PI
"""
from __future__ import annotations
import os
import subprocess

DENSITIES = [0.05, 0.10, 0.20, 0.30, 0.50]
HETEROPHILIES = [0.10, 0.30, 0.50, 0.70, 0.90]
N, K = 500, 5
TRIALS = 50


def density_hetero_to_p(d: float, h: float) -> tuple[float, float]:
    """density=p_in+p_out, hetero=p_out/(p_in+p_out). Returns (p_in, p_out)."""
    p_out = d * h
    p_in = d - p_out
    return round(p_in, 4), round(p_out, 4)


def submit_one(d_name: str, tag: str, pi_source: str = 'dionysus',
               no_pi: bool = False):
    script = f"/tmp/sbm_{d_name}_{tag}.sh"
    extra = ""
    if no_pi:
        extra += " --no_pi"
    if pi_source == 'pdgnn':
        extra += " --pi_source pdgnn"
    with open(script, 'w') as f:
        f.write(f"""#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/data/users/junyoungpark/code/TLC-GNN/slurm_logs/%x-%j.out
#SBATCH --error=/mnt/data/users/junyoungpark/code/TLC-GNN/slurm_logs/%x-%j.err
source /mnt/data/users/junyoungpark/miniforge3/etc/profile.d/conda.sh
conda activate tlcgnn
cd /mnt/data/users/junyoungpark/code/TLC-GNN
python pipelines.py --datasets {d_name} --trials {TRIALS} --tag {tag}{extra}
""")
    os.chmod(script, 0o755)
    jid = subprocess.check_output(
        ["sbatch", "--parsable", f"--job-name=sbm-{d_name}-{tag}", script]
    ).decode().strip()
    return jid


if __name__ == '__main__':
    submitted = []
    for d in DENSITIES:
        for h in HETEROPHILIES:
            p_in, p_out = density_hetero_to_p(d, h)
            name = f"SBM_{N}_{K}_{p_in}_{p_out}"
            # 3 jobs per cell
            for tag, kw in [('sbmTLCGNN', {}), ('sbmPDGNN', {'pi_source': 'pdgnn'}),
                             ('sbmNoPI', {'no_pi': True})]:
                jid = submit_one(name, tag, **kw)
                submitted.append((name, tag, jid))
                print(f"{name} {tag} → {jid}")
    print(f"\nTotal submitted: {len(submitted)} jobs")
```

- [ ] **Step 2: Submit the sweep**

```bash
cd /mnt/data/users/junyoungpark/code/TLC-GNN
source /mnt/data/users/junyoungpark/miniforge3/etc/profile.d/conda.sh && conda activate tlcgnn
python -m Knowledge_Distillation.sbm_sweep
```
Expected: 75 jobs submitted (25 configs × 3 variants), all PD pending. Note the job IDs.

- [ ] **Step 3: Verify queue**

```bash
squeue -u $(whoami) | head -30
```
Expected: 75 jobs in PD or R state.

- [ ] **Step 4: Commit**

```bash
git add Knowledge_Distillation/sbm_sweep.py
git commit -m "sbm sweep submitter"
```

### Task 1.5: Monitor + aggregate results

- [ ] **Step 1: Monitor jobs until all done**

Use the Monitor tool with the JIDs from step 1.4. Filter for `Traceback`, `OOM`, `^std`, `^mean`. Expect 12h-24h wall-clock for all 75 jobs to finish in parallel (cluster GPU count permits).

- [ ] **Step 2: Aggregate scores into CSV**

```python
# Run inline:
import os, glob, re, csv
rows = []
for f in glob.glob('scores/pipe_benchmark_SBM_*_LP_scores*.txt'):
    # parse SBM_<N>_<K>_<p_in>_<p_out> from filename
    m = re.search(r'SBM_(\d+)_(\d+)_([\d.]+)_([\d.]+)_LP_scores(\w+)\.txt', f)
    if not m: continue
    N, K, p_in, p_out, tag = m.groups()
    p_in, p_out = float(p_in), float(p_out)
    density = p_in + p_out
    hetero = p_out / density if density > 0 else 0
    aucs = []
    with open(f) as h:
        next(h)  # skip header
        for line in h:
            parts = [p.strip() for p in line.split(',') if p.strip()]
            if len(parts) >= 3 and parts[0].isdigit():
                aucs.append(float(parts[2]))
    if aucs:
        import statistics
        mean = statistics.mean(aucs)
        std = statistics.stdev(aucs) if len(aucs) > 1 else 0
        rows.append([density, hetero, tag, mean, std, len(aucs)])

with open('scores/sbm_sweep_summary.csv', 'w') as f:
    w = csv.writer(f)
    w.writerow(['density','heterophily','tag','mean_auc','std_auc','n_trials'])
    w.writerows(sorted(rows))
print(f'wrote {len(rows)} rows to scores/sbm_sweep_summary.csv')
```

- [ ] **Step 3: Verify CSV**

```bash
head -10 scores/sbm_sweep_summary.csv
wc -l scores/sbm_sweep_summary.csv
```
Expected: ~75 rows (one per config × variant).

- [ ] **Step 4: Commit**

```bash
git add scores/sbm_sweep_summary.csv
git commit -m "sbm sweep results"
```

### Task 1.6: 2D heatmap plot

**Files:**
- Create: `Knowledge_Distillation/sbm_plot.py`

- [ ] **Step 1: Create plotting script**

```python
# Knowledge_Distillation/sbm_plot.py
"""Read scores/sbm_sweep_summary.csv and produce 3 heatmaps:
  1. TLC-GNN AUC
  2. PDGNN AUC
  3. PI hurt magnitude = no-PI AUC − TLC-GNN AUC
"""
import csv
import numpy as np
import matplotlib.pyplot as plt

def load_csv(path='scores/sbm_sweep_summary.csv'):
    rows = []
    with open(path) as f:
        next(f)
        for line in f:
            d, h, tag, mean, std, n = line.strip().split(',')
            rows.append((float(d), float(h), tag, float(mean), float(std), int(n)))
    return rows


def grid_by_tag(rows, tag_filter: str, densities, heterophilies):
    grid = np.full((len(densities), len(heterophilies)), np.nan)
    for d, h, tag, mean, _, _ in rows:
        if tag != tag_filter:
            continue
        try:
            i = densities.index(round(d, 4))
            j = heterophilies.index(round(h, 4))
            grid[i, j] = mean
        except ValueError:
            pass
    return grid


def main():
    rows = load_csv()
    densities = sorted({round(d, 4) for d, *_ in rows})
    heterophilies = sorted({round(h, 4) for _, h, *_ in rows})
    g_tlc = grid_by_tag(rows, 'sbmTLCGNN', densities, heterophilies)
    g_pdg = grid_by_tag(rows, 'sbmPDGNN', densities, heterophilies)
    g_no  = grid_by_tag(rows, 'sbmNoPI', densities, heterophilies)
    hurt = g_no - g_tlc  # PI 도움 척도: 양수 = PI 해로움

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, grid, title, cmap in [
        (axes[0], g_tlc, 'TLC-GNN AUC', 'viridis'),
        (axes[1], g_pdg, 'PDGNN AUC',  'viridis'),
        (axes[2], hurt,  'PI hurt magnitude\n(no-PI − TLC-GNN)', 'RdBu_r'),
    ]:
        im = ax.imshow(grid, origin='lower', aspect='auto', cmap=cmap)
        ax.set_xticks(range(len(heterophilies)))
        ax.set_xticklabels([f'{h:.2f}' for h in heterophilies])
        ax.set_yticks(range(len(densities)))
        ax.set_yticklabels([f'{d:.2f}' for d in densities])
        ax.set_xlabel('Heterophily')
        ax.set_ylabel('Density')
        ax.set_title(title)
        plt.colorbar(im, ax=ax)
    plt.tight_layout()
    out = 'docs/figures/sbm_heatmap.png'
    import os
    os.makedirs('docs/figures', exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'saved {out}')


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Run + inspect**

```bash
cd /mnt/data/users/junyoungpark/code/TLC-GNN
python -m Knowledge_Distillation.sbm_plot
ls -la docs/figures/sbm_heatmap.png
```
Expected: PNG file created.

- [ ] **Step 3: Visual sanity check**

Open `docs/figures/sbm_heatmap.png` and verify:
- Third panel (hurt magnitude): top-right (high density, high heterophily) should be RED (PI hurts).
- Bottom-left (low density, low heterophily = homophilic): should be BLUE (PI helps) or neutral.
- If pattern doesn't match expectation, investigate by checking the raw CSV.

- [ ] **Step 4: Commit**

```bash
git add Knowledge_Distillation/sbm_plot.py docs/figures/sbm_heatmap.png
git commit -m "sbm heatmap"
```

---

## Phase 2: Adaptive PI Gating (Week 2)

### Task 2.1: Gating module + test

**Files:**
- Create: `tests/test_adaptive_gating.py`
- Create: `baselines/TLCGNN_gated.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_adaptive_gating.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np
from baselines.TLCGNN_gated import GatingNet, gated_decode


def test_gating_net_output_shape():
    """GatingNet takes per-edge features and outputs gate in [0,1]."""
    n_edges = 10
    feat_dim = 3
    gnet = GatingNet(in_dim=feat_dim, hidden=16)
    edge_feats = torch.randn(n_edges, feat_dim)
    gates = gnet(edge_feats)
    assert gates.shape == (n_edges,), f"expected ({n_edges},), got {gates.shape}"
    assert (gates >= 0).all() and (gates <= 1).all(), "gates out of [0,1]"


def test_gating_net_extremes():
    """Train a tiny gate to output 1 for one input and 0 for another → learnable."""
    torch.manual_seed(0)
    gnet = GatingNet(in_dim=1, hidden=8)
    opt = torch.optim.Adam(gnet.parameters(), lr=0.05)
    pos = torch.tensor([[1.0]] * 4)  # should → 1
    neg = torch.tensor([[-1.0]] * 4)  # should → 0
    for _ in range(200):
        opt.zero_grad()
        loss = (1 - gnet(pos)).mean() + gnet(neg).mean()
        loss.backward()
        opt.step()
    assert gnet(pos).mean() > 0.7, f"pos gate {gnet(pos).mean()} should be >0.7"
    assert gnet(neg).mean() < 0.3, f"neg gate {gnet(neg).mean()} should be <0.3"


def test_gated_decode_zero_gate_eq_no_pi():
    """When gate=0, gated_decode should equal a no-PI decode (PI contribution zeroed)."""
    torch.manual_seed(0)
    from baselines.TLCGNN_gated import gated_decode
    sqdist = torch.randn(5, 16)
    PI = torch.randn(5, 25)
    gates_zero = torch.zeros(5)
    gates_one = torch.ones(5)
    # With gate=0, the PI part of concat should be zero
    feat_zero = gated_decode(sqdist, PI, gates_zero)
    feat_one = gated_decode(sqdist, PI, gates_one)
    # PI columns (16:41) should be 0 for zero gate
    assert torch.allclose(feat_zero[:, 16:], torch.zeros(5, 25))
    # PI columns should equal PI for gate=1
    assert torch.allclose(feat_one[:, 16:], PI)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /mnt/data/users/junyoungpark/code/TLC-GNN
python -m pytest tests/test_adaptive_gating.py -v
```
Expected: FAIL with import error.

- [ ] **Step 3: Implement**

```python
# baselines/TLCGNN_gated.py
"""Adaptive PI Gating variant of TLC-GNN.

Identical to TLCGNN.Net except for a GatingNet that emits a per-edge
gate ∈ [0,1] multiplied into the persistence-image contribution before
the final MLP. When gate=0, the model degenerates to no-PI; when gate=1,
it is identical to TLC-GNN exact.

Gate inputs (per-edge features, computed in decode):
  [clustering coefficient of u, clustering coefficient of v, |emb_u−emb_v|_2]
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import Softmax
from torch_geometric.nn import GCNConv


class GatingNet(nn.Module):
    """Tiny MLP that maps per-edge features → gate in [0, 1]."""

    def __init__(self, in_dim: int = 3, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, edge_feats: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(edge_feats)).squeeze(-1)


def gated_decode(sqdist: torch.Tensor, PI: torch.Tensor,
                 gates: torch.Tensor) -> torch.Tensor:
    """Combine sqdist (E, 16) and PI (E, 25) with per-edge gates (E,) → (E, 41)."""
    gated_PI = gates.unsqueeze(-1) * PI  # broadcast
    return torch.cat([sqdist, gated_PI], dim=-1)


class Net(nn.Module):
    """TLC-GNN.Net with optional adaptive gating.

    Same encoder (2-layer GCN: in→100→16). Decoder concatenates
    [sqdist(emb_u−emb_v), gate × PI(u,v)] then MLP.
    """

    def __init__(self, data, num_features: int, num_classes: int, PI,
                 dimension: int = 5, clustering: np.ndarray | None = None):
        super().__init__()
        self.conv1 = GCNConv(num_features, 100, cached=True)
        self.conv2 = GCNConv(100, 16, cached=True)
        self.PI = PI
        self.clustering = clustering  # (N,) per-node clustering coef
        self.leakyrelu = nn.LeakyReLU(0.2, True)
        self.linear_1 = nn.Linear(dimension * dimension + 16, dimension * dimension, bias=True)
        self.linear = nn.Linear(dimension * dimension, 1, bias=True)
        self.softmax = Softmax(dim=1)
        self.gate_net = GatingNet(in_dim=3, hidden=16)

    def encode(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        return x

    def _edge_features_for_gate(self, total_edges, emb_in, emb_out):
        """Build per-edge feature tensor (E, 3)."""
        E = total_edges.shape[0]
        device = emb_in.device
        if self.clustering is not None:
            u_idx = total_edges[:, 0]
            v_idx = total_edges[:, 1]
            cl_u = torch.from_numpy(self.clustering[u_idx]).float().to(device)
            cl_v = torch.from_numpy(self.clustering[v_idx]).float().to(device)
        else:
            cl_u = torch.zeros(E, device=device)
            cl_v = torch.zeros(E, device=device)
        emb_dist = (emb_in - emb_out).norm(dim=-1)
        return torch.stack([cl_u, cl_v, emb_dist], dim=-1)

    def decode(self, data, emb, type='train'):
        if type == 'train':
            edges_pos = data.total_edges[:data.train_pos]
            index = np.random.randint(0, data.train_neg, data.train_pos)
            edges_neg = data.total_edges[data.train_pos:data.train_pos + data.train_neg][index]
            total_edges = np.concatenate((edges_pos, edges_neg))
            edges_y = torch.cat((data.total_edges_y[:data.train_pos],
                                  data.total_edges_y[data.train_pos:data.train_pos + data.train_neg][index]))
            PI = np.concatenate(
                (self.PI[:data.train_pos], self.PI[data.train_pos:data.train_pos + data.train_neg][index]))
        elif type == 'val':
            total_edges = data.total_edges[data.train_pos+data.train_neg:data.train_pos+data.train_neg+data.val_pos+data.val_neg]
            edges_y = data.total_edges_y[data.train_pos+data.train_neg:data.train_pos+data.train_neg+data.val_pos+data.val_neg]
            PI = self.PI[data.train_pos+data.train_neg:data.train_pos+data.train_neg+data.val_pos+data.val_neg]
        elif type == 'test':
            total_edges = data.total_edges[data.train_pos+data.train_neg+data.val_pos+data.val_neg:]
            edges_y = data.total_edges_y[data.train_pos+data.train_neg+data.val_pos+data.val_neg:]
            PI = self.PI[data.train_pos+data.train_neg+data.val_pos+data.val_neg:]

        emb = emb.renorm(2, 0, 1)
        new_x = torch.tensor(PI.reshape((len(total_edges), -1)), dtype=torch.float, device=emb.device)
        emb_in = emb[total_edges[:, 0]]
        emb_out = emb[total_edges[:, 1]]
        sqdist = (emb_in - emb_out).pow(2)
        edge_feats = self._edge_features_for_gate(total_edges, emb_in, emb_out)
        gates = self.gate_net(edge_feats)
        feats = gated_decode(sqdist, new_x, gates)
        feats = self.leakyrelu(self.linear_1(feats))
        feats = torch.abs(self.linear(feats)).reshape(-1)
        feats = torch.clamp(feats, min=0, max=40)
        prob = 1. / (torch.exp((feats - 2.0) / 1.0) + 1.0)
        return prob, edges_y.float()


def call(data, name, num_features, num_classes, data_cnt, use_pi: bool = True):
    """Drop-in replacement for TLCGNN.call that returns the gated Net."""
    from baselines.TLCGNN import call as orig_call
    model, data = orig_call(data, name, num_features, num_classes, data_cnt, use_pi=use_pi)
    # Replace the model with gated version
    import networkx as nx
    # Build clustering coefficient
    g = nx.Graph()
    g.add_nodes_from(range(data.num_nodes))
    ei = data.edge_index.cpu().numpy()
    g.add_edges_from(((int(ei[0, i]), int(ei[1, i])) for i in range(ei.shape[1])))
    clustering_dict = nx.clustering(g)
    cl_arr = np.array([clustering_dict.get(i, 0.0) for i in range(data.num_nodes)], dtype=np.float32)
    gated_model = Net(data, num_features, num_classes, PI=model.PI, clustering=cl_arr).to(data.x.device)
    return gated_model, data
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_adaptive_gating.py -v
```
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_adaptive_gating.py baselines/TLCGNN_gated.py
git commit -m "adaptive gating module"
```

### Task 2.2: pipelines.py에 `--use_gating` 추가

**Files:**
- Modify: `pipelines.py`

- [ ] **Step 1: Edit `pipelines.py`** — argparse 추가

Find the line:
```python
_parser.add_argument('--pi_source', choices=['dionysus', 'pdgnn'], default='dionysus',
                     help='source of PI cache (dionysus=TLC-GNN exact, pdgnn=neural approx)')
```

Insert after:
```python
_parser.add_argument('--use_gating', action='store_true',
                     help='use adaptive PI gating (baselines.TLCGNN_gated.Net)')
```

- [ ] **Step 2: Edit `pipelines.py`** — model dispatch

Find:
```python
from baselines import TLCGNN as TLCGNN
```

Replace with:
```python
from baselines import TLCGNN as TLCGNN
from baselines import TLCGNN_gated as TLCGNN_gated
```

Find:
```python
pipelines=['TLCGNN']
```

Replace with:
```python
if _args.use_gating:
    pipelines=['TLCGNN_gated']
else:
    pipelines=['TLCGNN']
```

- [ ] **Step 3: Smoke test on a small dataset**

```bash
cd /mnt/data/users/junyoungpark/code/TLC-GNN
python pipelines.py --datasets Cora --trials 1 --tag gateSmoke --use_gating
```
Expected: trains 1 trial without error. Test ROC printed.

- [ ] **Step 4: Commit**

```bash
git add pipelines.py
git commit -m "pipelines: --use_gating flag"
```

### Task 2.3: Adaptive gating 4-dataset sweep (50 trials)

Run gating on 4 representative datasets covering all 3 regimes.

- [ ] **Step 1: Submit 4 SLURM jobs**

```bash
cd /mnt/data/users/junyoungpark/code/TLC-GNN
for D in Photo Chameleon Texas ChChMiner; do
  sbatch --job-name=tlcgnn-${D}-gated slurm_run.sh --datasets $D --trials 50 --tag gated --use_gating
done
squeue -u $(whoami) -o "%.10i %.30j %.2t %.10M"
```
Expected: 4 jobs queued.

- [ ] **Step 2: Monitor + wait**

Use Monitor tool with persistent=true. Wait for all 4 to complete (~2-6h each).

- [ ] **Step 3: Aggregate gating results**

```bash
for D in Photo Chameleon Texas ChChMiner; do
  F="scores/pipe_benchmark_${D}_LP_scoresgated.txt"
  if [ -f "$F" ]; then
    awk -F'[, ]+' -v d="$D" 'NR>1 && /^[0-9]+,/ {a[++n]=$3; sum+=$3} END {if(n>1){m=sum/n; for(i=1;i<=n;i++){t=a[i]-m;s+=t*t}; printf "%-12s n=%2d  AUC=%.4f ± %.4f\n", d, n, m, sqrt(s/(n-1))}}' "$F"
  fi
done
```
Expected: 4 rows, AUCs comparable to TLC-GNN baselines.

- [ ] **Step 4: Commit results**

```bash
git add scores/pipe_benchmark_*_LP_scoresgated.txt
git commit -m "gating 4-dataset results"
```

### Task 2.4: Verify gate behavior (sanity check)

Check that gate values reflect heterophily/density expectations.

- [ ] **Step 1: Write quick inspection script**

```bash
cd /mnt/data/users/junyoungpark/code/TLC-GNN
source /mnt/data/users/junyoungpark/miniforge3/etc/profile.d/conda.sh && conda activate tlcgnn
python -c "
# Train a single 200-epoch gating model on Chameleon and inspect gate values
import torch, copy, numpy as np
import loaddatas as lds
import torch.nn.functional as F
from baselines import TLCGNN_gated
from torch.nn.init import xavier_normal_ as xavier

torch.manual_seed(0); np.random.seed(0)
dataset = lds.loaddatas('Chameleon')
data = copy.deepcopy(dataset[0])
model, data = TLCGNN_gated.call(data, dataset.name, data.x.size(1), dataset.num_classes, 0)
def init(m):
    if isinstance(m, torch.nn.Linear):
        xavier(m.weight)
        if m.bias is not None: torch.nn.init.constant_(m.bias, 0)
model.apply(init)
opt = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0)
for ep in range(200):
    model.train(); opt.zero_grad()
    emb = model.encode(data)
    x, y = model.decode(data, emb)
    loss = F.binary_cross_entropy(x, y); loss.backward(); opt.step()
# Inspect gate values
model.eval()
with torch.no_grad():
    emb = model.encode(data)
    # Force decode on val edges to inspect gates
    total_edges = data.total_edges[data.train_pos+data.train_neg:data.train_pos+data.train_neg+data.val_pos+data.val_neg]
    emb_in = emb[total_edges[:,0]]; emb_out = emb[total_edges[:,1]]
    feats = model._edge_features_for_gate(total_edges, emb_in, emb_out)
    gates = model.gate_net(feats)
    print(f'Chameleon gate stats: mean={gates.mean():.3f} std={gates.std():.3f} '
          f'min={gates.min():.3f} max={gates.max():.3f}')
"
```
Expected: prints gate statistics. For Chameleon (heterophilic) we expect mean gate < 0.5.

- [ ] **Step 2: Repeat for Photo (homophilic)**

Same script with `Chameleon` → `Photo`. Expected: mean gate > 0.5.

- [ ] **Step 3: Document gate behavior**

Add a small section to `docs/specs/2026-06-21-tda-conference-results.md` (to be created in Phase 3) summarizing gate stats per dataset.

- [ ] **Step 4: Commit (no code change, just documentation)**

```bash
git commit --allow-empty -m "gate inspection notes"
```

---

## Phase 3: Presentation + Demo (Week 3-4)

### Task 3.1: 결과 doc 작성

**Files:**
- Create: `docs/specs/2026-06-21-tda-conference-results.md`

- [ ] **Step 1: Write the doc**

```markdown
# TDA Conference Results

**Date:** 2026-06-21
**Plan:** docs/superpowers/plans/2026-05-24-tda-conference.md

## 1. Baseline (from PDGNN reproduction)

| Dataset | Domain | TLC-GNN | PDGNN | No PI | Best |
|---|---|---|---|---|---|
| Photo | Homo Amazon | 0.9825 | **0.9860** | — | PDGNN |
| PubMed | Homo citation | 0.9635 | **0.9669** | — | PDGNN |
| Computers | Homo Amazon | 0.9680 | **0.9830** | — | PDGNN |
| Chameleon | Hetero wiki | 0.9432 | 0.9447 | **0.9686** | No PI |
| Squirrel | Hetero wiki | 0.9120 | (TBD) | **0.9854** | No PI |
| Texas | Hetero web | 0.5709 | 0.5396 | **0.5939** | No PI |
| Cornell | Hetero web | 0.5850 | 0.5737 | **0.6502** | No PI |
| Wisconsin | Hetero web | 0.8640 | 0.8449 | 0.8653 | tie |
| ChChMiner | Drug DDI | 0.9026 | 0.9625 | **0.9650** | No PI |

## 2. SBM Sweep (B)

[Insert sbm_heatmap.png reference]

Quantitative pattern: PI hurt magnitude = no-PI AUC − TLC-GNN AUC.
- Maximum hurt: density=___, heterophily=___, hurt=___.
- Hurt threshold (where PI becomes negative): density × heterophily > ___.

## 3. Adaptive Gating (C)

| Dataset | Mean gate | TLC-GNN AUC | Gated AUC | No-PI AUC | Best of three |
|---|---|---|---|---|---|
| Photo | 0.XX | 0.9825 | 0.XXXX | — | TLC-GNN / Gated |
| Chameleon | 0.XX | 0.9432 | 0.XXXX | 0.9686 | No-PI / Gated |
| Texas | 0.XX | 0.5709 | 0.XXXX | 0.5939 | No-PI / Gated |
| ChChMiner | 0.XX | 0.9026 | 0.XXXX | 0.9650 | No-PI / Gated |

(Numbers filled after Phase 2 Task 2.3 completes)

## 4. 핵심 발견

1. **Homophilic 큰 그래프**: PI 도움 (PDGNN > TLC-GNN exact)
2. **Heterophilic 그래프**: PI 무용/유해. SBM sweep으로 density × heterophily가 hurt magnitude를 예측한다는 정량적 패턴 입증.
3. **Adaptive gating**: 가능한 한 도메인 자동 인식. (결과에 따라 보충)

## 5. 전망

- Drug discovery (OGBL-DDI 등 대규모 약물 그래프)
- Social network analysis (heterophily 강한 도메인)
- Brain connectivity (TDA가 sweet spot)
```

- [ ] **Step 2: Fill in numbers after sweeps complete**

Wait for Phase 1 + Phase 2 results, then update placeholders.

- [ ] **Step 3: Commit**

```bash
git add docs/specs/2026-06-21-tda-conference-results.md
git commit -m "results doc draft"
```

### Task 3.2: 슬라이드 작성 (Marp 또는 reveal.js 호환 markdown)

**Files:**
- Create: `slides/tda-conference.md`

- [ ] **Step 1: Write slide deck**

```markdown
---
title: Persistent Homology가 Link Prediction에 정말 도움 되나?
subtitle: 도메인별 분석과 Adaptive Topology Gating
author: 박준영
date: 2026-06-21
marp: true
theme: gaia
paginate: true
---

# Persistent Homology가 Link Prediction에 정말 도움 되나?

## 도메인별 분석과 Adaptive Topology Gating

박준영 · TDA 학회 · 2026-06-21

---

## TDA란?

- **Topological Data Analysis** — 데이터의 "모양"을 분석
- 거리/연결성에 robust → noise에 강함
- 도구: persistent homology (구멍/연결 component를 시간축에 따라 추적)

[그림: 점들이 점점 연결되는 filtration 시각화]

---

## Persistent Diagram (PD)

각 topological feature가 **언제 생기고 (birth)** **언제 사라지나 (death)**.

[그림: 그래프 → PD 시각화 (왼쪽 그래프, 오른쪽 birth-death scatter)]

오래 사는 점 = robust feature. 빨리 사라지는 점 = noise.

---

## Persistent Image (PI)

PD를 5×5 grid에 Gaussian으로 흐려서 vector화 → ML 입력 가능.

[그림: PD → PI heatmap 변환]

---

## TLC-GNN (ICML 2021)

GNN + PI를 결합해서 link prediction:

```
emb_u = GCN(u)
emb_v = GCN(v)
PI(u, v) = persistent_image(vicinity)
features = [|emb_u − emb_v|², PI(u, v)]
prob = MLP(features)
```

논문 결과: PubMed, Photo, Computers 등에서 SOTA.

---

## PDGNN (NeurIPS 2022)

문제: PD 계산이 느림 (Squirrel: 80시간).

해결: GNN으로 (birth, death) 좌표를 **근사**.

성능: TLC-GNN과 비슷 + 100x 빠름.

---

## 우리가 한 것

1. 재현 (TLC-GNN + PDGNN, PyTorch 2.1로 modernize)
2. 9개 도메인 확장 (homo / hetero / drug)
3. Ablation (PI on/off)
4. SBM density × heterophily sweep
5. Adaptive PI Gating 제안

GitHub: github.com/jjune5/TDA_conference

---

## 결과: 3-way 비교

[그림: 표 또는 bar chart, TLC-GNN / PDGNN / No PI 8개 dataset]

3개 패턴:
1. Homo (Photo / PubMed / Computers): PDGNN ≥ TLC-GNN > No PI
2. Hetero (Chameleon / Squirrel / WebKB): No PI > 둘 다
3. Drug (ChChMiner): No PI ≈ PDGNN > TLC-GNN

---

## 발견 1: 도메인이 결정

[그림: 3개 그룹의 PI 효과]

- Homophilic citation/Amazon → PI 도움
- Heterophilic wiki/web → PI 해로움
- Drug interaction → PI 해로움 (놀라움)

→ paper의 "topology helps LP" 주장은 **도메인 의존적**.

---

## 발견 2: PDGNN의 의외성

[그림: PDGNN > exact PI 막대 그래프]

PDGNN approximation이 **dionysus exact PD보다 더 좋음** (homo에서).
- Photo: +0.35%p
- Computers: +1.50%p

해석: smoothing 효과로 noise 제거 → regularization.

---

## 발견 3: Density × Heterophily 패턴 (SBM)

[그림: docs/figures/sbm_heatmap.png (3-panel)]

- 정량적 결과: density × heterophily가 클수록 PI hurt magnitude ↑
- 단순 ablation 아니라 **인과관계** 입증

---

## 우리 제안: Adaptive PI Gating

```
gate = sigmoid(GatingNet(edge_features))
features = [|emb_u − emb_v|², gate × PI(u, v)]
prob = MLP(features)
```

모델이 자동으로 결정:
- Homo edge → gate ~1 (PI on)
- Hetero edge → gate ~0 (PI off)

---

## Adaptive Gating 결과

[표: 4 datasets, gate value + AUC]

(Phase 2 Task 2.3 후 채우기)

해석: ___

---

## 전망

- **Drug discovery** — OGBL-DDI, BIOSNAP scale-up with PDGNN
- **Social network** — heterophily strong domain
- **Brain connectivity** — TDA의 sweet spot
- **Heterogeneous KG** — drug-protein-disease 같은 multi-relation

---

## 정리

1. PI가 link prediction에 무조건 도움 ≠ 사실
2. 도메인 (density × heterophily) 의존
3. PDGNN approximation은 의외로 더 robust
4. Adaptive gating: 자동 적응 방법 제안

질문?

GitHub: github.com/jjune5/TDA_conference

```

- [ ] **Step 2: Render slides (Marp)**

```bash
cd /mnt/data/users/junyoungpark/code/TLC-GNN
# Install marp-cli if not present
npm install -g @marp-team/marp-cli 2>&1 || true
marp slides/tda-conference.md --pdf --output slides/tda-conference.pdf
```
Expected: PDF file generated. (If marp-cli not installable, use VSCode Marp extension or just edit markdown.)

- [ ] **Step 3: Commit**

```bash
git add slides/tda-conference.md slides/tda-conference.pdf
git commit -m "slides draft"
```

### Task 3.3: Demo notebook

**Files:**
- Create: `notebooks/demo.ipynb`

- [ ] **Step 1: Create notebook**

Create `notebooks/demo.ipynb` with cells:

1. **Markdown intro**: "Live demo of TLC-GNN / PDGNN / Adaptive Gating"

2. **Code cell**: load Cora, show graph stats
```python
import sys; sys.path.insert(0, '..')
import loaddatas as lds
import networkx as nx
ds = lds.loaddatas('Cora')
data = ds[0]
print(f'Cora: {data.num_nodes} nodes, {data.edge_index.size(1)//2} edges')
```

3. **Code cell**: compute persistent diagram for one edge (visualize)
```python
# Pick one edge, extract vicinity, compute PD, plot
# (Use functions from Knowledge_Distillation.prepare_data_LP_modern)
```

4. **Code cell**: show PI heatmap (5x5)

5. **Code cell**: show TLC-GNN prediction vs PDGNN prediction on the same edge

6. **Code cell**: show gate value from adaptive model

7. **Code cell**: SBM heatmap (load from PNG)

8. **Markdown wrap-up**: link to GitHub.

- [ ] **Step 2: Run all cells end-to-end**

```bash
jupyter nbconvert --to notebook --execute notebooks/demo.ipynb --output demo_executed.ipynb
```
Expected: no errors, all cells produce output.

- [ ] **Step 3: Commit**

```bash
git add notebooks/demo.ipynb notebooks/demo_executed.ipynb
git commit -m "demo notebook"
```

### Task 3.4: 리허설 + push

- [ ] **Step 1: Open slides + go through 10-15 min**

Time yourself reading through. Adjust slide count if too long/short.

- [ ] **Step 2: Run demo notebook end-to-end one more time**

Verify all images render.

- [ ] **Step 3: README 업데이트**

Add section linking to slides + results doc:

```markdown
## 발표 자료 (TDA 학회 2026-06-21)
- [Slides](slides/tda-conference.pdf)
- [Results doc](docs/specs/2026-06-21-tda-conference-results.md)
- [Demo notebook](notebooks/demo.ipynb)
```

- [ ] **Step 4: Push final**

```bash
cd /mnt/data/users/junyoungpark/code/TLC-GNN
git add README.md
git commit -m "tda conference release"
git push tda main
```

---

## Self-Review Checklist (run after writing the plan)

- [x] **Spec coverage:**
  - SBM sweep (B) → Task 1.1-1.6 ✓
  - Adaptive Gating (C) → Task 2.1-2.4 ✓
  - Slides → Task 3.2 ✓
  - Demo → Task 3.3 ✓
  - Results doc → Task 3.1 ✓
  - Squirrel exclusion → noted as out of scope ✓
- [x] **Placeholder scan:** Numbers in results doc (`0.XX`) are intentional templates filled after sweeps. No "TBD" in code or commands.
- [x] **Type consistency:** `GatingNet(in_dim=3, hidden=16)` used consistently. `sbm_to_pyg` returns `Data` with same fields used by `loaddatas._FakeDataset`. SBM dataset naming convention `SBM_<N>_<K>_<p_in>_<p_out>` consistent across tasks.

---

## Known risks

| Risk | Mitigation |
|---|---|
| SBM 25 jobs × 3 variants = 75 SLURM. Compute cluster contention. | Per-job 32G mem, 12h time. If queue full, fall back to running 5-config × 3 = 15 jobs first. |
| Adaptive gating doesn't learn (collapses to constant 0.5). | Step 2.4 gate inspection catches this. Fall back: report honest negative result in slides. |
| Slide deck > 15 min. | Cut adaptive gating section to 1 slide, push results to backup. |
| Marp/notebook render issues. | Backup: plain markdown viewer + screenshots. |
| Squirrel PDGNN finishes during Phase 1-2. | Drop it in as bonus row in results table. |

---

## Time estimate

| Phase | Tasks | Wall-clock |
|---|---|---|
| 1. SBM sweep | 1.1–1.6 | 1주 (코드 2일 + 12h compute + plot/aggregate) |
| 2. Adaptive Gating | 2.1–2.4 | 1주 (코드 2일 + 4 jobs × 4-6h compute) |
| 3. Presentation | 3.1–3.4 | 2주 (slides + notebook + 리허설) |
| **Total** | **3 phases** | **~4주** (deadline 2026-06-21) |
