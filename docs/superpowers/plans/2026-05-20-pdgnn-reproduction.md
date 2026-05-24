# PDGNN Reproduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reproduce PDGNN (Yan et al., *Neural Approximation of Graph Topological Features*, NeurIPS 2022) on the **same 9 datasets** where we have TLC-GNN (exact PI) and no-PI results, enabling a 3-way comparison (exact PI vs neural-approx PI vs no PI).

**Architecture:** PDGNN is a GNN that approximates the persistence-diagram (PD) computation of TLC-GNN. The downstream link-prediction pipeline (GCN encoder + MLP decoder + Fermi-Dirac) is unchanged. Concretely: (1) train PDGNN once on supervised data — edge-centered vicinity subgraphs with ground-truth PDs from dionysus; (2) at inference, generate a `data/PDGNN/<dataset>.npy` cache with the same layout as the existing TLC-GNN cache; (3) reuse `pipelines.py` with a `--pi-source pdgnn` flag that picks the right cache directory; (4) run 50-trial sweep per dataset; (5) compare against TLC-GNN and no-PI results.

**Tech Stack:** PyTorch 2.1 / torch-geometric 2.5 / dionysus 2.1.8 / existing `Knowledge_Distillation/pdgnn_modern.py` (PDGNN architecture in PyG 2.x) / existing `sg2dgm/PersistenceImager` Cython module / SLURM (A100-80GB × N) / Python 3.9.

**Datasets (9):** Photo, PubMed, Computers (homophilic, paper Table 1) — Texas, Cornell, Wisconsin, Chameleon, Squirrel (heterophilic) — ChChMiner (drug-drug interaction).

---

## Pre-flight notes

- The existing `pipelines_LP_GIN.py` and `Knowledge_Distillation/ConvCurv_GIN.py` contain absolute paths from the original authors' machine (`/home/yzy/...`, `/data1/curvGN_LP/...`) and tangled cross-imports. **Do not try to fix them.** This plan introduces a clean integration via `pipelines.py` + `--pi-source` flag.
- `Knowledge_Distillation/pdgnn_modern.py` provides the PDGNN architecture (PyG 2.x). It has never been executed. Phase A validates it works.
- `Knowledge_Distillation/prepare_data_modern.py` provides *node-centered* (NC) data prep. We need *edge-centered* (LP) for link prediction. Phase B writes it.
- `sg2dgm/accelerated_PD.py` provides `perturb_filter_function / Union_find / Accelerate_PD` — used to compute ground-truth PDs. Already validated (works in `prepare_data_modern.py`).
- All datasets have working loaders in `loaddatas.py:loaddatas()`.

---

## File map

**New files to create:**
- `Knowledge_Distillation/prepare_data_LP_modern.py` — edge-centered vicinity data generation
- `Knowledge_Distillation/train_pdgnn_lp.py` — PDGNN supervised training script
- `Knowledge_Distillation/pdgnn_inference.py` — produce `data/PDGNN/<dataset>.npy` cache
- `tests/test_pdgnn_modern.py` — smoke tests for PDGNN model
- `data/PDGNN/checkpoints/pdgnn_lp.pt` — trained model checkpoint (output)
- `data/PDGNN/<dataset>.npy` — predicted PI cache per dataset (output)
- `docs/specs/2026-05-20-pdgnn-reproduction-results.md` — final results doc

**Existing files to modify:**
- `loaddatas.py` — add `pi_source` parameter to `compute_persistence_image`
- `pipelines.py` — add `--pi-source` argparse, pass through to model

---

## Phase A — Validate PDGNN architecture works at all

`pdgnn_modern.py` has never been executed. Run a synthetic forward + backward to catch any wiring bugs before investing in data prep.

### Task A.1: Smoke test forward pass

**Files:**
- Create: `tests/test_pdgnn_modern.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pdgnn_modern.py
import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Knowledge_Distillation.pdgnn_modern import PDGNN, PDGNNLayer


def test_forward_synthetic_chain():
    """5-node chain. PDGNN should run forward without error and return (E, 2)."""
    n = 5
    # chain: 0-1-2-3-4 (undirected)
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4],
                                [1, 0, 2, 1, 3, 2, 4, 3]], dtype=torch.long)
    filt = torch.arange(n, dtype=torch.float).view(-1, 1)
    model = PDGNN(hidden_dim=16, num_layers=2)
    out = model(filt, edge_index)
    assert out.shape == (edge_index.size(1), 2), f"expected ({edge_index.size(1)}, 2), got {out.shape}"


def test_backward_synthetic_chain():
    """Backward pass should produce gradients on all model params."""
    n = 5
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4],
                                [1, 0, 2, 1, 3, 2, 4, 3]], dtype=torch.long)
    filt = torch.arange(n, dtype=torch.float).view(-1, 1)
    target = torch.zeros(edge_index.size(1), 2)
    model = PDGNN(hidden_dim=16, num_layers=2)
    out = model(filt, edge_index)
    loss = (out - target).pow(2).mean()
    loss.backward()
    grads_ok = all(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in model.parameters())
    assert grads_ok, "some params have None or NaN gradient"
```

- [ ] **Step 2: Run test to verify it fails (or surfaces real bug)**

Run: `cd /mnt/data/users/junyoungpark/code/TLC-GNN && python -m pytest tests/test_pdgnn_modern.py -v`

Expected: either PASS (lucky), or FAIL with a real error from `pdgnn_modern.py`. Do not modify test until you've inspected the error.

- [ ] **Step 3: Fix bugs in `pdgnn_modern.py` if test fails**

Likely failure modes (based on PyG 2.x API changes):
- `scatter_min` import path: in newer torch-scatter, `from torch_scatter import scatter_min` still works. If error, replace with `from torch_geometric.utils import scatter` and use `reduce='min'`.
- `MessagePassing` signature: `dim_size` may need to be passed explicitly to `propagate`.
- `softmax` import unused; remove if it causes warnings.

For each fix:
1. Identify the line and exact error.
2. Apply minimal fix.
3. Re-run the test.

- [ ] **Step 4: Verify both tests pass**

Run: `cd /mnt/data/users/junyoungpark/code/TLC-GNN && python -m pytest tests/test_pdgnn_modern.py -v`

Expected:
```
tests/test_pdgnn_modern.py::test_forward_synthetic_chain PASSED
tests/test_pdgnn_modern.py::test_backward_synthetic_chain PASSED
```

- [ ] **Step 5: Commit**

```bash
cd /mnt/data/users/junyoungpark/code/TLC-GNN
git add tests/test_pdgnn_modern.py Knowledge_Distillation/pdgnn_modern.py
git commit -m "pdgnn smoke"
```

---

## Phase B — Generate edge-centered training data

PDGNN is trained on (filtration values, vicinity edge_index, ground-truth (birth, death) per edge). For LP comparison with TLC-GNN we must use *edge-centered* vicinities (one subgraph per source edge), with the **same Ollivier-Ricci-based filtration as TLC-GNN**, so predictions transfer.

### Task B.1: Write edge-centered data generator

**Files:**
- Create: `Knowledge_Distillation/prepare_data_LP_modern.py`

The generator mirrors `loaddatas.compute_persistence_image` but stores per-edge supervision tuples instead of bare PI vectors.

- [ ] **Step 1: Create the file with this content**

```python
# Knowledge_Distillation/prepare_data_LP_modern.py
"""Generate PDGNN training data for link prediction (edge-centered vicinities).

For each (u, v) edge in the source graph, extracts the intersection of
hop-k neighborhoods, computes the same Ollivier-Ricci-based 'sum' filtration
as loaddatas.compute_persistence_image, then records (filt_values,
edge_index, ground-truth (birth, death) coords).
"""

from __future__ import annotations
import os
import sys
import argparse
import pickle
import numpy as np
import networkx as nx
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import loaddatas as lds
from sg2dgm import riccidist2dgm as sg2dgm
from sg2dgm import PersistenceImager as pimg_mod


def _edge_vicinity(g: nx.Graph, u: int, v: int, hop: int) -> nx.Graph:
    """Edge-centered vicinity: intersection of k-hop neighborhoods of u and v.
    Matches TLC-GNN's V_12 (cf. loaddatas.compute_persistence_image)."""
    Nu = {u} | {w for _, w in nx.bfs_edges(g, u, depth_limit=hop)}
    Nv = {v} | {w for _, w in nx.bfs_edges(g, v, depth_limit=hop)}
    V12 = Nu & Nv
    if u not in V12: V12.add(u)
    if v not in V12: V12.add(v)
    return g.subgraph(V12).copy()


def _ollivier_ricci_filt(sub: nx.Graph, u: int, v: int,
                          ricci_lookup: dict) -> np.ndarray:
    """For each node x in sub, filter value = d(x,u) + d(x,v) under
    Ollivier-Ricci-weighted shortest paths. Matches TLC-GNN's 'sum' filter."""
    nodes = list(sub.nodes())
    vals = []
    for x in nodes:
        if x in (u, v):
            vals.append(0.0)
            continue
        try:
            p1 = nx.dijkstra_path(sub, x, u, weight='weight')
            d1 = sum(ricci_lookup.get((p1[i], p1[i+1]), 1.0) + 1
                     for i in range(len(p1)-1))
        except Exception:
            d1 = 100.0
        try:
            p2 = nx.dijkstra_path(sub, x, v, weight='weight')
            d2 = sum(ricci_lookup.get((p2[i], p2[i+1]), 1.0) + 1
                     for i in range(len(p2)-1))
        except Exception:
            d2 = 100.0
        vals.append(d1 + d2)
    arr = np.array(vals, dtype=np.float64)
    m = arr.max() + 1e-10
    return arr / m


def build_lp_vicinity_dataset(name: str, hop: int | None = None,
                               max_edges: int | None = None,
                               cache_dir: str = './data/PDGNN'):
    """Returns dict of {edge_id: (filt_values, edge_index_local, PD_pairs)}.

    PD_pairs is (K, 2) numpy: ground-truth (birth, death) coordinates for
    the K extended-persistence pairs computed via dionysus on this vicinity.
    Stored as the per-edge label for PDGNN supervised training."""
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, f'{name}_LP_train.pkl')
    if os.path.exists(cache):
        with open(cache, 'rb') as f:
            return pickle.load(f), cache

    ds = lds.loaddatas(name)
    data = ds[0]
    if hop is None:
        hop = 2 if name in ('PubMed',) else 1

    # Build full graph + Ricci curvature lookup
    g = nx.Graph()
    g.add_nodes_from(range(data.num_nodes))
    ei = np.array(data.edge_index)
    g.add_edges_from((int(ei[0, i]), int(ei[1, i])) for i in range(ei.shape[1]))
    ricci_list = lds.compute_ricci_curvature(data)
    ricci_lookup = {(int(a), int(b)): float(c) for a, b, c in ricci_list}
    for a, b in g.edges():
        w = ricci_lookup.get((a, b), ricci_lookup.get((b, a), 0.0)) + 1
        g[a][b]['weight'] = max(w, 1e-6)

    # Sample source edges
    edges = list(g.edges())
    if max_edges is not None and len(edges) > max_edges:
        rng = np.random.RandomState(1234)
        edges = [edges[i] for i in rng.choice(len(edges), max_edges, replace=False)]

    samples = {}
    imager = pimg_mod.PersistenceImager(resolution=5)
    g2pi = sg2dgm.graph2pi(g, ricci_curv=ricci_list)  # holds PD compute method
    for idx, (u, v) in enumerate(tqdm(edges, desc=f'{name}/LP')):
        sub = _edge_vicinity(g, u, v, hop)
        if sub.number_of_edges() == 0:
            continue
        filt_vals = _ollivier_ricci_filt(sub, u, v, ricci_lookup)
        # ground-truth PD via dionysus (same path TLC-GNN uses)
        try:
            dgms = g2pi.compute_extended_pd_for_edge(sub, u, v, filt_vals)
        except AttributeError:
            # If sg2dgm doesn't expose a single-edge method, fall back to
            # the local accelerated_PD path:
            from Knowledge_Distillation.accelerated_PD import (
                perturb_filter_function, Union_find, Accelerate_PD)
            sub_re = nx.convert_node_labels_to_integers(sub)
            sf = perturb_filter_function(sub_re, filt_vals)
            PD_up, ess0, PD_down, Pos, Neg = Union_find(sf)
            PD_one = Accelerate_PD(Pos, Neg, sf)
            dgms = np.concatenate([
                np.asarray(PD_up, dtype=np.float64).reshape(-1, 2) if len(PD_up) else np.empty((0,2)),
                np.asarray(ess0, dtype=np.float64).reshape(-1, 2) if len(ess0) else np.empty((0,2)),
                np.asarray(PD_down, dtype=np.float64).reshape(-1, 2) if len(PD_down) else np.empty((0,2)),
                np.asarray(PD_one, dtype=np.float64).reshape(-1, 2) if len(PD_one) else np.empty((0,2)),
            ], axis=0)
        ei_sub = np.array(list(sub.edges()), dtype=np.int64).T  # (2, E)
        if ei_sub.size:
            ei_sub = np.concatenate([ei_sub, ei_sub[[1, 0]]], axis=1)
        else:
            ei_sub = np.zeros((2, 0), dtype=np.int64)
        # Map sub-graph node ids to local 0..n-1
        node_list = list(sub.nodes())
        remap = {n: i for i, n in enumerate(node_list)}
        ei_sub_local = np.array([[remap[ei_sub[0, k]] for k in range(ei_sub.shape[1])],
                                  [remap[ei_sub[1, k]] for k in range(ei_sub.shape[1])]],
                                 dtype=np.int64) if ei_sub.size else np.zeros((2, 0), dtype=np.int64)
        samples[idx] = (filt_vals.astype(np.float32),
                        ei_sub_local,
                        dgms.astype(np.float64),
                        int(remap[u]), int(remap[v]))

    with open(cache, 'wb') as f:
        pickle.dump(samples, f, pickle.HIGHEST_PROTOCOL)
    print(f'[PDGNN-LP data] saved {len(samples)} samples to {cache}')
    return samples, cache


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='PubMed')
    parser.add_argument('--hop', type=int, default=None)
    parser.add_argument('--max_edges', type=int, default=5000)
    args = parser.parse_args()
    build_lp_vicinity_dataset(args.name, hop=args.hop, max_edges=args.max_edges)
```

- [ ] **Step 2: Verify by generating a small batch (200 edges, Cora)**

Run:
```bash
cd /mnt/data/users/junyoungpark/code/TLC-GNN
conda activate tlcgnn
python -m Knowledge_Distillation.prepare_data_LP_modern --name Cora --max_edges 200
```

Expected: progress bar runs to 200/200, output `[PDGNN-LP data] saved {N} samples to ./data/PDGNN/Cora_LP_train.pkl`.

- [ ] **Step 3: Sanity-check the produced pickle**

```bash
python -c "
import pickle
with open('./data/PDGNN/Cora_LP_train.pkl','rb') as f:
    s = pickle.load(f)
print(f'samples: {len(s)}')
k = next(iter(s))
filt, ei, pd, u, v = s[k]
print(f'sample[{k}]: filt={filt.shape} ei={ei.shape} pd={pd.shape} u={u} v={v}')
print(f'pd sample rows: {pd[:3]}')
"
```

Expected: ~150-200 samples (some edges skipped for empty vicinity), filt is (n,), ei is (2, E), pd is (K, 2) with K≥1.

- [ ] **Step 4: Commit**

```bash
git add Knowledge_Distillation/prepare_data_LP_modern.py
git commit -m "edge-centered data prep"
```

### Task B.2: Generate full PubMed training data

PubMed is sparse and large enough to give PDGNN diverse training samples. Paper trains on ~10K vicinity subgraphs.

- [ ] **Step 1: Submit SLURM job**

```bash
cd /mnt/data/users/junyoungpark/code/TLC-GNN
cat > pdgnn_data_prep.sh <<'EOF'
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/data/users/junyoungpark/code/TLC-GNN/slurm_logs/%x-%j.out
#SBATCH --error=/mnt/data/users/junyoungpark/code/TLC-GNN/slurm_logs/%x-%j.err
source /mnt/data/users/junyoungpark/miniforge3/etc/profile.d/conda.sh
conda activate tlcgnn
cd /mnt/data/users/junyoungpark/code/TLC-GNN
python -m Knowledge_Distillation.prepare_data_LP_modern --name PubMed --max_edges 10000
EOF
chmod +x pdgnn_data_prep.sh
sbatch --job-name=pdgnn-data-PubMed pdgnn_data_prep.sh
```

- [ ] **Step 2: Wait for completion + verify output**

```bash
ls -la data/PDGNN/PubMed_LP_train.pkl
python -c "
import pickle
s = pickle.load(open('./data/PDGNN/PubMed_LP_train.pkl','rb'))
print(f'PubMed LP samples: {len(s)}')
"
```

Expected: ~9000-10000 samples (some skipped for empty vicinity), file size ~50-100 MB.

- [ ] **Step 3: Commit**

```bash
git add pdgnn_data_prep.sh
git commit -m "pdgnn data sbatch script"
```

---

## Phase C — Train PDGNN

Supervised training: minimize MSE between predicted (b, d) and ground-truth pairs.

### Task C.1: Write training script

**Files:**
- Create: `Knowledge_Distillation/train_pdgnn_lp.py`

PDGNN outputs one (b, d) per edge in the vicinity. Ground-truth PDs have K pairs of (b, d) where K can be larger or smaller than #edges. We use a **bipartite-matching loss**: each predicted (b, d) matches to its closest ground-truth pair (greedy or Hungarian), then MSE on matched pairs. This is the paper's "Wasserstein-loss-lite" approach.

- [ ] **Step 1: Create the file**

```python
# Knowledge_Distillation/train_pdgnn_lp.py
"""Train PDGNN on edge-centered LP supervision data."""

from __future__ import annotations
import os
import sys
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Knowledge_Distillation.pdgnn_modern import PDGNN


def _bipartite_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Pred: (E, 2). GT: (K, 2). Match each gt to its closest pred (or vice versa).
    Use the Hungarian algorithm on a small cost matrix; clamp at the smaller side."""
    if pred.numel() == 0 or gt.numel() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    # Cost: pairwise sq distance
    cost = ((pred.unsqueeze(1) - gt.unsqueeze(0)) ** 2).sum(dim=-1)  # (E, K)
    cost_np = cost.detach().cpu().numpy()
    row_idx, col_idx = linear_sum_assignment(cost_np)
    matched = cost[row_idx, col_idx]
    return matched.mean()


def train_one_epoch(model, optimizer, samples, device):
    model.train()
    losses = []
    keys = list(samples.keys())
    np.random.shuffle(keys)
    for k in tqdm(keys, desc='train'):
        filt, ei, pd_gt, _, _ = samples[k]
        if ei.size == 0 or pd_gt.size == 0:
            continue
        filt_t = torch.tensor(filt, dtype=torch.float, device=device).view(-1, 1)
        ei_t = torch.tensor(ei, dtype=torch.long, device=device)
        gt_t = torch.tensor(pd_gt, dtype=torch.float, device=device)
        optimizer.zero_grad()
        pred = model(filt_t, ei_t)  # (E, 2)
        loss = _bipartite_loss(pred, gt_t)
        if not torch.isfinite(loss):
            continue
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else float('nan')


def evaluate(model, samples, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for k, (filt, ei, pd_gt, _, _) in samples.items():
            if ei.size == 0 or pd_gt.size == 0:
                continue
            filt_t = torch.tensor(filt, dtype=torch.float, device=device).view(-1, 1)
            ei_t = torch.tensor(ei, dtype=torch.long, device=device)
            gt_t = torch.tensor(pd_gt, dtype=torch.float, device=device)
            pred = model(filt_t, ei_t)
            loss = _bipartite_loss(pred, gt_t)
            losses.append(loss.item())
    return float(np.mean(losses)) if losses else float('nan')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='./data/PDGNN/PubMed_LP_train.pkl')
    parser.add_argument('--out', default='./data/PDGNN/checkpoints/pdgnn_lp.pt')
    parser.add_argument('--hidden', type=int, default=32)
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=1234)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(args.data, 'rb') as f:
        samples = pickle.load(f)
    keys = list(samples.keys())
    np.random.RandomState(1234).shuffle(keys)
    n_train = int(0.9 * len(keys))
    train_keys, val_keys = keys[:n_train], keys[n_train:]
    train_samples = {k: samples[k] for k in train_keys}
    val_samples = {k: samples[k] for k in val_keys}
    print(f'train={len(train_samples)} val={len(val_samples)}')

    model = PDGNN(hidden_dim=args.hidden, num_layers=args.layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    best_val = float('inf')
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    for ep in range(1, args.epochs + 1):
        tr = train_one_epoch(model, optimizer, train_samples, device)
        va = evaluate(model, val_samples, device)
        improved = va < best_val
        if improved:
            best_val = va
            torch.save({'state_dict': model.state_dict(),
                        'config': {'hidden_dim': args.hidden, 'num_layers': args.layers}},
                       args.out)
        print(f'ep {ep:3d}  train_mse={tr:.4f}  val_mse={va:.4f}  '
              f'best_val={best_val:.4f}  saved={improved}')


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Verify import + dry-run on the Cora 200-sample pickle**

```bash
cd /mnt/data/users/junyoungpark/code/TLC-GNN
conda activate tlcgnn
python -m Knowledge_Distillation.train_pdgnn_lp --data ./data/PDGNN/Cora_LP_train.pkl --epochs 2 --out /tmp/pdgnn_cora_smoke.pt
```

Expected: 2 epochs run, train_mse + val_mse printed each epoch, MSE should decrease (not necessarily monotonically) and final val_mse < 1.0.

- [ ] **Step 3: Commit**

```bash
git add Knowledge_Distillation/train_pdgnn_lp.py
git commit -m "pdgnn training script"
```

### Task C.2: Train PDGNN on PubMed data

- [ ] **Step 1: Write training sbatch**

```bash
cat > pdgnn_train.sh <<'EOF'
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/data/users/junyoungpark/code/TLC-GNN/slurm_logs/%x-%j.out
#SBATCH --error=/mnt/data/users/junyoungpark/code/TLC-GNN/slurm_logs/%x-%j.err
source /mnt/data/users/junyoungpark/miniforge3/etc/profile.d/conda.sh
conda activate tlcgnn
cd /mnt/data/users/junyoungpark/code/TLC-GNN
python -m Knowledge_Distillation.train_pdgnn_lp \
  --data ./data/PDGNN/PubMed_LP_train.pkl \
  --out ./data/PDGNN/checkpoints/pdgnn_lp.pt \
  --hidden 32 --layers 3 --epochs 50 --lr 1e-3
EOF
chmod +x pdgnn_train.sh
sbatch --job-name=pdgnn-train pdgnn_train.sh
```

- [ ] **Step 2: Validate checkpoint**

```bash
python -c "
import torch
ckpt = torch.load('./data/PDGNN/checkpoints/pdgnn_lp.pt', map_location='cpu')
print('config:', ckpt['config'])
print('state_dict keys:', list(ckpt['state_dict'].keys())[:5], '...')
"
```

Expected: config dict + ~10-15 parameter tensor names.

- [ ] **Step 3: Pass gate — val MSE < 0.05**

If val MSE > 0.05, PDGNN approximation is too coarse to be useful. Diagnose:
- Are PDs being computed correctly on the training set? Spot-check a few `pd_gt` arrays — should not all be empty.
- Hidden dim too small? Try 64.
- Learning rate too low/high? Try 5e-4, 5e-3.

Re-train until val MSE < 0.05 or document why it's not achievable (and proceed anyway with note).

- [ ] **Step 4: Commit**

```bash
git add pdgnn_train.sh
git commit -m "pdgnn train sbatch"
```

---

## Phase D — Inference: generate PDGNN-PI cache per dataset

For each of the 9 datasets, run the trained PDGNN to produce a `data/PDGNN/<name>.npy` cache with the same shape/layout as the TLC-GNN cache.

### Task D.1: Write inference script

**Files:**
- Create: `Knowledge_Distillation/pdgnn_inference.py`

Mirrors `loaddatas.compute_persistence_image` row-for-row, but replaces dionysus calls with PDGNN forward.

- [ ] **Step 1: Create the file**

```python
# Knowledge_Distillation/pdgnn_inference.py
"""Generate PDGNN-predicted PI cache for a dataset.

Produces ./data/PDGNN/<name>.npy with shape (N_edges, 25),
where N_edges and the row order match exactly
loaddatas.compute_persistence_image's output (so cached files are
interchangeable via --pi-source flag in pipelines.py)."""

from __future__ import annotations
import os
import sys
import argparse
import numpy as np
import networkx as nx
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import loaddatas as lds
from Knowledge_Distillation.pdgnn_modern import PDGNN
from Knowledge_Distillation.prepare_data_LP_modern import _edge_vicinity, _ollivier_ricci_filt
from sg2dgm import PersistenceImager as pimg_mod


def _pd_to_pi(pd: np.ndarray, imager) -> np.ndarray:
    """Convert (K, 2) PD coords to 5x5 PI flatten = (25,)."""
    if pd.size == 0:
        return np.zeros(25, dtype=np.float64)
    return imager.transform(pd).reshape(-1)


@torch.no_grad()
def run_inference(name: str, ckpt_path: str = './data/PDGNN/checkpoints/pdgnn_lp.pt',
                  out_dir: str = './data/PDGNN', hop: int | None = None):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{name}.npy')
    if os.path.exists(out_path):
        print(f'cache exists: {out_path}; skipping')
        return out_path

    # Load model
    ckpt = torch.load(ckpt_path, map_location='cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = ckpt['config']
    model = PDGNN(hidden_dim=cfg['hidden_dim'], num_layers=cfg['num_layers']).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    # Load dataset + compute the SAME edge ordering as loaddatas
    ds = lds.loaddatas(name)
    data = ds[0]
    val_prop = 0.2 if name in ('PPI',) else 0.05
    test_prop = 0.2 if name in ('PPI',) else 0.1
    tr, trf, va, vaf, te, tef = lds.get_edges_split(data, val_prop=val_prop,
                                                     test_prop=test_prop)
    total_edges = np.concatenate((tr, trf, va, vaf, te, tef))

    if hop is None:
        hop = 2 if name == 'PubMed' else 1

    # Full graph + Ricci
    g = nx.Graph()
    g.add_nodes_from(range(data.num_nodes))
    ei = np.array(data.edge_index)
    g.add_edges_from((int(ei[0, i]), int(ei[1, i])) for i in range(ei.shape[1]))
    ricci_list = lds.compute_ricci_curvature(data)
    ricci_lookup = {(int(a), int(b)): float(c) for a, b, c in ricci_list}
    for a, b in g.edges():
        w = ricci_lookup.get((a, b), ricci_lookup.get((b, a), 0.0)) + 1
        g[a][b]['weight'] = max(w, 1e-6)

    imager = pimg_mod.PersistenceImager(resolution=5)
    PIs = np.zeros((len(total_edges), 25), dtype=np.float64)
    from tqdm import tqdm
    for i, (u, v) in enumerate(tqdm(total_edges, desc=f'PDGNN-PI {name}')):
        u, v = int(u), int(v)
        sub = _edge_vicinity(g, u, v, hop)
        if sub.number_of_edges() == 0:
            continue
        filt_vals = _ollivier_ricci_filt(sub, u, v, ricci_lookup)
        node_list = list(sub.nodes())
        remap = {n: idx for idx, n in enumerate(node_list)}
        ei_sub = np.array([(remap[a], remap[b]) for a, b in sub.edges()],
                          dtype=np.int64).T
        if ei_sub.size:
            ei_sub = np.concatenate([ei_sub, ei_sub[[1, 0]]], axis=1)
        else:
            ei_sub = np.zeros((2, 0), dtype=np.int64)
        filt_t = torch.tensor(filt_vals, dtype=torch.float, device=device).view(-1, 1)
        ei_t = torch.tensor(ei_sub, dtype=torch.long, device=device)
        if ei_t.size(1) == 0:
            continue
        pred = model(filt_t, ei_t).cpu().numpy()  # (E_sub, 2)
        PIs[i] = _pd_to_pi(pred, imager)

    np.save(out_path, PIs)
    print(f'saved PDGNN PI cache: {out_path} shape={PIs.shape}')
    return out_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('--ckpt', default='./data/PDGNN/checkpoints/pdgnn_lp.pt')
    parser.add_argument('--hop', type=int, default=None)
    args = parser.parse_args()
    run_inference(args.name, ckpt_path=args.ckpt, hop=args.hop)
```

- [ ] **Step 2: Smoke test on Cora (small dataset, fast)**

```bash
cd /mnt/data/users/junyoungpark/code/TLC-GNN
conda activate tlcgnn
python -m Knowledge_Distillation.pdgnn_inference --name Cora
ls -la data/PDGNN/Cora.npy
python -c "
import numpy as np
a = np.load('./data/PDGNN/Cora.npy')
print('shape:', a.shape, 'dtype:', a.dtype)
print('nonzero rows:', int((a != 0).any(axis=1).sum()), '/', a.shape[0])
print('value range:', float(a.min()), '..', float(a.max()))
"
```

Expected: cache file written, shape (~N_edges, 25), >50% nonzero rows.

- [ ] **Step 3: Compare PDGNN PI vs TLC-GNN PI on Cora (sanity check)**

(Only if TLC-GNN Cora cache also exists. If not, skip — generate a small dionysus cache for comparison.)

```bash
python -c "
import numpy as np
pd = np.load('./data/PDGNN/Cora.npy')
tg = np.load('./data/TLCGNN/Cora.npy') if __import__('os').path.exists('./data/TLCGNN/Cora.npy') else None
if tg is not None:
    n = min(len(pd), len(tg))
    mse = ((pd[:n] - tg[:n]) ** 2).mean()
    print(f'PDGNN vs dionysus PI MSE: {mse:.4f}')
    print(f'PDGNN nonzero ratio: {(pd[:n]!=0).mean():.3f}')
    print(f'dionysus nonzero ratio: {(tg[:n]!=0).mean():.3f}')
else:
    print('no TLC-GNN Cora cache; skipping comparison')
"
```

Expected MSE: < 1.0. Higher means PDGNN approximation is too rough; revisit training (Phase C).

- [ ] **Step 4: Commit**

```bash
git add Knowledge_Distillation/pdgnn_inference.py
git commit -m "pdgnn inference"
```

### Task D.2: Generate PDGNN-PI cache for all 9 datasets

- [ ] **Step 1: Submit one SLURM job per dataset**

```bash
cd /mnt/data/users/junyoungpark/code/TLC-GNN
for D in Photo PubMed Computers Texas Cornell Wisconsin Chameleon Squirrel ChChMiner; do
  cat > /tmp/pdgnn_inf_$D.sh <<EOF
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/data/users/junyoungpark/code/TLC-GNN/slurm_logs/%x-%j.out
#SBATCH --error=/mnt/data/users/junyoungpark/code/TLC-GNN/slurm_logs/%x-%j.err
source /mnt/data/users/junyoungpark/miniforge3/etc/profile.d/conda.sh
conda activate tlcgnn
cd /mnt/data/users/junyoungpark/code/TLC-GNN
python -m Knowledge_Distillation.pdgnn_inference --name $D
EOF
  chmod +x /tmp/pdgnn_inf_$D.sh
  sbatch --job-name=pdgnn-inf-$D /tmp/pdgnn_inf_$D.sh
done
squeue -u $(whoami)
```

- [ ] **Step 2: Wait for all 9 to complete + verify**

```bash
for D in Photo PubMed Computers Texas Cornell Wisconsin Chameleon Squirrel ChChMiner; do
  if [ -f "data/PDGNN/$D.npy" ]; then
    python -c "import numpy as np; a=np.load('data/PDGNN/$D.npy'); print('$D', a.shape)"
  else
    echo "$D MISSING"
  fi
done
```

Expected: all 9 files present, shapes matching the corresponding TLC-GNN cache.

- [ ] **Step 3: Commit any sbatch script changes (if templated)**

```bash
git add pdgnn_data_prep.sh pdgnn_train.sh
git commit -m "pdgnn batch scripts" --allow-empty
```

---

## Phase E — Pipeline integration: `--pi-source` flag

### Task E.1: Add `pi_source` parameter to `loaddatas.compute_persistence_image`

**Files:**
- Modify: `loaddatas.py` (the `compute_persistence_image` function and `filename` line)

- [ ] **Step 1: Edit `loaddatas.py`**

Find the line:
```python
filename = './data/TLCGNN/' + data_name + '.npy'
```

Replace with:
```python
# Allow pluggable PI sources: dionysus (default, TLC-GNN) or pdgnn (neural approx)
pi_source = os.environ.get('TLCGNN_PI_SOURCE', 'dionysus')
if pi_source == 'pdgnn':
    filename = './data/PDGNN/' + data_name + '.npy'
else:
    filename = './data/TLCGNN/' + data_name + '.npy'
```

This keeps `loaddatas.py`'s function signature unchanged — the source is selected via environment variable, set by `pipelines.py`.

- [ ] **Step 2: Add `--pi-source` to `pipelines.py`**

Find:
```python
_parser.add_argument('--no_pi', action='store_true', ...)
```

Insert above it:
```python
_parser.add_argument('--pi-source', choices=['dionysus', 'pdgnn'], default='dionysus',
                     help='where PI cache comes from')
```

Then right after `_args = _parser.parse_args()`, add:
```python
os.environ['TLCGNN_PI_SOURCE'] = _args.pi_source
```

- [ ] **Step 3: Smoke test PubMed 1-trial with `--pi-source pdgnn`**

```bash
cd /mnt/data/users/junyoungpark/code/TLC-GNN
conda activate tlcgnn
python pipelines.py --datasets PubMed --trials 1 --tag pdgnnSmoke --pi-source pdgnn
```

Expected: trial runs, Test ROC > 0.85 (lower bound — at minimum the GCN encoder works; PDGNN PI quality determines how close to TLC-GNN's 0.97 we get).

- [ ] **Step 4: Commit**

```bash
git add loaddatas.py pipelines.py
git commit -m "pi-source flag"
```

---

## Phase F — 50-trial sweep with PDGNN PI

### Task F.1: Submit 9 SLURM jobs

- [ ] **Step 1: Submit**

```bash
cd /mnt/data/users/junyoungpark/code/TLC-GNN
for D in Photo PubMed Computers Texas Cornell Wisconsin Chameleon Squirrel ChChMiner; do
  sbatch --job-name=tlcgnn-${D}-pdgnn slurm_run.sh --datasets $D --trials 50 --tag pdgnn --pi-source pdgnn
done
squeue -u $(whoami)
```

- [ ] **Step 2: Wait + collect results**

```bash
for D in Photo PubMed Computers Texas Cornell Wisconsin Chameleon Squirrel ChChMiner; do
  F="scores/pipe_benchmark_${D}_LP_scorespdgnn.txt"
  if [ -f "$F" ]; then
    awk -F'[, ]+' 'NR>1 && /^[0-9]+,/ {a[++n]=$3; sum+=$3} END {if(n>1){m=sum/n; for(i=1;i<=n;i++){d=a[i]-m;s+=d*d}; printf "%-12s %d  AUC=%.4f ± %.4f\n", "'$D'", n, m, sqrt(s/(n-1))} else print "'$D'", "no data"}' "$F"
  else
    echo "$D: missing"
  fi
done
```

- [ ] **Step 3: Commit results**

```bash
git add scores/pipe_benchmark_*_LP_scorespdgnn.txt
git commit -m "pdgnn 50-trial scores"
```

---

## Phase G — 3-way comparison report

### Task G.1: Write results doc

**Files:**
- Create: `docs/specs/2026-05-20-pdgnn-reproduction-results.md`

- [ ] **Step 1: Create doc**

```markdown
# PDGNN Reproduction Results

**Date:** 2026-05-20
**Plan:** `docs/superpowers/plans/2026-05-20-pdgnn-reproduction.md`

## Setup
- PDGNN trained on PubMed edge-centered LP vicinities (10K samples)
- Hidden dim 32, 3 layers
- Trained 50 epochs, Adam lr=1e-3, bipartite-matching MSE loss
- Final val MSE: <FILL IN>

## 3-way comparison (50 trials per dataset, AUC ± std)

| Dataset    | TLC-GNN (dionysus) | PDGNN (neural)     | No PI       | PDGNN vs TLC-GNN | PDGNN vs No PI |
|------------|---------------------|--------------------|-------------|-------------------|------------------|
| Photo      | 0.9825 ± 0.0008     | <fill>             | <fill>      | <fill>            | <fill>           |
| PubMed     | 0.9635 ± 0.0025     | <fill>             | <fill>      | <fill>            | <fill>           |
| Computers  | 0.9680 ± 0.0023     | <fill>             | <fill>      | <fill>            | <fill>           |
| Texas      | 0.5709 ± 0.111      | <fill>             | 0.5939 ± 0.133 | <fill>         | <fill>           |
| Cornell    | 0.5850 ± 0.113      | <fill>             | 0.6502 ± 0.143 | <fill>         | <fill>           |
| Wisconsin  | 0.8640 ± 0.062      | <fill>             | 0.8653 ± 0.061 | <fill>         | <fill>           |
| Chameleon  | 0.9432 ± 0.007      | <fill>             | 0.9686 ± 0.006 | <fill>         | <fill>           |
| Squirrel   | (incomplete TLC-GNN) | <fill>            | 0.9854 ± 0.001 | <fill>         | <fill>           |
| ChChMiner  | 0.9026 ± 0.007      | <fill>             | 0.9650 ± 0.006 | <fill>         | <fill>           |

## Findings
1. **PDGNN approximation quality vs exact PI:** <fill — within X%p of TLC-GNN AUC?>
2. **Compute cost:** PDGNN inference <fill> s/edge vs dionysus <fill> s/edge.
3. **Heterophilic / drug findings hold?** Does PDGNN PI also fail to help (or hurt) on Chameleon / Cornell / Texas / ChChMiner — i.e., is the topology useless property of the data, or an artifact of exact PD computation?

## Conclusion
<fill>
```

- [ ] **Step 2: Fill in the table with actual numbers from `scores/*.txt`**

Replace each `<fill>` with computed mean ± std from the corresponding score file. Use the same awk one-liner from Phase F Step 2.

- [ ] **Step 3: Commit**

```bash
git add docs/specs/2026-05-20-pdgnn-reproduction-results.md
git commit -m "pdgnn results"
```

### Task G.2: Update README

- [ ] **Step 1: Append a row to the README results table**

Find the existing `## 재현 결과` section, replace the table with:

```markdown
## 결과

| Dataset   | TLC-GNN | PDGNN | No PI |
|-----------|---------|-------|-------|
| Photo     | 0.9825  | <fill> | <fill> |
| PubMed    | 0.9635  | <fill> | <fill> |
| Computers | 0.9680  | <fill> | <fill> |
| Chameleon | 0.9432  | <fill> | 0.9686 |
| ChChMiner | 0.9026  | <fill> | 0.9650 |
| ...       | ...     | ...    | ...    |

자세한 결과: [docs/specs/2026-05-20-pdgnn-reproduction-results.md](docs/specs/2026-05-20-pdgnn-reproduction-results.md)
```

- [ ] **Step 2: Commit + push to remote**

```bash
git add README.md
git commit -m "readme update"
git push tda main
```

---

## Self-Review Checklist (run after writing the plan)

- [x] **Spec coverage:** PDGNN architecture (Phase A), edge-centered LP training data (Phase B), training (Phase C), per-dataset inference (Phase D), pipeline integration (Phase E), 50-trial sweep (Phase F), comparison (Phase G).
- [x] **Placeholder scan:** `<fill in>` markers in Phase G are *intentional* (template for engineer to populate with real numbers after experiments). No other TODOs.
- [x] **Type consistency:** PDGNN forward returns `(E, 2)` throughout (Phase A test, Phase C training, Phase D inference). `samples` pickle structure is `dict[int, tuple]` with consistent unpacking `(filt, ei, pd_gt, u, v)` in Phases B/C, and `(filt_vals, ei_sub_local, dgms, remap[u], remap[v])` in producer matches consumer.
- [x] **Cache layout:** `data/PDGNN/<name>.npy` shape `(N_edges, 25)` matches `data/TLCGNN/<name>.npy` exactly, enabling drop-in swap via env var.
- [x] **Scope:** Reproduces PDGNN end-to-end on the 9 already-evaluated datasets. Out of scope: training PDGNN on multiple source graphs (paper does single-source PubMed too); evaluating PDGNN on graph classification or node classification tasks; OGBL-DDI (deferred separately).

## Time estimate

| Phase | Wall-clock |
|---|---|
| A. PDGNN smoke + bug fixes | 2-4 hours |
| B. Data prep (Cora + PubMed) | 6-8 hours (mostly waiting for PubMed gen) |
| C. PDGNN training | 4-8 hours |
| D. Inference cache for 9 datasets | 6-12 hours (parallel SLURM) |
| E. Pipeline integration + smoke | 1-2 hours |
| F. 50-trial sweeps (9 datasets parallel) | 4-12 hours |
| G. Results write-up | 2-4 hours |
| **Total** | **~1 week** |

## Known risks

| Risk | Mitigation |
|---|---|
| `pdgnn_modern.py` has bugs (untested) | Phase A smoke tests surface them early. |
| PDGNN val MSE >0.1 → PI predictions too noisy | Phase C gate; tune hidden_dim / lr if needed. |
| PDGNN trained on PubMed doesn't generalize to ChChMiner (drug) | Document in results. If gap is large, consider training on mixed sources. |
| Long PD compute on training-data prep | Phase B uses `--max_edges 10000` cap. |
| Pipeline cache layout drift | `pdgnn_inference.py` mirrors `loaddatas.compute_persistence_image` exactly — same `get_edges_split` seed=1234, same concat order. |
