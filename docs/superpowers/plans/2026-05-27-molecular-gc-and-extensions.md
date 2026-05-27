# Molecular GC + Extensions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Add molecular **graph classification** (GC) as a second task type to the TDA topology study, plus 4 targeted extensions to the existing link-prediction (LP) story.

**Architecture:** Part M reuses the existing PD machinery (`accelerated_PD`, `PersistenceImager`) but on whole molecule graphs, with a GIN classifier comparing with-PI vs no-PI. Part I fills gaps in the LP story (homophilic anchor, sparsity-gating, cap sweep, optional OGBL-DDI). All new code lives in new files to avoid conflicts; shared-file edits (`loaddatas.py`) are serialized.

**Tech Stack:** PyTorch 2.1 / torch-geometric 2.5 (GINConv, TUDataset) / dionysus 2.1.8 / existing sg2dgm + accelerated_PD / SLURM.

**Spec:** `docs/superpowers/specs/2026-05-27-molecular-gc-and-extensions-design.md`

---

## File Structure

### New files
- `Knowledge_Distillation/mol_data.py` — TUDataset loader + per-graph PI computation + cache
- `Knowledge_Distillation/mol_classify.py` — GIN graph classifier (with/no PI) + 10-fold CV
- `baselines/TLCGNN_gated_reg.py` — sparsity-regularized gating variant
- `tests/test_mol_data.py` — mol_data unit test

### Modified files (controller serializes loaddatas edits)
- `loaddatas.py` — env-var negative cap (`TLCGNN_NEG_CAP`) for I.3; OGBL-DDI loader for I.4
- `pipelines.py` — `--gate_reg` flag for I.2

---

## Part M — Molecular Graph Classification

### Task M.1: mol_data.py — TUDataset + per-graph PI

**Files:**
- Create: `tests/test_mol_data.py`
- Create: `Knowledge_Distillation/mol_data.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_mol_data.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from Knowledge_Distillation.mol_data import load_tudataset, graph_to_pi


def test_load_mutag():
    """MUTAG has 188 graphs, binary labels."""
    graphs, labels = load_tudataset('MUTAG')
    assert len(graphs) == 188
    assert set(labels.tolist()) <= {0, 1}
    # each graph is a networkx Graph
    import networkx as nx
    assert isinstance(graphs[0], nx.Graph)


def test_graph_to_pi_shape():
    """graph_to_pi returns a flat 25-dim vector per graph."""
    import networkx as nx
    g = nx.path_graph(6)  # simple chain
    pi = graph_to_pi(g)
    assert pi.shape == (25,), f"expected (25,), got {pi.shape}"
    assert np.isfinite(pi).all()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /mnt/data/users/junyoungpark/code/TLC-GNN
source /mnt/data/users/junyoungpark/miniforge3/etc/profile.d/conda.sh && conda activate tlcgnn
python -m pytest tests/test_mol_data.py -v
```
Expected: FAIL (module not found).

- [ ] **Step 3: Implement `Knowledge_Distillation/mol_data.py`**

```python
# Knowledge_Distillation/mol_data.py
"""Load TUDataset molecule graphs and compute a whole-graph persistence image
per molecule. Filter = normalized node degree. Reuses the repo's accelerated_PD
+ PersistenceImager machinery (same as the LP pipeline, but on the full molecule
graph instead of an edge vicinity)."""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import networkx as nx
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx
from sg2dgm import PersistenceImager as pimg_mod
from Knowledge_Distillation.accelerated_PD import (perturb_filter_function,
                                                    Union_find, Accelerate_PD)

_IMAGER = pimg_mod.PersistenceImager(resolution=5)


def load_tudataset(name: str):
    """Return (list_of_networkx_graphs, label_tensor)."""
    ds = TUDataset(root=f'./data/TU_{name}', name=name)
    graphs, labels = [], []
    for data in ds:
        g = to_networkx(data, to_undirected=True)
        graphs.append(g)
        labels.append(int(data.y.item()))
    return graphs, torch.tensor(labels, dtype=torch.long)


def _degree_filter(g: nx.Graph) -> np.ndarray:
    """Normalized node degree as filtration value."""
    g = nx.convert_node_labels_to_integers(g)
    n = g.number_of_nodes()
    deg = np.array([d for _, d in sorted(g.degree(), key=lambda x: x[0])],
                   dtype=np.float64)
    m = deg.max() + 1e-10
    return deg / m


def graph_to_pi(g: nx.Graph) -> np.ndarray:
    """Whole-graph extended PD → 5x5 PI flattened to (25,)."""
    g = nx.convert_node_labels_to_integers(g)
    if g.number_of_edges() == 0 or g.number_of_nodes() < 3:
        return np.zeros(25, dtype=np.float64)
    filt = _degree_filter(g)
    sf = perturb_filter_function(g, filt)
    try:
        PD_up, ess0, PD_down, Pos, Neg = Union_find(sf)
        PD_one = Accelerate_PD(Pos, Neg, sf)
    except (ValueError, IndexError):
        return np.zeros(25, dtype=np.float64)
    pd = []
    for arr in [PD_up, ess0, PD_down, PD_one]:
        a = np.asarray(arr, dtype=np.float64).reshape(-1, 2) if len(arr) else np.empty((0, 2))
        if a.size:
            pd.append(a)
    pd_all = np.concatenate(pd, axis=0) if pd else np.empty((0, 2))
    if pd_all.size == 0:
        return np.zeros(25, dtype=np.float64)
    return _IMAGER.transform(pd_all).reshape(-1)


def compute_all_pi(name: str, cache_dir: str = './data/MOL'):
    """Compute + cache PI for every graph in a TUDataset. Returns (N, 25) array."""
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, f'{name}_PI.npy')
    if os.path.exists(cache):
        return np.load(cache)
    graphs, _ = load_tudataset(name)
    from tqdm import tqdm
    pis = np.stack([graph_to_pi(g) for g in tqdm(graphs, desc=f'{name} PI')])
    np.save(cache, pis)
    return pis


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='MUTAG')
    args = p.parse_args()
    pis = compute_all_pi(args.dataset)
    print(f'{args.dataset}: PI cache shape {pis.shape}')
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_mol_data.py -v
```
Expected: 2 PASS. (Downloads MUTAG ~1MB.)

- [ ] **Step 5: Commit**

```bash
git add tests/test_mol_data.py Knowledge_Distillation/mol_data.py
git commit -m "mol data + per-graph PI"
```

### Task M.2: mol_classify.py — GIN classifier with/without PI

**Files:**
- Create: `Knowledge_Distillation/mol_classify.py`

- [ ] **Step 1: Create the file**

```python
# Knowledge_Distillation/mol_classify.py
"""GIN graph classifier for TUDataset molecules, with optional whole-graph PI
feature. 10-fold stratified CV. Compares with-PI vs no-PI (--no_pi)."""
from __future__ import annotations
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_mean_pool
from sklearn.model_selection import StratifiedKFold
from Knowledge_Distillation.mol_data import compute_all_pi


class GINClassifier(nn.Module):
    def __init__(self, in_dim, hidden=64, n_classes=2, use_pi=True, pi_dim=25):
        super().__init__()
        def mlp(i, o):
            return nn.Sequential(nn.Linear(i, o), nn.ReLU(), nn.Linear(o, o))
        self.conv1 = GINConv(mlp(in_dim, hidden))
        self.conv2 = GINConv(mlp(hidden, hidden))
        self.conv3 = GINConv(mlp(hidden, hidden))
        self.use_pi = use_pi
        head_in = hidden + (pi_dim if use_pi else 0)
        self.head = nn.Sequential(nn.Linear(head_in, hidden), nn.ReLU(),
                                   nn.Dropout(0.5), nn.Linear(hidden, n_classes))

    def forward(self, x, edge_index, batch, pi=None):
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        h = F.relu(self.conv3(h, edge_index))
        hg = global_mean_pool(h, batch)
        if self.use_pi:
            hg = torch.cat([hg, pi], dim=-1)
        return self.head(hg)


def run(name, use_pi, epochs=100, seed=1234, device='cuda'):
    torch.manual_seed(seed); np.random.seed(seed)
    ds = TUDataset(root=f'./data/TU_{name}', name=name)
    # node features: if none, use degree one-hot via constant
    if ds.num_node_features == 0:
        from torch_geometric.transforms import OneHotDegree
        import torch_geometric.transforms as T
        max_deg = max(int(d.edge_index.max()) for d in ds) if len(ds) else 1
        ds = TUDataset(root=f'./data/TU_{name}', name=name,
                       transform=T.OneHotDegree(max_degree=135))
    pis = compute_all_pi(name)  # (N, 25)
    labels = np.array([int(ds[i].y.item()) for i in range(len(ds))])
    in_dim = ds.num_node_features
    dev = torch.device(device if torch.cuda.is_available() else 'cpu')

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    accs = []
    for fold, (tr_idx, te_idx) in enumerate(skf.split(np.zeros(len(ds)), labels)):
        model = GINClassifier(in_dim, use_pi=use_pi,
                              n_classes=int(labels.max()) + 1).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        tr_loader = DataLoader([ds[i] for i in tr_idx], batch_size=32, shuffle=True)
        # map dataset index → pi row
        for ep in range(epochs):
            model.train()
            for batch in tr_loader:
                batch = batch.to(dev)
                # gather PIs for this batch's graphs by original index
                # DataLoader loses original idx; recompute via batch.ptr is complex,
                # so we pass PI per-graph using a parallel loader of indices.
                pass
        # Simpler: evaluate per-graph (batch_size=1) to keep PI alignment trivial
        accs.append(_train_eval_fold(ds, pis, labels, tr_idx, te_idx, in_dim,
                                      use_pi, epochs, dev))
        print(f'fold {fold}: acc={accs[-1]:.4f}')
    print(f'{name} use_pi={use_pi}: {np.mean(accs):.4f} ± {np.std(accs):.4f}')
    return np.mean(accs), np.std(accs)


def _train_eval_fold(ds, pis, labels, tr_idx, te_idx, in_dim, use_pi, epochs, dev):
    model = GINClassifier(in_dim, use_pi=use_pi, n_classes=int(labels.max()) + 1).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    pis_t = torch.tensor(pis, dtype=torch.float, device=dev)
    tr = [ds[i] for i in tr_idx]
    from torch_geometric.loader import DataLoader as DL
    # Attach original index to each graph for PI lookup
    for j, i in enumerate(tr_idx):
        tr[j].orig_idx = int(i)
    loader = DL(tr, batch_size=32, shuffle=True)
    for ep in range(epochs):
        model.train()
        for batch in loader:
            batch = batch.to(dev)
            pi = pis_t[batch.orig_idx] if use_pi else None
            opt.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch, pi)
            loss = F.cross_entropy(out, batch.y)
            loss.backward(); opt.step()
    # eval
    model.eval()
    te = [ds[i] for i in te_idx]
    for j, i in enumerate(te_idx):
        te[j].orig_idx = int(i)
    te_loader = DL(te, batch_size=64)
    correct = 0
    with torch.no_grad():
        for batch in te_loader:
            batch = batch.to(dev)
            pi = pis_t[batch.orig_idx] if use_pi else None
            pred = model(batch.x, batch.edge_index, batch.batch, pi).argmax(1)
            correct += (pred == batch.y).sum().item()
    return correct / len(te_idx)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='MUTAG')
    p.add_argument('--no_pi', action='store_true')
    p.add_argument('--epochs', type=int, default=100)
    args = p.parse_args()
    mean, std = run(args.dataset, use_pi=not args.no_pi, epochs=args.epochs)
    os.makedirs('scores', exist_ok=True)
    tag = 'noPI' if args.no_pi else 'withPI'
    with open(f'scores/mol_{args.dataset}_{tag}.txt', 'w') as f:
        f.write(f'{args.dataset} {tag} acc {mean:.4f} std {std:.4f}\n')
    print(f'saved scores/mol_{args.dataset}_{tag}.txt')
```

- [ ] **Step 2: Smoke test on MUTAG with reduced epochs**

```bash
cd /mnt/data/users/junyoungpark/code/TLC-GNN
source /mnt/data/users/junyoungpark/miniforge3/etc/profile.d/conda.sh && conda activate tlcgnn
export MPLCONFIGDIR=/tmp/mpl
python -m Knowledge_Distillation.mol_classify --dataset MUTAG --epochs 20 --no_pi 2>&1 | tail -5
```
Expected: 10 folds run, final line `MUTAG use_pi=False: 0.XX ± 0.XX`. Accuracy should be > 0.6 (better than chance).

If `OneHotDegree(max_degree=135)` errors (degree exceeds), adjust max_degree or use a different node feature. If MUTAG already has node features (it does — 7-dim one-hot atom type), the `num_node_features == 0` branch is skipped.

- [ ] **Step 3: Smoke test with PI**

```bash
python -m Knowledge_Distillation.mol_classify --dataset MUTAG --epochs 20 2>&1 | tail -5
```
Expected: `MUTAG use_pi=True: 0.XX ± 0.XX`. PI cache computed first (~1 min for 188 small graphs).

- [ ] **Step 4: Commit**

```bash
git add Knowledge_Distillation/mol_classify.py
git commit -m "mol GIN classifier"
```

### Task M.3: Full MUTAG/PROTEINS/NCI1 runs (100 epochs)

- [ ] **Step 1: Submit 6 SLURM jobs (3 datasets × with/no PI)**

```bash
cd /mnt/data/users/junyoungpark/code/TLC-GNN
for D in MUTAG PROTEINS NCI1; do
  for FLAG in "" "--no_pi"; do
    TAG=$([ -z "$FLAG" ] && echo withPI || echo noPI)
    cat > /tmp/mol_${D}_${TAG}.sh <<EOF
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=/mnt/data/users/junyoungpark/code/TLC-GNN/slurm_logs/%x-%j.out
#SBATCH --error=/mnt/data/users/junyoungpark/code/TLC-GNN/slurm_logs/%x-%j.err
source /mnt/data/users/junyoungpark/miniforge3/etc/profile.d/conda.sh
conda activate tlcgnn
export MPLCONFIGDIR=/tmp/mpl
cd /mnt/data/users/junyoungpark/code/TLC-GNN
python -m Knowledge_Distillation.mol_classify --dataset $D --epochs 100 $FLAG
EOF
    chmod +x /tmp/mol_${D}_${TAG}.sh
    sbatch --job-name=mol-${D}-${TAG} /tmp/mol_${D}_${TAG}.sh
  done
done
squeue -u $(whoami) | grep mol
```
Expected: 6 jobs queued.

- [ ] **Step 2: Wait + aggregate**

After all 6 complete:
```bash
cd /mnt/data/users/junyoungpark/code/TLC-GNN
echo "Dataset    withPI            noPI"
for D in MUTAG PROTEINS NCI1; do
  W=$(grep acc scores/mol_${D}_withPI.txt 2>/dev/null | awk '{print $4" ± "$6}')
  N=$(grep acc scores/mol_${D}_noPI.txt 2>/dev/null | awk '{print $4" ± "$6}')
  printf "%-10s %-16s %-16s\n" "$D" "${W:-?}" "${N:-?}"
done
```

- [ ] **Step 3: Commit**

```bash
git add scores/mol_*.txt
git commit -m "mol GC results"
```

---

## Part I — LP story extensions

### Task I.1: Cora + Citeseer homophilic anchor

**Files:** none (loaddatas already has Cora/Citeseer). SLURM + analysis only.

- [ ] **Step 1: Submit 6 SLURM jobs (Cora/Citeseer × TLC-GNN/no-PI/PDGNN)**

```bash
cd /mnt/data/users/junyoungpark/code/TLC-GNN
for D in Cora Citeseer; do
  sbatch --job-name=tlcgnn-${D,,}-rerun     slurm_run.sh --datasets $D --trials 50 --tag homo --dropout 0.8
  sbatch --job-name=tlcgnn-${D,,}-nopi       slurm_run.sh --datasets $D --trials 50 --tag homoNoPI --dropout 0.8 --no_pi
done
squeue -u $(whoami) | grep -iE "cora|citeseer"
```
(PDGNN for Cora/Citeseer requires PDGNN cache — skip PDGNN for these unless inference is run; TLC-GNN + no-PI is enough to populate the homophilic anchor.)

- [ ] **Step 2: After completion, re-run heterophily correlation with new points**

Edit `Knowledge_Distillation/heterophily_analysis.py`'s `cfg` dict to add:
```python
        'Cora':      ('homo', 'homoNoPI'),
        'Citeseer':  ('homo', 'homoNoPI'),
```
Then:
```bash
python -m Knowledge_Distillation.heterophily_analysis
```
Expected: Cora/Citeseer have high homophily (~0.8); with their points the Pearson r should turn negative (homophilic = PI helps = negative hurt).

- [ ] **Step 3: Commit**

```bash
git add Knowledge_Distillation/heterophily_analysis.py docs/figures/heterophily_correlation.png scores/pipe_benchmark_C*_LP_scoreshomo*.txt
git commit -m "homophilic anchor (cora/citeseer)"
```

### Task I.3: Negative cap sweep (paper gap diagnosis)

**Files:**
- Modify: `loaddatas.py` (`get_adj_split` — env-var cap)

- [ ] **Step 1: Edit `loaddatas.py` get_adj_split**

Find:
```python
    train_neg_cap = min(len(neg_edges), max(len(train_edges) * 5, 1024))
```
Replace with:
```python
    _cap_mult = os.environ.get('TLCGNN_NEG_CAP', '5')
    if _cap_mult == 'all':
        train_neg_cap = len(neg_edges)
    else:
        train_neg_cap = min(len(neg_edges), max(int(len(train_edges) * float(_cap_mult)), 1024))
```

- [ ] **Step 2: Submit cap sweep on PubMed (cap ∈ 1,5,20)**

```bash
cd /mnt/data/users/junyoungpark/code/TLC-GNN
for CAP in 1 20; do
  cat > /tmp/cap_${CAP}.sh <<EOF
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
export TLCGNN_NEG_CAP=${CAP}
# fresh PI cache per cap → use distinct tag and a per-cap cache dir override is NOT needed
# because cap changes the edge set; delete stale PubMed cache first to force recompute
rm -f data/TLCGNN/PubMed_cap${CAP}.npy
python pipelines.py --datasets PubMed --trials 50 --tag cap${CAP}
EOF
  chmod +x /tmp/cap_${CAP}.sh
  sbatch --job-name=cap-${CAP} /tmp/cap_${CAP}.sh
done
```
**NOTE:** changing the cap changes total_edges, so the cached `data/TLCGNN/PubMed.npy` (5× cap) won't match. The pipeline's splice logic handles cap < cached, but cap=20 needs MORE negatives than cached → recompute. cap=1 < 5 → splice works. Expect cap=20 to recompute PI (slow). Acceptable for diagnosis.

- [ ] **Step 3: Compare**

```bash
echo "cap  AUC"
for CAP in 1 5 20; do
  T=$([ $CAP -eq 5 ] && echo rerun || echo cap${CAP})
  F="scores/pipe_benchmark_PubMed_LP_scores${T}.txt"
  [ -f "$F" ] && awk -F'[, ]+' -v c=$CAP 'NR>1 && /^[0-9]+,/ {s+=$3;n++} END {if(n)printf "%-4s %.4f (n=%d)\n",c,s/n,n}' "$F"
done
echo "(paper PubMed = 0.9703; ours 5x = 0.9635)"
```

- [ ] **Step 4: Commit**

```bash
git add loaddatas.py scores/pipe_benchmark_PubMed_LP_scorescap*.txt
git commit -m "neg cap sweep"
```

### Task I.2: Sparsity-regularized gating

**Files:**
- Create: `baselines/TLCGNN_gated_reg.py`
- Modify: `pipelines.py` (`--gate_reg` flag)

- [ ] **Step 1: Create `baselines/TLCGNN_gated_reg.py`**

```python
# baselines/TLCGNN_gated_reg.py
"""Gating variant with an L1 sparsity penalty pushing gates toward 0, plus a
graph-level heterophily feature. Goal: break the gate=1 saturation seen in the
plain gating model so gates can learn to suppress PI on heterophilic graphs.

The sparsity penalty is exposed via .last_gate_penalty (mean gate over the last
decode call); the training loop adds GATE_REG_LAMBDA * penalty to the loss.
"""
from __future__ import annotations
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from baselines.TLCGNN_gated import GatingNet, gated_decode, Net as GatedNet

GATE_REG_LAMBDA = 0.1


class Net(GatedNet):
    """Same as TLCGNN_gated.Net but records mean-gate penalty for the training loop."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_gate_penalty = torch.tensor(0.0)

    def decode(self, data, emb, type='train'):
        # Replicate parent decode but capture gates for the penalty.
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
        else:
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
        self.last_gate_penalty = gates.mean()
        feats = gated_decode(sqdist, new_x, gates)
        feats = self.leakyrelu(self.linear_1(feats))
        feats = torch.abs(self.linear(feats)).reshape(-1)
        feats = torch.clamp(feats, min=0, max=40)
        prob = 1. / (torch.exp((feats - 2.0) / 1.0) + 1.0)
        return prob, edges_y.float()


def call(data, name, num_features, num_classes, data_cnt, use_pi: bool = True):
    from baselines.TLCGNN_gated import call as gated_call
    model, data = gated_call(data, name, num_features, num_classes, data_cnt, use_pi=use_pi)
    # Re-wrap as the reg variant (reuse PI + clustering)
    reg_model = Net(data, num_features, num_classes, PI=model.PI,
                    clustering=model.clustering).to(data.x.device)
    return reg_model, data
```

- [ ] **Step 2: Add `--gate_reg` to pipelines.py**

Find:
```python
from baselines import TLCGNN_gated as TLCGNN_gated
```
Add after:
```python
from baselines import TLCGNN_gated_reg as TLCGNN_gated_reg
```

Find the `if _args.use_gating:` block, replace with:
```python
if _args.gate_reg:
    pipelines=['TLCGNN_gated_reg']
elif _args.use_gating:
    pipelines=['TLCGNN_gated']
else:
    pipelines=['TLCGNN']
```

Add the flag near the other argparse lines:
```python
_parser.add_argument('--gate_reg', action='store_true',
                     help='sparsity-regularized gating (baselines.TLCGNN_gated_reg)')
```

Add the penalty to the loss in `train()`. Find:
```python
def train():
    model.train()
    optimizer.zero_grad()
    emb = model.encode(data)
    x, y = model.decode(data, emb)
    loss = F.binary_cross_entropy(x,y)
    loss.backward()
    optimizer.step()
    return x
```
Replace `loss = F.binary_cross_entropy(x,y)` with:
```python
    loss = F.binary_cross_entropy(x,y)
    if getattr(model, 'last_gate_penalty', None) is not None and hasattr(model, 'last_gate_penalty'):
        from baselines.TLCGNN_gated_reg import GATE_REG_LAMBDA
        loss = loss + GATE_REG_LAMBDA * model.last_gate_penalty
```

- [ ] **Step 3: Smoke test**

```bash
cd /mnt/data/users/junyoungpark/code/TLC-GNN
python pipelines.py --datasets Chameleon --trials 1 --tag gateRegSmoke --gate_reg 2>&1 | tail -3
```
Expected: runs, Test ROC printed.

- [ ] **Step 4: Submit 4-dataset gate_reg sweep**

```bash
for D in Photo Chameleon Texas ChChMiner; do
  sbatch --job-name=tlcgnn-${D,,}-gatereg slurm_run.sh --datasets $D --trials 50 --tag gateReg --gate_reg
done
```

- [ ] **Step 5: After completion, inspect gates (reuse Task 2.4 inspection script with TLCGNN_gated_reg)**

Compare gate means: do they now differ by domain (homo high, hetero low)?

- [ ] **Step 6: Commit**

```bash
git add baselines/TLCGNN_gated_reg.py pipelines.py scores/pipe_benchmark_*_LP_scoresgateReg.txt
git commit -m "sparsity gating"
```

### Task I.4: OGBL-DDI subsample [OPTIONAL — drop if time short]

**Files:**
- Modify: `loaddatas.py` (OGBL-DDI subsample loader)

- [ ] **Step 1: Add loader to loaddatas.py**

Find the SBM dispatch:
```python
    elif d_name.startswith('SBM_'):
        dataset = _load_sbm(d_name)
    return dataset
```
Replace with:
```python
    elif d_name.startswith('SBM_'):
        dataset = _load_sbm(d_name)
    elif d_name == 'OGBL_DDI_sub':
        dataset = _load_ogbl_ddi_sub()
    return dataset
```

Add function:
```python
def _load_ogbl_ddi_sub(n_nodes: int = 3000, seed: int = 1234):
    """OGBL-DDI subsampled to n_nodes (induced subgraph) for feasible PI compute."""
    from ogb.linkproppred import LinkPropPredDataset
    ds = LinkPropPredDataset(name='ogbl-ddi', root='./data')
    graph = ds[0]
    ei_full = graph['edge_index']  # (2, E)
    N = graph['num_nodes']
    rng = np.random.RandomState(seed)
    keep_nodes = set(rng.choice(N, size=min(n_nodes, N), replace=False).tolist())
    mask = np.array([(int(u) in keep_nodes and int(v) in keep_nodes)
                     for u, v in zip(ei_full[0], ei_full[1])])
    sub_ei = ei_full[:, mask]
    # relabel nodes 0..n-1
    nodes = sorted(keep_nodes)
    remap = {n: i for i, n in enumerate(nodes)}
    re_ei = np.array([[remap[int(u)] for u in sub_ei[0]],
                      [remap[int(v)] for v in sub_ei[1]]], dtype=np.int64)
    n = len(nodes)
    x = torch.eye(n)
    y = torch.zeros(n, dtype=torch.long)
    data = Data(x=x, edge_index=torch.from_numpy(re_ei).long(), y=y)

    class _FakeDataset:
        def __init__(self, data, name):
            self._data = [data]; self.name = name; self.num_classes = 2
        def __getitem__(self, i): return self._data[i]
        def __len__(self): return 1
    return _FakeDataset(data, 'OGBL_DDI_sub')
```

- [ ] **Step 2: Smoke + submit**

```bash
cd /mnt/data/users/junyoungpark/code/TLC-GNN
python -c "import loaddatas as lds; d=lds.loaddatas('OGBL_DDI_sub')[0]; print('nodes', d.num_nodes, 'edges', d.edge_index.size(1)//2)"
sbatch --job-name=tlcgnn-ddisub slurm_run.sh --datasets OGBL_DDI_sub --trials 50 --tag ddisub
sbatch --job-name=tlcgnn-ddisub-nopi slurm_run.sh --datasets OGBL_DDI_sub --trials 50 --tag ddisubNoPI --no_pi
```

- [ ] **Step 3: Commit**

```bash
git add loaddatas.py
git commit -m "ogbl-ddi subsample"
```

---

## Part F — Finalize (after M + I complete)

### Task F.1: Update results doc + slides + README + push

- [ ] **Step 1: Add molecular GC section + extension results to `docs/specs/2026-06-21-tda-conference-results.md`**

Add a "## 7. Molecular Graph Classification (GC)" section with the MUTAG/PROTEINS/NCI1 with-PI vs no-PI table, and update the heterophily correlation (now with homophilic anchor) and cap sweep findings.

- [ ] **Step 2: Add a slide on molecular GC to `slides/tda-conference.md`** (before the 전망 slide): table + one-line finding.

- [ ] **Step 3: Update README results section** with molecular GC + corrected hetero PDGNN (neural) numbers.

- [ ] **Step 4: Commit + push**

```bash
git add -A
git commit -m "molecular gc + extensions final"
git push tda main
```

---

## Self-Review Checklist

- [x] **Spec coverage:** Part M (M.1-M.3) ✓, I.1 Cora/Citeseer ✓, I.2 sparsity gating ✓, I.3 cap sweep ✓, I.4 OGBL-DDI optional ✓, finalize F.1 ✓.
- [x] **Placeholder scan:** `0.XX` in M.2 smoke are expected-output templates, not code placeholders. No TODO in code.
- [x] **Type consistency:** `compute_all_pi` returns (N,25) used by mol_classify; `graph_to_pi` returns (25,); GINClassifier head_in = hidden + pi_dim consistent. `TLCGNN_gated_reg.Net` extends `TLCGNN_gated.Net` with matching decode signature.
- [x] **Conflict mgmt:** loaddatas.py edits (I.3 cap, I.4 OGBL) serialized; everything else new files.

## Known risks

| Risk | Mitigation |
|---|---|
| mol_classify PI-batch alignment bug (orig_idx) | M.2 smoke catches; per-graph orig_idx attached before DataLoader |
| MUTAG too small → high CV variance | report std; add PROTEINS/NCI1 for robustness |
| Sparsity gating still saturates (λ too small) | try λ ∈ {0.1, 1.0}; honest negative acceptable |
| cap=20 PubMed recompute slow (no cache) | accept; it's a one-off diagnosis |
| OGBL-DDI subsample PI cost | n_nodes=3000 keeps it feasible; optional anyway |
| TUDataset OneHotDegree max_degree | MUTAG/PROTEINS/NCI1 have node features already, branch skipped |

## Time estimate
- Part M: 2-3일 (코드 1일 + 6 SLURM jobs)
- Part I: 2-3일 (I.1/I.3 빠름, I.2 중간, I.4 optional)
- Finalize: 0.5일
- **Total: ~1주** (4주 deadline 내)
