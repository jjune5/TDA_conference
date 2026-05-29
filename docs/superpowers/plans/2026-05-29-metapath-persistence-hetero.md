# Meta-path-induced Persistence for Heterogeneous Node Classification — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 이종 그래프 node classification에서 meta-path 유도 동종 그래프의 persistent homology(PI) feature가 genuine 신호를 주는지 exact-first로 검증하고(Phase 1), PDGNN으로 ogbn-mag에 확장(Phase 2).

**Architecture:** HGBDataset(ACM/DBLP) → meta-path별 weighted 동종 그래프(sparse matmul) → 각 target 노드의 ego-graph exact PH → PI(25d) → GCN 백본 임베딩에 concat → node classification. 3-way(plain/+PH/controls) 비교. PH 코드는 DNP의 `node_ph_features.py`를 재사용, PDGNN은 `Knowledge_Distillation/pdgnn_modern.py` 재사용. 전부 additive(`hetero/` 신규 디렉토리), 기존 코드 수정 없음.

**Tech Stack:** Python `tlcgnn` env (PyG 2.5.3 HGBDataset/OGB_MAG, gudhi 3.11, scipy sparse, networkx, torch), 재사용: `node_ph_features.py`, `sg2dgm.PersistenceImager`, `Knowledge_Distillation/pdgnn_modern.py`.

**Env (every command):**
```bash
source /mnt/data/users/junyoungpark/miniforge3/etc/profile.d/conda.sh
conda activate tlcgnn
cd /mnt/data/users/junyoungpark/code/TLC-GNN
```

**Branch:** main에서 직접 작업(이 프로젝트의 확립된 워크플로우; worktree는 [[worktree-stale-base-gotcha]] 때문에 회피).

**Verified grounding (non-phantom):** ACM HeteroData = node types [paper, author, subject, term]; target=`paper` (n=3025, x=1902d, y=3 classes, train_mask=907); edge types 포함 `('paper','to','author')`,`('author','to','paper')`,`('paper','to','subject')`,`('subject','to','paper')`,`('paper','cite','paper')`. → meta-path PAP=[(paper,to,author),(author,to,paper)], PSP=[(paper,to,subject),(subject,to,paper)].

---

## File Structure

- **Create `hetero/__init__.py`** — 빈 패키지 마커.
- **Create `hetero/metapath_graph.py`** — HGBDataset 로드 + meta-path weighted 동종 그래프 빌더(scipy sparse matmul) + networkx 변환. 단일 책임: hetero→homo meta-path 그래프.
- **Create `hetero/leakage_audit.py`** — meta-path 그래프 구조만으로 라벨을 얼마나 맞히나(label-propagation) 측정 = §14식 누수 감사.
- **Create `hetero/metapath_ph.py`** — meta-path 그래프 → 각 target 노드 exact PI(25d) 행렬. `node_ph_features._ego_sublevel_pi`/`_diagram_points` 재사용.
- **Create `hetero/hetero_nc_pipeline.py`** — GCN 백본 node classification 드라이버: 3-way(no-PH / +PH / shuffled-PH control) + random-filter control. ACM/DBLP.
- **Create `tests/test_hetero_metapath.py`** — 위 모듈 단위/스모크 테스트.
- **Reuse (수정 금지):** `node_ph_features.py`(repo root), `sg2dgm/`, `Knowledge_Distillation/pdgnn_modern.py`.

---

## Phase 1 — meta-path exact PH + 누수감사 + 3-way (ACM/DBLP)

### Task 1: meta-path weighted 동종 그래프 빌더

**Files:**
- Create: `hetero/__init__.py` (빈 파일)
- Create: `hetero/metapath_graph.py`
- Test: `tests/test_hetero_metapath.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_hetero_metapath.py
import os, sys
import numpy as np
import networkx as nx
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_metapath_adjacency_counts_cooccurrence():
    """PAP 류 meta-path: A1 @ A2 가 공동출현 횟수(가중치)를 준다."""
    import scipy.sparse as sp
    from hetero.metapath_graph import compose_metapath_adj
    # paper(2) - author(2): p0-a0, p0-a1, p1-a1
    pa = sp.csr_matrix(np.array([[1, 1], [0, 1]], dtype=np.float64))  # paper x author
    ap = pa.T.tocsr()                                                 # author x paper
    W = compose_metapath_adj([pa, ap])                                # paper x paper
    W = np.asarray(W.todense())
    # p0,p1 share author a1 -> off-diagonal = 1
    assert W[0, 1] == 1 and W[1, 0] == 1
    # diagonal = #authors per paper (p0 has 2) -> will be zeroed by builder later
    assert W[0, 0] == 2 and W[1, 1] == 1


def test_build_metapath_graph_acm_pap_smoke():
    """ACM PAP meta-path -> paper 동종 weighted nx graph (no self loops)."""
    from hetero.metapath_graph import load_hgb, build_metapath_graph
    d = load_hgb('ACM')
    g, y, masks = build_metapath_graph(d, 'PAP')
    assert g.number_of_nodes() == int(d['paper'].num_nodes)   # 3025
    assert g.number_of_edges() > 0
    assert not any(u == v for u, v in g.edges())              # diagonal removed
    # weights are positive integers (co-authored paper counts)
    w = [dd['weight'] for _, _, dd in g.edges(data=True)]
    assert min(w) >= 1
    assert y.shape[0] == 3025 and int(y.max()) == 2           # 3 classes
    assert masks['train'].sum() > 0 and masks['test'].sum() > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_hetero_metapath.py -x -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'hetero'`

- [ ] **Step 3: Write minimal implementation**

```python
# hetero/__init__.py
```
(빈 파일)

```python
# hetero/metapath_graph.py
"""Heterogeneous graph -> meta-path induced weighted homogeneous graph.

A meta-path P = e1 e2 ... ek (a sequence of relation types that starts and ends
at the TARGET node type) induces a homogeneous graph over the target nodes whose
weighted adjacency is the product of the per-relation adjacency matrices
(A_e1 @ A_e2 @ ... ). Off-diagonal entry (i,j) = number of meta-path instances
connecting target nodes i and j (= relation strength). The diagonal (self
co-occurrence count) is removed. This is the hetero->homo step (CS224W lec09).
"""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
import networkx as nx
import torch

# canonical meta-paths per dataset (target node type first/last). Each entry is a
# list of edge_type triples in HGBDataset naming.
METAPATHS = {
    'ACM': {
        'PAP': [('paper', 'to', 'author'), ('author', 'to', 'paper')],
        'PSP': [('paper', 'to', 'subject'), ('subject', 'to', 'paper')],  # leak-audit
    },
    'DBLP': {
        # target = author
        'APA': [('author', 'to', 'paper'), ('paper', 'to', 'author')],
        'APCPA': [('author', 'to', 'paper'), ('paper', 'to', 'term'),
                  ('term', 'to', 'paper'), ('paper', 'to', 'author')],  # leak-audit
    },
}
TARGET = {'ACM': 'paper', 'DBLP': 'author'}


def load_hgb(name: str):
    from torch_geometric.datasets import HGBDataset
    return HGBDataset(root=f'./data/HGB_{name}', name=name)[0]


def _edge_adj(d, etype) -> sp.csr_matrix:
    """Sparse adjacency (src_count x dst_count) for one edge type."""
    src, _, dst = etype
    ei = d[etype].edge_index.numpy()
    n_src = int(d[src].num_nodes)
    n_dst = int(d[dst].num_nodes)
    data = np.ones(ei.shape[1], dtype=np.float64)
    return sp.csr_matrix((data, (ei[0], ei[1])), shape=(n_src, n_dst))


def compose_metapath_adj(adjs):
    """Multiply a list of sparse adjacencies left-to-right -> target x target."""
    W = adjs[0]
    for A in adjs[1:]:
        W = W @ A
    return W.tocsr()


def build_metapath_graph(d, metapath_name: str):
    """Return (nx weighted graph over target nodes, y (N,), masks dict)."""
    # infer dataset name from which METAPATHS table contains metapath_name
    ds_name = next(k for k, v in METAPATHS.items() if metapath_name in v)
    etypes = METAPATHS[ds_name][metapath_name]
    adjs = [_edge_adj(d, et) for et in etypes]
    W = compose_metapath_adj(adjs)
    W = W.tocoo()
    tgt = TARGET[ds_name]
    n = int(d[tgt].num_nodes)
    g = nx.Graph()
    g.add_nodes_from(range(n))
    for i, j, w in zip(W.row, W.col, W.data):
        if i < j and w > 0:                  # upper triangle, drop diagonal
            g.add_edge(int(i), int(j), weight=float(w))
    y = d[tgt].y.numpy()
    masks = {k: getattr(d[tgt], f'{k}_mask').numpy()
             for k in ('train', 'val', 'test') if hasattr(d[tgt], f'{k}_mask')}
    # HGB ACM has train/test masks; synthesize val from train tail if absent
    if 'val' not in masks:
        tr = masks['train'].copy()
        idx = np.where(tr)[0]
        cut = idx[int(0.85 * len(idx)):]
        masks['val'] = np.zeros_like(tr); masks['val'][cut] = True
        masks['train'][cut] = False
    return g, y, masks
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_hetero_metapath.py -x -q`
Expected: PASS (2 passed). ACM 첫 다운로드로 1–2분 걸릴 수 있음.

- [ ] **Step 5: Commit**

```bash
git add hetero/__init__.py hetero/metapath_graph.py tests/test_hetero_metapath.py
git commit -m "feat(hetero): meta-path induced weighted homogeneous graph builder (ACM/DBLP)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: 라벨 누수 감사 (§14 교훈)

**Files:**
- Create: `hetero/leakage_audit.py`
- Test: `tests/test_hetero_metapath.py` (add `test_leakage_audit_runs`)

- [ ] **Step 1: Write the failing test**

```python
def test_leakage_audit_runs():
    """meta-path 그래프 구조만으로 라벨 예측(LP) 정확도를 반환한다."""
    from hetero.metapath_graph import load_hgb, build_metapath_graph
    from hetero.leakage_audit import structure_only_label_acc
    d = load_hgb('ACM')
    g, y, masks = build_metapath_graph(d, 'PAP')
    acc = structure_only_label_acc(g, y, masks)
    assert 0.0 <= acc <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_hetero_metapath.py::test_leakage_audit_runs -x -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'hetero.leakage_audit'`

- [ ] **Step 3: Write minimal implementation**

```python
# hetero/leakage_audit.py
"""§14-style leakage audit for meta-path graphs.

If a meta-path routes through a label-defining node type, the meta-path graph's
STRUCTURE ALONE can predict the label -> the topological feature built on it would
be a label-membership artifact, not genuine topology. We probe this with simple
weighted majority-vote label propagation from train labels: test accuracy from
STRUCTURE ALONE (no node features). Suspiciously high acc => leak flag.
"""
from __future__ import annotations
import numpy as np


def structure_only_label_acc(g, y, masks, n_iter: int = 10) -> float:
    """Weighted majority-vote label propagation on the meta-path graph using only
    train labels; return TEST accuracy (structure-only, no features)."""
    n = g.number_of_nodes()
    C = int(y.max()) + 1
    train, test = masks['train'], masks['test']
    # one-hot label distribution, fixed on train nodes
    P = np.zeros((n, C), dtype=np.float64)
    P[train, y[train]] = 1.0
    fixed = train.copy()
    # neighbor lists with weights
    nbrs = {u: list(g[u].items()) for u in g.nodes()}
    for _ in range(n_iter):
        Pn = P.copy()
        for u in range(n):
            if fixed[u]:
                continue
            acc = np.zeros(C)
            for v, dd in nbrs[u]:
                acc += dd['weight'] * P[v]
            if acc.sum() > 0:
                Pn[u] = acc / acc.sum()
        P = Pn
    pred = P.argmax(1)
    # nodes with no propagated signal -> predict train majority
    nosig = (P.sum(1) == 0)
    if nosig.any():
        pred[nosig] = np.bincount(y[train]).argmax()
    return float((pred[test] == y[test]).mean())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_hetero_metapath.py::test_leakage_audit_runs -x -q`
Expected: PASS.

- [ ] **Step 5: Run the actual audit on ACM PAP vs PSP and record**

Run:
```bash
python - <<'PY'
from hetero.metapath_graph import load_hgb, build_metapath_graph
from hetero.leakage_audit import structure_only_label_acc
d = load_hgb('ACM')
for mp in ['PAP', 'PSP']:
    g, y, masks = build_metapath_graph(d, mp)
    acc = structure_only_label_acc(g, y, masks)
    print(f'ACM {mp}: structure-only test acc = {acc:.4f}  (edges={g.number_of_edges()})')
PY
```
Expected: 두 줄 출력. **해석 기준**: PSP(라벨-인접 subject 경유) acc가 PAP보다 현저히 높거나 ~1.0이면 **누수 의심 → PSP를 clean 비교에서 제외**(leaky 대조로만 사용). 결과를 커밋 메시지에 기록.

- [ ] **Step 6: Commit**

```bash
git add hetero/leakage_audit.py tests/test_hetero_metapath.py
git commit -m "feat(hetero): §14-style label-leakage audit (structure-only LP acc)

ACM PAP vs PSP audit: <PAP acc> vs <PSP acc> -> <clean / leaky 판정 기록>

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: meta-path 그래프 per-node exact PI

**Files:**
- Create: `hetero/metapath_ph.py`
- Test: `tests/test_hetero_metapath.py` (add `test_metapath_pi_shape`)

- [ ] **Step 1: Write the failing test**

```python
def test_metapath_pi_shape():
    """각 target 노드 -> (25,) PI. 반환 (N,25), 유한, 일부 nonzero."""
    from hetero.metapath_graph import load_hgb, build_metapath_graph
    from hetero.metapath_ph import metapath_node_pi
    d = load_hgb('ACM')
    g, y, masks = build_metapath_graph(d, 'PAP')
    pi = metapath_node_pi(g, hop=1, max_nodes=200, verbose=False)
    assert pi.shape == (g.number_of_nodes(), 25)
    assert np.isfinite(pi).all()
    assert pi.sum() > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_hetero_metapath.py::test_metapath_pi_shape -x -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'hetero.metapath_ph'`

- [ ] **Step 3: Write minimal implementation** (DNP의 `node_ph_features` 재사용)

```python
# hetero/metapath_ph.py
"""Per-target-node exact persistence image on a meta-path graph.

For each target node v: take its k-hop ego-graph in the (weighted) meta-path
graph, use weighted node degree as the sublevel filter, compute exact extended
persistence (gudhi) and vectorize to a 5x5=25-dim persistence image. Reuses the
DNP machinery in node_ph_features.py (_ego_sublevel_pi, PI_RES). Exact-first:
prove the signal is genuine before approximating with PDGNN (Phase 2).
"""
from __future__ import annotations
import os, sys
import numpy as np
import networkx as nx
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from node_ph_features import _ego_sublevel_pi, PI_RES   # reuse DNP helpers
from sg2dgm import PersistenceImager as pimg_mod


def metapath_node_pi(g: nx.Graph, hop: int = 1, max_nodes: int = 200,
                     verbose: bool = False) -> np.ndarray:
    """(N, 25) per-node exact PI on the meta-path graph; filter = weighted degree."""
    imager = pimg_mod.PersistenceImager(resolution=PI_RES)
    n = g.number_of_nodes()
    # weighted degree as the node filter value (sum of incident meta-path counts)
    wdeg = {u: float(sum(dd['weight'] for _, dd in g[u].items())) for u in g.nodes()}
    out = np.zeros((n, PI_RES * PI_RES), dtype=np.float64)
    for v in range(n):
        out[v] = _ego_sublevel_pi(g, v, hop, wdeg, imager, max_nodes)
        if verbose and (v + 1) % 1000 == 0:
            print(f'    metapath_pi {v+1}/{n}')
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_hetero_metapath.py::test_metapath_pi_shape -x -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add hetero/metapath_ph.py tests/test_hetero_metapath.py
git commit -m "feat(hetero): per-node exact PI on meta-path graph (reuse DNP node_ph_features)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: node classification 파이프라인 (3-way + controls)

**Files:**
- Create: `hetero/hetero_nc_pipeline.py`
- (no new test; smoke via CLI in Task 5)

- [ ] **Step 1: Write the driver**

```python
# hetero/hetero_nc_pipeline.py
"""Heterogeneous node classification: does meta-path PH add genuine signal?

Backbone: 2-layer GCN on the meta-path homogeneous graph of target nodes, node
features = target node features. Variants compared on the SAME backbone:
  - 'none'     : GCN only (no PH)
  - 'ph'       : GCN + per-node meta-path PI concatenated to node features
  - 'shuffled' : GCN + PI rows randomly permuted (control: is PH genuine?)
  - 'random'   : GCN + PI from a RANDOM node filter (control: does structure matter?)
Multiple meta-paths -> concat their PIs. Reports test accuracy per variant.
"""
from __future__ import annotations
import os, sys, json, csv, time, argparse
import numpy as np
import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hetero.metapath_graph import load_hgb, build_metapath_graph, METAPATHS, TARGET
from hetero.metapath_ph import metapath_node_pi
from hetero.leakage_audit import structure_only_label_acc
from node_ph_features import _ego_sublevel_pi, PI_RES
from sg2dgm import PersistenceImager as pimg_mod

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
HIDDEN, DROPOUT, LR, EPOCHS = 64, 0.5, 0.01, 200


def _graph_to_edge_index(g, n):
    ei = np.array(list(g.edges())).T if g.number_of_edges() else np.zeros((2, 0), int)
    if ei.size:
        ei = np.concatenate([ei, ei[[1, 0]]], axis=1)   # symmetric
    return torch.tensor(ei, dtype=torch.long, device=device)


def _random_filter_pi(g, hop, max_nodes, seed=0):
    """Control: PI from a RANDOM node filter (structure used, filter meaningless)."""
    rng = np.random.RandomState(seed)
    imager = pimg_mod.PersistenceImager(resolution=PI_RES)
    n = g.number_of_nodes()
    rfilt = {u: float(rng.rand()) for u in g.nodes()}
    out = np.zeros((n, PI_RES * PI_RES))
    for v in range(n):
        out[v] = _ego_sublevel_pi(g, v, hop, rfilt, imager, max_nodes)
    return out


class GCNNet(torch.nn.Module):
    def __init__(self, in_dim, n_cls):
        super().__init__()
        self.c1 = GCNConv(in_dim, HIDDEN, cached=True)
        self.c2 = GCNConv(HIDDEN, n_cls, cached=True)

    def forward(self, x, ei):
        x = F.dropout(x, DROPOUT, self.training)
        x = F.relu(self.c1(x, ei))
        x = F.dropout(x, DROPOUT, self.training)
        return self.c2(x, ei)


def run_variant(x, ei, y, masks, n_cls, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = GCNNet(x.size(1), n_cls).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)
    yt = torch.tensor(y, dtype=torch.long, device=device)
    tr = torch.tensor(masks['train'], device=device)
    va = torch.tensor(masks['val'], device=device)
    te = torch.tensor(masks['test'], device=device)
    best_va, best_te = 0.0, 0.0
    for _ in range(EPOCHS):
        model.train(); opt.zero_grad()
        out = model(x, ei)
        loss = F.cross_entropy(out[tr], yt[tr])
        loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(x, ei).argmax(1)
            va_acc = float((pred[va] == yt[va]).float().mean())
            if va_acc >= best_va:
                best_va = va_acc
                best_te = float((pred[te] == yt[te]).float().mean())
    return best_te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='ACM')
    ap.add_argument('--metapaths', nargs='+', default=None,
                    help='default: all non-leaky for the dataset')
    ap.add_argument('--hop', type=int, default=1)
    ap.add_argument('--max_nodes', type=int, default=200)
    ap.add_argument('--trials', type=int, default=10)
    ap.add_argument('--outdir', default=None)
    args = ap.parse_args()
    outdir = args.outdir or f'results/hetero_{args.dataset}'
    os.makedirs(outdir, exist_ok=True)
    d = load_hgb(args.dataset)
    mps = args.metapaths or list(METAPATHS[args.dataset].keys())
    tgt = TARGET[args.dataset]
    x_feat = d[tgt].x.numpy().astype(np.float64)
    print(f'{args.dataset} target={tgt} x={x_feat.shape} metapaths={mps}')

    # build the (shared) backbone graph from the FIRST metapath; PH from all metapaths
    g0, y, masks = build_metapath_graph(d, mps[0])
    n = g0.number_of_nodes()
    ei = _graph_to_edge_index(g0, n)
    n_cls = int(y.max()) + 1

    # leakage audit per metapath (record, don't silently use leaky ones)
    audit = {}
    pis, pis_rand = [], []
    for mp in mps:
        g, _, _ = build_metapath_graph(d, mp)
        audit[mp] = structure_only_label_acc(g, y, masks)
        print(f'  leakage audit {mp}: structure-only acc={audit[mp]:.4f}')
        pis.append(metapath_node_pi(g, args.hop, args.max_nodes))
        pis_rand.append(_random_filter_pi(g, args.hop, args.max_nodes))
    PI = np.concatenate(pis, axis=1)
    PI_rand = np.concatenate(pis_rand, axis=1)
    rng = np.random.RandomState(0)
    PI_shuf = PI[rng.permutation(n)]

    def feats(extra):
        if extra is None:
            return torch.tensor(x_feat, dtype=torch.float32, device=device)
        return torch.tensor(np.concatenate([x_feat, extra], 1),
                            dtype=torch.float32, device=device)

    variants = {'none': None, 'ph': PI, 'shuffled': PI_shuf, 'random': PI_rand}
    results = {}
    for name, extra in variants.items():
        x = feats(extra)
        accs = [run_variant(x, ei, y, masks, n_cls, s) for s in range(args.trials)]
        results[name] = accs
        print(f'  [{name:8}] test acc = {np.mean(accs):.4f} ± {np.std(accs):.4f}')

    with open(os.path.join(outdir, 'nc_acc.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['dataset', 'variant', 'mean_acc', 'std_acc', 'n'])
        for name, accs in results.items():
            a = np.array(accs)
            w.writerow([args.dataset, name, f'{a.mean():.6f}', f'{a.std():.6f}', len(a)])
    with open(os.path.join(outdir, 'audit.json'), 'w') as f:
        json.dump(audit, f, indent=2)
    print(f'Outputs -> {outdir}/')


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Commit**

```bash
git add hetero/hetero_nc_pipeline.py
git commit -m "feat(hetero): node classification pipeline (3-way + shuffled/random controls)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Phase-1 GATE run (ACM, then DBLP)

**Files:** none (run + record)

- [ ] **Step 1: Smoke (G1) — ACM PAP only, 2 trials**

Run:
```bash
python -m hetero.hetero_nc_pipeline --dataset ACM --metapaths PAP --trials 2 --outdir results/hetero_smoke
```
Expected: leakage audit 줄 1개, 4개 variant(none/ph/shuffled/random) acc 줄, `results/hetero_smoke/nc_acc.csv` 생성. 예외 없음.

- [ ] **Step 2: Full gate (G2) — ACM (non-leaky metapaths, 10 trials)**

Run:
```bash
python -m hetero.hetero_nc_pipeline --dataset ACM --trials 10 > results/hetero_ACM.log 2>&1
column -s, -t results/hetero_ACM/nc_acc.csv; cat results/hetero_ACM/audit.json
```
**GATE 판정 (커밋 메시지에 기록):**
- **PASS** iff `ph` 평균 > `none` 평균 (유의) **AND** `ph` > `shuffled` (genuine, 노드↔PI 대응이 의미있음) **AND** `ph` ≥ `random` (구조 filter가 random filter보다 나음).
- **누수 처리**: audit.json에서 어떤 metapath가 structure-only acc ≫ baseline이면 그 metapath는 leaky로 표시하고 clean 비교에서 제외(재실행 `--metapaths`로 clean만).
- **null이어도 finding**: `ph ≈ none`이면 "ACM 포화(~93%)에서 위상 무신호" 로 정직 보고 → Phase 2(ogbn-mag 헤드룸)로 직행 근거.

- [ ] **Step 3: DBLP 반복**

Run:
```bash
python -m hetero.hetero_nc_pipeline --dataset DBLP --trials 10 > results/hetero_DBLP.log 2>&1
column -s, -t results/hetero_DBLP/nc_acc.csv; cat results/hetero_DBLP/audit.json
```
Expected: ACM과 동일 형식. APCPA는 누수 감사 결과 보고 leaky면 제외.

- [ ] **Step 4: Commit Phase-1 결과 + 판정**

```bash
git add results/hetero_ACM/ results/hetero_DBLP/ results/hetero_*.log
git commit -m "exp(hetero): Phase-1 gate — meta-path PH 3-way+controls on ACM/DBLP

<ACM/DBLP에서 ph vs none vs shuffled vs random, leakage audit 결과 한 줄 요약>

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Phase 2 — PDGNN으로 ogbn-mag 확장 (골격; Phase 1 PASS 후 상세화)

**목표:** ogbn-mag(1.9M 노드, 헤드룸 ~58%)의 meta-path 그래프는 dense → exact PH 불가 → **PDGNN(`Knowledge_Distillation/pdgnn_modern.py`)으로 EPD 근사**. **Simple-HGN 능가가 contribution gate.**

골격 (Phase 1 PASS 후 bite-sized로 확장):
- **T6.1:** `hetero/metapath_graph.py`에 OGB_MAG 로더 + PAP/PP meta-path 추가 (sparse matmul은 대형이므로 메모리/blocking 처리).
- **T6.2:** `hetero/pdgnn_metapath.py` — meta-path 그래프 위에서 PDGNN으로 per-node EPD 예측. PDGNN을 meta-path 그래프 분포에서 (재)학습: exact EPD(작은 subgraph 샘플)로 supervision → `train_pdgnn_lp._bipartite_loss` 패턴(Wasserstein/Hungarian) 재사용.
- **T6.3:** ogbn-mag node classification: PDGNN-PI feature를 백본에 concat, Simple-HGN baseline과 비교 (year split: train≤2017/val2018/test2019).
- **T6.4:** 결과 → results doc 새 챕터 + push. 신규성 framing(meta-path PH for HIN, HTGNN 인용·구분, HL-HGAT disambiguate).

---

## Self-Review

**1. Spec coverage:**
- spec "Idea 1 meta-path 그래프 (A₁×A₂)" → Task 1 (`compose_metapath_adj`). ✓
- spec "weighted filtration (인스턴스 수)" → Task 1 weight + Task 3 weighted-degree filter. ✓
- spec "라벨 누수 감사" → Task 2 (`structure_only_label_acc`) + Task 5 판정. ✓
- spec "exact-first EPD/PI" → Task 3 (gudhi, no PDGNN). ✓
- spec "3-way + controls(random/shuffled)" → Task 4. ✓
- spec "ACM/DBLP 검증, Simple-HGN 비교" → Task 5 (3-way; Simple-HGN 전체 비교는 Phase 2/contribution으로 — Phase 1은 동일 백본 +PH 신호 검증에 집중, 정직히 명시). ✓ (부분: Simple-HGN 강baseline은 Phase 2)
- spec "PDGNN ogbn-mag" → Phase 2 골격 T6.*. ✓
- spec "additive only, hetero/" → 모든 신규 파일 hetero/, 기존 수정 0. ✓

**2. Placeholder scan:** Phase 1 모든 step에 완전 코드/명령/예상출력. Phase 2는 의도적 골격(Phase 1 게이트 후 상세화, active path 아님). ✓

**3. Type consistency:** `build_metapath_graph` → (nx.Graph, y(np), masks dict); `metapath_node_pi(g,...)`→(N,25); `_ego_sublevel_pi`/`PI_RES` = node_ph_features 실제 시그니처(검증됨); `structure_only_label_acc(g,y,masks)`; 파이프라인이 이들 그대로 소비. METAPATHS/TARGET dict 일관. ✓

**Gap (정직):** Phase 1의 "baseline"은 *동일 GCN 백본 ± PH*라서 spec의 "Simple-HGN 능가"를 Phase 1에선 직접 안 함 — Phase 1은 "PH가 genuine 신호를 *추가*하나"만 깨끗이 격리 검증, Simple-HGN 강비교는 Phase 2(기여)로 미룸. 의도된 분리.

---

## Execution Handoff

Plan saved to `docs/superpowers/plans/2026-05-29-metapath-persistence-hetero.md`. 사용자는 executing-plans(인라인)를 요청함. Phase 1이 critical path (exact-first gate). Phase 1 PASS 후 Phase 2(PDGNN) 상세화.
