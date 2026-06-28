"""Toy heterogeneous graph + type-constrained link-prediction data utilities.

Node types : author, paper, field
Edge types : (author, writes, paper), (paper, written_by, author),
             (paper, has_topic, field), (field, rev_has_topic, paper),
             (paper, cites, paper)

All functions are deterministic given a seed. Nothing here mutates the input
HeteroData in place.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch_geometric.data import HeteroData

NodeType = str
EdgeType = Tuple[str, str, str]

NODE_TYPES: Tuple[NodeType, ...] = ("author", "paper", "field")
EDGE_TYPES: Tuple[EdgeType, ...] = (
    ("author", "writes", "paper"),
    ("paper", "written_by", "author"),
    ("paper", "has_topic", "field"),
    ("field", "rev_has_topic", "paper"),
    ("paper", "cites", "paper"),
)
DEFAULT_FEAT_DIMS: Dict[NodeType, int] = {"author": 16, "paper": 12, "field": 8}


@dataclass
class LinkSplit:
    """Disjoint positive target-edge splits (each a (2, n) LongTensor)."""
    train_pos: torch.Tensor
    val_pos: torch.Tensor
    test_pos: torch.Tensor


@dataclass
class PairBatch:
    """A batch of candidate (src, dst) pairs with binary labels and endpoint types."""
    src: torch.Tensor      # (B,) long
    dst: torch.Tensor      # (B,) long
    label: torch.Tensor    # (B,) float in {0,1}
    src_type: str
    dst_type: str


# --------------------------------------------------------------------------- #
def make_toy_hetero_graph(
    seed: int = 0,
    n_author: int = 30,
    n_paper: int = 60,
    n_field: int = 8,
    feat_dims: Optional[Dict[NodeType, int]] = None,
    planted: bool = False,
    n_comm: Optional[int] = None,
) -> HeteroData:
    """Reproducible toy HeteroData with all 5 edge types non-empty.

    Per-type feature dims differ on purpose (so a single shared filtration MLP
    would be ill-typed). ``planted=True`` is an optional positive control that
    makes citations field-assortative (topology then carries signal while node
    features stay random); default ``False`` gives random (near-chance) citations.
    """
    feat_dims = feat_dims or DEFAULT_FEAT_DIMS
    rng = np.random.RandomState(seed)
    data = HeteroData()
    counts = {"author": n_author, "paper": n_paper, "field": n_field}
    for nt in NODE_TYPES:
        data[nt].x = torch.tensor(rng.randn(counts[nt], feat_dims[nt]), dtype=torch.float32)
        data[nt].num_nodes = counts[nt]

    # author -writes-> paper : 1..3 authors per paper
    w_src, w_dst = [], []
    for p in range(n_paper):
        for a in rng.choice(n_author, size=min(rng.randint(1, 4), n_author), replace=False):
            w_src.append(int(a)); w_dst.append(p)
    writes = torch.tensor([w_src, w_dst], dtype=torch.long)

    if planted:
        n_comm = n_comm or n_field
        comm = rng.randint(0, n_comm, size=n_paper)
        t_src, t_dst = [], []
        for p in range(n_paper):
            t_src.append(p); t_dst.append(int(comm[p] % n_field))
            if rng.rand() < 0.25:
                t_src.append(p); t_dst.append(int(rng.randint(n_field)))
        cset = set()
        for p in range(n_paper):
            same = [q for q in range(n_paper) if q != p and comm[q] == comm[p]]
            for _ in range(rng.randint(1, 4)):
                if same and rng.rand() < 0.85:
                    q = int(rng.choice(same))
                else:
                    q = int(rng.choice([x for x in range(n_paper) if x != p]))
                cset.add((p, q))
        c_src = [a for a, _ in sorted(cset)]; c_dst = [b for _, b in sorted(cset)]
    else:
        t_src, t_dst = [], []
        for p in range(n_paper):
            for f in rng.choice(n_field, size=min(rng.randint(1, 3), n_field), replace=False):
                t_src.append(p); t_dst.append(int(f))
        cset = set()
        for p in range(n_paper):
            others = [q for q in range(n_paper) if q != p]
            for q in rng.choice(others, size=min(rng.randint(1, 4), len(others)), replace=False):
                cset.add((p, int(q)))
        c_src = [a for a, _ in sorted(cset)]; c_dst = [b for _, b in sorted(cset)]

    has_topic = torch.tensor([t_src, t_dst], dtype=torch.long)
    cites = torch.tensor([c_src, c_dst], dtype=torch.long).reshape(2, -1)

    data["author", "writes", "paper"].edge_index = writes
    data["paper", "written_by", "author"].edge_index = writes.flip(0)
    data["paper", "has_topic", "field"].edge_index = has_topic
    data["field", "rev_has_topic", "paper"].edge_index = has_topic.flip(0)
    data["paper", "cites", "paper"].edge_index = cites
    return data


# --------------------------------------------------------------------------- #
def split_target_edges(
    data: HeteroData, target_rel: EdgeType,
    val_frac: float = 0.1, test_frac: float = 0.1, seed: int = 0,
) -> LinkSplit:
    """Shuffle and partition the target relation's edges into train/val/test."""
    ei = data[target_rel].edge_index
    E = ei.shape[1]
    perm = np.random.RandomState(seed).permutation(E)
    n_test = max(1, int(round(test_frac * E)))
    n_val = max(1, int(round(val_frac * E)))
    test_idx, val_idx = perm[:n_test], perm[n_test:n_test + n_val]
    train_idx = perm[n_test + n_val:]
    sel = lambda idx: ei[:, torch.as_tensor(np.sort(idx), dtype=torch.long)]
    return LinkSplit(sel(train_idx), sel(val_idx), sel(test_idx))


def typed_negative_sampling(
    data: HeteroData, target_rel: EdgeType, num_samples: int,
    seed: int = 0, exclude: Optional[Sequence[Tuple[int, int]]] = None,
) -> torch.Tensor:
    """Sample ``num_samples`` non-edges of ``target_rel`` respecting endpoint types.

    Excludes existing positives (and optional ``exclude`` pairs); for same-type
    relations also excludes self-loops. Returns (2, num_samples) LongTensor.
    """
    src_t, _, dst_t = target_rel
    n_src, n_dst = int(data[src_t].num_nodes), int(data[dst_t].num_nodes)
    same = src_t == dst_t
    forbidden = {(int(a), int(b)) for a, b in data[target_rel].edge_index.t()}
    if exclude:
        forbidden |= {(int(a), int(b)) for a, b in exclude}
    rng = np.random.RandomState(seed)
    out: List[Tuple[int, int]] = []
    seen: set = set()
    tries, max_tries = 0, num_samples * 1000
    while len(out) < num_samples and tries < max_tries:
        tries += 1
        u, v = int(rng.randint(n_src)), int(rng.randint(n_dst))
        if (same and u == v) or (u, v) in forbidden or (u, v) in seen:
            continue
        seen.add((u, v)); out.append((u, v))
    if len(out) < num_samples:
        raise RuntimeError(f"could not sample {num_samples} negatives for {target_rel} "
                           f"(got {len(out)}); graph too dense for node counts.")
    return torch.tensor(out, dtype=torch.long).t().contiguous()


def make_pair_batches(
    pos: torch.Tensor, neg: torch.Tensor, src_type: str, dst_type: str,
    batch_size: Optional[int] = None, shuffle: bool = True, seed: int = 0,
) -> List[PairBatch]:
    """Combine positive/negative edges into labelled PairBatches."""
    src = torch.cat([pos[0], neg[0]]); dst = torch.cat([pos[1], neg[1]])
    label = torch.cat([torch.ones(pos.shape[1]), torch.zeros(neg.shape[1])])
    order = (torch.as_tensor(np.random.RandomState(seed).permutation(src.numel()))
             if shuffle else torch.arange(src.numel()))
    src, dst, label = src[order], dst[order], label[order]
    bs = batch_size or int(src.numel())
    batches = []
    for i in range(0, max(int(src.numel()), 1), max(bs, 1)):
        sl = slice(i, i + bs)
        batches.append(PairBatch(src[sl], dst[sl], label[sl], src_type, dst_type))
    return batches


def remove_target_edges_for_split(
    data: HeteroData, edges_to_remove: torch.Tensor, target_rel: EdgeType,
) -> HeteroData:
    """Return a deep copy of ``data`` with the given target edges removed.

    The original HeteroData is never mutated (all tensors are cloned).
    """
    out = HeteroData()
    for nt in data.node_types:
        if hasattr(data[nt], "x") and data[nt].x is not None:
            out[nt].x = data[nt].x.clone()
        out[nt].num_nodes = int(data[nt].num_nodes)
    for et in data.edge_types:
        out[et].edge_index = data[et].edge_index.clone()
    remove = {(int(a), int(b)) for a, b in edges_to_remove.t()}
    ei = out[target_rel].edge_index
    keep = [i for i in range(ei.shape[1]) if (int(ei[0, i]), int(ei[1, i])) not in remove]
    out[target_rel].edge_index = (ei[:, torch.tensor(keep, dtype=torch.long)] if keep
                                  else torch.zeros((2, 0), dtype=torch.long))
    return out
