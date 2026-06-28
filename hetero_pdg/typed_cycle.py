"""Phase-3 (experimental, research-inspired): TYPED CYCLE *PROXY* DESCRIPTORS.

honesty label:
    typed_cycle_proxy_descriptors (proxy, NOT exact persistent homology,
    NOT real PDGNN/EPD)

Motivation
----------
Two cycles with similar birth/death can be semantically very different -- e.g. an
Author-Paper-Author loop versus a Paper-Field-Paper loop. A plain (untyped)
persistence diagram throws away the relation/type semantics that distinguish them.
This module computes cheap, deterministic *proxy* descriptors around each target
pair that try to PRESERVE those typed semantics. They are NOT a persistence
diagram, NOT extended persistence (EPD), and NOT a learned PDGNN; no claim is made
that they recover, approximate, or outperform any of those. They are simple typed
counting statistics, useful as interpretable side-features.

Per target pair (src, dst) we compute:

  1. node_type_histogram  -- #nodes of each NODE_TYPE in the k-hop heterogeneous
                             neighborhood of the pair (typed "where am I" context).
  2. edge_type_histogram  -- #edges of each EDGE_TYPE inside that neighborhood
                             (typed relation context; reverse relations are listed
                             separately, matching EDGE_TYPES, and are counted as
                             distinct columns by design).
  3. per-meta-path common-neighbor count -- |N(u) & N(v)| on each projected
                             meta-path graph (a length-2 "cycle"/cherry through the
                             pair in that typed channel).
  4. per-meta-path short-cycle proxy -- #edges among N(u) | N(v) on that projected
                             graph (a local typed motif/short-cycle richness proxy,
                             distinct from the raw common-neighbor count).
  5. (optional) local typed motif density -- edge density of the k-hop hetero
                             neighborhood (one scalar).

Missing structure is handled gracefully: if a meta-path graph does not cover a
pair (endpoint out of range, wrong target type, or both endpoints isolated) that
meta-path's block is left as zeros and its validity mask entry is False. No crash.

Determinism: every quantity is an integer/ratio count derived from the (fixed)
graph; no randomness, no learned parameters. Two runs are bit-identical.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

import networkx as nx
import numpy as np
import torch

from hetero_pdg.data import NODE_TYPES, EDGE_TYPES, PairBatch
from hetero_pdg.filtration import build_homogeneous_view_with_type_ids

# Honesty constants -- never call these exact PH / real PDGNN / EPD.
TYPED_CYCLE_BACKEND = "typed_cycle_proxy_descriptors"
TYPED_CYCLE_HONESTY_LABEL = (
    "typed_cycle_proxy_descriptors (proxy, NOT exact persistent homology, "
    "NOT real PDGNN/EPD)"
)

# Fixed-size histogram blocks (independent of which meta-paths are passed).
NODE_TYPE_HIST_NAMES: List[str] = [f"node_type_hist[{nt}]" for nt in NODE_TYPES]
EDGE_TYPE_HIST_NAMES: List[str] = [
    f"edge_type_hist[{s}-{r}-{d}]" for (s, r, d) in EDGE_TYPES
]
# Per-meta-path descriptor suffixes (2 per channel).
PER_METAPATH_SUFFIXES: List[str] = ["common_neighbors", "short_cycle_proxy_edges"]
DENSITY_NAME = "local_typed_neighborhood_density"


# --------------------------------------------------------------------------- #
def typed_cycle_descriptor_names(
    metapath_names: Sequence[str], include_density: bool = True
) -> List[str]:
    """Ordered descriptor names for the produced feature vector.

    Layout: [node_type_hist..] + [edge_type_hist..] + per-meta-path blocks
    (each = common_neighbors, short_cycle_proxy_edges) + [density?].
    """
    names: List[str] = list(NODE_TYPE_HIST_NAMES) + list(EDGE_TYPE_HIST_NAMES)
    for mp in metapath_names:
        for suf in PER_METAPATH_SUFFIXES:
            names.append(f"mp[{mp}].{suf}")
    if include_density:
        names.append(DENSITY_NAME)
    return names


def typed_cycle_dim(n_metapaths: int, include_density: bool = True) -> int:
    """Total feature dimension for ``n_metapaths`` channels."""
    return (
        len(NODE_TYPE_HIST_NAMES)
        + len(EDGE_TYPE_HIST_NAMES)
        + len(PER_METAPATH_SUFFIXES) * int(n_metapaths)
        + (1 if include_density else 0)
    )


# --------------------------------------------------------------------------- #
def _build_typed_index(data):
    """Pre-compute typed adjacency / per-edge-type edge lists once for the graph.

    Returns offsets, node_type_ids (np), num_nodes, undirected adjacency
    (global id -> set of global ids), and per-edge-type list of global (a, b).
    """
    view = build_homogeneous_view_with_type_ids(data)
    offsets: Dict[str, Tuple[int, int]] = view["offsets"]
    node_type_ids = view["node_type_ids"].cpu().numpy()
    num_nodes = int(view["num_nodes"])

    adj: Dict[int, Set[int]] = defaultdict(set)
    edges_by_type: Dict[Tuple[str, str, str], List[Tuple[int, int]]] = {}
    for et in EDGE_TYPES:
        s, _, d = et
        ei = data[et].edge_index.cpu().numpy()
        os_, od = offsets[s][0], offsets[d][0]
        lst: List[Tuple[int, int]] = []
        for a, b in zip(ei[0], ei[1]):
            ga, gb = os_ + int(a), od + int(b)
            adj[ga].add(gb)
            adj[gb].add(ga)
            lst.append((ga, gb))
        edges_by_type[et] = lst
    return offsets, node_type_ids, num_nodes, adj, edges_by_type


def _khop_nodes(adj: Dict[int, Set[int]], seeds: Sequence[int], k: int) -> Set[int]:
    """k-hop neighborhood (inclusive of seeds) over an undirected adjacency map."""
    seen: Set[int] = set(seeds)
    frontier: Set[int] = set(seeds)
    for _ in range(max(k, 0)):
        nxt: Set[int] = set()
        for n in frontier:
            nxt |= adj.get(n, set())
        nxt -= seen
        if not nxt:
            break
        seen |= nxt
        frontier = nxt
    return seen


def _hetero_histograms(
    node_type_ids: np.ndarray,
    adj: Dict[int, Set[int]],
    edges_by_type,
    gu: int,
    gv: int,
    k: int,
    include_density: bool,
) -> np.ndarray:
    """node-type hist + edge-type hist (+ density) for the pair's k-hop hood."""
    nodes = _khop_nodes(adj, [gu, gv], k)

    node_hist = np.zeros(len(NODE_TYPES), dtype=np.float64)
    for n in nodes:
        node_hist[int(node_type_ids[n])] += 1.0

    edge_hist = np.zeros(len(EDGE_TYPES), dtype=np.float64)
    for ci, et in enumerate(EDGE_TYPES):
        c = 0
        for a, b in edges_by_type[et]:
            if a in nodes and b in nodes:
                c += 1
        edge_hist[ci] = float(c)

    out = np.concatenate([node_hist, edge_hist])
    if include_density:
        m = len(nodes)
        if m > 1:
            e = 0
            for a in nodes:
                e += len(adj.get(a, set()) & nodes)
            e //= 2  # undirected double count
            density = (2.0 * e) / (m * (m - 1))
        else:
            density = 0.0
        out = np.concatenate([out, [density]])
    return out


def _per_metapath_block(
    bundle, u: int, v: int, valid: bool
) -> Tuple[np.ndarray, bool]:
    """(common_neighbors, short_cycle_proxy_edges) for the pair on one bundle.

    ``valid`` is a pre-check (correct target type & in-range indices). Returns a
    zero block + False if the pair has no structure (endpoints absent/isolated).
    """
    block = np.zeros(len(PER_METAPATH_SUFFIXES), dtype=np.float64)
    if not valid:
        return block, False
    g: nx.Graph = bundle.graph
    if u not in g or v not in g:
        return block, False
    nu = set(g.neighbors(u))
    nv = set(g.neighbors(v))
    if not nu and not nv:
        return block, False  # both endpoints isolated -> missing structure

    common = nu & nv
    block[0] = float(len(common))

    # short-cycle proxy: #edges among the union of the endpoints' neighbors
    union = (nu | nv) - {u, v}
    short_edges = 0
    for w in union:
        short_edges += len(set(g.neighbors(w)) & union)
    block[1] = float(short_edges // 2)

    is_valid = bool(len(nu) > 0 or len(nv) > 0)
    return block, is_valid


# --------------------------------------------------------------------------- #
def typed_cycle_signature_features(
    data,
    pair_batch: PairBatch,
    bundles: Dict[str, object],
    k: int = 1,
    include_density: bool = True,
    return_mask: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Typed-cycle PROXY descriptors for every pair in ``pair_batch``.

    Parameters
    ----------
    data : HeteroData
        Used for the typed (node/edge-type) neighborhood histograms.
    pair_batch : PairBatch
        Candidate (src, dst) pairs with endpoint types.
    bundles : dict[str, ProjectedGraphBundle]
        Projected meta-path graphs supplying the per-channel cycle proxies.
        Iteration order (dict order) fixes the per-meta-path block order.
    k : int
        Hop radius for the heterogeneous neighborhood histograms.
    include_density : bool
        Append the local typed neighborhood density scalar.
    return_mask : bool
        Also return a (B, n_metapaths) bool tensor: True where that meta-path
        actually covered the pair (else its block is zeros).

    Returns
    -------
    Tensor[B, typed_cycle_dim] (float32), or (features, mask) if ``return_mask``.
    """
    src = [int(x) for x in pair_batch.src.tolist()]
    dst = [int(x) for x in pair_batch.dst.tolist()]
    B = len(src)
    src_type = pair_batch.src_type
    dst_type = pair_batch.dst_type

    mp_names = list(bundles.keys())
    n_mp = len(mp_names)
    dim = typed_cycle_dim(n_mp, include_density=include_density)

    offsets, node_type_ids, num_nodes, adj, edges_by_type = _build_typed_index(data)

    F = np.zeros((B, dim), dtype=np.float64)
    mask = np.zeros((B, n_mp), dtype=bool)

    hist_len = len(NODE_TYPE_HIST_NAMES) + len(EDGE_TYPE_HIST_NAMES)
    dens_extra = 1 if include_density else 0

    s_off = offsets[src_type][0]
    d_off = offsets[dst_type][0]
    n_src_type = offsets[src_type][1] - offsets[src_type][0]
    n_dst_type = offsets[dst_type][1] - offsets[dst_type][0]

    for i in range(B):
        u, v = src[i], dst[i]

        # --- heterogeneous typed histograms (always computable for valid ids) ---
        if 0 <= u < n_src_type and 0 <= v < n_dst_type:
            gu, gv = s_off + u, d_off + v
            hist = _hetero_histograms(
                node_type_ids, adj, edges_by_type, gu, gv, k, include_density
            )
            F[i, : hist_len + dens_extra] = hist
        # else leave the histogram block as zeros

        # --- per-meta-path cycle proxies ---
        for j, name in enumerate(mp_names):
            bundle = bundles[name]
            same_type = (
                src_type == dst_type == getattr(bundle, "target_type", None)
            )
            in_range = (
                0 <= u < int(bundle.num_nodes) and 0 <= v < int(bundle.num_nodes)
            )
            valid = bool(same_type and in_range)
            block, ok = _per_metapath_block(bundle, u, v, valid)
            base = hist_len + len(PER_METAPATH_SUFFIXES) * j
            F[i, base : base + len(PER_METAPATH_SUFFIXES)] = block
            mask[i, j] = ok

    feats = torch.from_numpy(F.astype(np.float32))
    if return_mask:
        return feats, torch.from_numpy(mask)
    return feats


def combine_topology_with_typed_cycle_signature(
    topology_tensor: torch.Tensor,
    typed_cycle_tensor: torch.Tensor,
    dim: int = 1,
) -> torch.Tensor:
    """Concatenate a topology feature tensor with the typed-cycle tensor.

    Both must share the batch dimension (dim 0). Returns
    ``[B, topo_dim + typed_cycle_dim]`` (when ``dim=1``).
    """
    if topology_tensor.shape[0] != typed_cycle_tensor.shape[0]:
        raise ValueError(
            "batch-size mismatch: topology has "
            f"{topology_tensor.shape[0]} rows, typed-cycle has "
            f"{typed_cycle_tensor.shape[0]}"
        )
    return torch.cat(
        [topology_tensor, typed_cycle_tensor.to(topology_tensor.dtype)], dim=dim
    )
