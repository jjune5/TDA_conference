"""Phase-1 pairwise topology features: DETERMINISTIC FALLBACK DESCRIPTORS.

These descriptors are explicitly **fallback_topology_descriptors** -- they are
NOT PDGNN features and NOT extended-persistence-diagram (EPD) features. They exist
so the end-to-end link-prediction pipeline runs and can be validated without the
heavier neural topology machinery.

Per (pair, projected meta-path graph) we compute:
  common_neighbors, shortest_path_distance, src_degree, dst_degree,
  local_edge_count (k-hop enclosing subgraph), local_density (k-hop enclosing subgraph).

Leakage rule: if the target edge (u, v) exists in the projected graph it is removed
before measuring, then restored (no permanent mutation).

Phase-2 seam: ``compute_topology_features(..., backend="pdgnn")`` is where a real
PDGNN/TLC EPD adapter will plug in; in Phase 1 it raises NotImplementedError.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import networkx as nx
import numpy as np
import torch

from hetero_pdg.data import PairBatch

FALLBACK_TOPOLOGY_DESCRIPTOR_NAMES: List[str] = [
    "common_neighbors",
    "shortest_path_distance",
    "src_degree",
    "dst_degree",
    "local_edge_count",
    "local_density",
]
TOPOLOGY_BACKEND = "fallback_topology_descriptors"
TOPO_DIM = len(FALLBACK_TOPOLOGY_DESCRIPTOR_NAMES)
NO_PATH_DISTANCE = -1.0  # sentinel when no path connects u and v


@dataclass
class TopologyFeatures:
    """Per-meta-path topology features + validity masks for a PairBatch."""
    features_by_metapath: Dict[str, torch.Tensor]   # name -> (B, TOPO_DIM)
    mask_by_metapath: Dict[str, torch.Tensor]       # name -> (B,) bool
    backend: str
    feature_names: List[str]


def fallback_topology_descriptors(
    g: nx.Graph, u: int, v: int, k: int = 2, remove_target: bool = True
) -> np.ndarray:
    """Deterministic NON-PDGNN descriptors for the pair (u, v). Returns (6,) float64."""
    restored = None
    if remove_target and g.has_edge(u, v):
        restored = dict(g[u][v])
        g.remove_edge(u, v)
    try:
        nu, nv = set(g.neighbors(u)), set(g.neighbors(v))
        common = len(nu & nv)
        du, dv = g.degree(u), g.degree(v)
        try:
            spd = float(nx.shortest_path_length(g, u, v))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            spd = NO_PATH_DISTANCE
        nodes = {u, v}
        for s in (u, v):
            nodes |= set(nx.single_source_shortest_path_length(g, s, cutoff=k).keys())
        sub = g.subgraph(nodes)
        m, e = sub.number_of_nodes(), sub.number_of_edges()
        density = (2.0 * e) / (m * (m - 1)) if m > 1 else 0.0
        feats = np.array([common, spd, du, dv, e, density], dtype=np.float64)
    finally:
        if restored is not None:
            g.add_edge(u, v, **restored)
    return feats


def compute_fallback_features_by_metapath(
    bundles: Dict[str, "object"], pair_batch: PairBatch, k: int = 2
) -> TopologyFeatures:
    """Fallback descriptors for every pair in every projected meta-path graph.

    A pair is invalid for a graph if an endpoint is out of range or both endpoints
    are isolated (no coverage); such entries are zero-filled and masked False.
    """
    src = pair_batch.src.tolist()
    dst = pair_batch.dst.tolist()
    B = len(src)
    feats: Dict[str, torch.Tensor] = {}
    masks: Dict[str, torch.Tensor] = {}
    for name, bundle in bundles.items():
        g = bundle.graph
        F = np.zeros((B, TOPO_DIM), dtype=np.float32)
        M = np.zeros(B, dtype=bool)
        for i, (u, v) in enumerate(zip(src, dst)):
            u, v = int(u), int(v)
            if u >= bundle.num_nodes or v >= bundle.num_nodes or u not in g or v not in g:
                continue  # invalid -> zeros + mask False
            d = fallback_topology_descriptors(g, u, v, k=k)
            if d[2] > 0 or d[3] > 0:           # covered: at least one endpoint participates
                F[i] = d
                M[i] = True
            # else: uncovered -> leave zeros + mask False
        feats[name] = torch.from_numpy(F)
        masks[name] = torch.from_numpy(M)
    return TopologyFeatures(feats, masks, TOPOLOGY_BACKEND,
                            list(FALLBACK_TOPOLOGY_DESCRIPTOR_NAMES))


def compute_topology_features(
    bundles: Dict[str, "object"], pair_batch: PairBatch,
    backend: str = "fallback", k: int = 2,
) -> TopologyFeatures:
    """Dispatcher / Phase-2 seam.

    backend='fallback' -> deterministic descriptors (Phase 1, default).
    backend='pdgnn'    -> real PDGNN/TLC EPD adapter (Phase 2; not yet implemented).
    """
    if backend in ("fallback", "fallback_topology_descriptors"):
        return compute_fallback_features_by_metapath(bundles, pair_batch, k=k)
    if backend in ("pdgnn", "tlc", "epd"):
        raise NotImplementedError(
            "Phase 2: real PDGNN/TLC EPD topology adapter is not implemented yet. "
            "Phase 1 uses backend='fallback' (fallback_topology_descriptors). "
            "See README 'Next steps for real PDGNN/TLC integration'."
        )
    raise ValueError(f"unknown topology backend {backend!r}")
