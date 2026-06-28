"""Relation-aware 0-dim EPD where edge TYPE changes the persistence events.

This is the Phase-4 step that fixes the key limitation noted in the report: until now
edge type did NOT enter the topology computation (the EPD still used g(e)=max(f_i,f_j)).
Here we build a multi-relation graph (each edge tagged with its meta-path / relation id),
lift the node filtration to a **relation-aware edge filtration**
``g(i,j,rho)=max(f_i,f_j)+softplus(delta_rho)`` (reusing edge_filtration.py, which
guarantees ``g >= max``), and run a 0-dim sublevel-persistence Union-Find that consumes
those EXPLICIT per-edge values (edge_filtration.zero_dim_sublevel_persistence). The
resulting birth/death pairs are vectorized into a small self-contained persistence image.

So edge type genuinely shifts when components merge: with distinct per-relation delays the
typed EPD differs from the untyped (``max``) EPD.

Honesty: this is a relation-aware **0-dim** sublevel EPD, NOT exact (extended) persistent
homology and NOT a PDGNN. The per-relation delays are FIXED hyper-parameters, not learned
end-to-end: the Union-Find is non-differentiable, so the LP loss cannot train them (a soft /
differentiable EPD would be needed). No performance claims. Self-contained (own persistence
image; no TLC-GNN engine needed).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import torch

from hetero_pdg.data import PairBatch
from hetero_pdg.edge_filtration import relation_edge_filtration, zero_dim_sublevel_persistence

RELATION_EPD_HONESTY_LABEL = (
    "relation-aware 0-dim sublevel EPD via typed edge filtration -> persistence image "
    "(NOT exact extended persistence)"
)
PI_RESOLUTION = 5
PI_DIM = PI_RESOLUTION * PI_RESOLUTION


def persistence_image(pairs: List[Tuple[float, float]], resolution: int = PI_RESOLUTION,
                      bandwidth_frac: float = 0.25) -> np.ndarray:
    """Tiny self-contained persistence image of (birth, death) pairs -> (resolution^2,).

    Finite bars only (the one infinite component bar is dropped). Each point is placed in
    (birth, persistence) space, ranges are auto-scaled to the data, and a persistence-weighted
    Gaussian is accumulated. Deterministic; empty input -> zeros.
    """
    finite = [(float(b), float(d) - float(b)) for (b, d) in pairs
              if np.isfinite(d) and float(d) > float(b)]
    img = np.zeros((resolution, resolution), dtype=np.float64)
    if not finite:
        return img.reshape(-1)
    bs = np.array([b for b, _ in finite]); ps = np.array([p for _, p in finite])
    b_lo, b_rng = bs.min(), max(bs.max() - bs.min(), 1e-6)
    p_rng = max(ps.max(), 1e-6)
    centers = (np.arange(resolution) + 0.5) / resolution
    bw = max(bandwidth_frac, 1e-3)
    for b, p in zip(bs, ps):
        bn, pn = (b - b_lo) / b_rng, p / p_rng
        gx = np.exp(-((centers - bn) ** 2) / (2 * bw * bw))
        gy = np.exp(-((centers - pn) ** 2) / (2 * bw * bw))
        img += p * np.outer(gy, gx)
    return img.reshape(-1)


def build_relation_tagged_graph(bundles: Dict[str, object]) -> Tuple[nx.Graph, int]:
    """Union of projected meta-path graphs over the (shared) target nodes, each edge
    tagged with ``rel`` = the meta-path's index. Returns (graph, num_relations)."""
    names = list(bundles.keys())
    n = int(next(iter(bundles.values())).num_nodes)
    g = nx.Graph(); g.add_nodes_from(range(n))
    for ridx, name in enumerate(names):
        for u, v in bundles[name].graph.edges():
            if not g.has_edge(u, v):            # first meta-path to claim the edge tags it
                g.add_edge(int(u), int(v), rel=ridx)
    return g, len(names)


class RelationAwareEPDTopology:
    """Per-pair relation-aware 0-dim EPD -> persistence image on a relation-tagged graph."""

    is_fallback = False

    def __init__(self, graph: nx.Graph, num_relations: int,
                 delays: Optional[torch.Tensor] = None, hop: int = 2, max_nodes: int = 30,
                 node_filt: Optional[dict] = None, resolution: int = PI_RESOLUTION, seed: int = 0):
        self.g = graph
        self.num_relations = int(num_relations)
        self.hop, self.max_nodes, self.res = hop, max_nodes, resolution
        self.feat_dim = resolution * resolution
        # distinct fixed delays -> edge type changes event timing (typed != untyped)
        if delays is None:
            delays = torch.arange(max(num_relations, 1), dtype=torch.float32)
        self.delays = torch.as_tensor(delays, dtype=torch.float32)
        if node_filt is None:                    # degree-based node filtration in [0,1]
            deg = dict(graph.degree())
            m = max(deg.values()) if deg else 1
            node_filt = {nd: deg.get(nd, 0) / max(m, 1) for nd in graph.nodes()}
        self.node_filt = node_filt

    def pair_pi(self, u: int, v: int, mode: str = "relation_delay",
                remove_target: bool = True) -> np.ndarray:
        restored = None
        if remove_target and self.g.has_edge(u, v):
            restored = dict(self.g[u][v]); self.g.remove_edge(u, v)
        try:
            nodes = set(nx.ego_graph(self.g, u, radius=self.hop).nodes()) | set(
                nx.ego_graph(self.g, v, radius=self.hop).nodes())
            if len(nodes) > self.max_nodes:
                nodes = set(sorted(nodes, key=lambda nd: self.node_filt.get(nd, 0.0))[:self.max_nodes]) | {u, v}
            sub = self.g.subgraph(nodes)
            if sub.number_of_edges() == 0:
                return np.zeros(self.feat_dim)
            nl = list(sub.nodes()); remap = {nd: i for i, nd in enumerate(nl)}
            nf = np.array([self.node_filt.get(nd, 0.0) for nd in nl], dtype=np.float64)
            ei = np.array([(remap[a], remap[b]) for a, b in sub.edges()], dtype=np.int64).T
            et = np.array([int(data.get("rel", 0)) for _, _, data in sub.edges(data=True)], dtype=np.int64)
            nf_t = torch.tensor(nf, dtype=torch.float32)
            ei_t = torch.tensor(ei, dtype=torch.long)
            et_t = torch.tensor(et, dtype=torch.long)
            ef = relation_edge_filtration(nf_t, ei_t, et_t, mode=mode, delays=self.delays)
            pairs = zero_dim_sublevel_persistence(nf_t, ei_t, ef)
            return persistence_image(pairs, resolution=self.res)
        finally:
            if restored is not None:
                self.g.add_edge(u, v, **restored)


def compute_relation_epd_features(bundles: Dict[str, object], pair_batch: PairBatch,
                                  mode: str = "relation_delay", hop: int = 2, max_nodes: int = 30,
                                  delays: Optional[torch.Tensor] = None,
                                  resolution: int = PI_RESOLUTION) -> torch.Tensor:
    """(B, resolution^2) relation-aware 0-dim EPD persistence-image features for a batch.

    mode='relation_delay' -> typed (edge type changes events); mode='max' -> untyped baseline.
    """
    g, n_rel = build_relation_tagged_graph(bundles)
    topo = RelationAwareEPDTopology(g, n_rel, delays=delays, hop=hop, max_nodes=max_nodes,
                                    resolution=resolution)
    src, dst = pair_batch.src.tolist(), pair_batch.dst.tolist()
    F = np.zeros((len(src), topo.feat_dim), dtype=np.float32)
    for i, (u, v) in enumerate(zip(src, dst)):
        u, v = int(u), int(v)
        if u in g and v in g:
            F[i] = topo.pair_pi(u, v, mode=mode)
    return torch.from_numpy(F)
