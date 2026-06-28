"""Experimental PAIR-CONDITIONED filtration for hetero link prediction (Phase 3).

HONESTY: this module implements
``experimental pair-conditioned filtration (Hetero-PDGNN extension inspired by
TLC-GNN pairwise topology + SEAL enclosing subgraphs + HGT relation-typing; NOT a
reproduction)``.

It is NOT exact persistent homology, NOT a real PDGNN, and NOT an
extended-persistence-diagram (EPD) feature. It produces *learnable scalar filter
values per node that are conditioned on a target pair (u, v) and relation r* -- a
heuristic, research-inspired construction. No performance claims are made.

Motivation
----------
In link prediction a node's topological importance should depend on the target pair
(u, v) and the relation r, not only on the node in isolation. We therefore make the
type-aware filter function pair-conditioned:

    f_theta(x | u, v, r) = MLP_type(x)( [ h_x, h_u, h_v, rel_emb(r),
                                          relation_degree_vector_x,
                                          typed_distance(x, u),
                                          typed_distance(x, v) ] )

where
  * h_x / h_u / h_v are node embeddings (a per-type Linear over raw features by
    default; custom embeddings can be supplied);
  * rel_emb(r) is a learnable relation embedding indexed by an integer relation id
    (for the toy graph a single target relation maps to id 0);
  * relation_degree_vector_x comes from
    :func:`hetero_pdg.filtration.compute_relation_degree_vectors`;
  * typed_distance(x, *) is the shortest-path distance on a projected homogeneous
    meta-path graph (``ProjectedGraphBundle.graph``), with a sentinel
    :data:`NO_PATH_DISTANCE` when no path exists.

Each node type has its OWN filtration MLP (HGT-style relation typing); the output is
ONE scalar filter value per node, but that scalar now varies with the conditioning
pair. A batched helper summarises the pair-conditioned filtration over the
pair's local vicinity into a fixed-width feature vector ``[B, feat_dim]`` so it can
be fed to :class:`hetero_pdg.models.HeteroTopoLinkPredictor` as its ``topo`` tensor.

Assumption: the pair endpoints (u, v) are indexed in the projected graph's node
space, i.e. the target relation's endpoint type equals ``bundle.target_type``
(true for the toy paper-cites-paper task with the PFP / PCP meta-paths). This is
documented rather than silently assumed: out-of-range endpoints are masked.

Everything is deterministic given a fixed torch seed and runs on CPU.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import networkx as nx
import numpy as np
import torch
import torch.nn as nn

from hetero_pdg.data import NODE_TYPES, PairBatch
from hetero_pdg.filtration import (
    compute_relation_degree_vectors,
    relation_schema,
    _placeholder_features,
)

# --------------------------------------------------------------------------- #
# Honesty / naming constants (exact wording required by the task).
PAIR_CONDITIONED_FILTRATION_LABEL: str = (
    "experimental pair-conditioned filtration (Hetero-PDGNN extension inspired by "
    "TLC-GNN pairwise topology + SEAL enclosing subgraphs + HGT relation-typing; "
    "NOT a reproduction)"
)
FILTRATION_MODE: str = "pair_conditioned"

# Sentinel distance when no path connects two nodes on the projected graph.
NO_PATH_DISTANCE: float = -1.0

# Names of the per-pair LP topology features produced by the batched helper. These
# are vicinity statistics of the pair-conditioned filter values -- explicitly NOT
# persistence pairs / EPD coordinates.
PAIR_CONDITIONED_FEATURE_NAMES: List[str] = [
    "pcf_value_at_src",
    "pcf_value_at_dst",
    "pcf_vicinity_mean",
    "pcf_vicinity_std",
    "pcf_vicinity_min",
    "pcf_vicinity_max",
]


@dataclass
class PairConditionedFeatures:
    """Per-pair LP topology tensor derived from the pair-conditioned filtration.

    ``features`` is ``(B, len(PAIR_CONDITIONED_FEATURE_NAMES))`` and ``mask`` is
    ``(B,)`` bool (False where a pair is invalid / has no vicinity coverage; such
    rows are zero-filled). ``feature_names`` and ``label`` document provenance.
    """
    features: torch.Tensor
    mask: torch.Tensor
    feature_names: List[str]
    label: str


# --------------------------------------------------------------------------- #
class PairConditionedFiltrationMLP(nn.Module):
    """Per-type pair-conditioned filtration network.

    For each node type a Linear maps raw features (or deterministic placeholder
    features for featureless types) to a common ``emb_dim`` embedding; a learnable
    relation embedding table is shared across types; and each type owns a small MLP
    mapping the concatenated conditioning vector to a single scalar filter value.

    Parameters
    ----------
    data : HeteroData
        Used only to read per-type raw-feature dimensions (no mutation, no caching
        of node data); relation-degree widths come from the fixed edge schema.
    emb_dim : int
        Width of the per-type node embeddings (h_x / h_u / h_v).
    rel_emb_dim : int
        Width of the learnable relation embedding.
    hidden : int
        Hidden width of each per-type filtration MLP.
    n_relations : int
        Size of the relation embedding table (toy: 1 target relation -> id 0).
    placeholder_dim : int
        Width of synthesised features for featureless node types.
    """

    def __init__(self, data, emb_dim: int = 8, rel_emb_dim: int = 4,
                 hidden: int = 16, n_relations: int = 1, placeholder_dim: int = 4):
        super().__init__()
        self.emb_dim = emb_dim
        self.rel_emb_dim = rel_emb_dim
        self.placeholder_dim = placeholder_dim
        self.n_relations = n_relations

        self.embed = nn.ModuleDict()
        self.filters = nn.ModuleDict()
        self._featureless: Dict[str, bool] = {}
        for nt in NODE_TYPES:
            x = getattr(data[nt], "x", None)
            featureless = x is None
            self._featureless[nt] = featureless
            raw_dim = placeholder_dim if featureless else int(x.size(1))
            self.embed[nt] = nn.Linear(raw_dim, emb_dim)
            rd_dim = len(relation_schema(nt))
            # conditioning vector = h_x | h_u | h_v | rel_emb | rdv | dist_u | dist_v
            in_dim = 3 * emb_dim + rel_emb_dim + rd_dim + 2
            self.filters[nt] = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, 1),
            )
        self.relation_embedding = nn.Embedding(n_relations, rel_emb_dim)

    def _raw_features(self, data, nt: str) -> torch.Tensor:
        x = getattr(data[nt], "x", None)
        if x is None:
            x = _placeholder_features(int(data[nt].num_nodes), self.placeholder_dim)
        return x

    def node_embeddings(self, data) -> Dict[str, torch.Tensor]:
        """Default node embeddings: per-type Linear over (raw|placeholder) features.

        Returns ``{node_type: (n, emb_dim)}``. Callers may instead build their own
        embedding dict and pass it to the filtration functions below.
        """
        dev = next(self.parameters()).device
        out: Dict[str, torch.Tensor] = {}
        for nt in NODE_TYPES:
            x = self._raw_features(data, nt).to(dev)
            out[nt] = self.embed[nt](x)
        return out


# --------------------------------------------------------------------------- #
def _typed_distances(graph: nx.Graph, source: int, num_nodes: int) -> np.ndarray:
    """Shortest-path hop distances from ``source`` to every target node.

    Unreachable nodes (and out-of-graph sources) get :data:`NO_PATH_DISTANCE`.
    Distance is the unweighted hop count (edge weights are meta-path multiplicities,
    not lengths, so hops are the meaningful notion here).
    """
    d = np.full(num_nodes, NO_PATH_DISTANCE, dtype=np.float64)
    if source in graph:
        lengths = nx.single_source_shortest_path_length(graph, source)
        for node, dist in lengths.items():
            if node < num_nodes:
                d[node] = float(dist)
    return d


def pair_conditioned_node_filtration(
    mlp: PairConditionedFiltrationMLP,
    data,
    bundle,
    u: int,
    v: int,
    relation_id: int = 0,
    node_embeddings: Optional[Dict[str, torch.Tensor]] = None,
    src_type: Optional[str] = None,
    dst_type: Optional[str] = None,
) -> torch.Tensor:
    """Pair-conditioned filter value for every node of ``bundle.target_type``.

    Computes ``f_theta(x | u, v, r)`` for all target-type nodes x, where typed
    distances are measured on ``bundle.graph``. Endpoints ``u`` / ``v`` are assumed
    to live in the target type's node space (see module docstring).

    Returns a ``(bundle.num_nodes,)`` float tensor (no NaN/inf).
    """
    nt = bundle.target_type
    src_type = src_type or nt
    dst_type = dst_type or nt
    if node_embeddings is None:
        node_embeddings = mlp.node_embeddings(data)

    dev = next(mlp.parameters()).device
    n = int(bundle.num_nodes)

    H = node_embeddings[nt].to(dev)                       # (n, emb_dim)  -> h_x
    h_u = node_embeddings[src_type][int(u)].to(dev)       # (emb_dim,)
    h_v = node_embeddings[dst_type][int(v)].to(dev)       # (emb_dim,)
    rel = mlp.relation_embedding(
        torch.tensor(int(relation_id), device=dev))      # (rel_emb_dim,)

    rdv = compute_relation_degree_vectors(data)[nt].to(dev)   # (n, rd_dim)

    dist_u = torch.from_numpy(_typed_distances(bundle.graph, int(u), n)).to(
        dev, dtype=torch.float32).unsqueeze(-1)          # (n, 1)
    dist_v = torch.from_numpy(_typed_distances(bundle.graph, int(v), n)).to(
        dev, dtype=torch.float32).unsqueeze(-1)          # (n, 1)

    h_u_b = h_u.unsqueeze(0).expand(n, -1)
    h_v_b = h_v.unsqueeze(0).expand(n, -1)
    rel_b = rel.unsqueeze(0).expand(n, -1)

    cond = torch.cat([H, h_u_b, h_v_b, rel_b, rdv, dist_u, dist_v], dim=-1)
    return mlp.filters[nt](cond).squeeze(-1)


def _vicinity_nodes(graph: nx.Graph, u: int, v: int, k: int) -> List[int]:
    """k-hop enclosing-subgraph node set around the pair (u, v) (SEAL-style)."""
    nodes = set()
    for s in (u, v):
        if s in graph:
            nodes |= set(nx.single_source_shortest_path_length(graph, s, cutoff=k).keys())
    nodes |= {u, v}
    return sorted(nodes)


def compute_pair_conditioned_lp_features(
    mlp: PairConditionedFiltrationMLP,
    data,
    bundle,
    pair_batch: PairBatch,
    relation_id: int = 0,
    k: int = 2,
    node_embeddings: Optional[Dict[str, torch.Tensor]] = None,
) -> PairConditionedFeatures:
    """Batched LP topology features from the pair-conditioned filtration.

    For each pair (u, v) in ``pair_batch`` we compute the pair-conditioned filter
    values over the target nodes, then summarise them over the pair's k-hop
    enclosing subgraph into the fixed vector named by
    :data:`PAIR_CONDITIONED_FEATURE_NAMES`:
    ``[value@src, value@dst, vicinity mean, std, min, max]``.

    Invalid pairs (endpoint out of range, or both endpoints absent from the
    projected graph) are zero-filled and masked ``False``. Returns a
    :class:`PairConditionedFeatures`. Deterministic for a fixed seed.
    """
    if node_embeddings is None:
        node_embeddings = mlp.node_embeddings(data)

    src = pair_batch.src.tolist()
    dst = pair_batch.dst.tolist()
    B = len(src)
    feat_dim = len(PAIR_CONDITIONED_FEATURE_NAMES)
    n = int(bundle.num_nodes)
    g = bundle.graph

    F = torch.zeros((B, feat_dim), dtype=torch.float32)
    M = torch.zeros(B, dtype=torch.bool)

    for i, (u, v) in enumerate(zip(src, dst)):
        u, v = int(u), int(v)
        if u >= n or v >= n or u < 0 or v < 0:
            continue  # invalid endpoint -> zeros + mask False
        if u not in g and v not in g:
            continue  # no coverage in this projected graph

        vals = pair_conditioned_node_filtration(
            mlp, data, bundle, u, v, relation_id=relation_id,
            node_embeddings=node_embeddings,
            src_type=pair_batch.src_type, dst_type=pair_batch.dst_type,
        )

        vic = _vicinity_nodes(g, u, v, k)
        vic_vals = vals[torch.tensor(vic, dtype=torch.long)]
        std = vic_vals.std(unbiased=False) if vic_vals.numel() > 1 else torch.zeros(())

        F[i, 0] = vals[u]
        F[i, 1] = vals[v]
        F[i, 2] = vic_vals.mean()
        F[i, 3] = std
        F[i, 4] = vic_vals.min()
        F[i, 5] = vic_vals.max()
        M[i] = True

    F = torch.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    return PairConditionedFeatures(
        features=F, mask=M,
        feature_names=list(PAIR_CONDITIONED_FEATURE_NAMES),
        label=PAIR_CONDITIONED_FILTRATION_LABEL,
    )
