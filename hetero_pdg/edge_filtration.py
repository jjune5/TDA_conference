"""Experimental relation-aware edge filtration (PREPARED INTERFACE).

Honesty label (use verbatim):
    "experimental relation-aware edge filtration (prepared interface; extends
     standard node->edge max filtration)"

This module is NOT exact persistent homology, NOT a real PDGNN, and NOT an EPD
feature extractor. It is an experimental, research-inspired construction with a
prepared interface; it makes NO performance claims.

Motivation
----------
The standard way to lift a node (vertex) filtration ``f`` to edges in lower-star /
sublevel persistence is the *max* rule::

    g(i, j) = max(f(i), f(j))

This rule is *relation-blind*: every edge type activates at the same height as its
later endpoint. In a heterogeneous / multi-relational graph we may want some
relation (edge) types to activate *later* than others -- e.g. a noisy relation
should "delay" before it can create / merge topological features.

We therefore allow a per-relation, non-negative *delay*::

    g(i, j, rho) = max(f(i), f(j)) + softplus(delta_rho)

where ``rho`` is the relation type of edge (i, j) and ``softplus(x) = log(1+e^x)``
is strictly >= 0. Because the added term is non-negative, the construction always
satisfies the validity condition required for a well-defined edge filtration::

    g(i, j, rho) >= max(f(i), f(j))          (HARD CONSTRAINT, enforced & tested)

``delta_rho`` may be a fixed (deterministic) vector or a learnable parameter
(see :class:`RelationDelayEdgeFiltration`).

Integration limitation (read me)
--------------------------------
The existing exact-EPD path (``hetero.pdgnn_metapath._exact_epd``) consumes a NODE
filtration and computes its edge filtration *internally* as ``max`` over endpoints;
it does NOT accept an explicit per-edge filtration. So wiring this relation-aware
edge filtration into that EPD/PDGNN routine would require a *custom* union-find
0-dim sublevel-persistence routine that reads explicit edge values. To demonstrate
that the produced edge filtration is actually usable, this module ships a tiny,
standalone, dependency-free :func:`zero_dim_sublevel_persistence` (union-find over
sorted edge filtration values). It is illustrative only -- not wired into the main
training path.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------------- #
# Honesty constants (used verbatim; never claim exact PH / real PDGNN / EPD).
EDGE_FILTRATION_HONESTY_LABEL: str = (
    "experimental relation-aware edge filtration (prepared interface; "
    "extends standard node->edge max filtration)"
)
EDGE_FILTRATION_FEATURE_NAME: str = "experimental_relation_aware_edge_filtration"
EDGE_FILTRATION_MODES: Tuple[str, ...] = ("max", "relation_delay")


# ----------------------------------------------------------------------------- #
def relation_edge_filtration(
    node_filt: torch.Tensor,
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    mode: str = "max",
    delays: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Lift a node filtration to a (relation-aware) edge filtration.

    Parameters
    ----------
    node_filt : Tensor (N,)
        Scalar filtration value ``f(i)`` per node.
    edge_index : LongTensor (2, E)
        Endpoints ``(i, j)`` of each edge.
    edge_type : LongTensor (E,)
        Relation-type id ``rho`` in ``[0, R)`` per edge.
    mode : {"max", "relation_delay"}
        * ``"max"`` (default): ``g(i,j) = max(f(i), f(j))`` -- the standard,
          relation-blind lower-star edge rule.
        * ``"relation_delay"``: ``g(i,j,rho) = max(f(i),f(j)) + softplus(delta_rho)``
          where ``delta_rho = delays[rho]``. ``softplus >= 0`` guarantees the
          validity constraint ``g >= max(f(i), f(j))``.
    delays : Tensor (R,), optional
        Raw (pre-softplus) per-relation delays. Required logical input for
        ``"relation_delay"``; if ``None`` it defaults to zeros (so every relation
        gets the same positive delay ``softplus(0) = ln 2``). Ignored for ``"max"``.

    Returns
    -------
    Tensor (E,)
        Edge filtration values, guaranteed ``>= max(f(i), f(j))`` element-wise.
    """
    if mode not in EDGE_FILTRATION_MODES:
        raise ValueError(
            f"unknown edge_filtration mode {mode!r}; expected one of {EDGE_FILTRATION_MODES}"
        )
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError(f"edge_index must be (2, E); got {tuple(edge_index.shape)}")

    src, dst = edge_index[0].long(), edge_index[1].long()
    endpoint_max = torch.maximum(node_filt[src], node_filt[dst])

    if mode == "max":
        return endpoint_max

    # mode == "relation_delay"
    et = edge_type.long()
    if et.shape[0] != endpoint_max.shape[0]:
        raise ValueError(
            f"edge_type length {et.shape[0]} != number of edges {endpoint_max.shape[0]}"
        )
    if delays is None:
        num_rel = int(et.max().item()) + 1 if et.numel() > 0 else 1
        delays = torch.zeros(num_rel, dtype=node_filt.dtype, device=node_filt.device)
    else:
        delays = delays.to(dtype=node_filt.dtype, device=node_filt.device)
        if et.numel() > 0 and int(et.max().item()) >= delays.shape[0]:
            raise ValueError(
                f"edge_type id {int(et.max().item())} out of range for "
                f"delays of size {delays.shape[0]}"
            )
    # softplus(delta) >= 0  =>  g >= max(f_i, f_j)
    per_edge_delay = F.softplus(delays[et])
    return endpoint_max + per_edge_delay


# ----------------------------------------------------------------------------- #
class RelationDelayEdgeFiltration(nn.Module):
    """Holds learnable per-relation (raw, pre-softplus) delays.

    forward(node_filt, edge_index, edge_type) -> Tensor (E,) equal to
    ``relation_edge_filtration(..., mode="relation_delay", delays=self.raw_delays)``.

    The non-negativity of ``softplus`` guarantees the validity constraint
    ``g(i,j,rho) >= max(f(i), f(j))`` for any value the parameter takes during
    training, so optimisation can never violate the constraint.
    """

    def __init__(self, num_relations: int, init: float = 0.0):
        super().__init__()
        if num_relations < 1:
            raise ValueError(f"num_relations must be >= 1; got {num_relations}")
        self.num_relations = int(num_relations)
        self.raw_delays = nn.Parameter(torch.full((num_relations,), float(init)))

    def forward(
        self,
        node_filt: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        return relation_edge_filtration(
            node_filt, edge_index, edge_type,
            mode="relation_delay", delays=self.raw_delays,
        )

    def effective_delays(self) -> torch.Tensor:
        """Non-negative per-relation delays actually applied (``softplus(raw)``)."""
        return F.softplus(self.raw_delays)

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return f"num_relations={self.num_relations}"


# ----------------------------------------------------------------------------- #
def _DisjointSet(n: int):
    parent = list(range(n))

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:  # path compression
            parent[x], x = root, parent[x]
        return root

    return parent, find


def zero_dim_sublevel_persistence(
    node_filt: torch.Tensor,
    edge_index: torch.Tensor,
    edge_filt: torch.Tensor,
    num_nodes: Optional[int] = None,
) -> List[Tuple[float, float]]:
    """Standalone 0-dim sublevel persistence consuming EXPLICIT edge filtration.

    This is a tiny union-find ("elder rule") computation that, unlike the existing
    ``_exact_epd`` routine, reads a per-edge filtration value ``edge_filt`` instead
    of recomputing ``max`` over endpoints. It exists to demonstrate that the
    relation-aware edge filtration produced above is consumable; it is NOT wired
    into the main PDGNN/EPD training path (see module docstring).

    Components are born at node filtration values; an edge merges the two
    components of its endpoints at the edge's filtration value, killing the
    younger component (elder rule). One component survives with death ``+inf``.

    Returns a list of ``(birth, death)`` pairs (one infinite bar for the final
    surviving component).
    """
    f = np.asarray(node_filt.detach().cpu() if torch.is_tensor(node_filt) else node_filt,
                   dtype=np.float64).reshape(-1)
    n = int(num_nodes) if num_nodes is not None else f.shape[0]
    ei = (edge_index.detach().cpu().numpy() if torch.is_tensor(edge_index)
          else np.asarray(edge_index))
    ef = np.asarray(edge_filt.detach().cpu() if torch.is_tensor(edge_filt) else edge_filt,
                    dtype=np.float64).reshape(-1)
    if ei.shape[1] != ef.shape[0]:
        raise ValueError(
            f"edge_index has {ei.shape[1]} edges but edge_filt has {ef.shape[0]} values"
        )

    parent, find = _DisjointSet(n)
    # birth time of each component = min node value among its members (elder rule).
    birth = {i: float(f[i]) for i in range(n)}

    order = np.argsort(ef, kind="stable")  # process edges by increasing filtration
    pairs: List[Tuple[float, float]] = []
    for e in order:
        u, v = int(ei[0, e]), int(ei[1, e])
        ru, rv = find(u), find(v)
        if ru == rv:
            continue  # creates a 1-cycle; irrelevant for 0-dim
        bu, bv = birth[ru], birth[rv]
        death = float(ef[e])
        # younger (larger birth) dies; elder survives.
        if bu <= bv:
            elder, younger = ru, rv
            younger_birth = bv
        else:
            elder, younger = rv, ru
            younger_birth = bu
        parent[younger] = elder
        birth[elder] = min(bu, bv)
        # a finite bar only if the merge actually raises the death above birth
        pairs.append((younger_birth, death))

    # surviving components -> infinite bars
    roots = {find(i) for i in range(n)}
    for r in roots:
        pairs.append((birth[r], float("inf")))
    return pairs
