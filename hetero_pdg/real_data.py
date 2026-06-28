"""Real HGB datasets (two small ones) wired into the hetero_pdg LP pipeline.

We reuse the link-prediction pipeline as-is by always predicting a **same-type**
target relation over the target node type, so the meta-path topology framework applies:

  acm  : target = stored (paper, cite, paper); topology meta-paths = PAP + cite-obs.
  imdb : no stored same-type edge, so we *derive* (movie, codir, movie) from the
         movie-director-movie meta-path (movies sharing a director) and predict that;
         topology meta-paths = MAM + codir-obs.

Real datasets need the TLC-GNN repo on PYTHONPATH (for `hetero.metapath_graph.load_hgb`
= PyG `HGBDataset`) plus the cached HGB data. If unavailable, a clear
`RealDatasetUnavailable` is raised. This module does NOT modify the TLC-GNN repo.

NOTE: `unified_filter_topology` is toy-schema-specific in Phase 1 (its type-aware
filtration hardcodes the author/paper/field schema); real datasets therefore support
no_topology / collapsed_topology / metapath_topology_{concat,attention}.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from hetero_pdg.metapath import MetaPathSpec, project_metapath

EdgeType = Tuple[str, str, str]


class RealDatasetUnavailable(RuntimeError):
    """Raised when a real HGB dataset cannot be loaded (engine/data missing)."""


# name -> (HGB name, target spec). For derived targets we give the meta-path to build.
REAL_REGISTRY: Dict[str, dict] = {
    "acm": {
        "hgb": "ACM",
        "target_type": "paper",
        "stored_target": ("paper", "cite", "paper"),
        "side_metapaths": [("PAP", [("paper", "to", "author"), ("author", "to", "paper")])],
    },
    "imdb": {
        "hgb": "IMDB",
        "target_type": "movie",
        "derived_target": ("movie", "codir", "movie"),         # built from MDM below
        "derive_from": [("movie", "to", "director"), ("director", "to", "movie")],
        "side_metapaths": [("MAM", [("movie", ">actorh", "actor"), ("actor", "to", "movie")])],
    },
}


def _load_hgb(hgb_name: str):
    try:
        from hetero.metapath_graph import load_hgb
    except Exception as e:  # noqa: BLE001
        raise RealDatasetUnavailable(
            f"real datasets require the TLC-GNN repo on PYTHONPATH "
            f"(export PYTHONPATH=/mnt/data/users/junyoungpark/code/TLC-GNN) and cached HGB "
            f"data: {type(e).__name__}: {e}")
    try:
        return load_hgb(hgb_name)
    except Exception as e:  # noqa: BLE001
        raise RealDatasetUnavailable(f"failed to load HGB {hgb_name!r}: {e}")


def _subsample_cols(ei: torch.Tensor, k: int, seed: int) -> torch.Tensor:
    if ei.shape[1] <= k:
        return ei
    idx = np.random.RandomState(seed).choice(ei.shape[1], size=k, replace=False)
    return ei[:, torch.as_tensor(np.sort(idx), dtype=torch.long)]


def load_real_hetero(
    name: str, max_target_edges: Optional[int] = None, seed: int = 0,
) -> Tuple[object, EdgeType, List[MetaPathSpec]]:
    """Return (HeteroData, target_rel, topo_metaspecs) for a small real dataset.

    The target is always same-type; topo_metaspecs = side meta-path(s) + the target
    relation's own observed structure (the standard LP "predict missing edges from
    observed edges + side info" setup). ``max_target_edges`` caps the target relation
    for fast smoke runs.
    """
    name = name.lower()
    if name not in REAL_REGISTRY:
        raise ValueError(f"unknown real dataset {name!r}; have {sorted(REAL_REGISTRY)}")
    cfg = REAL_REGISTRY[name]
    d = _load_hgb(cfg["hgb"])
    tt = cfg["target_type"]

    if "stored_target" in cfg:
        target_rel = cfg["stored_target"]
        if max_target_edges:
            d[target_rel].edge_index = _subsample_cols(d[target_rel].edge_index,
                                                       max_target_edges, seed)
    else:  # derive a same-type target from a meta-path (e.g. IMDB co-director)
        target_rel = cfg["derived_target"]
        mp = MetaPathSpec("__derive__", cfg["derive_from"])
        A = project_metapath(d, mp).adjacency
        A = A.tocoo() if hasattr(A, "tocoo") else None
        pairs = ([(int(i), int(j)) for i, j, v in zip(A.row, A.col, A.data) if i < j and v > 0]
                 if A is not None else [])
        rng = np.random.RandomState(seed)
        if max_target_edges and len(pairs) > max_target_edges:
            pairs = [pairs[k] for k in np.sort(rng.choice(len(pairs), max_target_edges, replace=False))]
        if not pairs:
            raise RealDatasetUnavailable(f"{name}: derived target {target_rel} is empty")
        # store one directed column per undirected pair (i<j) so split_target_edges
        # cannot place (i,j) and (j,i) in different splits (leakage); topology
        # symmetrises the relation anyway.
        d[target_rel].edge_index = torch.tensor(pairs, dtype=torch.long).t().contiguous()

    # topology meta-paths = side meta-path(s) + the (observed) target relation channel
    metaspecs = [MetaPathSpec(n, ets) for (n, ets) in cfg["side_metapaths"]]
    metaspecs.append(MetaPathSpec(f"{tt[:1].upper()}REL", [target_rel]))
    return d, target_rel, metaspecs
