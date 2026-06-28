"""Topology backend abstraction (Phase 2).

Backends:
  fallback   -- deterministic ``fallback_topology_descriptors`` (Phase 1; default).
                NOT PDGNN, NOT EPD. Always available, zero external dependency.
  real_tlc   -- exact extended persistence (gudhi lower-star) -> persistence image.
                Real TLC/EPD features.
  real_pdgnn -- neural PDGNN-approximated EPD -> persistence image (train on the
                projected graph's exact labels, or load a checkpoint).

The real backends reuse the EXISTING TLC-GNN engine
(``Knowledge_Distillation.pdgnn_modern``, ``hetero.pdgnn_metapath``,
``sg2dgm.PersistenceImager``) via lazy import, so the fallback path stays fully
self-contained. If a real backend is requested but the engine is not importable, a
clear ``RealBackendUnavailable`` error is raised -- there is NO silent fallback
unless ``allow_fallback=True``.

Leakage: the target edge (u, v) is removed from the projected graph before its
edge-vicinity persistence is read, then restored.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import networkx as nx
import torch

from hetero_pdg.data import PairBatch
from hetero_pdg.topology_features import (
    compute_fallback_features_by_metapath, TOPOLOGY_BACKEND,
)

BACKEND_FALLBACK = "fallback"
BACKEND_PDGNN = "real_pdgnn"
BACKEND_TLC = "real_tlc"
BACKENDS = (BACKEND_FALLBACK, BACKEND_PDGNN, BACKEND_TLC)

FEATURE_KIND = {
    BACKEND_FALLBACK: "fallback_topology_descriptors",
    BACKEND_TLC: "real_tlc_exact_epd_persistence_image",
    BACKEND_PDGNN: "real_pdgnn_approx_epd_persistence_image",
}

# What the real backends need (reported in the clear error).
REAL_BACKEND_REQUIREMENTS = (
    "TLC-GNN repo on PYTHONPATH (export PYTHONPATH=/mnt/data/users/junyoungpark/code/TLC-GNN), "
    "plus packages: gudhi, torch_scatter, sg2dgm; modules: "
    "Knowledge_Distillation.pdgnn_modern, hetero.pdgnn_metapath, node_ph_features."
)


class RealBackendUnavailable(RuntimeError):
    """Raised when a real backend is requested but its engine cannot be imported."""


@dataclass
class AdapterResult:
    features_by_metapath: Dict[str, torch.Tensor]
    mask_by_metapath: Dict[str, torch.Tensor]
    backend: str                 # backend actually used
    feature_kind: str            # human-readable provenance
    fallback_triggered: bool     # True iff a real backend was requested but fell back
    requested_backend: str
    feat_dim: int
    reason: str = ""             # why fallback was triggered (if any)


def _require_engine() -> dict:
    """Lazy-import the TLC-GNN real-topology engine; raise a clear error if missing."""
    try:
        from hetero.pdgnn_metapath import (
            _graph_hks, _ego_filt_edges, _exact_epd,
            gen_training_samples, train_pdgnn_metapath, device,
        )
        from Knowledge_Distillation.pdgnn_modern import PDGNN
        from sg2dgm import PersistenceImager as pimg_mod
        from node_ph_features import PI_RES
    except Exception as e:  # noqa: BLE001 - report any import failure clearly
        raise RealBackendUnavailable(
            f"real PDGNN/TLC backend is unavailable: {type(e).__name__}: {e}. "
            f"Requires {REAL_BACKEND_REQUIREMENTS}"
        )
    return {
        "_graph_hks": _graph_hks, "_ego_filt_edges": _ego_filt_edges,
        "_exact_epd": _exact_epd, "gen_training_samples": gen_training_samples,
        "train_pdgnn_metapath": train_pdgnn_metapath, "PDGNN": PDGNN,
        "imager": pimg_mod.PersistenceImager(resolution=PI_RES), "PI_RES": PI_RES,
        "device": device,
    }


def _pair_ego(g: nx.Graph, u: int, v: int, hop: int, node_filt: dict, max_nodes: int):
    """Union k-hop ego of {u, v} with local re-indexing -> (filt(m,), ei(2,E), nodes)."""
    nodes = set(nx.ego_graph(g, u, radius=hop).nodes()) | set(
        nx.ego_graph(g, v, radius=hop).nodes())
    if len(nodes) > max_nodes:
        nodes = set(sorted(nodes, key=lambda nd: node_filt.get(nd, 0.0))[:max_nodes]) | {u, v}
    sub = g.subgraph(nodes)
    nl = list(sub.nodes())
    remap = {nd: i for i, nd in enumerate(nl)}
    filt = np.array([node_filt.get(nd, 0.0) for nd in nl], dtype=np.float64)
    if sub.number_of_edges() == 0:
        ei = np.zeros((2, 0), dtype=np.int64)
    else:
        e = np.array([(remap[a], remap[b]) for a, b in sub.edges()], dtype=np.int64).T
        ei = np.concatenate([e, e[[1, 0]]], axis=1)
    return filt, ei, nl


class _RealMetapathTopology:
    """Per projected-graph real persistence-image extractor (TLC exact or PDGNN)."""

    def __init__(self, g, eng, backend, K=2, hop=2, max_nodes=40, hidden=32,
                 layers=3, epochs=15, n_train_samples=40, seed=0, checkpoint=None):
        self.g, self.eng, self.backend = g, eng, backend
        self.K, self.hop, self.max_nodes = K, hop, max_nodes
        self.per_leg = eng["PI_RES"] * eng["PI_RES"]
        self.feat_dim = self.per_leg * K
        self.hks = eng["_graph_hks"](g, K)
        self.node_filt = [{nd: float(self.hks[nd, k]) for nd in g.nodes()} for k in range(K)]
        self.model = None
        if backend == BACKEND_PDGNN:
            if checkpoint:
                try:
                    ckpt = torch.load(checkpoint, map_location=eng["device"], weights_only=False)
                    if isinstance(ckpt, dict) and "state_dict" in ckpt:        # {state_dict, config}
                        cfg = ckpt.get("config", {}) or {}
                        h = int(cfg.get("hidden_dim", hidden)); L = int(cfg.get("num_layers", layers))
                        state = ckpt["state_dict"]
                    elif hasattr(ckpt, "state_dict"):                          # a full nn.Module
                        h, L, state = hidden, layers, ckpt.state_dict()
                    else:                                                      # a bare state_dict
                        h, L, state = hidden, layers, ckpt
                    self.model = eng["PDGNN"](hidden_dim=h, num_layers=L).to(eng["device"])
                    self.model.load_state_dict(state)
                except Exception as e:  # noqa: BLE001
                    raise RealBackendUnavailable(
                        f"failed to load --pdgnn-checkpoint {checkpoint!r}: {e}")
            else:
                samples = eng["gen_training_samples"](g, self.hks, hop, max_nodes,
                                                      n_train_samples, seed)
                self.model = eng["train_pdgnn_metapath"](
                    samples, hidden=hidden, layers=layers, epochs=epochs, verbose=False)
            self.model.eval()

    @torch.no_grad()
    def pair_pi(self, u: int, v: int, remove_target: bool = True):
        """(feat_dim,) PI over the (u,v) edge-vicinity (target edge removed); + covered flag."""
        restored = None
        if remove_target and self.g.has_edge(u, v):
            restored = dict(self.g[u][v]); self.g.remove_edge(u, v)
        try:
            out = np.zeros(self.feat_dim, dtype=np.float64)
            covered = False
            for k in range(self.K):
                filt, ei, _ = _pair_ego(self.g, u, v, self.hop, self.node_filt[k], self.max_nodes)
                if ei.shape[1] == 0:
                    continue
                covered = True
                if self.backend == BACKEND_TLC:
                    pts = self.eng["_exact_epd"](filt, ei)
                else:  # real_pdgnn
                    ft = torch.tensor(filt.reshape(-1, 1).astype(np.float32), device=self.eng["device"])
                    et = torch.tensor(ei.astype(np.int64), device=self.eng["device"])
                    pts = self.model(ft, et).cpu().numpy()
                    pts = pts[pts[:, 1] > pts[:, 0]]
                if pts is not None and len(pts):
                    out[k * self.per_leg:(k + 1) * self.per_leg] = np.asarray(
                        self.eng["imager"].transform(np.asarray(pts, dtype=np.float64))).reshape(-1)
        finally:
            if restored is not None:
                self.g.add_edge(u, v, **restored)
        return out, covered


def compute_topology_features(
    bundles: Dict[str, "object"], pair_batch: PairBatch,
    backend: str = BACKEND_FALLBACK, k: int = 2,
    pdgnn_checkpoint: Optional[str] = None, allow_fallback: bool = False,
    K: int = 2, hop: int = 2, max_nodes: int = 40, epochs: int = 15,
    n_train_samples: int = 40, hidden: int = 32, layers: int = 3, seed: int = 0,
) -> AdapterResult:
    """Compute per-meta-path topology features with the requested backend.

    fallback -> deterministic descriptors. real_tlc/real_pdgnn -> persistence images
    via the TLC-GNN engine (or a clear error / opt-in fallback if unavailable).
    """
    if backend not in BACKENDS:
        raise ValueError(f"unknown topology backend {backend!r}; expected {BACKENDS}")

    def _fallback(triggered: bool, reason: str = "") -> AdapterResult:
        tf = compute_fallback_features_by_metapath(bundles, pair_batch, k=k)
        return AdapterResult(tf.features_by_metapath, tf.mask_by_metapath,
                             BACKEND_FALLBACK, FEATURE_KIND[BACKEND_FALLBACK],
                             triggered, backend, feat_dim=6, reason=reason)

    if backend == BACKEND_FALLBACK:
        return _fallback(False)

    try:
        eng = _require_engine()
    except RealBackendUnavailable as e:
        if allow_fallback:
            return _fallback(True, reason=str(e))
        raise

    src, dst = pair_batch.src.tolist(), pair_batch.dst.tolist()
    B = len(src)
    feats: Dict[str, torch.Tensor] = {}
    masks: Dict[str, torch.Tensor] = {}
    feat_dim = eng["PI_RES"] * eng["PI_RES"] * K
    for name, bundle in bundles.items():
        rt = _RealMetapathTopology(bundle.graph, eng, backend, K=K, hop=hop,
                                   max_nodes=max_nodes, hidden=hidden, layers=layers,
                                   epochs=epochs, n_train_samples=n_train_samples,
                                   seed=seed, checkpoint=pdgnn_checkpoint)
        F = np.zeros((B, rt.feat_dim), dtype=np.float32)
        M = np.zeros(B, dtype=bool)
        for i, (u, v) in enumerate(zip(src, dst)):
            u, v = int(u), int(v)
            if u >= bundle.num_nodes or v >= bundle.num_nodes:
                continue
            vec, covered = rt.pair_pi(u, v)
            F[i] = vec
            M[i] = covered
        feats[name] = torch.from_numpy(F)
        masks[name] = torch.from_numpy(M)
    return AdapterResult(feats, masks, backend, FEATURE_KIND[backend], False, backend,
                         feat_dim=feat_dim)
