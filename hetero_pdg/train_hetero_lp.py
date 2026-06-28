"""Phase-1 end-to-end toy link-prediction training (fallback topology backend).

Pipeline: toy HeteroData -> split target edges -> observation graph (val/test edges
removed) -> meta-path projections -> topology features (FALLBACK descriptors, or the
type-aware unified filtration) -> HeteroTopoLinkPredictor -> BCE training -> AUC/AP.

Phase 1 uses fallback_topology_descriptors (NOT PDGNN / NOT EPD). Toy metrics only
validate pipeline correctness -- no performance claims.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn as nn

from hetero_pdg.data import (
    NODE_TYPES, make_toy_hetero_graph, split_target_edges,
    typed_negative_sampling, make_pair_batches, remove_target_edges_for_split, PairBatch,
)
from hetero_pdg.metapath import (
    DEFAULT_METAPATHS, project_metapath, project_all_metapaths, build_collapsed_baseline_graph,
)
from hetero_pdg.filtration import TypeAwareFiltrationMLP, build_homogeneous_view_with_type_ids
from hetero_pdg.models import HeteroTopoLinkPredictor
from hetero_pdg.topology_features import (
    compute_fallback_features_by_metapath, TOPOLOGY_BACKEND, TOPO_DIM,
)
from hetero_pdg.topology_adapter import compute_topology_features, BACKENDS

try:
    from sklearn.metrics import average_precision_score, roc_auc_score
    _HAS_SK = True
except Exception:  # pragma: no cover
    _HAS_SK = False

UNIFIED_DIM = 5
# Phase-3 train-level topology modes (the model itself still only sees its 5 base modes;
# these are mapped onto a base model mode + an advanced topology source).
ADVANCED_TOPO_MODES = ("typed_cycle_topology", "multi_slice_topology", "relation_epd_topology")
ALL_TOPO_MODES = tuple(HeteroTopoLinkPredictor.MODES) + ADVANCED_TOPO_MODES
FILTRATION_MODES = ("type_aware", "pair_conditioned")
EDGE_FILTRATION_MODES = ("max", "relation_delay")


def save_json(obj: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def _select_metapaths(target_type: str):
    specs = [s for s in DEFAULT_METAPATHS.values() if s.target_type == target_type]
    if not specs:
        raise ValueError(f"no default meta-path has target type {target_type!r}")
    return specs


def _pairs_to_set(t: torch.Tensor):
    return [(int(a), int(b)) for a, b in t.t()]


# ---- topology for collapsed / metapath modes (via the backend adapter) ---- #
def _fixed_topology(mode, obs, metaspecs, batch, backend, pdgnn_checkpoint, allow_fallback):
    bundles = ({"collapsed": build_collapsed_baseline_graph(obs, metaspecs)}
               if mode == "collapsed_topology"
               else project_all_metapaths(obs, metaspecs))
    res = compute_topology_features(
        bundles, batch, backend=backend, k=2, pdgnn_checkpoint=pdgnn_checkpoint,
        allow_fallback=allow_fallback, K=2, hop=2, max_nodes=30, epochs=15,
        n_train_samples=40, seed=0,
    )
    if mode == "collapsed_topology":
        feats, mask = res.features_by_metapath["collapsed"], None          # (B, dim)
    else:
        feats = torch.stack([res.features_by_metapath[s.name] for s in metaspecs], dim=1)
        mask = torch.stack([res.mask_by_metapath[s.name] for s in metaspecs], dim=1)
    return feats, mask, res                                                # +provenance


def _znorm_fit(x: torch.Tensor):
    mu = x.mean(dim=0, keepdim=True)
    sd = x.std(dim=0, keepdim=True) + 1e-8
    return mu, sd


# ---- unified-filtration differentiable topology ---- #
def _unified_vicinity(obs, batch, device):
    view = build_homogeneous_view_with_type_ids(obs)
    N = view["num_nodes"]
    off = view["offsets"]["paper"][0]
    g = nx.Graph(); g.add_nodes_from(range(N))
    ei = view["edge_index"].numpy()
    for a, b in zip(ei[0], ei[1]):
        g.add_edge(int(a), int(b))
    src = batch.src.tolist(); dst = batch.dst.tolist()
    B = len(src)
    A = np.zeros((B, N), dtype=np.float32)
    pu, pv, ncommon = [], [], []
    for i, (u, v) in enumerate(zip(src, dst)):
        a, b = off + int(u), off + int(v)
        had = g.has_edge(a, b)
        if had:
            g.remove_edge(a, b)
        common = list(set(g.neighbors(a)) & set(g.neighbors(b)))
        if had:
            g.add_edge(a, b)
        pu.append(a); pv.append(b); ncommon.append(len(common))
        if common:
            A[i, common] = 1.0 / len(common)
    return {
        "view": view, "N": N,
        "pu": torch.tensor(pu, dtype=torch.long, device=device),
        "pv": torch.tensor(pv, dtype=torch.long, device=device),
        "A": torch.tensor(A, device=device),
        "ncommon": torch.tensor(ncommon, dtype=torch.float32, device=device),
    }


def _unified_topo(filt, obs, vic) -> torch.Tensor:
    raw = torch.cat([filt(obs)[nt] for nt in NODE_TYPES])  # (N,) differentiable
    fu, fv = raw[vic["pu"]], raw[vic["pv"]]
    mc = vic["A"] @ raw
    return torch.stack([fu, fv, (fu - fv).abs(), mc, vic["ncommon"]], dim=1)  # (B, 5)


def _metrics(label: np.ndarray, prob: np.ndarray):
    if not _HAS_SK or len(set(label.tolist())) < 2:
        return None, None
    return float(roc_auc_score(label, prob)), float(average_precision_score(label, prob))


def run_experiment(
    topo_mode: str,
    target_rel: Tuple[str, str, str] = ("paper", "cites", "paper"),
    dataset: str = "toy",
    epochs: int = 2,
    hidden_dim: int = 32,
    topo_hidden_dim: int = 32,
    lr: float = 1e-3,
    neg_ratio: float = 1.0,
    seed: int = 0,
    device: str = "cpu",
    output_dir: Optional[str] = None,
    planted: bool = False,
    topo_backend: str = "fallback",
    pdgnn_checkpoint: Optional[str] = None,
    allow_fallback: bool = False,
    max_target_edges: Optional[int] = None,
    filtration_mode: str = "type_aware",
    edge_filtration_mode: str = "max",
    num_slices: int = 3,
) -> Dict:
    """Train + evaluate one topology mode on the toy OR a small real dataset.

    Phase-3 experimental knobs (default values reproduce Phase-1/2 behaviour exactly):
      filtration_mode='pair_conditioned' -> pair-conditioned filtration for collapsed/metapath modes;
      topo_mode in {typed_cycle_topology, multi_slice_topology} -> Phase-3 topology sources;
      edge_filtration_mode='relation_delay' -> recorded (prepared interface; see edge_filtration.py).
    """
    if topo_mode not in ALL_TOPO_MODES:
        raise ValueError(f"unknown topo_mode {topo_mode!r}; expected {ALL_TOPO_MODES}")
    if filtration_mode not in FILTRATION_MODES:
        raise ValueError(f"unknown filtration_mode {filtration_mode!r}")
    if edge_filtration_mode not in EDGE_FILTRATION_MODES:
        raise ValueError(f"unknown edge_filtration_mode {edge_filtration_mode!r}")
    torch.manual_seed(seed); np.random.seed(seed)
    dev = torch.device(device)
    # Real backends import the TLC-GNN engine, which os.chdir()s to the TLC-GNN repo
    # (diffusion_features.py). Pin the output dir to an absolute path up front so
    # metrics land where the user asked regardless of any cwd change.
    if output_dir:
        output_dir = os.path.abspath(output_dir)

    metaspecs_real = None
    if dataset == "toy":
        data = make_toy_hetero_graph(seed=seed, planted=planted)
    else:
        from hetero_pdg.real_data import load_real_hetero
        if topo_mode == "unified_filter_topology":
            raise ValueError(
                "unified_filter_topology is toy-schema-specific in Phase 1; use "
                "no_topology / collapsed_topology / metapath_topology_* for real datasets")
        data, target_rel, metaspecs_real = load_real_hetero(
            dataset, max_target_edges=max_target_edges, seed=seed)
    t = target_rel[0]
    if target_rel[0] != target_rel[2]:
        raise ValueError("target must be a same-type relation (e.g. paper,cites,paper)")
    sp = split_target_edges(data, target_rel, val_frac=0.15, test_frac=0.15, seed=seed)
    held = torch.cat([sp.val_pos, sp.test_pos], dim=1)
    obs = remove_target_edges_for_split(data, held, target_rel)

    def neg(npos, sd, excl):
        return typed_negative_sampling(data, target_rel, max(1, int(round(neg_ratio * npos))),
                                       seed=sd, exclude=excl)
    tr_neg = neg(sp.train_pos.shape[1], seed + 11, None)
    va_neg = neg(sp.val_pos.shape[1], seed + 12, _pairs_to_set(tr_neg))
    te_neg = neg(sp.test_pos.shape[1], seed + 13, _pairs_to_set(tr_neg) + _pairs_to_set(va_neg))

    tr = make_pair_batches(sp.train_pos, tr_neg, t, t, seed=seed)[0]
    va = make_pair_batches(sp.val_pos, va_neg, t, t, shuffle=False)[0]
    te = make_pair_batches(sp.test_pos, te_neg, t, t, shuffle=False)[0]

    node_in_dims = {nt: int(data[nt].x.size(1)) for nt in data.node_types
                    if getattr(data[nt], "x", None) is not None}
    node_feats = {nt: data[nt].x.to(dev) for nt in node_in_dims}

    _needs_metapaths = (topo_mode.startswith(("metapath", "collapsed"))
                        or topo_mode in ADVANCED_TOPO_MODES)
    if metaspecs_real is not None:
        metaspecs = metaspecs_real
    else:
        metaspecs = _select_metapaths(t) if _needs_metapaths else []
    n_mp = len(metaspecs) if topo_mode.startswith("metapath") else 2
    _meta_modes = ("collapsed_topology", "metapath_topology_concat", "metapath_topology_attention")
    requested_backend = (topo_backend if (topo_mode in _meta_modes and filtration_mode == "type_aware")
                         else "n/a")
    used_backend, feature_kind, fallback_triggered = "none", "none", False

    filt = None
    adv_params = []                 # extra trainable params (unified / pair-cond / multi-slice)
    recompute = None                # closure(split_name, batch) -> topo tensor (trainable modes)
    topo = {"train": None, "val": None, "test": None}
    mask = {"train": None, "val": None, "test": None}
    vic = {}
    topo_dim = TOPO_DIM
    model_mode = topo_mode          # what the predictor sees (advanced modes map onto base modes)

    if topo_mode == "no_topology":
        model_mode = "no_topology"

    elif topo_mode == "unified_filter_topology":
        filt = TypeAwareFiltrationMLP(obs).to(dev); adv_params += list(filt.parameters())
        for nm, b in (("train", tr), ("val", va), ("test", te)):
            vic[nm] = _unified_vicinity(obs, b, dev)
        topo_dim = UNIFIED_DIM
        used_backend = feature_kind = "type_aware_filtration"
        recompute = lambda nm, b: _unified_topo(filt, obs, vic[nm])

    elif topo_mode == "typed_cycle_topology":              # Phase-3: typed cycle proxy descriptors
        from hetero_pdg.typed_cycle import typed_cycle_signature_features, typed_cycle_dim
        bundles = project_all_metapaths(obs, metaspecs)
        model_mode = "collapsed_topology"
        topo_dim = typed_cycle_dim(len(metaspecs), include_density=True)
        used_backend, feature_kind = "typed_cycle", "typed_cycle_proxy_descriptors"
        for nm, b in (("train", tr), ("val", va), ("test", te)):
            topo[nm] = typed_cycle_signature_features(obs, b, bundles, k=1).to(dev)

    elif topo_mode == "multi_slice_topology":              # Phase-3: sliced filtration approximation
        from hetero_pdg.multi_slice import MultiSliceFiltration
        bundles = project_all_metapaths(obs, metaspecs)
        msf = MultiSliceFiltration(num_slices=num_slices, seed=seed).to(dev)
        _ = msf(obs, bundles, tr)                          # warm-up: build lazy params
        adv_params += list(msf.parameters())
        model_mode = "collapsed_topology"; topo_dim = int(msf.fused_dim)
        used_backend, feature_kind = "multi_slice", "sliced_filtration_approximation"
        recompute = lambda nm, b: msf(obs, bundles, b).to(dev)

    elif topo_mode == "relation_epd_topology":            # Phase-4: edge-type-aware 0-dim EPD
        from hetero_pdg.relation_epd import compute_relation_epd_features, PI_DIM
        bundles = project_all_metapaths(obs, metaspecs)
        model_mode = "collapsed_topology"; topo_dim = PI_DIM
        used_backend = "relation_epd"
        feature_kind = ("relation_aware_0dim_epd_typed" if edge_filtration_mode == "relation_delay"
                        else "relation_aware_0dim_epd_untyped")
        for nm, b in (("train", tr), ("val", va), ("test", te)):
            topo[nm] = compute_relation_epd_features(bundles, b, mode=edge_filtration_mode).to(dev)

    elif filtration_mode == "pair_conditioned":           # Phase-3: pair-conditioned filtration
        from hetero_pdg.pair_filtration import (PairConditionedFiltrationMLP,
                                                compute_pair_conditioned_lp_features,
                                                PAIR_CONDITIONED_FEATURE_NAMES)
        pcm = PairConditionedFiltrationMLP(obs).to(dev); adv_params += list(pcm.parameters())
        used_backend = "pair_conditioned"
        feature_kind = "pair_conditioned_filtration_approximate"   # matches docs' "approximate"
        topo_dim = len(PAIR_CONDITIONED_FEATURE_NAMES)             # was hardcoded 6
        if topo_mode == "collapsed_topology":
            bundle = build_collapsed_baseline_graph(obs, metaspecs)
            recompute = lambda nm, b: compute_pair_conditioned_lp_features(pcm, obs, bundle, b).features.to(dev)
        else:                                              # metapath concat / attention
            mp_bundles = [project_metapath(obs, s) for s in metaspecs]; n_mp = len(mp_bundles)
            recompute = lambda nm, b: torch.stack(
                [compute_pair_conditioned_lp_features(pcm, obs, bd, b).features for bd in mp_bundles],
                dim=1).to(dev)

    else:                                                  # Phase 1/2: fixed fallback / real backend
        ftr, mtr, res_tr = _fixed_topology(topo_mode, obs, metaspecs, tr, topo_backend,
                                           pdgnn_checkpoint, allow_fallback)
        fva, mva, _ = _fixed_topology(topo_mode, obs, metaspecs, va, topo_backend,
                                      pdgnn_checkpoint, allow_fallback)
        fte, mte, _ = _fixed_topology(topo_mode, obs, metaspecs, te, topo_backend,
                                      pdgnn_checkpoint, allow_fallback)
        used_backend, feature_kind = res_tr.backend, res_tr.feature_kind
        fallback_triggered = res_tr.fallback_triggered
        topo_dim = int(ftr.shape[-1])
        mu, sd = _znorm_fit(ftr)
        topo = {"train": ((ftr - mu) / sd).to(dev), "val": ((fva - mu) / sd).to(dev),
                "test": ((fte - mu) / sd).to(dev)}
        mask = {"train": mtr.to(dev) if mtr is not None else None,
                "val": mva.to(dev) if mva is not None else None,
                "test": mte.to(dev) if mte is not None else None}

    model = HeteroTopoLinkPredictor(node_in_dims, model_mode, hidden_dim=hidden_dim,
                                    topo_dim=topo_dim, topo_hidden_dim=topo_hidden_dim,
                                    n_metapaths=n_mp, unified_dim=UNIFIED_DIM).to(dev)
    opt = torch.optim.Adam(list(model.parameters()) + adv_params, lr=lr)
    bce = nn.BCEWithLogitsLoss()

    last_loss = float("nan")
    for _ in range(epochs):
        model.train()
        if recompute is not None:
            topo["train"] = recompute("train", tr)
        opt.zero_grad()
        logits = model(node_feats, tr, topo=None if model_mode == "no_topology" else topo["train"],
                       topo_mask=mask["train"])
        loss = bce(logits, tr.label.to(dev))
        loss.backward(); opt.step()
        last_loss = float(loss)

    model.eval()
    out = {}
    with torch.no_grad():
        for nm, b in (("val", va), ("test", te)):
            if recompute is not None:
                topo[nm] = recompute(nm, b)
            prob = torch.sigmoid(model(node_feats, b,
                   topo=None if model_mode == "no_topology" else topo[nm],
                   topo_mask=mask[nm])).cpu().numpy()
            auc, ap = _metrics(b.label.numpy(), prob)
            out[f"{nm}_auc"], out[f"{nm}_ap"] = auc, ap

    result = {
        "dataset": dataset, "topo_mode": topo_mode,
        "topo_backend": feature_kind,                 # backward-compat: feature-kind label
        "topo_backend_requested": requested_backend,
        "topo_backend_used": used_backend,
        "feature_kind": feature_kind,
        "fallback_triggered": bool(fallback_triggered),
        "filtration_mode": filtration_mode,
        "edge_filtration_mode": edge_filtration_mode,
        "num_slices": int(num_slices) if topo_mode == "multi_slice_topology" else None,
        "pdgnn_checkpoint": pdgnn_checkpoint,
        "target_rel": list(target_rel), "n_metapaths": int(n_mp) if topo_mode.startswith("metapath") else 0,
        "topo_dim": int(topo_dim) if topo_mode != "no_topology" else 0,
        "n_train": int(tr.label.numel()), "n_val": int(va.label.numel()), "n_test": int(te.label.numel()),
        "val_auc": out["val_auc"], "val_ap": out["val_ap"],
        "test_auc": out["test_auc"], "test_ap": out["test_ap"],
        "final_train_loss": last_loss, "epochs": int(epochs), "seed": int(seed),
        "hidden_dim": int(hidden_dim), "device": str(dev), "planted": bool(planted),
        "phase": 2,
        "note": ("Toy metrics validate pipeline correctness only; no performance claim. "
                 "feature_kind states provenance (fallback descriptors are NOT PDGNN/EPD)."),
    }
    if output_dir:
        save_json({"topo_mode": topo_mode, "target_rel": list(target_rel), "dataset": dataset,
                   "epochs": epochs, "hidden_dim": hidden_dim, "topo_hidden_dim": topo_hidden_dim,
                   "lr": lr, "neg_ratio": neg_ratio, "seed": seed, "device": device,
                   "planted": planted, "topo_backend": topo_backend,
                   "pdgnn_checkpoint": pdgnn_checkpoint, "allow_fallback": allow_fallback,
                   "max_target_edges": max_target_edges, "filtration_mode": filtration_mode,
                   "edge_filtration_mode": edge_filtration_mode, "num_slices": num_slices},
                  os.path.join(output_dir, "config.json"))
        save_json(result, os.path.join(output_dir, "metrics.json"))
    return result


def main():
    ap = argparse.ArgumentParser(description="hetero_pdg Phase-1 toy link prediction")
    ap.add_argument("--dataset", default="toy", choices=["toy", "acm", "imdb"])
    ap.add_argument("--target-rel", default="paper,cites,paper")
    ap.add_argument("--topo-mode", default="metapath_topology_concat",
                    choices=list(ALL_TOPO_MODES))
    ap.add_argument("--filtration-mode", default="type_aware", choices=list(FILTRATION_MODES))
    ap.add_argument("--edge-filtration-mode", default="max", choices=list(EDGE_FILTRATION_MODES))
    ap.add_argument("--num-slices", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--hidden-dim", type=int, default=32)
    ap.add_argument("--topo-hidden-dim", type=int, default=32)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--neg-ratio", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--planted", action="store_true")
    ap.add_argument("--topo-backend", default="fallback", choices=list(BACKENDS))
    ap.add_argument("--pdgnn-checkpoint", default=None)
    ap.add_argument("--allow-fallback", default="false", choices=["true", "false"])
    ap.add_argument("--max-target-edges", type=int, default=None,
                    help="cap target-relation edges (real-data smoke speed)")
    ap.add_argument("--output-dir", default="runs/hetero_pdg_toy")
    a = ap.parse_args()
    target_rel = tuple(a.target_rel.split(","))
    if len(target_rel) != 3:
        raise SystemExit("--target-rel must be SRC,REL,DST")
    res = run_experiment(
        topo_mode=a.topo_mode, target_rel=target_rel, dataset=a.dataset, epochs=a.epochs,
        hidden_dim=a.hidden_dim, topo_hidden_dim=a.topo_hidden_dim, lr=a.lr,
        neg_ratio=a.neg_ratio, seed=a.seed, device=a.device, planted=a.planted,
        topo_backend=a.topo_backend, pdgnn_checkpoint=a.pdgnn_checkpoint,
        allow_fallback=(a.allow_fallback == "true"), max_target_edges=a.max_target_edges,
        filtration_mode=a.filtration_mode, edge_filtration_mode=a.edge_filtration_mode,
        num_slices=a.num_slices, output_dir=a.output_dir,
    )
    print(json.dumps(res, indent=2))
    print(f"[saved] {a.output_dir}/config.json , {a.output_dir}/metrics.json")


if __name__ == "__main__":
    main()
