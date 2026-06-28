"""Phase-2 RED tests: topology backend abstraction (fallback / real_pdgnn / real_tlc).

The real backends need the TLC-GNN repo on PYTHONPATH. These tests are dual-accept:
run standalone they exercise the *clear-error* path; run with PYTHONPATH=TLC-GNN they
exercise the *real* path. Either way the fallback path is fully deterministic.
"""
import torch
import pytest

from hetero_pdg.data import make_toy_hetero_graph, PairBatch
from hetero_pdg.metapath import DEFAULT_METAPATHS, project_all_metapaths
from hetero_pdg.topology_adapter import (
    BACKENDS,
    FEATURE_KIND,
    AdapterResult,
    RealBackendUnavailable,
    compute_topology_features,
)


def _bundles():
    d = make_toy_hetero_graph(seed=0)
    return project_all_metapaths(d, [DEFAULT_METAPATHS["PFP"], DEFAULT_METAPATHS["PCP"]])


def _pb():
    return PairBatch(torch.tensor([0, 1, 2]), torch.tensor([1, 2, 3]),
                     torch.ones(3), "paper", "paper")


def test_backends_constant():
    assert set(BACKENDS) == {"fallback", "real_pdgnn", "real_tlc"}


def test_fallback_via_adapter_labeled_correctly():
    res = compute_topology_features(_bundles(), _pb(), backend="fallback")
    assert isinstance(res, AdapterResult)
    assert res.backend == "fallback"
    assert res.feature_kind == "fallback_topology_descriptors"
    assert res.fallback_triggered is False
    for name in ("PFP", "PCP"):
        assert res.features_by_metapath[name].shape == (3, 6)
    # fallback must never be mislabeled as PDGNN/EPD
    assert "pdgnn" not in res.feature_kind.lower() and "epd" not in res.feature_kind.lower()


def test_unknown_backend_raises_value_error():
    with pytest.raises(ValueError):
        compute_topology_features(_bundles(), _pb(), backend="bogus")


def test_real_backend_smoke_or_clear_error():
    try:
        res = compute_topology_features(_bundles(), _pb(), backend="real_tlc",
                                        K=2, hop=2, max_nodes=20)
    except RealBackendUnavailable as e:
        assert "PYTHONPATH" in str(e)             # actionable, names the fix
        return
    assert res.backend == "real_tlc"
    assert res.fallback_triggered is False
    assert "epd" in res.feature_kind.lower() and "fallback" not in res.feature_kind.lower()
    dim = next(iter(res.features_by_metapath.values())).shape[1]
    assert dim == 25 * 2                          # PI_RES^2 * K


def test_real_requested_unavailable_without_allow_fallback_raises_or_runs():
    try:
        res = compute_topology_features(_bundles(), _pb(), backend="real_pdgnn",
                                        allow_fallback=False, K=2, max_nodes=20,
                                        epochs=3, n_train_samples=8)
    except RealBackendUnavailable:
        return                                    # standalone: no silent fallback
    assert res.backend == "real_pdgnn"            # with engine: real ran


def test_allow_fallback_flag_when_real_unavailable():
    res = compute_topology_features(_bundles(), _pb(), backend="real_pdgnn",
                                    allow_fallback=True, K=2, max_nodes=20,
                                    epochs=3, n_train_samples=8)
    assert res.backend in ("real_pdgnn", "fallback")
    if res.backend == "fallback":
        assert res.fallback_triggered is True
        assert res.feature_kind == "fallback_topology_descriptors"
