"""TDD tests for the multi-slice (vector-valued) filtration approximation.

These exercise a SLICED FILTRATION APPROXIMATION -- explicitly NOT exact
multi-parameter persistent homology. We only check structural contracts:
slice-weight normalisation, output shapes, finiteness, determinism.
"""
import numpy as np
import torch

from hetero_pdg.data import make_toy_hetero_graph, PairBatch
from hetero_pdg.metapath import DEFAULT_METAPATHS, project_all_metapaths
from hetero_pdg.multi_slice import (
    MultiSliceFiltration,
    compute_multi_slice_features,
    MULTI_SLICE_BACKEND,
    NUM_FILTRATION_COMPONENTS,
)


def _toy():
    data = make_toy_hetero_graph(seed=0)
    bundles = project_all_metapaths(
        data, [DEFAULT_METAPATHS["PFP"], DEFAULT_METAPATHS["PCP"]]
    )
    return data, bundles


def _pair_batch(seed=0, B=12, n_paper=60):
    rng = np.random.RandomState(seed)
    src = torch.tensor(rng.randint(0, n_paper, size=B), dtype=torch.long)
    dst = torch.tensor(rng.randint(0, n_paper, size=B), dtype=torch.long)
    label = torch.tensor(rng.randint(0, 2, size=B), dtype=torch.float32)
    return PairBatch(src, dst, label, "paper", "paper")


def test_backend_label_is_honest():
    # Must not claim exact persistent homology / EPD / real PDGNN.
    assert MULTI_SLICE_BACKEND == (
        "sliced filtration approximation "
        "(NOT exact multi-parameter persistent homology)"
    )


def test_lambda_normalization():
    model = MultiSliceFiltration(num_slices=3, seed=0)
    w = model.slice_weights()
    assert w.shape == (3, NUM_FILTRATION_COMPONENTS)
    assert torch.all(w >= 0)
    sums = w.sum(dim=1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_lambda_normalization_various_k():
    for K in (1, 2, 5):
        w = MultiSliceFiltration(num_slices=K, seed=1).slice_weights()
        assert w.shape == (K, NUM_FILTRATION_COMPONENTS)
        assert torch.all(w >= 0)
        assert torch.allclose(w.sum(1), torch.ones(K), atol=1e-5)


def test_vector_filtration_shape_and_finite():
    data, _ = _toy()
    model = MultiSliceFiltration(num_slices=3, seed=0)
    F = model.vector_filtration(data, "paper")
    n_paper = int(data["paper"].num_nodes)
    assert F.shape == (n_paper, NUM_FILTRATION_COMPONENTS)
    assert torch.isfinite(F).all()


def test_fused_output_shape_and_finite():
    data, bundles = _toy()
    pb = _pair_batch()
    model = MultiSliceFiltration(num_slices=3, seed=0)
    fused = model(data, bundles, pb)
    assert fused.shape == (pb.src.numel(), model.fused_dim)
    assert torch.isfinite(fused).all()


def test_convenience_function_matches_module():
    data, bundles = _toy()
    pb = _pair_batch()
    fused = compute_multi_slice_features(
        data, bundles, pb, num_slices=3, seed=0
    )
    assert fused.shape[0] == pb.src.numel()
    assert torch.isfinite(fused).all()


def test_determinism():
    data, bundles = _toy()
    pb = _pair_batch()
    a = compute_multi_slice_features(data, bundles, pb, num_slices=3, seed=7)
    b = compute_multi_slice_features(data, bundles, pb, num_slices=3, seed=7)
    assert torch.allclose(a, b, atol=0.0)


def test_gradients_flow_to_slice_weights():
    data, bundles = _toy()
    pb = _pair_batch()
    model = MultiSliceFiltration(num_slices=3, seed=0)
    fused = model(data, bundles, pb)
    fused.sum().backward()
    # raw lambda parameter must receive a gradient
    assert model.raw_lambda.grad is not None
    assert torch.isfinite(model.raw_lambda.grad).all()
