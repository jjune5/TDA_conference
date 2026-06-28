"""Phase-3 integration: the advanced topology modes/flags run end-to-end through
train_hetero_lp on the toy graph, and the Phase-1 base modes are unchanged.
Toy metrics are sanity checks only (no performance claims)."""
import pytest

from hetero_pdg.train_hetero_lp import run_experiment


def _ok(res):
    assert res["test_auc"] is None or 0.0 <= res["test_auc"] <= 1.0
    assert res["n_test"] > 0


@pytest.mark.parametrize("mode", ["typed_cycle_topology", "multi_slice_topology"])
def test_advanced_topology_modes_train(mode):
    res = run_experiment(topo_mode=mode, epochs=2, seed=0, device="cpu", num_slices=3)
    assert res["topo_mode"] == mode
    _ok(res)


def test_typed_cycle_feature_kind():
    res = run_experiment(topo_mode="typed_cycle_topology", epochs=2, seed=0, device="cpu")
    assert res["feature_kind"] == "typed_cycle_proxy_descriptors"


def test_multi_slice_records_num_slices():
    res = run_experiment(topo_mode="multi_slice_topology", epochs=2, seed=0, device="cpu", num_slices=3)
    assert res["feature_kind"] == "sliced_filtration_approximation"
    assert res["num_slices"] == 3


def test_pair_conditioned_collapsed():
    res = run_experiment(topo_mode="collapsed_topology", filtration_mode="pair_conditioned",
                         epochs=2, seed=0, device="cpu")
    assert res["filtration_mode"] == "pair_conditioned"
    assert res["feature_kind"] == "pair_conditioned_filtration_approximate"
    _ok(res)


def test_pair_conditioned_metapath_attention():
    res = run_experiment(topo_mode="metapath_topology_attention", filtration_mode="pair_conditioned",
                         epochs=2, seed=0, device="cpu")
    assert res["feature_kind"] == "pair_conditioned_filtration_approximate"
    _ok(res)


def test_relation_delay_edge_mode_recorded():
    res = run_experiment(topo_mode="metapath_topology_concat", edge_filtration_mode="relation_delay",
                         epochs=2, seed=0, device="cpu")
    assert res["edge_filtration_mode"] == "relation_delay"
    _ok(res)


@pytest.mark.parametrize("mode", [
    "no_topology", "collapsed_topology", "metapath_topology_concat",
    "metapath_topology_attention", "unified_filter_topology"])
def test_phase1_base_modes_unbroken(mode):
    res = run_experiment(topo_mode=mode, epochs=2, seed=0, device="cpu")
    assert res["topo_mode"] == mode
    _ok(res)
