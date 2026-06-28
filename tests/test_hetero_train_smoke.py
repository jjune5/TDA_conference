"""Phase-1 RED smoke tests: 2-epoch toy training for every topology mode +
config/metrics JSON output."""
import json
import os

import pytest

from hetero_pdg.models import HeteroTopoLinkPredictor
from hetero_pdg.train_hetero_lp import run_experiment, save_json


@pytest.mark.parametrize("mode", list(HeteroTopoLinkPredictor.MODES))
def test_smoke_2epochs_each_mode_writes_json(mode, tmp_path):
    out = str(tmp_path / mode)
    res = run_experiment(topo_mode=mode, epochs=2, seed=0, device="cpu", output_dir=out)
    assert res["topo_mode"] == mode
    assert res["n_test"] > 0
    for k in ("test_auc", "test_ap", "val_auc", "val_ap"):
        assert res[k] is None or (0.0 <= res[k] <= 1.0)
    assert os.path.exists(os.path.join(out, "config.json"))
    assert os.path.exists(os.path.join(out, "metrics.json"))
    metrics = json.load(open(os.path.join(out, "metrics.json")))
    assert metrics["topo_mode"] == mode


def test_metapath_backend_labeled_fallback():
    res = run_experiment(topo_mode="metapath_topology_concat", epochs=2, seed=0,
                         device="cpu", output_dir=None)
    assert res["topo_backend"] == "fallback_topology_descriptors"


def test_no_topology_backend_is_none():
    res = run_experiment(topo_mode="no_topology", epochs=2, seed=0,
                         device="cpu", output_dir=None)
    assert res["topo_backend"] == "none"


def test_save_json_roundtrip(tmp_path):
    p = str(tmp_path / "x.json")
    save_json({"a": 1, "b": [1, 2]}, p)
    assert json.load(open(p)) == {"a": 1, "b": [1, 2]}
