"""Real-data loader tests (ACM stored cite; IMDB derived co-director).

Needs the TLC-GNN repo on PYTHONPATH + cached HGB data. Dual-accept: with the engine
it loads real graphs; without it, a clear RealDatasetUnavailable is raised.
"""
import pytest

from hetero_pdg.real_data import (
    REAL_REGISTRY, RealDatasetUnavailable, load_real_hetero,
)
from hetero_pdg.metapath import MetaPathSpec, validate_metapath


def test_registry_lists_two_small_datasets():
    assert set(REAL_REGISTRY) == {"acm", "imdb"}


@pytest.mark.parametrize("name", ["acm", "imdb"])
def test_load_real_or_clear_error(name):
    try:
        data, target_rel, metaspecs = load_real_hetero(name, max_target_edges=200, seed=0)
    except RealDatasetUnavailable as e:
        assert "PYTHONPATH" in str(e)
        return
    # same-type target (so the metapath pipeline applies)
    assert target_rel[0] == target_rel[2]
    assert data[target_rel].edge_index.shape[1] > 0
    assert data[target_rel].edge_index.shape[1] <= 200          # capped
    assert len(metaspecs) >= 1
    for s in metaspecs:
        assert isinstance(s, MetaPathSpec)
        validate_metapath(s, data)                               # all metapaths valid on the data
        assert s.target_type == target_rel[0]


def test_unknown_dataset_raises():
    with pytest.raises(ValueError):
        load_real_hetero("nope")
