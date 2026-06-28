"""RED-first tests for the experimental relation-aware edge filtration.

These cover:
  * shape (E,) and finiteness of edge filtration values in both modes;
  * the HARD validity constraint  g(i,j,rho) >= max(f(i), f(j))  for ALL edges
    in both "max" and "relation_delay" modes (random node filtration + random
    relation/edge types);
  * the learnable-delay nn.Module behaving identically to the functional form;
  * determinism (same seed -> same values);
  * the optional standalone union-find 0-dim sublevel persistence that DOES
    consume explicit edge filtration.
"""
import numpy as np
import torch
import pytest

from hetero_pdg.data import make_toy_hetero_graph, NODE_TYPES, EDGE_TYPES
from hetero_pdg.metapath import DEFAULT_METAPATHS, project_all_metapaths
from hetero_pdg.filtration import TypeAwareFiltrationMLP
from hetero_pdg.edge_filtration import (
    relation_edge_filtration,
    RelationDelayEdgeFiltration,
    zero_dim_sublevel_persistence,
    EDGE_FILTRATION_MODES,
    EDGE_FILTRATION_HONESTY_LABEL,
)


def _toy_homogeneous_edges(seed=0):
    """A small homogeneous edge_index + random relation types over papers (PFP+PCP)."""
    data = make_toy_hetero_graph(seed=seed)
    specs = [DEFAULT_METAPATHS["PFP"], DEFAULT_METAPATHS["PCP"]]
    bundles = project_all_metapaths(data, specs)
    n = bundles["PFP"].num_nodes  # number of papers
    rng = np.random.RandomState(seed)
    node_filt = torch.tensor(rng.randn(n), dtype=torch.float32)
    # build one edge_index by tagging each meta-path graph's edges with a relation id
    rows, cols, rels = [], [], []
    for rid, name in enumerate(("PFP", "PCP")):
        for u, v in bundles[name].graph.edges():
            rows.append(int(u)); cols.append(int(v)); rels.append(rid)
    edge_index = torch.tensor([rows, cols], dtype=torch.long)
    edge_type = torch.tensor(rels, dtype=torch.long)
    return node_filt, edge_index, edge_type, n


def test_honesty_label_is_experimental_and_not_overclaiming():
    lab = EDGE_FILTRATION_HONESTY_LABEL.lower()
    assert "experimental" in lab
    # must NOT claim to be exact persistent homology / real PDGNN / EPD
    for forbidden in ("exact persistent homology", "real pdgnn", "epd feature"):
        assert forbidden not in lab


def test_max_mode_shape_and_finite():
    node_filt, edge_index, edge_type, _ = _toy_homogeneous_edges(seed=0)
    g = relation_edge_filtration(node_filt, edge_index, edge_type, mode="max")
    assert g.shape == (edge_index.shape[1],)
    assert torch.isfinite(g).all()


def test_max_mode_equals_endpoint_max_exactly():
    node_filt, edge_index, edge_type, _ = _toy_homogeneous_edges(seed=1)
    g = relation_edge_filtration(node_filt, edge_index, edge_type, mode="max")
    fi = node_filt[edge_index[0]]
    fj = node_filt[edge_index[1]]
    assert torch.allclose(g, torch.maximum(fi, fj))


def test_relation_delay_shape_and_finite():
    node_filt, edge_index, edge_type, _ = _toy_homogeneous_edges(seed=2)
    delays = torch.tensor([0.3, -1.0])  # raw (pre-softplus) per-relation delays
    g = relation_edge_filtration(node_filt, edge_index, edge_type,
                                 mode="relation_delay", delays=delays)
    assert g.shape == (edge_index.shape[1],)
    assert torch.isfinite(g).all()


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_validity_constraint_holds_for_all_edges_both_modes(seed):
    node_filt, edge_index, edge_type, _ = _toy_homogeneous_edges(seed=seed)
    endpoint_max = torch.maximum(node_filt[edge_index[0]], node_filt[edge_index[1]])

    g_max = relation_edge_filtration(node_filt, edge_index, edge_type, mode="max")
    assert torch.all(g_max >= endpoint_max - 1e-6)

    rng = np.random.RandomState(seed)
    delays = torch.tensor(rng.randn(int(edge_type.max()) + 1), dtype=torch.float32)
    g_del = relation_edge_filtration(node_filt, edge_index, edge_type,
                                     mode="relation_delay", delays=delays)
    # softplus(delta) >= 0  =>  g >= max(f_i, f_j) strictly (up to tolerance)
    assert torch.all(g_del >= endpoint_max - 1e-6)


def test_relation_delay_default_delays_zero_still_valid():
    """delays=None defaults to zeros; softplus(0)=ln2>0 so still >= endpoint max."""
    node_filt, edge_index, edge_type, _ = _toy_homogeneous_edges(seed=7)
    endpoint_max = torch.maximum(node_filt[edge_index[0]], node_filt[edge_index[1]])
    g = relation_edge_filtration(node_filt, edge_index, edge_type, mode="relation_delay")
    assert torch.all(g >= endpoint_max - 1e-6)


def test_unknown_mode_raises():
    node_filt, edge_index, edge_type, _ = _toy_homogeneous_edges(seed=0)
    with pytest.raises(ValueError):
        relation_edge_filtration(node_filt, edge_index, edge_type, mode="nonsense")
    assert set(EDGE_FILTRATION_MODES) == {"max", "relation_delay"}


def test_learnable_module_matches_functional_and_is_deterministic():
    node_filt, edge_index, edge_type, _ = _toy_homogeneous_edges(seed=3)
    num_rel = int(edge_type.max()) + 1
    torch.manual_seed(123)
    mod = RelationDelayEdgeFiltration(num_relations=num_rel)
    g1 = mod(node_filt, edge_index, edge_type)
    g2 = mod(node_filt, edge_index, edge_type)
    assert torch.allclose(g1, g2)  # deterministic
    # equivalent to functional form with the module's raw delay parameter
    g_func = relation_edge_filtration(node_filt, edge_index, edge_type,
                                      mode="relation_delay",
                                      delays=mod.raw_delays.detach())
    assert torch.allclose(g1, g_func)
    endpoint_max = torch.maximum(node_filt[edge_index[0]], node_filt[edge_index[1]])
    assert torch.all(g1 >= endpoint_max - 1e-6)


def test_module_is_differentiable_wrt_delays():
    node_filt, edge_index, edge_type, _ = _toy_homogeneous_edges(seed=4)
    num_rel = int(edge_type.max()) + 1
    mod = RelationDelayEdgeFiltration(num_relations=num_rel)
    g = mod(node_filt, edge_index, edge_type)
    g.sum().backward()
    assert mod.raw_delays.grad is not None
    assert torch.isfinite(mod.raw_delays.grad).all()


def test_works_with_typeaware_node_filtration_paper_subgraph():
    """End-to-end smoke: TypeAwareFiltrationMLP paper values feed the edge filter."""
    data = make_toy_hetero_graph(seed=0)
    mlp = TypeAwareFiltrationMLP(data, hidden=8)
    node_filt = mlp(data)["paper"].detach()
    _, edge_index, edge_type, _ = _toy_homogeneous_edges(seed=0)
    endpoint_max = torch.maximum(node_filt[edge_index[0]], node_filt[edge_index[1]])
    g = relation_edge_filtration(node_filt, edge_index, edge_type,
                                 mode="relation_delay")
    assert g.shape == (edge_index.shape[1],)
    assert torch.all(g >= endpoint_max - 1e-6)


def test_zero_dim_sublevel_persistence_consumes_explicit_edge_filtration():
    # path graph 0-1-2-3 with node values; explicit edge filtration values
    node_filt = torch.tensor([0.0, 1.0, 2.0, 3.0])
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    # edge filtration that strictly respects g >= max(endpoints)
    edge_filt = torch.tensor([1.0, 2.0, 3.0])
    pairs = zero_dim_sublevel_persistence(node_filt, edge_index, edge_filt, num_nodes=4)
    # 4 components born (at 0,1,2,3), 3 merges -> 3 finite + 1 infinite
    finite = [p for p in pairs if np.isfinite(p[1])]
    infinite = [p for p in pairs if not np.isfinite(p[1])]
    assert len(infinite) == 1
    assert len(finite) == 3
    # every finite bar has death >= birth
    for b, d in finite:
        assert d >= b - 1e-6


def test_zero_dim_persistence_uses_edge_values_not_node_max():
    """If we delay an edge, the corresponding merge (death) time increases."""
    node_filt = torch.tensor([0.0, 1.0])
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    base = zero_dim_sublevel_persistence(node_filt, edge_index,
                                         torch.tensor([1.0]), num_nodes=2)
    delayed = zero_dim_sublevel_persistence(node_filt, edge_index,
                                            torch.tensor([5.0]), num_nodes=2)
    base_finite = sorted(d for _, d in base if np.isfinite(d))
    del_finite = sorted(d for _, d in delayed if np.isfinite(d))
    assert del_finite[0] > base_finite[0]
