import torch_geometric.datasets
from torch_geometric.data import Data
import torch_geometric.transforms as T
import torch
import sys
import networkx as nx
import os
import numpy as np
import scipy.sparse as sp
from torch_geometric.utils import remove_self_loops
import torch_geometric.datasets
from sg2dgm import riccidist2dgm as sg2dgm
#from sg2dgm import riccidist2dgm_c as sg2dgm

def loaddatas(d_name):
    if d_name in ["PPI"]:
        dataset = torch_geometric.datasets.PPI('./data/' + d_name)
    elif d_name == 'Cora':
        dataset = torch_geometric.datasets.Planetoid('./data/'+d_name,d_name,transform=T.NormalizeFeatures())
    elif d_name in ['Citeseer', 'PubMed']:
        dataset = torch_geometric.datasets.Planetoid('./data/' + d_name, d_name)
    elif d_name in ["Computers", "Photo"]:
        dataset = torch_geometric.datasets.Amazon('./data/'+d_name,d_name)
    elif d_name in ['Chameleon', 'Squirrel']:
        dataset = torch_geometric.datasets.WikipediaNetwork(
            './data/' + d_name, d_name.lower())
    elif d_name in ['Texas', 'Cornell', 'Wisconsin']:
        dataset = torch_geometric.datasets.WebKB('./data/' + d_name, d_name)
    elif d_name == 'ChChMiner':
        dataset = _load_chch_miner()
    elif d_name.startswith('SBM_'):
        dataset = _load_sbm(d_name)
    return dataset


def _load_chch_miner():
    """ChCh-Miner (DrugBank chem-chem interaction). 1514 drugs / 48514 edges.
    Source: http://snap.stanford.edu/biodata/datasets/10001/
    No node features → one-hot identity."""
    raw = './data/ChChMiner/raw.tsv'
    pairs = np.loadtxt(raw, dtype=str)
    drugs = sorted(set(pairs.flatten().tolist()))
    idx = {d: i for i, d in enumerate(drugs)}
    src = np.array([idx[a] for a in pairs[:, 0]])
    dst = np.array([idx[b] for b in pairs[:, 1]])
    # symmetric for undirected
    edge_index = torch.tensor(
        np.stack([np.concatenate([src, dst]), np.concatenate([dst, src])]),
        dtype=torch.long)
    n = len(drugs)
    x = torch.eye(n)  # one-hot drug ID
    y = torch.zeros(n, dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, y=y)

    class _FakeDataset:
        def __init__(self, data, name):
            self._data = [data]; self.name = name; self.num_classes = 2
        def __getitem__(self, i): return self._data[i]
        def __len__(self): return 1
    return _FakeDataset(data, 'ChChMiner')

def _load_sbm(d_name: str):
    """Parse SBM_<n>_<p_in>_<p_out> and return Data wrapper.

    Example: 'SBM_500_5_0.1_0.05' = 500 nodes, 5 blocks, p_in=0.1, p_out=0.05.
    """
    parts = d_name.split('_')
    if len(parts) != 5:
        raise ValueError(f"SBM dataset must be SBM_<N>_<K>_<p_in>_<p_out>, got {d_name}")
    N, K, p_in, p_out = int(parts[1]), int(parts[2]), float(parts[3]), float(parts[4])
    from Knowledge_Distillation.sbm_data import generate_sbm, sbm_to_pyg
    g = generate_sbm(n_per_block=N // K, n_blocks=K, p_in=p_in, p_out=p_out, seed=1234)
    data = sbm_to_pyg(g, n_blocks=K, feat_dim=16, seed=1234)

    class _FakeDataset:
        def __init__(self, data, name):
            self._data = [data]; self.name = name; self.num_classes = K
        def __getitem__(self, i): return self._data[i]
        def __len__(self): return 1
    return _FakeDataset(data, d_name)

def get_edges_split(data, val_prop = 0.2, test_prop = 0.2, seed = 1234):
    g = nx.Graph()
    g.add_nodes_from([i for i in range(len(data.y))])
    ricci_edge_index_ = np.array((data.edge_index))
    ricci_edge_index = [(ricci_edge_index_[0, i], ricci_edge_index_[1, i]) for i in
                        range(np.shape(ricci_edge_index_)[1])]
    g.add_edges_from(ricci_edge_index)
    adj = nx.adjacency_matrix(g)

    return get_adj_split(adj,val_prop = val_prop, test_prop = test_prop, seed = seed)

#def get_adj_split(adj, val_prop = 0.05, test_prop = 0.1, seed=1234):
def get_adj_split(adj, val_prop=0.05, test_prop=0.1, seed=1234):
    np.random.seed(seed)  # get tp edges
    x, y = sp.triu(adj).nonzero()
    pos_edges = np.stack([x, y], axis=1)
    np.random.shuffle(pos_edges)
    # get tn edges, memory-efficient bool computation
    N = adj.shape[0]
    adj_bool = adj.astype(bool).toarray()
    upper = np.triu(np.ones((N, N), dtype=bool), k=1)
    neg_mask = (~adj_bool) & upper
    nx_idx, ny_idx = np.where(neg_mask)
    neg_edges = np.stack([nx_idx, ny_idx], axis=1)
    np.random.shuffle(neg_edges)

    m_pos = len(pos_edges)
    n_val = int(m_pos * val_prop)
    n_test = int(m_pos * test_prop)
    val_edges, test_edges, train_edges = pos_edges[:n_val], pos_edges[n_val:n_test + n_val], pos_edges[n_test + n_val:]
    val_edges_false, test_edges_false = neg_edges[:n_val], neg_edges[n_val:n_test + n_val]
    # subsample train negatives to a manageable count (default 5x #train_pos), to
    # keep persistence-image computation feasible. Each negative still gets a PI.
    # TLCGNN_NEG_CAP env var overrides the 5x multiplier ('all' = no cap) for the
    # cap-sweep diagnosis (does the cap explain our gap vs paper?).
    _cap_mult = os.environ.get('TLCGNN_NEG_CAP', '5')
    if _cap_mult == 'all':
        train_neg_cap = len(neg_edges)
    else:
        train_neg_cap = min(len(neg_edges), max(int(len(train_edges) * float(_cap_mult)), 1024))
    train_edges_false = neg_edges[n_test + n_val: n_test + n_val + train_neg_cap]
    return train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false

def compute_persistence_image(data, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false, data_name, hop = 1):
    if data_name == "photo":
        data_name = "Photo"
    if data_name == "computers":
        data_name = "Computers"

    # Pluggable PI source: dionysus (exact, TLC-GNN) or pdgnn (neural approx).
    pi_source = os.environ.get('TLCGNN_PI_SOURCE', 'dionysus')
    if pi_source == 'pdgnn':
        filename = './data/PDGNN/' + data_name + '.npy'
    else:
        filename = './data/TLCGNN/' + data_name + '.npy'
    expected_total = (len(train_edges) + len(train_edges_false)
                      + len(val_edges) + len(val_edges_false)
                      + len(test_edges) + len(test_edges_false))
    if os.path.exists(filename):
        cached = np.load(filename)
        if cached.shape[0] == expected_total:
            return cached
        # Stale cache layout: written before the train_neg cap was added.
        # Layout was [train_pos | ALL_train_neg | val_pos | val_neg | test_pos | test_neg].
        # Splice to [train_pos | first cap-many train_neg | val_pos | val_neg | test_pos | test_neg]
        # so the model's offsets line up. Splits are deterministic (seed=1234) so the first
        # cap-many train_neg rows here are exactly the ones the current loader emits.
        n_train_pos = len(train_edges)
        val_test_total = (len(val_edges) + len(val_edges_false)
                          + len(test_edges) + len(test_edges_false))
        n_cap = len(train_edges_false)
        n_orig_train_neg = cached.shape[0] - n_train_pos - val_test_total
        if n_orig_train_neg >= n_cap and cached.shape[0] > expected_total:
            head = cached[:n_train_pos + n_cap]
            tail = cached[n_train_pos + n_orig_train_neg:]
            spliced = np.concatenate([head, tail], axis=0)
            assert spliced.shape[0] == expected_total, (
                f"splice mismatch for {data_name}: got {spliced.shape[0]} want {expected_total}")
            # Persist the spliced cache so subsequent runs hit the fast path.
            np.save(filename + '.spliced.npy', spliced)
            os.rename(filename, filename + '.uncapped.bak')
            os.rename(filename + '.spliced.npy', filename)
            return spliced
        # cache smaller than expected: must recompute
        os.rename(filename, filename + '.short.bak')
    total_edges = np.concatenate(
        (train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false))
    data.train_pos, data.train_neg = len(train_edges), len(train_edges_false)
    data.val_pos, data.val_neg = len(val_edges), len(val_edges_false)
    data.test_pos, data.test_neg = len(test_edges), len(test_edges_false)
    data.total_edges = total_edges

    # delete val_pos and test_pos (set-based, O(E) instead of O(E^2))
    _ei = np.array(data.edge_index)
    _mask_remove = set()
    for edges in val_edges.tolist():
        _mask_remove.add((edges[0], edges[1]))
        _mask_remove.add((edges[1], edges[0]))
    for edges in test_edges.tolist():
        _mask_remove.add((edges[0], edges[1]))
        _mask_remove.add((edges[1], edges[0]))
    _keep = np.array([(int(u), int(v)) not in _mask_remove
                      for u, v in zip(_ei[0], _ei[1])])
    data.edge_index = torch.from_numpy(_ei[:, _keep]).long()
    data.edge_index, _ = remove_self_loops(data.edge_index)

    # generate graph for computing persistence diagram
    g = nx.Graph()
    # Add all original nodes first so that isolated nodes (whose only edges were
    # val/test) remain in the graph and in graph2pi.dict_node.
    g.add_nodes_from(range(data.num_nodes))
    ricci_edge_index_ = np.array(remove_self_loops((data.edge_index.cpu()))[0])
    ricci_edge_index = [(ricci_edge_index_[0, i], ricci_edge_index_[1, i]) for i in
                        range(np.shape(ricci_edge_index_)[1])]
    g.add_edges_from(ricci_edge_index)
    print(len(g.edges()))

    # ricci_cur = compute_ricci_flow(data, d_name)
    ricci_cur = compute_ricci_curvature(data)

    # compute sg2dgm and save in a dict
    pi = sg2dgm.graph2pi(g, ricci_curv=ricci_cur)
    _cores = int(os.environ.get('TLCGNN_CORES', 32))
    pi.get_pimg_for_all_edges(total_edges, cores=_cores, hop=hop, norm=True, extended_flag=True,
                                  resolution=5, descriptor='sum')
    np.save(filename,pi.pi_sg)
    return pi.pi_sg

def compute_ricci_curvature(data):
    from GraphRicciCurvature.OllivierRicci import OllivierRicci
    print("start writing ricci curvature")
    Gd = nx.Graph()
    ricci_edge_index_ = np.array(data.edge_index)
    ricci_edge_index = [(ricci_edge_index_[0, i],
                         ricci_edge_index_[1, i]) for i in
                        range(np.shape(data.edge_index)[1])]
    Gd.add_edges_from(ricci_edge_index)
    Gd_OT = OllivierRicci(Gd, alpha=0.5, method="Sinkhorn", verbose="INFO")
    print("adding edges finished")
    Gd_OT.compute_ricci_curvature()
    ricci_list = []
    for n1, n2 in Gd_OT.G.edges():
        ricci_list.append([n1, n2, Gd_OT.G[n1][n2]['ricciCurvature']])
        ricci_list.append([n2, n1, Gd_OT.G[n1][n2]['ricciCurvature']])
    ricci_list = sorted(ricci_list)
    print("computing ricci curvature finished")
    return ricci_list


def num(strings):
    try:
        return int(strings)
    except ValueError:
        return float(strings)



