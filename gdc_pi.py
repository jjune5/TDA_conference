"""gdc_pi.py — Graph Diffusion Convolution pre-processing for PI computation.

Thread B of the Diffusion-meets-Topology chapter experiment:
  "Does GDC-denoising rescue Persistence Images on heterophilic graphs?"

This module exposes a single public function:

    diffuse_graph(data) -> (edge_index, num_nodes)

which applies PyG's GDC transform to the input PyG Data object and
returns a *new* sparse (edge_index, num_nodes) describing the diffused
graph.  The caller (loaddatas.py) feeds this edge_index into the
NetworkX graph used for computing Ollivier-Ricci curvature and
Persistence Images, while the GCN encoder continues to see the
*original* edge_index (no node-feature change, no structural change
to the training/test edge splits).

──────────────────────────────────────────────────────────────
GDC configuration (chosen parameters, rationale documented here)
──────────────────────────────────────────────────────────────
Method:    Heat kernel    S_t = exp(-t * L_sym)
Parameter: t = 5.0        (standard mid-range scale; ≈3–5 hops of
                           effective neighbourhood; GDC paper uses t=5)
Sparsification:
  method = "topk"
  k = 16, dim = 0         Keep the 16 strongest entries PER COLUMN
                           (column-stochastic → out-degree ≤16).
                           topk is more stable than threshold-eps on
                           small graphs like Texas/Cornell where
                           avg_degree is low, and avoids the
                           avg_degree*N > total-entries IndexError
                           that threshold raises for dense small graphs.
normalization_in  = "sym"  Symmetric Laplacian normalisation (standard)
normalization_out = "col"  Column-stochastic output (default GDC paper)
exact = True               Exact matrix-exponential.  Feasible for all
                           five datasets (Photo: 7650 nodes = 0.22 GB
                           dense float32, well within available RAM).

Result: the diffused graph is SYMMETRIC (we symmetrise after GDC),
undirected, with no self-loops (self-loops are removed before handing
to NetworkX so that Ricci curvature is well-defined on edges).

Env-var override  TLCGNN_GDC_T   (float) overrides heat-kernel t.
                  TLCGNN_GDC_K   (int)   overrides topk k.
"""

import os
import torch
import numpy as np
from torch_geometric.transforms import GDC
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops, to_undirected


# ── tuneable defaults (override via env) ──────────────────────────────────────
_DEFAULT_T = 5.0   # heat kernel time scale
_DEFAULT_K = 16    # topk neighbours per node after diffusion


def _get_config():
    t = float(os.environ.get("TLCGNN_GDC_T", _DEFAULT_T))
    k = int(os.environ.get("TLCGNN_GDC_K", _DEFAULT_K))
    return t, k


def diffuse_graph(data: Data):
    """Apply heat-kernel GDC to *data* and return the diffused sparse graph.

    Parameters
    ----------
    data : torch_geometric.data.Data
        Input graph.  Only ``edge_index`` and ``num_nodes`` are used;
        node features are ignored.

    Returns
    -------
    edge_index_diffused : LongTensor [2, E']
        Edge index of the GDC-diffused, symmetrised, self-loop-free graph.
    num_nodes : int
    """
    t, k = _get_config()
    n = data.num_nodes

    print(f"[GDC] heat kernel t={t}, topk k={k}, "
          f"n_nodes={n}, n_edges_orig={data.edge_index.shape[1]}")

    # Build a minimal Data object for GDC (only topology needed).
    # GDC.forward makes a copy internally, so this is safe.
    d = Data(edge_index=data.edge_index.clone(), num_nodes=n)

    gdc = GDC(
        self_loop_weight=1.0,
        normalization_in="sym",
        normalization_out="col",
        diffusion_kwargs={"method": "heat", "t": t},
        sparsification_kwargs={"method": "topk", "k": k, "dim": 0},
        exact=True,
    )

    d_out = gdc(d)
    ei = d_out.edge_index  # [2, E'] — possibly directed (column-stochastic)

    # Symmetrise: take the union of (u,v) and (v,u) so NetworkX sees an
    # undirected graph that still represents the diffused connectivity.
    ei = to_undirected(ei, num_nodes=n)

    # Remove self-loops so Ricci curvature is defined on proper edges.
    ei, _ = remove_self_loops(ei)

    print(f"[GDC] diffused edge_index: {ei.shape[1]} edges "
          f"(avg deg {ei.shape[1]/n:.1f})")

    return ei, n
