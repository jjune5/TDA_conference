"""hetero_pdg: MVP for heterogeneous-graph link prediction with hetero-aware
topology features (MetaPath-PDGNN + UnifiedFilter-PDGNN).

This is a clean, self-contained scaffold. Topology features come in two flavours,
ALWAYS clearly labelled:
  * fallback  -- deterministic graph descriptors (common neighbours, shortest path,
                 degrees, local edge count/density). These are NOT PDGNN output.
  * pdgnn     -- real PDGNN-predicted extended-persistence-image features, reusing
                 Knowledge_Distillation.pdgnn_modern + hetero.pdgnn_metapath.

No performance claims are made anywhere in this package; run the experiments and
read the JSON metrics yourself.

This package does not modify or import-break the existing homogeneous TLC-GNN/PDGNN
code; it only adds a new top-level package.
"""
