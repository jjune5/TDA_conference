# hetero_pdg_lp

Dedicated, **self-contained** project for *heterogeneous-graph link prediction with
PDGNN-style topological features* (Phase 1 MVP). Lifted out of the TLC-GNN work so it
can be developed on its own; the original homogeneous TLC-GNN/PDGNN code is untouched.

📖 **Full documentation:** [`hetero_pdg/README.md`](hetero_pdg/README.md)

> ⚠️ Phase 1 uses `fallback_topology_descriptors` — **not** PDGNN and **not** EPD
> features. No performance improvement is claimed; toy metrics validate pipeline
> correctness only. The Phase-2 seam for a real PDGNN/TLC adapter is in
> `topology_features.compute_topology_features(..., backend="pdgnn")`.

## Quickstart

Self-contained — needs only `torch`, `torch_geometric`, `scikit-learn`, `networkx`,
`scipy`, `numpy` (all in the `tlcgnn` conda env). No `PYTHONPATH` required.

```bash
conda activate tlcgnn
python -m pytest tests/ -q                                   # 46 tests
python -m hetero_pdg.train_hetero_lp --dataset toy \
    --target-rel paper,cites,paper --topo-mode metapath_topology_concat \
    --epochs 30 --output-dir runs/hetero_pdg_toy
```

Modes: `no_topology`, `collapsed_topology`, `metapath_topology_concat`,
`metapath_topology_attention`, `unified_filter_topology`.

## Layout

```
hetero_pdg/   data.py metapath.py filtration.py topology_features.py
              models.py train_hetero_lp.py README.md
tests/        6 files, 46 TDD tests
runs/         per-run config.json + metrics.json
slurm_hetero_pdg.sh   optional SLURM runner (args passed through to the trainer)
```
