# Hetero-PDGNN — Preserving Semantic Topology in Heterogeneous-Graph Link Prediction

**Status: proposal + MVP progress report.** The central claim below is our working
*hypothesis*. What is implemented and validated so far is a clean, honest pipeline + the
proposed mechanisms; we make **no performance claim** — toy results are sanity checks and
real-data results are reported exactly as they come out.

## Claim (hypothesis)

PDGNN approximates extended-persistence (EPD) computation efficiently, but on heterogeneous
graphs its **filtration ordering** and **Union-Find events** do not reflect node/edge
**type**. As a result, topological events with different semantics collapse onto the same
birth/death point. We propose **pair-conditioned relation-aware filtration**, **multi-slice
persistence**, and **typed cycle signatures** to preserve the *semantic* topology of
heterogeneous graphs — **Hetero-PDGNN**.

## Background

- **PDGNN** approximates a graph's extended persistence diagram with a GNN that emulates the
  Union-Find merge/relax events of a scalar filtration.
- **TLC-GNN** uses pairwise persistent-homology features for link prediction.
- These assume a single scalar filtration on a homogeneous graph. On heterogeneous graphs,
  filter values across node/edge types are not directly comparable, and an untyped Union-Find
  cannot tell apart events that are structurally identical but semantically different (e.g. an
  author-paper-author cycle vs a paper-field-paper cycle).

## Baseline (what we build on)

A self-contained hetero link-prediction MVP (`hetero_pdg`):

- toy `HeteroData` (author/paper/field, 5 edge types) + small real HGB datasets (ACM, IMDB);
- meta-path projection (APA / PFP / PCP) by adjacency multiplication;
- five LP modes: `no_topology`, `collapsed_topology`, `metapath_topology_{concat,attention}`,
  `unified_filter_topology` (type-aware filtration: per-type MLP + quantile calibration +
  lexicographic ordering);
- topology backends: **fallback** deterministic descriptors (default; NOT PDGNN/EPD),
  **real_tlc** (exact EPD via union-find), **real_pdgnn** (neural EPD approximation), the
  latter two reusing the existing PDGNN/TLC engine;
- leakage control: the observation graph drops val/test target edges, and each candidate
  pair's own edge is removed before reading its topology.

## Our additions (the three mechanisms)

| mechanism | idea | honesty status |
|---|---|---|
| Pair-conditioned filtration | the filter of a node depends on the queried pair (u,v) and relation r, not the node alone | **approximate** (learnable scalar field + vicinity stats; not PH coordinates) |
| Relation-aware edge filtration | `g(i,j,ρ)=max(f_i,f_j)+softplus(δ_ρ)` — edge type delays an event while keeping `g ≥ max(f_i,f_j)` | **prepared interface** (validity guaranteed + tested; not yet wired into exact EPD) |
| Multi-slice persistence | a vector filtration `F(x)` viewed through K convex-combination scalar slices, attention-fused | **approximate** (scalar slicing; not exact multi-parameter PH) |
| Typed cycle signature | typed node/edge histograms + per-meta-path common-neighbour / short-cycle proxies, so typed cycles do not collapse | **proxy** (typed counting; not persistent homology) |

## Experiments

**Toy (controlled).** A planted positive control makes citations field-assortative so
topology *must* carry signal (node features stay random); a random graph is the negative
control. Pipeline correctness is confirmed: on the positive control the typed and
meta-path channels track the planted signal while everything stays near chance on the
negative control. test-AUC (mean over 3 seeds, sanity only):

| mode | random (neg.) | planted (pos.) |
|---|---|---|
| no_topology | 0.59 | 0.48 |
| metapath_attention (fallback) | 0.60 | 0.75 |
| typed_cycle_topology | 0.52 | 0.81 |
| relation_delay (recorded) | 0.63 | 0.72 |
| pair_conditioned | 0.50 | 0.40 |
| multi_slice | 0.48 | 0.40 |

Honest reading: the simple typed / meta-path channels capture the planted signal; the more
elaborate learnable filtrations (pair-conditioned, multi-slice) do **not** on this toy — a
caution, not a result.

**Real data (ACM, IMDB).** Small HGB graphs; LP target = paper–cite–paper (ACM) and a
derived movie–co-director–movie relation (IMDB), target edges capped at 1000. test-AUC
(mean over 3 seeds):

| config | ACM | IMDB |
|---|---|---|
| no_topology (baseline) | 0.701 | 0.722 |
| metapath_concat (fallback) | 0.691 | 0.709 |
| metapath_attention (fallback) | 0.723 | 0.720 |
| metapath_concat (real_tlc, exact EPD) | 0.690 | 0.712 |
| metapath_concat (real_pdgnn, neural EPD) | 0.713 | 0.704 |

**Result (honest):** no topology variant — fallback descriptors OR real extended persistence
(exact `real_tlc` / neural `real_pdgnn`) — meaningfully beats the no-topology baseline on
either dataset (all within ~0.69–0.72; the baseline is among the best). Consistent with the
broader finding that topology-as-feature does not obviously help heterogeneous link prediction
once controlled. The Phase-3 advanced modes (pair-conditioned, multi-slice, typed-cycle) are
toy-schema-specific and did not run on the real datasets' featureless node types — a current
limitation, not a result.

## Limitations / honesty

- No performance improvement is claimed. Fallback descriptors are not PDGNN/EPD.
- Pair-conditioned / multi-slice are vicinity statistics of learnable scalar fields, not
  persistence-diagram coordinates; typed-cycle is a typed counting proxy; relation-aware edge
  filtration is a prepared interface (the exact-EPD path still uses node→edge max).
- Pair-conditioned / multi-slice require featured nodes of every type, so on real datasets
  with featureless types they are not run.
- Evaluated on small graphs only; results may not generalize.

## How to extend the claim

1. Larger real datasets (DBLP, Freebase) with standard splits + more seeds.
2. Wire relation-aware edge filtration into the exact EPD (a Union-Find that consumes explicit
   per-edge filtration), so edge type genuinely changes the events.
3. Replace scalar slicing with true multi-parameter persistence (e.g. signed-barcode
   vectorization).
4. Cross-type (chromatic / H1) persistence to capture cross-type mingling directly.
5. Controlled ablations isolating where, if anywhere, typed topology beats untyped topology
   and the no-topology baseline.
