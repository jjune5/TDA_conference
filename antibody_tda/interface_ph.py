"""
interface_ph.py — Thread C, Rung 0' (RE-GATE).

Persistent homology of the ANTIBODY--ANTIGEN INTERFACE contact geometry, rather
than the CDR-H3 loop's own shape (which was the Rung-0 signal that gave a NO-GO).

Rationale (vs Rung 0): the loop's intrinsic shape carried no within-target signal
for binding quality. The *interface* (which antibody residues actually touch which
antigen residues, and the geometry of that contact patch) is, a priori, far more
binding-relevant. This module encodes the interface Calpha point cloud as a
Vietoris-Rips persistence diagram and measures its topological distance to the
NATIVE interface PD, plus a native-free "interface loop-likeness".

Interface definition (symmetric, Calpha-based):
  An antibody residue is an interface residue if its Calpha is within `cutoff` A
  of ANY antigen Calpha; antigen interface residues are defined symmetrically.
  The interface point cloud = the union of those antibody + antigen interface
  Calphas. (Calpha-only keeps it consistent with the Rung-0 PH-on-CA lens and the
  reused cdrh3_ph helpers; it is a coarse, rigid-motion-invariant shape lens —
  stated as an honest limit.)

We reuse cdrh3_ph.compute_persistence / topo_distance / loop_likeness so the
PH machinery, the infinite-H0-bar handling and the distance metrics are identical
to Rung 0. Robust to missing chains / tiny interfaces (returns empty + a reason).

PDB parsing reuses pdb_to_cdrh3._parse_ca_atoms (fixed-column ATOM parser, no
Biopython), extended to return *all* chains at once for efficiency.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import pdb_to_cdrh3 as p2c
import cdrh3_ph as ph


# -----------------------------------------------------------------------------
# Multi-chain CA parsing (one pass over the file)
# -----------------------------------------------------------------------------
def parse_all_ca(pdb_path: str) -> Dict[str, List[Tuple[int, str, np.ndarray]]]:
    """Parse CA atoms of ALL chains in one pass.

    Returns {chain_id: [(resSeq, iCode, xyz), ...]} in file order, first-MODEL
    only, altLoc-deduplicated per (chain,resSeq,iCode). Mirrors the semantics of
    pdb_to_cdrh3._parse_ca_atoms but for every chain at once.
    """
    out: Dict[str, List[Tuple[int, str, np.ndarray]]] = {}
    seen = set()
    in_model = False
    saw_model_record = False
    with open(pdb_path, "r") as fh:
        for line in fh:
            rec = line[0:6]
            if rec.startswith("MODEL"):
                saw_model_record = True
                if in_model:
                    break
                in_model = True
                continue
            if rec.startswith("ENDMDL"):
                if saw_model_record:
                    break
                continue
            if not (rec.startswith("ATOM") or rec.startswith("HETATM")):
                continue
            if line[12:16].strip() != "CA":
                continue
            ch = line[21:22]
            try:
                resseq = int(line[22:26])
            except ValueError:
                continue
            icode = line[26:27]
            key = (ch, resseq, icode)
            if key in seen:
                continue
            try:
                x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
            except ValueError:
                continue
            seen.add(key)
            out.setdefault(ch, []).append((resseq, icode, np.array([x, y, z], dtype=np.float64)))
    return out


# -----------------------------------------------------------------------------
# Interface extraction
# -----------------------------------------------------------------------------
def _stack(ca_list: List[Tuple[int, str, np.ndarray]]) -> np.ndarray:
    if not ca_list:
        return np.zeros((0, 3), dtype=np.float64)
    return np.array([xyz for (_, _, xyz) in ca_list], dtype=np.float64)


def extract_interface(
    pdb_path: str,
    ab_chains: Sequence[str],
    ag_chains: Sequence[str],
    cutoff: float = 10.0,
) -> Dict[str, object]:
    """Extract the antibody--antigen interface Calpha point cloud + contact graph.

    An antibody (resp. antigen) residue is an interface residue if its CA lies
    within `cutoff` A of any antigen (resp. antibody) CA.

    Returns dict:
      'coords'      : (M,3) interface CA point cloud (antibody ∪ antigen sides)
      'ab_coords'   : (Ma,3) antibody-side interface CAs
      'ag_coords'   : (Mg,3) antigen-side interface CAs
      'n_ab'/'n_ag' : counts
      'edges'       : list of (i,j) index pairs into 'coords' with CA-CA < cutoff
      'n_edges'     : len(edges)
      'ok'          : bool (interface usable for PH)
      'reason'      : str (why not ok, if applicable)
    """
    ca = parse_all_ca(pdb_path)
    ab_list: List[Tuple[int, str, np.ndarray]] = []
    for c in ab_chains:
        ab_list.extend(ca.get(c, []))
    ag_list: List[Tuple[int, str, np.ndarray]] = []
    for c in ag_chains:
        ag_list.extend(ca.get(c, []))

    ab_xyz = _stack(ab_list)
    ag_xyz = _stack(ag_list)

    empty = {
        "coords": np.zeros((0, 3)), "ab_coords": np.zeros((0, 3)),
        "ag_coords": np.zeros((0, 3)), "n_ab": 0, "n_ag": 0,
        "edges": [], "n_edges": 0, "ok": False,
    }
    if ab_xyz.shape[0] == 0:
        empty["reason"] = f"no antibody CA (chains {list(ab_chains)} absent)"
        return empty
    if ag_xyz.shape[0] == 0:
        empty["reason"] = f"no antigen CA (chains {list(ag_chains)} absent)"
        return empty

    # pairwise CA-CA distances between the two sides
    from scipy.spatial.distance import cdist
    D = cdist(ab_xyz, ag_xyz)  # (Na, Ng)
    ab_mask = (D < cutoff).any(axis=1)   # antibody residues near some antigen CA
    ag_mask = (D < cutoff).any(axis=0)   # antigen residues near some antibody CA

    ab_if = ab_xyz[ab_mask]
    ag_if = ag_xyz[ag_mask]
    coords = np.vstack([ab_if, ag_if]) if (ab_if.shape[0] + ag_if.shape[0]) else np.zeros((0, 3))

    out = {
        "coords": coords,
        "ab_coords": ab_if,
        "ag_coords": ag_if,
        "n_ab": int(ab_if.shape[0]),
        "n_ag": int(ag_if.shape[0]),
    }
    if coords.shape[0] < 3:
        out.update({"edges": [], "n_edges": 0, "ok": False,
                    "reason": f"interface too small ({coords.shape[0]} CAs)"})
        return out

    # contact graph edges (CA-CA < cutoff) within the interface cloud
    from scipy.spatial.distance import pdist, squareform
    Dintra = squareform(pdist(coords))
    iu = np.triu_indices(coords.shape[0], k=1)
    edge_sel = Dintra[iu] < cutoff
    edges = list(zip(iu[0][edge_sel].tolist(), iu[1][edge_sel].tolist()))
    out.update({"edges": edges, "n_edges": len(edges), "ok": True, "reason": ""})
    return out


# -----------------------------------------------------------------------------
# Interface PD
# -----------------------------------------------------------------------------
def interface_pd(
    coords: np.ndarray,
    max_edge_length: Optional[float] = None,
    max_dimension: int = 2,
) -> Dict[str, object]:
    """Vietoris-Rips PH of the interface CA point cloud (reuses cdrh3_ph).

    Returns the same dict shape as cdrh3_ph.compute_persistence:
      'dgms' {0,1}, 'pd0', 'pd1', 'n_points', 'diameter'.

    max_edge_length default (None) -> 2x diameter (as in cdrh3_ph) so cycles in
    the contact patch can fully form. max_dimension=2 (H0,H1; H2 voids possible
    on a closed shell but we only read H0/H1, matching Rung 0).
    """
    return ph.compute_persistence(coords, max_edge_length=max_edge_length, max_dimension=max_dimension)


def interface_topo_distance(pd_cand: np.ndarray, pd_native: np.ndarray,
                            metric: str = "bottleneck", order: float = 1.0) -> float:
    """topo_distance between candidate and native interface PDs (one homology dim)."""
    return ph.topo_distance(pd_cand, pd_native, metric=metric, order=order)


def interface_loop_likeness(pd_h1: np.ndarray, mode: str = "total") -> float:
    """Native-FREE interface 'loop-likeness' = H1 persistence (total/max).

    Interpretable as: how much persistent 1-cycle structure the interface contact
    patch carries. Higher = more ring-like / topologically structured interface.
    """
    return ph.loop_likeness(pd_h1, mode=mode)


# -----------------------------------------------------------------------------
# Chain-identity helper (RabD / SAbDab JSONL annotation)
# -----------------------------------------------------------------------------
def load_chain_map(*json_paths: str) -> Dict[str, dict]:
    """Merge JSONL annotation files into {pdb: record}. Later files do not
    override earlier ones (first hit wins) so a primary source can be given first.
    Each record exposes heavy_chain / light_chain / antigen_chains and cdr*_pos.
    """
    import json as _json
    out: Dict[str, dict] = {}
    for path in json_paths:
        if not path or not os.path.exists(path):
            continue
        with open(path, "r") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = _json.loads(line)
                except Exception:
                    continue
                pdb = rec.get("pdb")
                if pdb and pdb not in out:
                    out[pdb] = rec
    return out


def chains_from_identifier(identifier: str) -> Optional[Tuple[str, str, List[str]]]:
    """Parse FlowDesign identifier '<pdb>_<H>_<L>_<Ag...>' -> (H, L, [Ag...]).

    Tokens after pdb: token[0]=heavy, token[1]=light, token[2:]=antigen chains.
    Single-character chain ids assumed (PDB standard). Returns None if too few
    tokens. This is a FALLBACK; the JSON annotation is preferred when available.
    """
    parts = identifier.split("_")
    if len(parts) < 4:
        return None
    H = parts[1]
    L = parts[2]
    Ag = [p for p in parts[3:] if len(p) == 1 and p.isalpha()]
    # stop antigen collection at the first non-single-char token (timestamp etc.)
    Ag2: List[str] = []
    for p in parts[3:]:
        if len(p) == 1 and (p.isalpha() or p.isdigit()):
            Ag2.append(p)
        else:
            break
    Ag = Ag2 if Ag2 else Ag
    return H, L, Ag
