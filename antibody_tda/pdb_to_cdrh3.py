"""
pdb_to_cdrh3.py — extract heavy-chain CDR-H3 (+/- K flanking anchors) Calpha coords.

Thread C, Rung 0 (GO/NO-GO gate). Reads PDB ATOM records directly using fixed
PDB column positions (Biopython NOT required). Robust to missing residues and
insertion codes.

Two ways to specify the CDR-H3 residue window:
  1. FlowDesign outputs: use the residue range from the target's metadata.json
     `items[*]` -> (chain, residue_first, residue_last). Use `load_metadata_range`.
  2. IMGT JSON annotation (SAbDab / RabD test.json): `cdrh3_pos = [start, end]`
     on the heavy chain, matched to PDB residue *numbering*.

We add K flanking anchor residues on each side (default K=3) so the open loop
acquires enough context to *close* into a ring -> a non-trivial H1 (1-cycle) in
Vietoris-Rips persistence. Increasing K closes the loop more aggressively.

Returns an (N, 3) float64 array of CA coordinates ordered by residue position.

PDB fixed columns (1-indexed, per the PDB spec):
  record name : 1-6
  atom name   : 13-16
  altLoc      : 17
  resName     : 18-20
  chainID     : 22
  resSeq      : 23-26
  iCode       : 27
  x           : 31-38
  y           : 39-46
  z           : 47-54
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np


# -----------------------------------------------------------------------------
# Low-level PDB CA parsing
# -----------------------------------------------------------------------------
def _parse_ca_atoms(pdb_path: str, chain: str) -> List[Tuple[int, str, np.ndarray]]:
    """Parse all CA atoms of `chain` from a PDB file.

    Returns a list of (resSeq:int, iCode:str, xyz:np.ndarray) in file order.
    Handles altLocs by keeping the first CA seen for a given (resSeq, iCode).
    Reads only the first MODEL if multiple MODEL records are present.
    """
    out: List[Tuple[int, str, np.ndarray]] = []
    seen = set()
    in_model = False
    saw_model_record = False
    with open(pdb_path, "r") as fh:
        for line in fh:
            rec = line[0:6]
            if rec.startswith("MODEL"):
                saw_model_record = True
                if in_model:
                    # second model begins -> stop (keep only first model)
                    break
                in_model = True
                continue
            if rec.startswith("ENDMDL"):
                if saw_model_record:
                    break
                continue
            if not (rec.startswith("ATOM") or rec.startswith("HETATM")):
                continue
            # atom name occupies cols 13-16; strip whitespace for comparison
            atom_name = line[12:16].strip()
            if atom_name != "CA":
                continue
            ch = line[21:22]
            if ch != chain:
                continue
            try:
                resseq = int(line[22:26])
            except ValueError:
                continue
            icode = line[26:27]
            key = (resseq, icode)
            if key in seen:  # altLoc duplicate -> keep first
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            seen.add(key)
            out.append((resseq, icode, np.array([x, y, z], dtype=np.float64)))
    return out


# -----------------------------------------------------------------------------
# metadata.json (FlowDesign) helper
# -----------------------------------------------------------------------------
def load_metadata_range(metadata_path: str, tag: str = "H_CDR3") -> Tuple[str, int, int]:
    """Read (chain, first_resseq, last_resseq) for a given CDR `tag` from
    a FlowDesign metadata.json. residue_first/last are [chain, resseq, icode]."""
    with open(metadata_path, "r") as fh:
        meta = json.load(fh)
    items = meta.get("items", [])
    chosen = None
    for it in items:
        if it.get("tag") == tag or it.get("cdr") == tag:
            chosen = it
            break
    if chosen is None and items:
        chosen = items[0]
    if chosen is None:
        raise ValueError(f"No items in metadata {metadata_path}")
    rf = chosen["residue_first"]
    rl = chosen["residue_last"]
    chain = rf[0]
    return chain, int(rf[1]), int(rl[1])


# -----------------------------------------------------------------------------
# Core extraction
# -----------------------------------------------------------------------------
def extract_cdrh3_ca(
    pdb_path: str,
    chain: str,
    res_first: int,
    res_last: int,
    k_anchor: int = 3,
) -> np.ndarray:
    """Extract CA coords for CDR-H3 residues [res_first, res_last] on `chain`,
    extended by `k_anchor` residues on each side (by *position in the chain's
    ordered CA list*, so gaps in numbering do not break anchoring).

    Returns (N, 3) array ordered along the chain. Robust to missing residues:
    only residues actually present in the PDB are returned.
    """
    cas = _parse_ca_atoms(pdb_path, chain)
    if not cas:
        raise ValueError(f"No CA atoms for chain {chain} in {pdb_path}")

    # Sort by (resSeq, iCode) to get chain order. Insertion codes sort after blank.
    cas_sorted = sorted(cas, key=lambda t: (t[0], t[1]))

    # Indices of residues whose resSeq lies within the CDR window.
    core_idx = [i for i, (rs, ic, xyz) in enumerate(cas_sorted) if res_first <= rs <= res_last]
    if not core_idx:
        # Numbering may not match (e.g. window outside present residues).
        return np.zeros((0, 3), dtype=np.float64)

    lo = max(0, core_idx[0] - k_anchor)
    hi = min(len(cas_sorted), core_idx[-1] + 1 + k_anchor)
    window = cas_sorted[lo:hi]
    coords = np.array([xyz for (_, _, xyz) in window], dtype=np.float64)
    return coords


def extract_from_flowdesign(
    pdb_path: str,
    metadata_path: str,
    tag: str = "H_CDR3",
    k_anchor: int = 3,
) -> np.ndarray:
    """Convenience: read residue window from metadata.json and extract CA coords."""
    chain, rf, rl = load_metadata_range(metadata_path, tag=tag)
    return extract_cdrh3_ca(pdb_path, chain, rf, rl, k_anchor=k_anchor)


# -----------------------------------------------------------------------------
# IMGT-JSON path (SAbDab / RabD annotation)
# -----------------------------------------------------------------------------
def load_imgt_map(json_path: str) -> Dict[str, dict]:
    """Load JSONL annotation into {pdb: record}."""
    out: Dict[str, dict] = {}
    with open(json_path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[rec["pdb"]] = rec
    return out


def extract_from_imgt(
    pdb_path: str,
    imgt_record: dict,
    k_anchor: int = 3,
) -> np.ndarray:
    """Extract CDR-H3 CA coords using an IMGT annotation record
    (`heavy_chain`, `cdrh3_pos=[start,end]`)."""
    chain = imgt_record["heavy_chain"]
    start, end = imgt_record["cdrh3_pos"]
    return extract_cdrh3_ca(pdb_path, chain, int(start), int(end), k_anchor=k_anchor)


# -----------------------------------------------------------------------------
# CLI smoke
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 3:
        pdb, meta = sys.argv[1], sys.argv[2]
        k = int(sys.argv[3]) if len(sys.argv) > 3 else 3
        ch, rf, rl = load_metadata_range(meta)
        coords = extract_from_flowdesign(pdb, meta, k_anchor=k)
        print(f"chain={ch} window=[{rf},{rl}] K={k} -> {coords.shape[0]} CA atoms")
        print(coords[:3])
