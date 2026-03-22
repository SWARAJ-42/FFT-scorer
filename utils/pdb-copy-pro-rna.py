"""
Dataset Preparation Script
===========================
Reads PRDBv3.json, copies unbound protein + RNA PDBs into ALL_PDBs/<COMPLEX_ID>/.
If U_pro_PDB or U_RNA_PDB is missing/null, extracts the protein or RNA directly
from the bound complex by chain classification, and saves them as:
    <COMPLEX_ID>_pro.pdb
    <COMPLEX_ID>_rna.pdb

Also writes a new JSON (PRDBv3_updated.json) where previously-null U_pro_PDB /
U_RNA_PDB fields are filled in with the generated IDs.

Output layout:
    ALL_PDBs/
        <COMPLEX_ID>/
            <COMPLEX_ID>.pdb          ← bound complex (always)
            <U_pro_PDB>.pdb           ← unbound protein (copied or extracted)
            <U_RNA_PDB>.pdb           ← unbound RNA     (copied or extracted)

    assets/PRDBv3_updated.json        ← updated metadata with no null IDs
"""

import json
import shutil
import copy
from pathlib import Path

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────

JSON_PATH            = "../assets/PRDBv3.json"
PRDB_DIR             = Path("../assets/PRDBv3.0")
UNBOUND_COMPLEX_DIR  = Path("../assets/PDBs/C_PDB")
OUT_ROOT             = Path("../ALL_PDBs_v2")
UPDATED_JSON_PATH    = "../assets/PRDBv3_updated.json"

# ──────────────────────────────────────────────


OUT_ROOT.mkdir(exist_ok=True)

with open(JSON_PATH, "r") as f:
    data = json.load(f)


# ═══════════════════════════════════════════════════════════════════════════
# Residue sets — mirrors PDBparser.py exactly
# ═══════════════════════════════════════════════════════════════════════════

RNA_RESIDUES = {
    "A", "U", "G", "C",
    "RA", "RU", "RG", "RC",
    "ADE", "URA", "GUA", "CYT",
}

PROTEIN_RESIDUES = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY",
    "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER",
    "THR", "TRP", "TYR", "VAL",
    "MSE",
}

SKIP_RESIDUES = {"HOH", "WAT", "H2O"}   # water — ignore entirely


def detect_mol_type(residue_names: set) -> str:
    res = residue_names - SKIP_RESIDUES
    n_protein = len(res & PROTEIN_RESIDUES)
    n_rna     = len(res & RNA_RESIDUES)
    if n_protein == 0 and n_rna == 0:
        return "unknown"
    return "protein" if n_protein >= n_rna else "rna"


# ═══════════════════════════════════════════════════════════════════════════
# PDB parser — minimal, self-contained (no dependency on PDBparser.py)
# ═══════════════════════════════════════════════════════════════════════════

def parse_pdb_chains(pdb_path: Path) -> dict:
    """
    Parse a PDB file and return a dict:
        { chain_id: { 'mol_type': str, 'lines': [raw PDB lines] } }

    Only ATOM / HETATM records are kept (no waters).
    Alt locations: keep first only.
    Hydrogens: skipped.
    """
    chains = {}   # chain_id -> {'residue_names': set, 'lines': [str]}

    with open(pdb_path, "r") as fh:
        for line in fh:
            rec = line[:6].strip()
            if rec not in ("ATOM", "HETATM"):
                continue

            atom_name = line[12:16].strip()
            element   = line[76:78].strip() if len(line) > 76 else ""

            # Skip hydrogens
            if atom_name.startswith("H") or element == "H":
                continue

            res_name = line[17:20].strip()

            # Skip water
            if res_name in SKIP_RESIDUES:
                continue

            chain_id = line[21].strip() or "_"
            alt_loc  = line[16].strip()

            # Keep first alt location only
            if alt_loc not in ("", "A", " "):
                continue

            if chain_id not in chains:
                chains[chain_id] = {"residue_names": set(), "lines": []}

            chains[chain_id]["residue_names"].add(res_name)
            chains[chain_id]["lines"].append(line)

    # Classify each chain
    result = {}
    for chain_id, data in chains.items():
        mol_type = detect_mol_type(data["residue_names"])
        result[chain_id] = {
            "mol_type":      mol_type,
            "residue_names": data["residue_names"],
            "lines":         data["lines"],
        }

    return result


def write_chains_to_pdb(chain_data_list: list, out_path: Path, remark: str = ""):
    """
    Write a list of chain dicts (each with 'lines') to a single PDB file.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        if remark:
            fh.write(f"REMARK  {remark}\n")
        for chain_data in chain_data_list:
            for line in chain_data["lines"]:
                fh.write(line)
        fh.write("END\n")


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def clean_id(pdb_id: str) -> str:
    return pdb_id.replace("*", "").strip().upper()


def find_pdb(pdb_id: str) -> Path | None:
    pdb_id = clean_id(pdb_id)
    matches = list(PRDB_DIR.rglob(f"{pdb_id}.pdb"))
    return matches[0] if matches else None


def is_missing(value) -> bool:
    """True if a JSON field is null, empty, or whitespace-only."""
    return not value or not str(value).strip()


# ═══════════════════════════════════════════════════════════════════════════
# Main loop
# ═══════════════════════════════════════════════════════════════════════════

unbound_complex_ids = {
    p.stem.upper() for p in UNBOUND_COMPLEX_DIR.glob("*.pdb")
}
print(f"Found {len(unbound_complex_ids)} unbound complexes\n")

updated_data  = copy.deepcopy(data)   # will become the new JSON
stats = {"copied": 0, "extracted": 0, "missing_complex": 0, "unknown_chain": 0}

for idx, obj in enumerate(data):

    complex_id = obj.get("C_PDB")
    if not complex_id:
        continue

    complex_id = complex_id.upper()

    if complex_id not in unbound_complex_ids:
        continue

    pro_id = obj.get("U_pro_PDB")
    rna_id = obj.get("U_RNA_PDB")

    complex_dir = OUT_ROOT / complex_id
    complex_dir.mkdir(exist_ok=True)

    # ── Always copy the bound complex ────────────────────────────────────
    complex_src = find_pdb(complex_id)
    if not complex_src:
        print(f"[{complex_id}]  ✗  bound complex PDB not found — skipping")
        stats["missing_complex"] += 1
        continue

    shutil.copy(complex_src, complex_dir / f"{complex_id}.pdb")

    # ── Parse complex chains once if either field is missing ─────────────
    complex_chains = None
    if is_missing(pro_id) or is_missing(rna_id):
        complex_chains = parse_pdb_chains(complex_src)

    # ── Protein ───────────────────────────────────────────────────────────
    if not is_missing(pro_id):
        pro_id_clean = clean_id(pro_id)
        pro_src      = find_pdb(pro_id_clean)
        if pro_src:
            shutil.copy(pro_src, complex_dir / f"{pro_id_clean}.pdb")
            print(f"[{complex_id}]  protein  {pro_id_clean}  (copied)")
            stats["copied"] += 1
        else:
            print(f"[{complex_id}]  ✗  protein PDB {pro_id_clean} not found in PRDB_DIR")
    else:
        pro_chains = [c for c in complex_chains.values() if c["mol_type"] == "protein"]
        if not pro_chains:
            print(f"[{complex_id}]  ✗  no protein chains detected in complex — cannot extract")
            stats["unknown_chain"] += 1
        else:
            generated_pro_id  = f"{complex_id}_pro"
            generated_pro_pdb = complex_dir / f"{generated_pro_id}.pdb"
            write_chains_to_pdb(
                pro_chains,
                generated_pro_pdb,
                remark=f"Extracted protein chains from {complex_id} — generated by prep script",
            )
            updated_data[idx]["U_pro_PDB"] = generated_pro_id
            print(f"[{complex_id}]  protein  {generated_pro_id}  (extracted from complex)")
            stats["extracted"] += 1

    # ── RNA ───────────────────────────────────────────────────────────────
    if not is_missing(rna_id):
        rna_id_clean = clean_id(rna_id)
        rna_src      = find_pdb(rna_id_clean)
        if rna_src:
            shutil.copy(rna_src, complex_dir / f"{rna_id_clean}.pdb")
            print(f"[{complex_id}]  rna      {rna_id_clean}  (copied)")
            stats["copied"] += 1
        else:
            print(f"[{complex_id}]  ✗  RNA PDB {rna_id_clean} not found in PRDB_DIR")
    else:
        rna_chains = [c for c in complex_chains.values() if c["mol_type"] == "rna"]
        if not rna_chains:
            print(f"[{complex_id}]  ✗  no RNA chains detected in complex — cannot extract")
            stats["unknown_chain"] += 1
        else:
            generated_rna_id  = f"{complex_id}_rna"
            generated_rna_pdb = complex_dir / f"{generated_rna_id}.pdb"
            write_chains_to_pdb(
                rna_chains,
                generated_rna_pdb,
                remark=f"Extracted RNA chains from {complex_id} — generated by prep script",
            )
            updated_data[idx]["U_RNA_PDB"] = generated_rna_id
            print(f"[{complex_id}]  rna      {generated_rna_id}  (extracted from complex)")
            stats["extracted"] += 1

# ═══════════════════════════════════════════════════════════════════════════
# Write updated JSON
# ═══════════════════════════════════════════════════════════════════════════

with open(UPDATED_JSON_PATH, "w") as fh:
    json.dump(updated_data, fh, indent=2)

print(f"\n{'═'*55}")
print(f"  Done.")
print(f"  Copied    : {stats['copied']} PDB files")
print(f"  Extracted : {stats['extracted']} PDB files from complexes")
print(f"  Missing complexes  : {stats['missing_complex']}")
print(f"  Unclassified chains: {stats['unknown_chain']}")
print(f"  Updated JSON written to: {UPDATED_JSON_PATH}")
print(f"{'═'*55}")