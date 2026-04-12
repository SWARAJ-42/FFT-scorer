"""
Dataset Preparation Script
===========================
Reads PRDBv3.json and for every complex:
  1. Copies the bound complex PDB into ALL_PDBs/<COMPLEX_ID>/<COMPLEX_ID>.pdb
  2. Splits it into protein and RNA by chain classification → always extracted
     from the complex (U_pro_PDB / U_RNA_PDB fields are ignored entirely).
  3. Saves:
       <COMPLEX_ID>/protein.pdb
       <COMPLEX_ID>/RNA.pdb
  4. Writes PRDBv3_updated.json with U_pro_PDB and U_RNA_PDB set to the
     generated filenames for every entry.

Output layout:
    ALL_PDBs/
        <COMPLEX_ID>/
            <COMPLEX_ID>.pdb   ← bound complex (always)
            protein.pdb        ← protein chains extracted from complex
            RNA.pdb            ← RNA chains extracted from complex

    assets/PRDBv3_updated.json ← updated metadata
"""

import json
import shutil
import copy
from pathlib import Path

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────

JSON_PATH           = "../assets/PRDBv3_legacy.json"
PRDB_DIR            = Path("../assets/PRDBv3.0")
UNBOUND_COMPLEX_DIR = Path("../assets/ALL_PDBs")
OUT_ROOT            = Path("../ALL_PDBs_v2")
UPDATED_JSON_PATH   = "../assets/PRDBv3_updated.json"

# ──────────────────────────────────────────────


OUT_ROOT.mkdir(exist_ok=True)

with open(JSON_PATH, "r") as f:
    data = json.load(f)


# ═══════════════════════════════════════════════════════════════════════════
# Residue sets
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

SKIP_RESIDUES = {"HOH", "WAT", "H2O"}


def detect_mol_type(residue_names: set) -> str:
    res = residue_names - SKIP_RESIDUES
    n_protein = len(res & PROTEIN_RESIDUES)
    n_rna     = len(res & RNA_RESIDUES)
    if n_protein == 0 and n_rna == 0:
        return "unknown"
    return "protein" if n_protein >= n_rna else "rna"


# ═══════════════════════════════════════════════════════════════════════════
# PDB parser
# ═══════════════════════════════════════════════════════════════════════════

def parse_pdb_chains(pdb_path: Path) -> dict:
    """
    Parse a PDB file and return:
        { chain_id: { 'mol_type': str, 'lines': [raw PDB lines] } }
    Skips waters, hydrogens, and non-primary alt locations.
    """
    chains = {}

    with open(pdb_path, "r") as fh:
        for line in fh:
            rec = line[:6].strip()
            if rec not in ("ATOM", "HETATM"):
                continue

            atom_name = line[12:16].strip()
            element   = line[76:78].strip() if len(line) > 76 else ""

            if atom_name.startswith("H") or element == "H":
                continue

            res_name = line[17:20].strip()
            if res_name in SKIP_RESIDUES:
                continue

            chain_id = line[21].strip() or "_"
            alt_loc  = line[16].strip()

            if alt_loc not in ("", "A", " "):
                continue

            if chain_id not in chains:
                chains[chain_id] = {"residue_names": set(), "lines": []}

            chains[chain_id]["residue_names"].add(res_name)
            chains[chain_id]["lines"].append(line)

    result = {}
    for chain_id, chain_data in chains.items():
        mol_type = detect_mol_type(chain_data["residue_names"])
        result[chain_id] = {
            "mol_type":      mol_type,
            "residue_names": chain_data["residue_names"],
            "lines":         chain_data["lines"],
        }

    return result


def write_chains_to_pdb(chain_data_list: list, out_path: Path, remark: str = ""):
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


# ═══════════════════════════════════════════════════════════════════════════
# Main loop
# ═══════════════════════════════════════════════════════════════════════════

unbound_complex_ids = {
    p.parent.name.upper()
    for p in UNBOUND_COMPLEX_DIR.glob("*/*.pdb")
    if p.stem.lower() == p.parent.name.lower()
}
print(f"Found {len(unbound_complex_ids)} unbound complexes\n")

updated_data = copy.deepcopy(data)
stats = {"ok": 0, "missing_complex": 0, "no_protein": 0, "no_rna": 0}

for idx, obj in enumerate(data):

    complex_id = obj.get("C_PDB")
    if not complex_id:
        continue

    complex_id = complex_id.upper()

    if complex_id not in unbound_complex_ids:
        continue

    # ── Locate bound complex ─────────────────────────────────────────────
    complex_src = find_pdb(complex_id)
    if not complex_src:
        print(f"[{complex_id}]  ✗  bound complex PDB not found — skipping")
        stats["missing_complex"] += 1
        continue

    complex_dir = OUT_ROOT / complex_id
    complex_dir.mkdir(exist_ok=True)

    # Copy bound complex
    shutil.copy(complex_src, complex_dir / f"{complex_id}.pdb")

    # ── Split complex into chains ────────────────────────────────────────
    chains = parse_pdb_chains(complex_src)

    protein_chains = [c for c in chains.values() if c["mol_type"] == "protein"]
    rna_chains     = [c for c in chains.values() if c["mol_type"] == "rna"]

    # ── Write protein.pdb ────────────────────────────────────────────────
    if protein_chains:
        write_chains_to_pdb(
            protein_chains,
            complex_dir / "protein.pdb",
            remark=f"Protein chains extracted from {complex_id}",
        )
        updated_data[idx]["U_pro_PDB"] = "protein"   # filename without .pdb
        print(f"[{complex_id}]  ✓  protein.pdb  ({len(protein_chains)} chain(s))")
    else:
        print(f"[{complex_id}]  ✗  no protein chains detected")
        stats["no_protein"] += 1

    # ── Write RNA.pdb ────────────────────────────────────────────────────
    if rna_chains:
        write_chains_to_pdb(
            rna_chains,
            complex_dir / "RNA.pdb",
            remark=f"RNA chains extracted from {complex_id}",
        )
        updated_data[idx]["U_RNA_PDB"] = "RNA"        # filename without .pdb
        print(f"[{complex_id}]  ✓  RNA.pdb      ({len(rna_chains)} chain(s))")
    else:
        print(f"[{complex_id}]  ✗  no RNA chains detected")
        stats["no_rna"] += 1

    if protein_chains and rna_chains:
        stats["ok"] += 1

# ═══════════════════════════════════════════════════════════════════════════
# Write updated JSON
# ═══════════════════════════════════════════════════════════════════════════

with open(UPDATED_JSON_PATH, "w") as fh:
    json.dump(updated_data, fh, indent=2)

print(f"\n{'═'*55}")
print(f"  Done.")
print(f"  Fully split       : {stats['ok']}")
print(f"  Missing complexes : {stats['missing_complex']}")
print(f"  No protein found  : {stats['no_protein']}")
print(f"  No RNA found      : {stats['no_rna']}")
print(f"  Updated JSON → {UPDATED_JSON_PATH}")
print(f"{'═'*55}")