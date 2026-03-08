"""
Phase 1 — PDB Parser & Preprocessor
=====================================
Reads PRDBv3.json, filters UU docking cases, resolves PDB file paths
from the folder structure, and parses protein + RNA chains from each PDB.

Folder structure assumed:
    <PDB_ROOT>/
        <C_PDB>/
            <C_PDB>.pdb          ← bound complex
            <U_pro_PDB>.pdb      ← unbound protein
            <U_RNA_PDB>.pdb      ← unbound RNA

Usage:
    python phase1_loader.py --json ./assets/PRDBv3.json --pdb_root ./UU_PDBS

Output:
    For each UU complex, prints resolved paths and parsed chain summary.
    Returns a list of DockingCase models for downstream use.
"""

import os
import json
import argparse
from typing import Optional, List, Dict, Set
from pydantic import BaseModel, Field


# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

# Standard RNA residue names (includes both plain and R-prefixed variants)
RNA_RESIDUES = {
    "A", "U", "G", "C",          # bare names (common in many PDBs)
    "RA", "RU", "RG", "RC",      # R-prefixed (AMBER convention)
    "ADE", "URA", "GUA", "CYT",  # 3-letter variants
}

# Standard amino acid residue names
PROTEIN_RESIDUES = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY",
    "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER",
    "THR", "TRP", "TYR", "VAL",
    "MSE",   # selenomethionine — treated as protein
}


# ──────────────────────────────────────────────
# Data containers
# ──────────────────────────────────────────────

class Atom(BaseModel):
    record: str       # ATOM or HETATM
    serial: int
    name: str         # atom name  e.g. CA, N, C4'
    alt_loc: str
    res_name: str     # residue name
    chain_id: str
    res_seq: int
    icode: str
    x: float
    y: float
    z: float
    occupancy: float = 1.0
    b_factor: float = 0.0
    element: str = ""


class Chain(BaseModel):
    chain_id: str
    mol_type: str                 # 'protein' | 'rna' | 'unknown'
    atoms: List[Atom] = Field(default_factory=list)

    def residue_names(self) -> Set[str]:
        return {a.res_name for a in self.atoms}

    def __repr__(self):
        return (
            f"Chain(id={self.chain_id!r}, type={self.mol_type!r}, "
            f"atoms={len(self.atoms)}, residues={len(self.residue_names())})"
        )


class Structure(BaseModel):
    pdb_id: str
    filepath: str
    chains: List[Chain] = Field(default_factory=list)

    def protein_chains(self):
        return [c for c in self.chains if c.mol_type == "protein"]

    def rna_chains(self):
        return [c for c in self.chains if c.mol_type == "rna"]

    def __repr__(self):
        return (
            f"Structure(pdb={self.pdb_id!r}, "
            f"chains={[c.chain_id for c in self.chains]}, "
            f"protein_chains={[c.chain_id for c in self.protein_chains()]}, "
            f"rna_chains={[c.chain_id for c in self.rna_chains()]})"
        )


class DockingCase(BaseModel):
    complex_id: str
    complex_pdb: str
    protein_pdb: str
    rna_pdb: str

    complex_struct: Optional[Structure] = None
    protein_struct: Optional[Structure] = None
    rna_struct: Optional[Structure] = None


# ──────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────

def clean_pdb_id(raw: str) -> str:
    """Strip trailing * or whitespace from a PDB ID string."""
    return raw.strip().rstrip("*").strip()


def detect_mol_type(residue_names: Set[str]) -> str:
    """
    Given the set of residue names in a chain, decide if it is
    protein, rna, or unknown.
    Majority-vote: whichever category has more matching residues wins.
    """
    n_protein = len(residue_names & PROTEIN_RESIDUES)
    n_rna = len(residue_names & RNA_RESIDUES)

    if n_protein == 0 and n_rna == 0:
        return "unknown"

    if n_protein >= n_rna:
        return "protein"

    return "rna"


# ──────────────────────────────────────────────
# PDB Parser
# ──────────────────────────────────────────────

def parse_pdb(filepath: str, pdb_id: str = "") -> Structure:
    """
    Parse a PDB file into a Structure object.
    Only ATOM and HETATM records are read.
    Chains are assembled and their mol_type is inferred automatically.
    """

    chains_dict: Dict[str, List[Atom]] = {}

    with open(filepath, "r") as fh:
        for line in fh:

            rec = line[:6].strip()

            if rec not in ("ATOM", "HETATM"):
                continue

            # Skip hydrogen atoms — not needed for shape/electrostatic grids
            atom_name = line[12:16].strip()
            element = line[76:78].strip() if len(line) > 76 else ""

            if atom_name.startswith("H") or element == "H":
                continue

            try:

                atom = Atom(
                    record=rec,
                    serial=int(line[6:11]),
                    name=atom_name,
                    alt_loc=line[16].strip(),
                    res_name=line[17:20].strip(),
                    chain_id=line[21].strip() or "_",
                    res_seq=int(line[22:26]),
                    icode=line[26].strip(),
                    x=float(line[30:38]),
                    y=float(line[38:46]),
                    z=float(line[46:54]),
                    occupancy=float(line[54:60]) if line[54:60].strip() else 1.0,
                    b_factor=float(line[60:66]) if line[60:66].strip() else 0.0,
                    element=element,
                )

            except (ValueError, IndexError):
                # Malformed line — skip silently
                continue

            # Only keep the first alt location
            if atom.alt_loc not in ("", "A", " "):
                continue

            chains_dict.setdefault(atom.chain_id, []).append(atom)

    # Build Chain objects with inferred mol_type
    chains: List[Chain] = []

    for chain_id, atoms in chains_dict.items():

        res_names = {a.res_name for a in atoms}
        mol_type = detect_mol_type(res_names)

        chains.append(
            Chain(
                chain_id=chain_id,
                mol_type=mol_type,
                atoms=atoms,
            )
        )

    return Structure(
        pdb_id=pdb_id or os.path.splitext(os.path.basename(filepath))[0],
        filepath=filepath,
        chains=chains,
    )


# ──────────────────────────────────────────────
# JSON loader & path resolver
# ──────────────────────────────────────────────

def resolve_pdb_path(pdb_root: str, complex_id: str, pdb_name: str) -> Optional[str]:
    """
    Given the root folder, the complex folder name, and a PDB ID,
    return the full path if the file exists, else None.
    """

    filename = pdb_name.upper() + ".pdb"
    path = os.path.join(pdb_root, complex_id.upper(), filename)

    return path if os.path.isfile(path) else None


def load_uu_cases(json_path: str, pdb_root: str):
    """
    Read PRDBv3.json, filter UU docking cases, resolve file paths,
    parse all three PDB structures per case.

    Returns list[DockingCase]
    """

    with open(json_path, "r") as fh:
        records = json.load(fh)

    # Handle both a top-level list and a dict wrapping a list
    if isinstance(records, dict):

        for key in ("data", "entries", "complexes", "records"):
            if key in records:
                records = records[key]
                break
        else:
            records = next(v for v in records.values() if isinstance(v, list))

    cases: List[DockingCase] = []
    skipped = []

    for rec in records:

        docking_case = rec.get("Docking_case", "").strip().upper()

        if docking_case != "UU":
            continue

        # ── Extract & clean IDs ──────────────────────────────────────────
        complex_id = clean_pdb_id(rec.get("C_PDB", ""))
        pro_id = clean_pdb_id(rec.get("U_pro_PDB", ""))
        rna_id = clean_pdb_id(rec.get("U_RNA_PDB", ""))

        if not complex_id or not pro_id or not rna_id:
            skipped.append({"record": rec, "reason": "missing field"})
            continue

        # ── Resolve file paths ───────────────────────────────────────────
        complex_path = resolve_pdb_path(pdb_root, complex_id, complex_id)
        protein_path = resolve_pdb_path(pdb_root, complex_id, pro_id)
        rna_path = resolve_pdb_path(pdb_root, complex_id, rna_id)

        missing = []

        if not complex_path:
            missing.append(f"complex({complex_id})")

        if not protein_path:
            missing.append(f"protein({pro_id})")

        if not rna_path:
            missing.append(f"rna({rna_id})")

        if missing:
            skipped.append(
                {"complex": complex_id, "reason": f"file not found: {missing}"}
            )
            continue

        case = DockingCase(
            complex_id=complex_id,
            complex_pdb=complex_path,
            protein_pdb=protein_path,
            rna_pdb=rna_path,
        )

        try:

            case.complex_struct = parse_pdb(complex_path, pdb_id=complex_id)
            case.protein_struct = parse_pdb(protein_path, pdb_id=pro_id)
            case.rna_struct = parse_pdb(rna_path, pdb_id=rna_id)

        except Exception as e:
            skipped.append({"complex": complex_id, "reason": f"parse error: {e}"})
            continue

        cases.append(case)

    return cases, skipped


# ──────────────────────────────────────────────
# Validation helpers
# ──────────────────────────────────────────────

def validate_case(case: DockingCase):

    warnings = []

    if not case.protein_struct.protein_chains():
        warnings.append(
            f"[{case.complex_id}] Unbound protein file has NO detected protein chains"
        )

    if not case.rna_struct.rna_chains():
        warnings.append(
            f"[{case.complex_id}] Unbound RNA file has NO detected RNA chains"
        )

    pro_atoms = sum(len(c.atoms) for c in case.protein_struct.chains)
    rna_atoms = sum(len(c.atoms) for c in case.rna_struct.chains)

    if pro_atoms < 10:
        warnings.append(f"[{case.complex_id}] Very few protein atoms: {pro_atoms}")

    if rna_atoms < 10:
        warnings.append(f"[{case.complex_id}] Very few RNA atoms: {rna_atoms}")

    return warnings


# ──────────────────────────────────────────────
# Pretty printer
# ──────────────────────────────────────────────

def print_summary(cases, skipped):

    SEP = "─" * 70

    print(f"\n{'═'*70}")
    print("  Phase 1 Summary — UU Protein-RNA Docking Cases")
    print(f"{'═'*70}")
    print(f"  Loaded : {len(cases)}")
    print(f"  Skipped: {len(skipped)}")
    print(f"{'═'*70}\n")

    for case in cases:

        print(SEP)

        print(f"  Complex  : {case.complex_id}")

        print("  Files")
        print(f"    Complex : {case.complex_pdb}")
        print(f"    Protein : {case.protein_pdb}")
        print(f"    RNA     : {case.rna_pdb}")

        print("  Parsed structures")
        print(f"    Complex : {case.complex_struct}")
        print(f"    Protein : {case.protein_struct}")
        print(f"    RNA     : {case.rna_struct}")

        warnings = validate_case(case)

        if warnings:
            print("  ⚠ Warnings:")
            for w in warnings:
                print(f"    {w}")
        else:
            print("  ✓ Validation passed")

    if skipped:
        print(f"\n{SEP}")
        print("  Skipped cases:")
        for s in skipped:
            print(f"    {s}")

    print(f"\n{'═'*70}\n")


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

def main():

    parser = argparse.ArgumentParser(
        description="Phase 1: Load and parse UU protein-RNA docking cases"
    )

    parser.add_argument(
        "--json",
        default="./assets/PRDBv3.json",
        help="Path to PRDBv3.json",
    )

    parser.add_argument(
        "--pdb_root",
        default="./UU_PDBS",
        help="Root folder containing complex subfolders",
    )

    args = parser.parse_args()

    print(f"Loading JSON : {args.json}")
    print(f"PDB root     : {args.pdb_root}")

    cases, skipped = load_uu_cases(args.json, args.pdb_root)

    # Visualize the first complex for quick inspection
    if cases:
        print("Opening visualization for first case...")
        visualize_structure(cases[0].complex_struct, title=f"Complex {cases[0].complex_id}")

    print_summary(cases, skipped)

    return cases


# ──────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────

def visualize_structure(struct: Structure, title: str = ""):
    """
    Interactive 3D visualization of a Structure using Plotly.
    Protein chains are colored blue, RNA chains red.
    """

    try:
        import plotly.graph_objects as go
    except ImportError:
        print("Plotly not installed. Run: pip install plotly")
        return

    protein_x, protein_y, protein_z = [], [], []
    rna_x, rna_y, rna_z = [], [], []
    unknown_x, unknown_y, unknown_z = [], [], []

    for chain in struct.chains:

        for atom in chain.atoms:

            if chain.mol_type == "protein":
                protein_x.append(atom.x)
                protein_y.append(atom.y)
                protein_z.append(atom.z)

            elif chain.mol_type == "rna":
                rna_x.append(atom.x)
                rna_y.append(atom.y)
                rna_z.append(atom.z)

            else:
                unknown_x.append(atom.x)
                unknown_y.append(atom.y)
                unknown_z.append(atom.z)

    fig = go.Figure()

    # Protein atoms
    fig.add_trace(go.Scatter3d(
        x=protein_x,
        y=protein_y,
        z=protein_z,
        mode="markers",
        marker=dict(size=3),
        name="Protein"
    ))

    # RNA atoms
    fig.add_trace(go.Scatter3d(
        x=rna_x,
        y=rna_y,
        z=rna_z,
        mode="markers",
        marker=dict(size=3),
        name="RNA"
    ))

    # Unknown atoms
    if unknown_x:
        fig.add_trace(go.Scatter3d(
            x=unknown_x,
            y=unknown_y,
            z=unknown_z,
            mode="markers",
            marker=dict(size=3),
            name="Unknown"
        ))

    fig.update_layout(
        title=title or struct.pdb_id,
        scene=dict(
            xaxis_title="X (Å)",
            yaxis_title="Y (Å)",
            zaxis_title="Z (Å)"
        ),
        width=900,
        height=700
    )

    fig.show()

if __name__ == "__main__":
    main()