import json
import shutil
from pathlib import Path

# -------- CONFIG --------
JSON_PATH = "../assets/PRDBv3.json"
PRDB_DIR = Path("../assets/PRDBv3.0")

UNBOUND_COMPLEX_DIR = Path("../assets/PDBs/C_PDB")
OUT_ROOT = Path("../ALL_PDBs")
# ------------------------

OUT_ROOT.mkdir(exist_ok=True)

# Load dataset
with open(JSON_PATH, "r") as f:
    data = json.load(f)


def clean_id(pdb_id):
    """Remove wildcard like 1U0B*"""
    return pdb_id.replace("*", "").upper()


def find_pdb(pdb_id):
    """Locate pdb file inside PRDBv3.0"""
    pdb_id = clean_id(pdb_id)
    matches = list(PRDB_DIR.rglob(f"{pdb_id}.pdb"))
    return matches[0] if matches else None


# ---------------------------------------------------
# Step 1: get unbound complex IDs
# ---------------------------------------------------

unbound_complex_ids = {
    p.stem.upper() for p in UNBOUND_COMPLEX_DIR.glob("*.pdb")
}

print(f"Found {len(unbound_complex_ids)} unbound complexes")

# ---------------------------------------------------
# Step 2: process dataset
# ---------------------------------------------------

for obj in data:

    complex_id = obj.get("C_PDB")
    if not complex_id:
        continue

    complex_id = complex_id.upper()

    if complex_id not in unbound_complex_ids:
        continue

    pro_id = obj.get("U_pro_PDB")
    rna_id = obj.get("U_RNA_PDB")

    # create complex folder
    complex_dir = OUT_ROOT / complex_id
    complex_dir.mkdir(exist_ok=True)

    # -----------------------
    # copy complex pdb
    # -----------------------
    complex_src = find_pdb(complex_id)
    if complex_src:
        shutil.copy(complex_src, complex_dir / f"{complex_id}.pdb")
        print(f"Copied complex {complex_id}")

    # -----------------------
    # copy protein pdb
    # -----------------------
    if pro_id:
        pro_id = clean_id(pro_id)
        pro_src = find_pdb(pro_id)
        if pro_src:
            shutil.copy(pro_src, complex_dir / f"{pro_id}.pdb")
            print(f"  protein {pro_id}")

    # -----------------------
    # copy RNA pdb
    # -----------------------
    if rna_id:
        rna_id = clean_id(rna_id)
        rna_src = find_pdb(rna_id)
        if rna_src:
            shutil.copy(rna_src, complex_dir / f"{rna_id}.pdb")
            print(f"  rna {rna_id}")

print("Finished.")