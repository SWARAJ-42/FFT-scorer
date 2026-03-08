import json
import shutil
from pathlib import Path

# -------- CONFIG --------
JSON_PATH = "../assets/PRDBv3.json"       # your JSON file
PDB_SOURCE_DIR = "../pdb_files"   # where all pdb files are currently
OUTPUT_DIR = "../uu_pdbs"         # filtered pdb files
# ------------------------

Path(OUTPUT_DIR).mkdir(exist_ok=True)

# Load JSON
with open(JSON_PATH, "r") as f:
    data = json.load(f)

selected_pdbs = []

# Find UU docking cases
for obj in data:
    pdb_id = obj.get("C_PDB")
    docking_case = obj.get("Docking_case")

    if docking_case == "UU":
        selected_pdbs.append(pdb_id)

print(f"Found {len(selected_pdbs)} UU cases")

# Copy corresponding PDB files
for pdb_id in selected_pdbs:
    src = Path(PDB_SOURCE_DIR) / f"{pdb_id}.pdb"
    dst = Path(OUTPUT_DIR) / f"{pdb_id}.pdb"

    if src.exists():
        shutil.copy(src, dst)
        print(f"Copied {pdb_id}.pdb")
    else:
        print(f"Missing file: {pdb_id}.pdb")

print("Done.")