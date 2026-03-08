# Run from current directory (Sorry I am lazy)

import os
import requests
from pathlib import Path

# Folder containing FASTA files
FASTA_DIR = "../assets/fasta-sequences"

# Where PDB files will be saved
OUT_DIR = "../pdb_files"
os.makedirs(OUT_DIR, exist_ok=True)

for fasta_file in Path(FASTA_DIR).glob("*.fasta"):
    
    # Get PDB ID from filename
    pdb_id = fasta_file.stem.upper()
    
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    out_path = Path(OUT_DIR) / f"{pdb_id}.pdb"

    print(f"Downloading {pdb_id}...")

    try:
        r = requests.get(url)
        if r.status_code == 200:
            with open(out_path, "wb") as f:
                f.write(r.content)
            print(f"Saved -> {out_path}")
        else:
            print(f"Failed: {pdb_id}")
    except Exception as e:
        print(f"Error downloading {pdb_id}: {e}")