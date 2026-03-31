#!/usr/bin/python3
# =============================================================================
# run_pipeline.py  —  FFT-scorer interface pipeline (freeSASA version)
# =============================================================================

import os
import sys
import json
import logging
import argparse
import traceback
from datetime import datetime

# ---------------------------------------------------------------------------
# Locate modules
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import multi_interface as mi
import compare_results as cr


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "pipeline.log")
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.FileHandler(log_path, mode='w'),
            logging.StreamHandler(sys.stdout),
        ]
    )


# ---------------------------------------------------------------------------
# freeSASA check (NEW)
# ---------------------------------------------------------------------------

def check_freesasa_available() -> bool:
    try:
        import freesasa
        logging.info("freeSASA found — OK")
        return True
    except ImportError:
        logging.error(
            "freeSASA NOT installed.\n"
            "Run: pip install freesasa"
        )
        return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def strip_star(pdb_id: str) -> str:
    return pdb_id.rstrip('*') if isinstance(pdb_id, str) else pdb_id


def expand_chains(chain_str: str) -> list:
    if not chain_str:
        return []
    return list(str(chain_str))


def pdb_folder(truth_dir: str, complex_id: str) -> str:
    return os.path.join(truth_dir, complex_id)


def rank_folders(gen_dir: str, complex_id: str) -> list:
    base = os.path.join(gen_dir, complex_id)
    if not os.path.isdir(base):
        return []
    folders = sorted(
        [
            os.path.join(base, d)
            for d in os.listdir(base)
            if os.path.isdir(os.path.join(base, d))
            and d.lower().startswith("rank")
        ],
        key=lambda p: int(''.join(filter(str.isdigit, os.path.basename(p))) or '0')
    )
    return folders


def skipped_result(reason: str) -> dict:
    return {
        "skipped": True, "reason": reason,
        "bsa_complex": "NA", "bsa_pro": "NA", "bsa_rna": "NA",
        "pro_int": "NA", "rna_int": "NA", "combined_int": "NA",
        "has_interface": False, "error": None,
    }


def safe_run(label: str, fn, *args, **kwargs) -> dict:
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        msg = f"EXCEPTION in {label}: {exc}\n{traceback.format_exc()}"
        logging.error(msg)
        return {
            "bsa_complex": "NA", "bsa_pro": "NA", "bsa_rna": "NA",
            "pro_int": "NA", "rna_int": "NA", "combined_int": "NA",
            "has_interface": False, "error": msg,
        }


# ---------------------------------------------------------------------------
# Mode A — complex
# ---------------------------------------------------------------------------

def run_complex_mode(entry: dict, truth_dir: str) -> dict:
    c_pdb = entry["C_PDB"]
    input_dir = pdb_folder(truth_dir, c_pdb)
    results_dir = os.path.join(input_dir, "complex_results")
    source_pdb = os.path.join(input_dir, c_pdb + ".pdb")

    if not os.path.isfile(source_pdb):
        return {"_error": skipped_result(f"Missing {source_pdb}")}

    pro_chains = expand_chains(entry.get("C_pro_chain", ""))
    rna_chains = expand_chains(entry.get("C_RNA_chain", ""))

    pair_results = {}

    for pro_ch in pro_chains:
        for rna_ch in rna_chains:
            key = f"{pro_ch}{rna_ch}"
            result = safe_run(
                f"complex/{c_pdb}/{key}",
                mi.run_interface,
                pdb_file=c_pdb,
                first_chain=pro_ch,
                second_chain=rna_ch,
                run_mode="complex",
                input_dir=input_dir,
                results_dir=results_dir,
            )
            pair_results[key] = result

    return pair_results


# ---------------------------------------------------------------------------
# Mode B — unbound
# ---------------------------------------------------------------------------

def run_unbound_mode(entry: dict, truth_dir: str) -> dict:
    c_pdb = entry["C_PDB"]

    if entry.get("Docking_case") != "UU":
        return {"_skipped": skipped_result("Not UU")}

    u_pro = strip_star(entry.get("U_pro_PDB"))
    u_rna = strip_star(entry.get("U_RNA_PDB"))

    folder = pdb_folder(truth_dir, c_pdb)

    pro_path = os.path.join(folder, u_pro + ".pdb")
    rna_path = os.path.join(folder, u_rna + ".pdb")

    key = f"{entry.get('U_PRO_chain')}{entry.get('U_RNA_chain')}"

    result = safe_run(
        f"unbound/{c_pdb}",
        mi.run_interface,
        pdb_file=c_pdb,
        first_chain=entry.get("U_PRO_chain"),
        second_chain=entry.get("U_RNA_chain"),
        run_mode="unbound",
        input_dir=folder,
        results_dir=os.path.join(folder, "unbound_results"),
        pre_split={"protein": pro_path, "rna": rna_path},
    )

    return {key: result}


# ---------------------------------------------------------------------------
# Mode C — generated
# ---------------------------------------------------------------------------

def run_generated_mode(entry: dict, gen_dir: str) -> dict:
    c_pdb = entry["C_PDB"]
    ranks = rank_folders(gen_dir, c_pdb)

    results = {}

    for rank_path in ranks:
        label = os.path.basename(rank_path)

        result = safe_run(
            f"generated/{c_pdb}/{label}",
            mi.run_interface,
            pdb_file=c_pdb,
            first_chain="A",
            second_chain="R",
            run_mode="generated",
            input_dir=rank_path,
            pre_split={
                "protein": os.path.join(rank_path, "protein.pdb"),
                "rna": os.path.join(rank_path, "rna.pdb"),
            },
        )

        results[label] = result

    return results


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def run_pipeline(json_path, truth_dir, gen_dir, out_dir):

    setup_logging(out_dir)
    start = datetime.now()

    logging.info("Starting pipeline")

    # ✅ NEW
    check_freesasa_available()

    with open(json_path, 'r') as f:
        entries = json.load(f)

    all_results = {}

    for entry in entries:
        cid = entry["C_PDB"]

        all_results[cid] = {
            "complex": run_complex_mode(entry, truth_dir),
            "unbound": run_unbound_mode(entry, truth_dir),
            "generated": run_generated_mode(entry, gen_dir),
        }

    out_file = os.path.join(out_dir, "pipeline_results.json")

    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)

    cr.run_comparison(all_results, out_dir)

    logging.info(f"Done in {datetime.now() - start}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--json", required=True)
    parser.add_argument("--truth_dir", required=True)
    parser.add_argument("--gen_dir", required=True)
    parser.add_argument("--out_dir", required=True)

    args = parser.parse_args()

    run_pipeline(
        args.json,
        args.truth_dir,
        args.gen_dir,
        args.out_dir
    )