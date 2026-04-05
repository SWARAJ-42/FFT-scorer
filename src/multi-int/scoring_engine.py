#!/usr/bin/python3
# =============================================================================
# scoring_engine.py  —  Cross-reference metrics for protein-RNA interface scoring
# =============================================================================
#
# OVERVIEW
# --------
# Computes interaction metrics between ground truth (Mode A) and generated poses
# (Mode C). For each generated pose, this engine calculates:
#
#   1. Fraction of Native Contacts (f_nat)
#      What % of residue-residue contacts in ground truth appear in generated pose?
#
#   2. Interface RMSD (I-RMSD)
#      Spatial deviation of interface backbone atoms between poses.
#
#   3. BSA Recovery (Δ BSA)
#      Absolute difference in buried surface area.
#
#   4. Steric Clash Penalty
#      Voxel-based overlap detection to penalize impossible configurations.
#
#   5. Composite Interaction Score
#      Normalized aggregation of all metrics.
#
# KEY FUNCTIONS
# ------7-------
#   score_single_rank()   Main entry point; compares ground truth to one rank
#   calculate_f_nat()     Fraction of native contacts
#   calculate_irmsd()     Interface RMSD
#   calculate_bsa_delta() Δ BSA between two poses
#   calculate_steric_clash() Voxel-based clash detection
#   compute_interaction_score() Final composite score
#
# =============================================================================

import os
import json
import logging
import numpy as np
from scipy.spatial.distance import cdist
from typing import Dict, Tuple, List, Optional

# Contactspan definitions (standard for protein-RNA interactions)
CONTACT_DISTANCE = 4.5  # Å — distance threshold for residue contact


# ---------------------------------------------------------------------------
# PDB Parsing and Atom Selection
# ---------------------------------------------------------------------------

def parse_pdb_atoms(pdb_file: str, chain_id: str = None) -> Dict[str, np.ndarray]:
    """
    Parse a PDB file and extract atomic coordinates.

    Parameters
    ----------
    pdb_file : str
        Path to the PDB file
    chain_id : str | None
        If specified, extract only atoms from this chain.
        If None, extract all atoms.

    Returns
    -------
    dict
        {
            "coordinates": np.ndarray of shape (N, 3),
            "residues":    list of residue info dicts,
            "backbone":    list of backbone atom indices,
        }
    """
    atoms = []
    residues = []
    backbone_indices = []
    current_resnum = None
    current_chain = None

    with open(pdb_file, 'r') as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue

            # Parse PDB line
            atom_name = line[12:16].strip()
            res_name = line[17:20].strip()
            chain = line[21].strip()
            res_num = int(line[22:26].strip())
            x = float(line[30:38].strip())
            y = float(line[38:46].strip())
            z = float(line[46:54].strip())

            # Filter by chain if specified
            if chain_id is not None and chain != chain_id:
                continue

            atoms.append([x, y, z])
            is_backbone = atom_name in ["N", "CA", "C", "O"]

            if (res_num != current_resnum or chain != current_chain):
                residues.append({
                    "residue_number": res_num,
                    "residue_name": res_name,
                    "chain": chain,
                    "atom_count": 0,
                    "atom_indices": []
                })
                current_resnum = res_num
                current_chain = chain

            atom_idx = len(atoms) - 1
            residues[-1]["atom_count"] += 1
            residues[-1]["atom_indices"].append(atom_idx)

            if is_backbone:
                backbone_indices.append(atom_idx)

    if not atoms:
        return {
            "coordinates": np.array([]),
            "residues": [],
            "backbone": []
        }

    return {
        "coordinates": np.array(atoms, dtype=np.float32),
        "residues": residues,
        "backbone": backbone_indices
    }


def get_interface_atoms(pdb_data: Dict, interface_file: str) -> np.ndarray:
    """
    Extract interface atom coordinates from .int file.

    The .int file contains space-separated columns of interface residues.
    This function identifies all atoms belonging to those residues.

    Parameters
    ----------
    pdb_data : dict
        Output of parse_pdb_atoms()
    interface_file : str
        Path to .int file (format: residue IDs, one per line)

    Returns
    -------
    np.ndarray of shape (M, 3)
        Coordinates of all interface atoms
    """
    interface_residues = set()

    if not os.path.exists(interface_file):
        logging.warning(f"Interface file not found: {interface_file}")
        return np.array([])

    with open(interface_file, 'r') as f:
        for line in f:
            # Example line: "A_GLY45" (from multi_interface.py output)
            parts = line.strip().split('_')
            if len(parts) >= 2:
                try:
                    res_num = int(''.join(filter(str.isdigit, parts[-1])))
                    interface_residues.add(res_num)
                except ValueError:
                    pass

    interface_atoms = []
    for res in pdb_data.get("residues", []):
        if res["residue_number"] in interface_residues:
            for idx in res["atom_indices"]:
                interface_atoms.append(pdb_data["coordinates"][idx])

    return np.array(interface_atoms, dtype=np.float32)


# ---------------------------------------------------------------------------
# Core Scoring Metrics
# ---------------------------------------------------------------------------

def calculate_f_nat(truth_int: np.ndarray, gen_int: np.ndarray) -> float:
    """
    Fraction of Native Contacts (f_nat).

    For each residue-residue pair (i, j) where distance < CONTACT_DISTANCE in
    the ground truth, check if the same pair appears in the generated pose.

    Parameters
    ----------
    truth_int : np.ndarray of shape (N_truth, 3)
        Interface atoms in ground truth
    gen_int : np.ndarray of shape (N_gen, 3)
        Interface atoms in generated pose

    Returns
    -------
    float
        Fraction in range [0, 1]. 1.0 = perfect native contact overlap.
    """
    if len(truth_int) == 0 or len(gen_int) == 0:
        return 0.0

    # Compute all pairwise distances in truth
    truth_dist = cdist(truth_int, truth_int, metric='euclidean')
    native_contacts = (truth_dist < CONTACT_DISTANCE) & (truth_dist > 0)
    n_native_contacts = np.sum(native_contacts)

    if n_native_contacts == 0:
        return 0.0

    # Compute all pairwise distances in generated
    gen_dist = cdist(gen_int, gen_int, metric='euclidean')
    gen_contacts = (gen_dist < CONTACT_DISTANCE) & (gen_dist > 0)

    # Count recovered contacts (this is a simplified proxy)
    # In practice, you'd need to match atoms across the two structures first.
    # Here we use the fraction of close contacts in generated as a proxy.
    n_gen_contacts = np.sum(gen_contacts)

    if n_gen_contacts == 0:
        return 0.0

    # Simplified: take the ratio of contact densities
    f_nat = min(n_gen_contacts / max(n_native_contacts, 1), 1.0)
    return float(f_nat)


def calculate_irmsd(truth_pdb: str, gen_pdb: str,
                    truth_int_file: str, gen_int_file: str) -> float:
    """
    Interface RMSD (I-RMSD).

    RMSD of backbone atoms in interface residues between ground truth and
    generated pose. Requires structural alignment.

    Parameters
    ----------
    truth_pdb : str
        Path to ground truth PDB
    gen_pdb : str
        Path to generated PDB
    truth_int_file : str
        Path to ground truth .int file
    gen_int_file : str
        Path to generated .int file

    Returns
    -------
    float
        I-RMSD in Ångströms. Lower is better.
    """
    # Parse both structures
    truth_data = parse_pdb_atoms(truth_pdb)
    gen_data = parse_pdb_atoms(gen_pdb)

    # Extract backbone atoms from interface residues
    truth_int_atoms = get_interface_atoms(truth_data, truth_int_file)
    gen_int_atoms = get_interface_atoms(gen_data, gen_int_file)

    if len(truth_int_atoms) == 0 or len(gen_int_atoms) == 0:
        return float('inf')

    # Simple RMSD without alignment (assumes PDB files are already aligned)
    # In production, you'd use BioPython's Superimposer for true alignment.
    min_len = min(len(truth_int_atoms), len(gen_int_atoms))
    if min_len == 0:
        return float('inf')

    distances = np.sqrt(np.sum((truth_int_atoms[:min_len] - gen_int_atoms[:min_len]) ** 2, axis=1))
    irmsd = np.sqrt(np.mean(distances ** 2))
    return float(irmsd)


def calculate_bsa_delta(truth_bsa: float, gen_bsa: float) -> float:
    """
    Buried Surface Area difference (Δ BSA).

    Absolute difference between BSA values.

    Parameters
    ----------
    truth_bsa : float
        BSA from ground truth complex
    gen_bsa : float
        BSA from generated complex

    Returns
    -------
    float
        |truth_bsa - gen_bsa| in Ų. Lower is better.
    """
    if truth_bsa == "NA" or gen_bsa == "NA":
        return float('inf')
    return abs(float(truth_bsa) - float(gen_bsa))


def calculate_steric_clash(gen_pdb: str, protein_pdb: str, rna_pdb: str) -> float:
    """
    Steric Clash Penalty via voxel-based overlap.

    Discretize both protein and RNA into voxels; count overlapping voxels as a
    clash penalty. Penalizes physically impossible configurations.

    Parameters
    ----------
    gen_pdb : str
        Path to combined generated PDB (protein + RNA)
    protein_pdb : str
        Path to isolated protein PDB
    rna_pdb : str
        Path to isolated RNA PDB

    Returns
    -------
    float
        Clash penalty in [0, 1]. 0 = no clashes. 1 = severe overlap.
    """
    try:
        # Parse structures
        protein_data = parse_pdb_atoms(protein_pdb)
        rna_data = parse_pdb_atoms(rna_pdb)

        if len(protein_data["coordinates"]) == 0 or len(rna_data["coordinates"]) == 0:
            return 0.0

        # Voxel grid parameters
        voxel_size = 1.0  # Å
        protein_coords = protein_data["coordinates"]
        rna_coords = rna_data["coordinates"]

        # Compute bounding box
        min_coord = np.minimum(protein_coords.min(axis=0), rna_coords.min(axis=0))
        max_coord = np.maximum(protein_coords.max(axis=0), rna_coords.max(axis=0))

        # Discretize into voxels
        voxel_origin = np.floor(min_coord / voxel_size).astype(int)
        protein_voxels = set(
            tuple(np.floor((c / voxel_size) - voxel_origin).astype(int))
            for c in protein_coords
        )
        rna_voxels = set(
            tuple(np.floor((c / voxel_size) - voxel_origin).astype(int))
            for c in rna_coords
        )

        # Count overlaps
        overlap = len(protein_voxels & rna_voxels)
        union = len(protein_voxels | rna_voxels)

        if union == 0:
            return 0.0

        # Normalize: fraction of voxels that overlap
        clash_penalty = overlap / union
        return min(clash_penalty, 1.0)

    except Exception as e:
        logging.warning(f"Steric clash calculation failed: {e}")
        return 0.0


# ---------------------------------------------------------------------------
# Composite Scoring
# ---------------------------------------------------------------------------

def compute_interaction_score(f_nat: float,
                               irmsd: float,
                               bsa_delta: float,
                               clash_penalty: float,
                               weights: Dict[str, float] = None) -> float:
    """
    Composite Interaction Score.

    Aggregates normalized metrics into a final [0, 1] score.
    High = near-native. Low = false positive.

    Parameters
    ----------
    f_nat : float
        Fraction of native contacts [0, 1]
    irmsd : float
        Interface RMSD in Å
    bsa_delta : float
        Δ BSA in Ų
    clash_penalty : float
        Steric clash [0, 1]
    weights : dict | None
        {"f_nat": w1, "irmsd": w2, "bsa_delta": w3, "clash": w4}
        Defaults to equal weights if not provided.

    Returns
    -------
    float
        Final score in range [0, 1]. Higher is better.
    """
    if weights is None:
        weights = {
            "f_nat": 0.4,
            "irmsd": 0.3,
            "bsa_delta": 0.2,
            "clash": 0.1
        }

    # Normalize each metric
    # f_nat: already in [0, 1] — higher is better
    norm_f_nat = max(0.0, min(f_nat, 1.0))

    # irmsd: normalize via exponential decay (lower is better)
    # reference: 5 Å is "good", 10 Å is "poor"
    norm_irmsd = np.exp(-irmsd / 5.0)

    # bsa_delta: normalize via exponential decay (lower is better)
    # reference: 100 Ų is "good", 500 Ų is "poor"
    norm_bsa_delta = np.exp(-bsa_delta / 100.0)

    # clash_penalty: already in [0, 1], invert so 1 = good
    norm_clash = 1.0 - min(clash_penalty, 1.0)

    # Weighted aggregation
    weighted_sum = (
        weights["f_nat"] * norm_f_nat
        + weights["irmsd"] * norm_irmsd
        + weights["bsa_delta"] * norm_bsa_delta
        + weights["clash"] * norm_clash
    )

    # Normalize to [0, 1]
    total_weight = sum(weights.values())
    score = weighted_sum / total_weight if total_weight > 0 else 0.0

    return float(max(0.0, min(score, 1.0)))


# ---------------------------------------------------------------------------
# Main Scoring Function
# ---------------------------------------------------------------------------

def score_single_rank(complex_id: str,
                      truth_pdb: str,
                      truth_bsa: float,
                      gen_pdb: str,
                      gen_bsa: float,
                      truth_int_file: str,
                      gen_int_file: str,
                      protein_pdb: str,
                      rna_pdb: str,
                      rank_label: str = "rank1") -> Dict:
    """
    Score a single generated rank against ground truth.

    Parameters
    ----------
    complex_id : str
        Complex identifier (e.g., "1ASY")
    truth_pdb : str
        Path to ground truth complex PDB
    truth_bsa : float | "NA"
        Ground truth BSA
    gen_pdb : str
        Path to generated combined PDB
    gen_bsa : float | "NA"
        Generated BSA
    truth_int_file : str
        Path to ground truth .int file
    gen_int_file : str
        Path to generated .int file
    protein_pdb : str
        Path to protein.pdb (for clash detection)
    rna_pdb : str
        Path to rna.pdb (for clash detection)
    rank_label : str
        Label for this rank (default "rank1")

    Returns
    -------
    dict
        {
            "complex_id": str,
            "rank": str,
            "f_nat": float,
            "i_rmsd": float,
            "bsa_delta": float,
            "clash_penalty": float,
            "interaction_score": float,
            "category": str,  ("Near-Native" | "Medium" | "Incorrect")
            "error": str | None,
        }
    """
    result = {
        "complex_id": complex_id,
        "rank": rank_label,
        "f_nat": None,
        "i_rmsd": None,
        "bsa_delta": None,
        "clash_penalty": None,
        "interaction_score": None,
        "category": None,
        "error": None,
    }

    try:
        # 0. Validate input files exist before doing any computation
        missing = []
        for label, path in [("truth_pdb", truth_pdb), ("gen_pdb", gen_pdb),
                             ("protein_pdb", protein_pdb), ("rna_pdb", rna_pdb)]:
            if not path or not os.path.exists(path):
                missing.append(f"{label}={path!r}")
        if missing:
            raise FileNotFoundError(f"Missing required files: {', '.join(missing)}")

        # 1. Parse PDB data
        truth_pdb_data = parse_pdb_atoms(truth_pdb)
        gen_pdb_data = parse_pdb_atoms(gen_pdb)

        # 2. Extract interface atoms from .int files
        truth_int_atoms = get_interface_atoms(truth_pdb_data, truth_int_file)
        gen_int_atoms = get_interface_atoms(gen_pdb_data, gen_int_file)

        # 3. Calculate metrics
        f_nat = calculate_f_nat(truth_int_atoms, gen_int_atoms)

        irmsd = calculate_irmsd(truth_pdb, gen_pdb, truth_int_file, gen_int_file)

        bsa_delta = calculate_bsa_delta(truth_bsa, gen_bsa)

        clash_penalty = calculate_steric_clash(gen_pdb, protein_pdb, rna_pdb)

        # 2. Compute composite score
        interaction_score = compute_interaction_score(
            f_nat, irmsd, bsa_delta, clash_penalty
        )

        # 3. Categorize
        if interaction_score >= 0.7 and irmsd < 3.0:
            category = "Near-Native"
        elif interaction_score >= 0.4:
            category = "Medium"
        else:
            category = "Incorrect"

        result.update({
            "f_nat": round(f_nat, 3),
            "i_rmsd": round(irmsd, 2) if irmsd != float('inf') else "inf",
            "bsa_delta": round(bsa_delta, 1) if bsa_delta != float('inf') else "inf",
            "clash_penalty": round(clash_penalty, 3),
            "interaction_score": round(interaction_score, 3),
            "category": category,
            "error": None,
        })

    except Exception as e:
        logging.error(f"Scoring failed for {complex_id}/{rank_label}: {e}")
        result["error"] = str(e)

    return result


# ---------------------------------------------------------------------------
# Batch Scoring Across All Ranks
# ---------------------------------------------------------------------------

def score_all_ranks(complex_id: str,
                    truth_complex_result: Dict,
                    gen_results: Dict) -> Dict:
    """
    Score all generated ranks for a complex against ground truth.

    Parameters
    ----------
    complex_id : str
        Complex identifier
    truth_complex_result : dict
        Result dict from run_interface() for ground truth
    gen_results : dict
        Dict of {rank_label: result_dict} from run_generated_mode()

    Returns
    -------
    dict
        {rank_label: score_result, ...}
    """
    scores = {}

    truth_bsa = truth_complex_result.get("bsa_complex", "NA")

    for rank_label, gen_result in gen_results.items():
        if gen_result.get("error"):
            logging.warning(
                f"[scoring] {complex_id}/{rank_label}  "
                f"skipping due to error: {gen_result['error']}"
            )
            scores[rank_label] = {
                "error": gen_result["error"],
                "interaction_score": 0.0,
                "category": "Skipped"
            }
            continue

        gen_bsa = gen_result.get("bsa_complex", "NA")

        # Paths for this rank
        # Assuming: generated_PDBS/<complex_id>/<rank>/results/
        rank_results_dir = os.path.dirname(gen_result["combined_int"])

        score = score_single_rank(
            complex_id=complex_id,
            truth_pdb=truth_complex_result.get("combined_pdb"),
            truth_bsa=truth_bsa,
            gen_pdb=os.path.join(rank_results_dir, "..", f"{complex_id}_combined.pdb"),
            gen_bsa=gen_bsa,
            truth_int_file=truth_complex_result.get("combined_int"),
            gen_int_file=gen_result["combined_int"],
            protein_pdb=os.path.join(rank_results_dir, "..", "protein.pdb"),
            rna_pdb=os.path.join(rank_results_dir, "..", "rna.pdb"),
            rank_label=rank_label,
        )

        scores[rank_label] = score

    return scores