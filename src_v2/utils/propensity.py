"""
propensity.py — Interface Propensity Rescoring
===============================================
Builds a log-odds propensity table for amino acid–nucleotide contacts from
the PRDBv3 bound complex structures, then uses it to re-rank top FFT poses.

Theory
------
The log-odds propensity measures how much more (or less) often a given
(amino acid, nucleotide) pair appears at protein-RNA interfaces compared
to what would be expected by chance:

    P(aa, nt) = log₂ [ (count_obs(aa,nt) + ψ) / (count_exp(aa,nt) + ψ) ]

where ψ is a Laplace pseudocount and:

    count_exp(aa, nt) = freq(aa) × freq(nt) × total_contacts

A positive propensity means the pair is enriched at interfaces (e.g. Arg–G
due to the arginine fork motif); negative means depleted.

Rescoring
---------
For each candidate docked pose:
    1. Apply the docking transform (R, t) to RNA residue centroids.
    2. Find all (amino acid, nucleotide) pairs within CONTACT_CUTOFF Å.
    3. propensity_score = Σ P(aa_i, nt_j)  for all interface pairs
    4. combined_score   = α × shape_score + β × propensity_score

Integration
-----------
• phase5.py — rescore_poses() is called at the start of run_phase5()
              before PDB writing and RMSD benchmarking.
• run.py    — PropensityTable is built from all loaded cases when
              --propensity is passed, then forwarded to run_phase5().
"""

import json
import math
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from phase1 import DockingCase, PROTEIN_RESIDUES, RNA_RESIDUES, Structure
from phase4 import DockingResult


# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

CONTACT_CUTOFF = 5.0      # Å  — centroid-centroid distance for a contact
PSEUDOCOUNT    = 0.5      # Laplace smoothing — prevents log(0) on rare pairs
DEFAULT_SCORE  = -0.10    # score returned for (aa, nt) pairs never observed

# Canonical RNA nucleotide names accepted in the propensity table
CANONICAL_NTS = ("A", "U", "G", "C")

# Aliases → canonical form
NT_ALIAS: Dict[str, str] = {
    "RA": "A", "RU": "U", "RG": "G", "RC": "C",
    "ADE": "A", "URA": "U", "GUA": "G", "CYT": "C",
}

# Standard amino acid three-letter codes accepted in the table
AA_LIST = sorted(PROTEIN_RESIDUES)   # 21 entries including MSE


def canonical_nt(name: str) -> Optional[str]:
    """Map any RNA residue name to its canonical single-letter code, or None."""
    if name in CANONICAL_NTS:
        return name
    return NT_ALIAS.get(name)


# ═══════════════════════════════════════════════════════════════════════════
# Contact extraction from bound complex structures
# ═══════════════════════════════════════════════════════════════════════════

def _residue_centroids(
    structure: Structure,
    mol_type:  str,
) -> List[Tuple[str, np.ndarray]]:
    """
    Compute per-residue centroids (mean of heavy-atom coordinates).

    Returns
    -------
    list of (residue_name, centroid_xyz)  — one entry per residue
    """
    chains = (structure.protein_chains() if mol_type == "protein"
              else structure.rna_chains())

    by_res: Dict[Tuple, List] = defaultdict(list)
    for chain in chains:
        for atom in chain.atoms:
            key = (chain.chain_id, atom.res_seq, atom.res_name)
            by_res[key].append([atom.x, atom.y, atom.z])

    out = []
    for (_, _, res_name), coords in by_res.items():
        centroid = np.mean(coords, axis=0)
        out.append((res_name, centroid))
    return out


def extract_contacts(
    complex_struct: Structure,
    cutoff:         float = CONTACT_CUTOFF,
) -> List[Tuple[str, str]]:
    """
    Find all (amino_acid, nucleotide) contact pairs in a bound complex.

    Two residues are 'in contact' when their heavy-atom centroids are
    within `cutoff` Ångström of each other.

    Parameters
    ----------
    complex_struct : parsed bound complex Structure (from phase1)
    cutoff         : centroid-centroid distance threshold in Å

    Returns
    -------
    list of (aa_name, canonical_nt_name) tuples
    """
    pro_residues = _residue_centroids(complex_struct, "protein")
    rna_residues = _residue_centroids(complex_struct, "rna")

    if not pro_residues or not rna_residues:
        return []

    pro_names     = [r[0] for r in pro_residues]
    pro_centroids = np.array([r[1] for r in pro_residues])   # (Np, 3)
    rna_names_raw = [r[0] for r in rna_residues]
    rna_centroids = np.array([r[1] for r in rna_residues])   # (Nr, 3)

    contacts = []
    for aa_name, aa_xyz in zip(pro_names, pro_centroids):
        if aa_name not in PROTEIN_RESIDUES:
            continue
        dists = np.linalg.norm(rna_centroids - aa_xyz, axis=1)   # (Nr,)
        for j in np.where(dists <= cutoff)[0]:
            nt = canonical_nt(rna_names_raw[j])
            if nt is not None:
                contacts.append((aa_name, nt))

    return contacts


# ═══════════════════════════════════════════════════════════════════════════
# PropensityTable — build and query
# ═══════════════════════════════════════════════════════════════════════════

class PropensityTable:
    """
    Amino acid–nucleotide contact propensity table derived from PRDBv3.

    Workflow
    --------
    table = PropensityTable()
    table.fit(cases)          # uses case.complex_struct for all cases
    table.print_table()       # inspect propensities

    score = table.score_contacts([('LYS','A'), ('ARG','G'), ...])

    Persistence
    -----------
    table.save("propensity.json")
    table.load("propensity.json")
    """

    def __init__(self, pseudocount: float = PSEUDOCOUNT):
        self.pseudocount = pseudocount
        self._log_odds:  Dict[Tuple[str, str], float] = {}
        self._n_cases:   int = 0
        self._n_contacts: int = 0
        self._is_fit:    bool = False

    # ── Fitting ──────────────────────────────────────────────────────────

    def fit(self, cases: List[DockingCase]) -> "PropensityTable":
        """
        Build the propensity table from all bound complex structures.

        Only case.complex_struct is used — the unbound structures are
        not touched here (they are used later during rescoring).
        """
        count_obs: Dict[Tuple[str, str], int] = defaultdict(int)
        count_aa:  Dict[str, int]             = defaultdict(int)
        count_nt:  Dict[str, int]             = defaultdict(int)
        total = 0

        valid_cases = 0
        for case in cases:
            if case.complex_struct is None:
                continue
            contacts = extract_contacts(case.complex_struct)
            valid_cases += 1
            for aa, nt in contacts:
                count_obs[(aa, nt)] += 1
                count_aa[aa]        += 1
                count_nt[nt]        += 1
                total               += 1

        self._n_cases    = valid_cases
        self._n_contacts = total

        if total == 0:
            print("[PropensityTable] Warning: zero contacts extracted — "
                  "all queries will return DEFAULT_SCORE.")
            self._is_fit = True
            return self

        # Marginal frequencies (with pseudocount)
        N_aa = len(AA_LIST)
        N_nt = len(CANONICAL_NTS)
        psi  = self.pseudocount

        freq_aa = {
            aa: (count_aa[aa] + psi) / (total + N_aa * psi)
            for aa in AA_LIST
        }
        freq_nt = {
            nt: (count_nt[nt] + psi) / (total + N_nt * psi)
            for nt in CANONICAL_NTS
        }

        # Log-odds for every (aa, nt) combination
        for aa in AA_LIST:
            for nt in CANONICAL_NTS:
                obs      = count_obs[(aa, nt)] + psi
                expected = freq_aa[aa] * freq_nt[nt] * (total + psi)
                self._log_odds[(aa, nt)] = math.log2(obs / expected)

        self._is_fit = True
        print(
            f"[PropensityTable] Fitted on {valid_cases} cases  "
            f"— {total:,} contacts  "
            f"— {len(self._log_odds)} (aa, nt) pairs"
        )
        return self

    # ── Querying ─────────────────────────────────────────────────────────

    def get(self, aa: str, nt: str) -> float:
        """
        Log-odds score for a single (amino acid, nucleotide) pair.
        Returns DEFAULT_SCORE for unrecognised residue names.
        """
        if not self._is_fit:
            raise RuntimeError("PropensityTable.fit() must be called first.")
        return self._log_odds.get((aa, canonical_nt(nt) or nt), DEFAULT_SCORE)

    def score_contacts(self, contacts: List[Tuple[str, str]]) -> float:
        """
        Sum log-odds scores over a list of (aa, nt) contact pairs.
        An empty contact list returns 0.0.
        """
        return sum(self.get(aa, nt) for aa, nt in contacts)

    # ── Serialisation ────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Persist the propensity table to a JSON file."""
        data = {
            "__meta__": {
                "n_cases": self._n_cases,
                "n_contacts": self._n_contacts,
                "pseudocount": self.pseudocount,
            },
            "log_odds": {
                f"{aa}_{nt}": v
                for (aa, nt), v in self._log_odds.items()
            },
        }
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(data, fh, indent=2)
        print(f"[PropensityTable] Saved to {path}")

    def load(self, path: str) -> "PropensityTable":
        """Load a previously saved propensity table from JSON."""
        with open(path) as fh:
            data = json.load(fh)

        meta = data.get("__meta__", {})
        self._n_cases    = meta.get("n_cases",    0)
        self._n_contacts = meta.get("n_contacts", 0)
        self.pseudocount = meta.get("pseudocount", self.pseudocount)

        self._log_odds = {
            tuple(k.split("_", 1)): v    # "LYS_A" → ("LYS", "A")
            for k, v in data["log_odds"].items()
        }
        self._is_fit = True
        print(
            f"[PropensityTable] Loaded from {path}  "
            f"— {len(self._log_odds)} pairs  "
            f"({self._n_cases} training cases)"
        )
        return self

    # ── Display ──────────────────────────────────────────────────────────

    def print_table(self) -> None:
        """Pretty-print the (aa × nt) propensity matrix to stdout."""
        nts    = list(CANONICAL_NTS)
        col_w  = 9
        header = f"{'AA':>5}" + "".join(f"{nt:>{col_w}}" for nt in nts)
        sep    = "  " + "─" * (5 + col_w * len(nts))

        print("\n  Amino Acid – Nucleotide Contact Log-Odds Propensity Table")
        print(f"  (fitted on {self._n_cases} bound complexes, "
              f"{self._n_contacts:,} contacts)")
        print(sep)
        print("  " + header)
        print(sep)
        for aa in sorted(AA_LIST):
            row = f"{aa:>5}" + "".join(
                f"{self.get(aa, nt):>{col_w}.3f}" for nt in nts
            )
            print("  " + row)
        print(sep)

    def top_pairs(self, n: int = 10) -> List[Tuple[float, str, str]]:
        """Return the n most enriched (aa, nt) pairs by log-odds score."""
        pairs = [
            (score, aa, nt)
            for (aa, nt), score in self._log_odds.items()
        ]
        pairs.sort(reverse=True)
        return pairs[:n]


# ═══════════════════════════════════════════════════════════════════════════
# Pose rescoring
# ═══════════════════════════════════════════════════════════════════════════

def _transform_centroids(
    centroids: np.ndarray,
    R:         np.ndarray,
    center:    np.ndarray,
    t:         np.ndarray,
) -> np.ndarray:
    """
    Apply the standard docking transform to an array of 3-D points.
        coords_docked = R @ (coords - center) + center + t
    """
    return (R @ (centroids - center).T).T + center + t


def rescore_poses(
    case:    DockingCase,
    results: List[DockingResult],
    table:   PropensityTable,
    top_n:   int   = 2000,
    alpha:   float = 1.0,
    beta:    float = 0.5,
    cutoff:  float = CONTACT_CUTOFF,
) -> List[DockingResult]:
    """
    Rescore the top `top_n` docking results with interface propensity and
    re-sort.  Poses beyond top_n are left at the end, unmodified.

    Combined score = α × shape_score + β × propensity_score

    The original shape score is preserved in result._shape_score, and
    the propensity score in result._prop_score for Phase 5 reporting.

    Parameters
    ----------
    case    : DockingCase  (protein_struct and rna_struct must be populated)
    results : List[DockingResult] sorted by shape score descending
    table   : fitted PropensityTable
    top_n   : only rescore the top top_n poses
    alpha   : weight for FFT shape score
    beta    : weight for propensity score
    cutoff  : centroid-centroid contact distance in Å

    Returns
    -------
    Re-sorted list with combined scores. Unscored poses appended at the end.
    """
    if not table._is_fit:
        raise RuntimeError("PropensityTable must be fitted before rescoring.")

    # ── Pre-compute protein residue centroids (never move) ────────────────
    pro_residues = _residue_centroids(case.protein_struct, "protein")
    if not pro_residues:
        print("[rescore_poses] Warning: no protein residues found — "
              "returning results unchanged.")
        return results

    pro_names     = [r[0] for r in pro_residues]
    pro_centroids = np.array([r[1] for r in pro_residues])   # (Np, 3)

    # ── Pre-compute native RNA residue centroids ──────────────────────────
    rna_residues = _residue_centroids(case.rna_struct, "rna")
    if not rna_residues:
        print("[rescore_poses] Warning: no RNA residues found — "
              "returning results unchanged.")
        return results

    rna_names_raw  = [r[0] for r in rna_residues]
    rna_centroids0 = np.array([r[1] for r in rna_residues])   # (Nr, 3)
    rna_center     = rna_centroids0.mean(axis=0)

    # ── Rescore top_n poses ───────────────────────────────────────────────
    to_rescore  = results[:top_n]
    untouched   = results[top_n:]
    rescored    = []

    for res in to_rescore:
        R = res.rotation_matrix
        t = res.translation_vector

        # Transform RNA centroids to this pose's position
        rna_transformed = _transform_centroids(
            rna_centroids0, R, rna_center, t
        )   # (Nr, 3)

        # Find interface contacts at this pose
        contacts = []
        for aa_name, aa_xyz in zip(pro_names, pro_centroids):
            if aa_name not in PROTEIN_RESIDUES:
                continue
            dists = np.linalg.norm(rna_transformed - aa_xyz, axis=1)
            for j in np.where(dists <= cutoff)[0]:
                nt = canonical_nt(rna_names_raw[j])
                if nt is not None:
                    contacts.append((aa_name, nt))

        prop_score = table.score_contacts(contacts)
        combined   = alpha * res.score + beta * prop_score

        # Build a new DockingResult with the combined score
        new_res = DockingResult(
            score              = combined,
            rotation_matrix    = res.rotation_matrix,
            translation_vector = res.translation_vector,
        )
        # Attach sub-scores as extra attributes for Phase 5 reporting
        new_res._shape_score = res.score          # original FFT shape score
        new_res._prop_score  = prop_score         # raw propensity sum
        new_res._n_contacts  = len(contacts)      # interface contact count
        rescored.append(new_res)

    rescored.sort(key=lambda x: x.score, reverse=True)
    return rescored + untouched


# ═══════════════════════════════════════════════════════════════════════════
# Entry-point / self-test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    from phase1 import load_cases

    parser = argparse.ArgumentParser(
        description="propensity.py — build and inspect propensity table"
    )
    parser.add_argument("--json",       default="../assets/PRDBv3.json")
    parser.add_argument("--pdb_root",   default="../assets/ALL_PDBs")
    parser.add_argument("--save",       default="propensity.json",
                        help="Save table to this JSON file")
    parser.add_argument("--load",       default=None,
                        help="Load table from JSON instead of fitting")
    parser.add_argument("--top_pairs",  type=int, default=10,
                        help="Show N most enriched (aa, nt) pairs")
    args = parser.parse_args()

    table = PropensityTable()

    if args.load and os.path.isfile(args.load):
        table.load(args.load)
    else:
        cases, skipped = load_cases(args.json, args.pdb_root)
        if not cases:
            print("No cases loaded. Check --json and --pdb_root.")
            raise SystemExit(1)
        print(f"Fitting propensity table on {len(cases)} cases …")
        table.fit(cases)
        table.save(args.save)

    table.print_table()

    print(f"\n  Top {args.top_pairs} enriched (aa, nt) pairs:")
    print(f"  {'Score':>8}  {'AA':<6}  {'NT'}")
    print(f"  {'─'*8}  {'──':<6}  {'──'}")
    for score, aa, nt in table.top_pairs(args.top_pairs):
        print(f"  {score:>8.3f}  {aa:<6}  {nt}")