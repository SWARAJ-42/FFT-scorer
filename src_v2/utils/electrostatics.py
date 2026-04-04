"""
electrostatic.py — Atom Charge Assignment & Electrostatic Potential Grid
=========================================================================
Assigns partial charges to protein and RNA atoms and computes a Coulombic
electrostatic potential on the same voxel grid used by Phase 2.

Charge scheme: simplified AMBER99 point charges for the ionisable groups
that dominate protein-RNA recognition.

    Protein : Arg (+), Lys (+), His (partial +), Asp (-), Glu (-)
              backbone N, C, O (small fractional charges)
    RNA     : phosphate P, O1P, O2P (strongly −), ribose O2' (−), O3'/O5' (−)

Distance-dependent dielectric (Mehler & Solmajer 1991):
    ε(r) = max(4.0, DIEL_SLOPE × r)

This avoids the singularity at r=0 and approximates the shielding effect
of water at longer distances.

Output: ElecGrid  — same spatial domain as MolGrid (phase2.py), values in
                    kcal/mol/e (clamped to ±MAX_CLIP to prevent FFT overflow).

Integration
-----------
• phase2.py  — ElecGridBuilder can be called alongside GridBuilder to
               produce an electrostatic grid for visualisation or analysis.
• phase4.py  — A second FFT cross-correlation is run on the electrostatic
               grids and added to the shape correlation score:
                   corr_total = corr_shape + ELEC_WEIGHT × corr_elec
"""

import math
import dataclasses
import numpy as np
from typing import Dict, List, Optional, Tuple

from phase1 import Atom, Structure


# ═══════════════════════════════════════════════════════════════════════════
# Partial charge dictionary
# key: (residue_name, atom_name)  →  charge in units of e
# Wildcard residue "*" applies to every residue not matched by a specific key.
# Source: AMBER99SB partial charges (Wang et al. 2000, JACS)
# ═══════════════════════════════════════════════════════════════════════════

PARTIAL_CHARGES: Dict[Tuple[str, str], float] = {
    # ── Protein — positively charged side chains ──────────────────────────
    ("ARG", "CZ"):   +0.64,
    ("ARG", "NH1"):  +0.46,
    ("ARG", "NH2"):  +0.46,
    ("ARG", "NE"):   +0.29,
    ("ARG", "HE"):   +0.00,   # no H atoms in our PDB parser
    ("LYS", "NZ"):   +0.69,
    ("HIS", "ND1"):  +0.10,
    ("HIS", "NE2"):  +0.10,
    ("HIS", "CE1"):  +0.08,
    # ── Protein — negatively charged side chains ──────────────────────────
    ("ASP", "OD1"):  -0.80,
    ("ASP", "OD2"):  -0.80,
    ("ASP", "CG"):   +0.70,
    ("GLU", "OE1"):  -0.80,
    ("GLU", "OE2"):  -0.80,
    ("GLU", "CD"):   +0.70,
    # ── Protein — backbone (partial charges present in every residue) ─────
    ("*",   "N"):    -0.41,
    ("*",   "C"):    +0.60,
    ("*",   "O"):    -0.57,
    ("*",   "CA"):   +0.02,
    # ── RNA — phosphate backbone (dominant negative patch) ────────────────
    ("A",   "P"):   +1.17,  ("U",   "P"):   +1.17,
    ("G",   "P"):   +1.17,  ("C",   "P"):   +1.17,
    ("A",   "O1P"): -0.78,  ("U",   "O1P"): -0.78,
    ("G",   "O1P"): -0.78,  ("C",   "O1P"): -0.78,
    ("A",   "O2P"): -0.78,  ("U",   "O2P"): -0.78,
    ("G",   "O2P"): -0.78,  ("C",   "O2P"): -0.78,
    # ── RNA — ribose oxygens ──────────────────────────────────────────────
    ("A",   "O5'"): -0.37,  ("U",   "O5'"): -0.37,
    ("G",   "O5'"): -0.37,  ("C",   "O5'"): -0.37,
    ("A",   "O3'"): -0.37,  ("U",   "O3'"): -0.37,
    ("G",   "O3'"): -0.37,  ("C",   "O3'"): -0.37,
    ("A",   "C1'"): +0.06,  ("U",   "C1'"): +0.06,
    ("G",   "C1'"): +0.06,  ("C",   "C1'"): +0.06,
    # ── RNA — 2'-OH (key for A-form helix recognition) ───────────────────
    ("A",   "O2'"): -0.61,  ("U",   "O2'"): -0.61,
    ("G",   "O2'"): -0.61,  ("C",   "O2'"): -0.61,
    # ── RNA — adenine base (partial positive at N1, N6) ──────────────────
    ("A",   "N1"):  -0.59,  ("A",   "N6"):  -0.91,
    ("A",   "C6"):  +0.52,  ("A",   "N3"):  -0.69,
    # ── RNA — guanine base (O6 negative, N1/N2 partial positive) ─────────
    ("G",   "O6"):  -0.54,  ("G",   "N1"):  -0.47,
    ("G",   "N2"):  -0.92,  ("G",   "N7"):  -0.57,
    # ── RNA — uracil base (O2, O4 negative) ──────────────────────────────
    ("U",   "O2"):  -0.55,  ("U",   "O4"):  -0.55,
    ("U",   "N3"):  -0.35,
    # ── RNA — cytosine base (O2 negative, N3/N4 partial) ─────────────────
    ("C",   "O2"):  -0.58,  ("C",   "N3"):  -0.72,
    ("C",   "N4"):  -0.95,  ("C",   "C4"):  +0.71,
}

# Physical constants
COULOMB_K  = 332.0   # kcal·Å / (mol·e²)
R_CUT      = 1.5     # Å — minimum distance (prevents 1/r singularity)
DIEL_SLOPE = 0.24    # ε(r) ≈ DIEL_SLOPE × r  (Mehler & Solmajer)


def get_partial_charge(atom: Atom) -> float:
    """
    Return the partial charge (in e) for a given atom.
    Look-up priority:
        1. Exact (res_name, atom_name) match
        2. Wildcard ("*", atom_name)  — backbone atoms
        3. 0.0 if no match
    """
    q = PARTIAL_CHARGES.get((atom.res_name, atom.name))
    if q is not None:
        return q
    q = PARTIAL_CHARGES.get(("*", atom.name))
    return q if q is not None else 0.0


# ═══════════════════════════════════════════════════════════════════════════
# ElecGrid — output container  (mirrors MolGrid from phase2.py)
# ═══════════════════════════════════════════════════════════════════════════

@dataclasses.dataclass
class ElecGrid:
    """
    Stores the 3-D Coulombic potential grid and spatial metadata.

    elec_grid : (Nx, Ny, Nz)  float32,  kcal/mol/e  (clamped to ±max_clip)
    origin    : (3,)  lower-left corner of the grid in Ångström
    resolution: Å/voxel
    """
    pdb_id:     str
    mol_type:   str
    elec_grid:  np.ndarray
    origin:     np.ndarray
    resolution: float

    @property
    def grid_shape(self) -> Tuple[int, int, int]:
        return self.elec_grid.shape

    def summary(self) -> str:
        Nx, Ny, Nz = self.grid_shape
        g = self.elec_grid
        n_pos = int((g > 0.01).sum())
        n_neg = int((g < -0.01).sum())
        return (
            f"ElecGrid  pdb={self.pdb_id!r}  type={self.mol_type}\n"
            f"  grid shape   : ({Nx}, {Ny}, {Nz})  —  {Nx*Ny*Nz:,} voxels\n"
            f"  resolution   : {self.resolution} Å/voxel\n"
            f"  origin       : ({self.origin[0]:.2f}, "
            f"{self.origin[1]:.2f}, {self.origin[2]:.2f}) Å\n"
            f"  potential    : min={g.min():.2f}  max={g.max():.2f}  "
            f"mean={g.mean():.3f}  kcal/mol/e\n"
            f"  positive vox : {n_pos:,}   negative vox : {n_neg:,}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# ElecGridBuilder
# ═══════════════════════════════════════════════════════════════════════════

class ElecGridBuilder:
    """
    Computes a Coulombic electrostatic potential grid for a Structure.

    For each grid voxel v:
        φ(v) = COULOMB_K × Σᵢ  qᵢ / [ ε(|v − rᵢ|) × |v − rᵢ| ]

    where  ε(r) = max(4.0, DIEL_SLOPE × r)   (distance-dependent dielectric).

    A spherical cut-off is applied per atom: only voxels within the radius
    where |φ| would exceed 0.01 kcal/mol/e are updated, keeping the loop fast.

    The final grid is clamped to [−max_clip, +max_clip] to prevent extreme
    values near atom centres from dominating the FFT cross-correlation.

    Parameters
    ----------
    resolution : Å/voxel  (must match the shape grid used in Phase 4)
    padding    : Å of extra space around the molecule
    max_clip   : kcal/mol/e  — clip potential to this absolute value
    """

    def __init__(
        self,
        resolution: float = 1.0,
        padding:    float = 8.0,
        max_clip:   float = 10.0,
    ):
        self.resolution = resolution
        self.padding    = padding
        self.max_clip   = max_clip

    # ── Public API ──────────────────────────────────────────────────────────

    def build(
        self,
        structure: Structure,
        mol_type:  str,
        origin:    np.ndarray,
        dims:      Tuple[int, int, int],
    ) -> ElecGrid:
        """
        Build the electrostatic potential grid on a pre-defined spatial domain.

        The domain (origin, dims) must be shared with the shape grid so that
        the FFT cross-correlations can be combined.  Call this after
        CommonGridManager.determine_common_shape() to get (origin, dims).

        Parameters
        ----------
        structure : parsed Structure from Phase 1
        mol_type  : 'protein' or 'rna'
        origin    : (3,) lower-left corner of the grid in Å
        dims      : (Nx, Ny, Nz) — must match the shape grid
        """
        charged_atoms = self._collect_charged_atoms(structure, mol_type)

        Nx, Ny, Nz = dims
        if not charged_atoms:
            # Return a zero grid — no charged atoms detected
            return ElecGrid(
                pdb_id     = structure.pdb_id,
                mol_type   = mol_type,
                elec_grid  = np.zeros((Nx, Ny, Nz), dtype=np.float32),
                origin     = origin.copy(),
                resolution = self.resolution,
            )

        grid = self._compute_coulomb_grid(charged_atoms, origin, dims)

        return ElecGrid(
            pdb_id     = structure.pdb_id,
            mol_type   = mol_type,
            elec_grid  = grid,
            origin     = origin.copy(),
            resolution = self.resolution,
        )

    # ── Private helpers ─────────────────────────────────────────────────────

    def _collect_charged_atoms(
        self,
        structure: Structure,
        mol_type:  str,
    ) -> List[Tuple[Atom, float]]:
        """Return (atom, charge) pairs where |charge| > 1e-6 e."""
        chains = (structure.protein_chains() if mol_type == "protein"
                  else structure.rna_chains())
        charged = []
        for chain in chains:
            for atom in chain.atoms:
                q = get_partial_charge(atom)
                if abs(q) > 1e-6:
                    charged.append((atom, float(q)))
        return charged

    def _compute_coulomb_grid(
        self,
        charged_atoms: List[Tuple[Atom, float]],
        origin:        np.ndarray,
        dims:          Tuple[int, int, int],
    ) -> np.ndarray:
        """
        Compute the Coulombic potential at every voxel centre.
        Uses a local bounding-box trick to avoid O(N_atoms × N_voxels) loops.
        """
        Nx, Ny, Nz = dims
        r          = self.resolution
        grid       = np.zeros((Nx, Ny, Nz), dtype=np.float64)

        # Voxel centre coordinates for each axis
        xs = origin[0] + np.arange(Nx) * r   # (Nx,)
        ys = origin[1] + np.arange(Ny) * r   # (Ny,)
        zs = origin[2] + np.arange(Nz) * r   # (Nz,)

        for atom, q in charged_atoms:
            ax, ay, az = atom.x, atom.y, atom.z

            # Radius (voxels) at which |φ| drops below 0.01 kcal/mol/e
            # From φ = K*q/(ε*d) ≈ K*q/(DIEL_SLOPE*d²): d²=K*q/(0.01*DIEL)
            r_cutoff_ang = math.sqrt(abs(COULOMB_K * q) / (0.01 * DIEL_SLOPE))
            r_cutoff_vox = int(math.ceil(r_cutoff_ang / r)) + 2
            r_cutoff_vox = min(r_cutoff_vox, max(Nx, Ny, Nz))

            # Grid index closest to atom centre
            ci = int(round((ax - origin[0]) / r))
            cj = int(round((ay - origin[1]) / r))
            ck = int(round((az - origin[2]) / r))

            # Bounding box in voxel coordinates (clipped to grid)
            i0 = max(0, ci - r_cutoff_vox)
            i1 = min(Nx, ci + r_cutoff_vox + 1)
            j0 = max(0, cj - r_cutoff_vox)
            j1 = min(Ny, cj + r_cutoff_vox + 1)
            k0 = max(0, ck - r_cutoff_vox)
            k1 = min(Nz, ck + r_cutoff_vox + 1)

            if i0 >= i1 or j0 >= j1 or k0 >= k1:
                continue

            # Broadcast distance computation — no explicit Python loops
            dx = (xs[i0:i1] - ax)[:, None, None]   # (ni, 1, 1)
            dy = (ys[j0:j1] - ay)[None, :, None]   # (1, nj, 1)
            dz = (zs[k0:k1] - az)[None, None, :]   # (1, 1, nk)

            dist = np.sqrt(dx**2 + dy**2 + dz**2)
            dist = np.maximum(dist, R_CUT)                    # clamp to R_CUT
            eps  = np.maximum(4.0, DIEL_SLOPE * dist)         # ε(r)

            grid[i0:i1, j0:j1, k0:k1] += COULOMB_K * q / (eps * dist)

        # Clamp to prevent extreme values near atom centres
        grid = np.clip(grid, -self.max_clip, self.max_clip)
        return grid.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# Convenience wrapper — mirrors build_grids_for_case() in phase2.py
# ═══════════════════════════════════════════════════════════════════════════

def build_elec_grids_for_case(
    case,
    origin:  np.ndarray,
    dims:    Tuple[int, int, int],
    builder: Optional[ElecGridBuilder] = None,
) -> Tuple[ElecGrid, ElecGrid]:
    """
    Build protein and RNA electrostatic grids for a DockingCase on a shared grid.

    Parameters
    ----------
    case    : DockingCase with protein_struct and rna_struct populated
    origin  : (3,)  shared grid origin (from CommonGridManager)
    dims    : (Nx, Ny, Nz)  shared grid dimensions
    builder : optional pre-configured ElecGridBuilder

    Returns
    -------
    (pro_elec_grid, rna_elec_grid)
    """
    if builder is None:
        builder = ElecGridBuilder()

    pro_eg = builder.build(case.protein_struct, "protein", origin, dims)
    rna_eg = builder.build(case.rna_struct,     "rna",     origin, dims)
    return pro_eg, rna_eg


# ═══════════════════════════════════════════════════════════════════════════
# Visualization  (requires plotly)
# ═══════════════════════════════════════════════════════════════════════════

def visualize_elec_grid(egrid: ElecGrid, isosurface_value: float = 1.0):
    """
    Render the electrostatic potential grid as two isosurface slabs:
        Positive (blue) : regions attractive to anions (e.g. RNA phosphate)
        Negative (red)  : regions attractive to cations (e.g. Arg, Lys)
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("Plotly not installed. Run: pip install plotly")
        return

    g = egrid.elec_grid
    pos_vox = np.argwhere(g >=  isosurface_value)
    neg_vox = np.argwhere(g <= -isosurface_value)

    def to_ang(vox):
        return egrid.origin + vox * egrid.resolution

    fig = go.Figure()

    if len(pos_vox):
        c = to_ang(pos_vox)
        fig.add_trace(go.Scatter3d(
            x=c[:,0], y=c[:,1], z=c[:,2],
            mode="markers",
            marker=dict(size=2, color="royalblue", opacity=0.5),
            name=f"φ ≥ +{isosurface_value} kcal/mol/e (electropositive)",
        ))

    if len(neg_vox):
        c = to_ang(neg_vox)
        fig.add_trace(go.Scatter3d(
            x=c[:,0], y=c[:,1], z=c[:,2],
            mode="markers",
            marker=dict(size=2, color="crimson", opacity=0.5),
            name=f"φ ≤ −{isosurface_value} kcal/mol/e (electronegative)",
        ))

    fig.update_layout(
        title=f"Electrostatic Potential — {egrid.pdb_id} ({egrid.mol_type})",
        scene=dict(xaxis_title="X (Å)", yaxis_title="Y (Å)", zaxis_title="Z (Å)"),
        width=900, height=750,
    )
    fig.show()


# ═══════════════════════════════════════════════════════════════════════════
# Entry-point / self-test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    from phase1 import load_cases
    from phase4 import CommonGridManager

    parser = argparse.ArgumentParser(
        description="electrostatic.py — build and visualise electrostatic grids"
    )
    parser.add_argument("--json",       default="../assets/PRDBv3.json")
    parser.add_argument("--pdb_root",   default="../assets/ALL_PDBs")
    parser.add_argument("--resolution", type=float, default=1.0)
    parser.add_argument("--clip",       type=float, default=10.0,
                        help="Potential clip value in kcal/mol/e (default 10)")
    parser.add_argument("--iso",        type=float, default=1.0,
                        help="Isosurface threshold for visualisation (default 1.0)")
    args = parser.parse_args()

    cases, skipped = load_cases(args.json, args.pdb_root)
    if not cases:
        print("No cases loaded.")
        raise SystemExit(1)

    case = cases[0]
    print(f"Building electrostatic grids for {case.complex_id} …")

    mgr   = CommonGridManager(resolution=args.resolution)
    import numpy as _np
    from phase2 import get_vdw_radius
    pro_atoms  = mgr.builder._collect_atoms(case.protein_struct, "protein")
    rna_atoms  = mgr.builder._collect_atoms(case.rna_struct,     "rna")
    pro_coords = _np.array([[a.x, a.y, a.z] for a in pro_atoms])
    rna_coords = _np.array([[a.x, a.y, a.z] for a in rna_atoms])
    origin, dims = mgr.determine_common_shape(pro_coords, rna_coords)

    builder  = ElecGridBuilder(resolution=args.resolution, max_clip=args.clip)
    pro_eg, rna_eg = build_elec_grids_for_case(case, origin, dims, builder)

    print(pro_eg.summary())
    print()
    print(rna_eg.summary())

    print("\nOpening protein electrostatic grid visualisation …")
    visualize_elec_grid(pro_eg, isosurface_value=args.iso)
    print("Opening RNA electrostatic grid visualisation …")
    visualize_elec_grid(rna_eg, isosurface_value=args.iso)