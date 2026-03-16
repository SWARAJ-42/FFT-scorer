"""
Phase 4 — FFT Correlation Docking (GPU Accelerated)
===================================
Executes the core FFT-based shape complementarity search using PyTorch 
to push grid arrays to the GPU, massively accelerating computation.

Issue fixes applied:
  1. RNA grid no longer binarised — preserves -15/+1/0 encoding
  2. RNA grid rotated via affine_transform instead of re-voxelising
     atom coordinates every rotation (10-50x faster)
  3. RNA radii computed once before loop, not re-derived per rotation
"""

import math
import time
import numpy as np
import argparse
import torch
from scipy.ndimage import affine_transform
from typing import List, Tuple
from dataclasses import dataclass

from phase1 import load_uu_cases, Structure
from phase2 import GridBuilder, MolGrid, _next_power_of_two, get_vdw_radius
from phase3 import SO3Sampler, rotate_coords


@dataclass
class DockingResult:
    score: float
    rotation_matrix: np.ndarray
    translation_vector: np.ndarray  # in Angstroms


# ═══════════════════════════════════════════════════════════════════════════
# Common Grid Manager
# ═══════════════════════════════════════════════════════════════════════════

class CommonGridManager:
    """
    Ensures that both the Protein and RNA are embedded into a shared,
    large enough grid to prevent FFT circular convolution artifacts.
    """
    def __init__(self, resolution: float = 1.0, padding: float = 10.0):
        self.resolution = resolution
        self.padding = padding
        self.builder = GridBuilder(resolution=resolution, padding=padding)

    def determine_common_shape(
        self,
        pro_coords: np.ndarray,
        rna_coords: np.ndarray,
    ) -> Tuple[np.ndarray, Tuple[int, int, int]]:
        p_min = pro_coords.min(axis=0)
        p_max = pro_coords.max(axis=0)

        rna_center   = rna_coords.mean(axis=0)
        rna_max_dist = np.max(np.linalg.norm(rna_coords - rna_center, axis=1))

        lo   = p_min - rna_max_dist - self.padding
        hi   = p_max + rna_max_dist + self.padding
        dims = tuple(
            _next_power_of_two(math.ceil(s / self.resolution))
            for s in (hi - lo)
        )
        return lo, dims

    def build_protein_grid(
        self,
        struct: Structure,
        origin: np.ndarray,
        dims:   Tuple[int, int, int],
    ) -> np.ndarray:
        atoms  = self.builder._collect_atoms(struct, "protein")
        coords = np.array([[a.x, a.y, a.z] for a in atoms], dtype=np.float64)
        radii  = np.array([get_vdw_radius(a) for a in atoms], dtype=np.float64)
        return self.builder._build_shape_grid(coords, radii, origin, dims)

    def build_rna_grid_native(
        self,
        struct: Structure,
        origin: np.ndarray,
        dims:   Tuple[int, int, int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build the RNA shape grid ONCE at its native (unrotated) orientation.

        Returns
        -------
        grid  : (Nx, Ny, Nz) float32  — the native RNA shape grid
        radii : (N,) float64          — per-atom VDW radii (cached for nothing;
                                        kept here so caller doesn't need to
                                        re-collect atoms)
        """
        atoms  = self.builder._collect_atoms(struct, "rna")
        coords = np.array([[a.x, a.y, a.z] for a in atoms], dtype=np.float64)
        radii  = np.array([get_vdw_radius(a) for a in atoms], dtype=np.float64)
        grid   = self.builder._build_shape_grid(coords, radii, origin, dims)
        return grid, radii


# ═══════════════════════════════════════════════════════════════════════════
# Grid rotation + rethresholding
# ═══════════════════════════════════════════════════════════════════════════

def rotate_grid(
    grid: np.ndarray,
    R:    np.ndarray,
) -> np.ndarray:
    """
    Rotate a 3D voxel grid by rotation matrix R using trilinear
    interpolation (scipy affine_transform).

    affine_transform applies the INVERSE mapping: for each output voxel,
    it looks up where in the INPUT grid that voxel came from.
    So we pass R.T (= R⁻¹ for rotation matrices) as the transform matrix,
    and compute the offset so the rotation is about the grid centre.

    Parameters
    ----------
    grid : (Nx, Ny, Nz) float32
    R    : (3, 3) rotation matrix

    Returns
    -------
    Rotated grid, rethresholded back to {-15, 0, +1}.
    """
    center = np.array(grid.shape) / 2.0
    offset = center - R.T @ center     # keeps grid centre fixed

    rotated = affine_transform(
        grid,
        matrix=R.T,
        offset=offset,
        order=1,           # trilinear interpolation
        mode='constant',
        cval=0.0,
        output=np.float32,
    )

    return _rethreshold(rotated)


def _rethreshold(grid: np.ndarray) -> np.ndarray:
    """
    Snap interpolated voxel values back to the canonical {-15, 0, +1} set.

    After trilinear interpolation, boundary voxels have intermediate
    values.  We snap:
        value >  0.5  →  +1    (surface)
        value < -0.5  → -15    (interior)
        otherwise     →   0    (exterior)
    """
    out = np.zeros_like(grid)
    out[grid >  0.5] =  1.0
    out[grid < -0.5] = -15.0
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Core FFT Docking Engine (GPU Enabled)
# ═══════════════════════════════════════════════════════════════════════════

class FFTDocker:
    def __init__(self, angular_step: float = 30.0, resolution: float = 1.0):
        self.resolution  = resolution
        self.sampler     = SO3Sampler(angular_step_deg=angular_step)
        self.grid_manager = CommonGridManager(resolution=resolution)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n[Hardware Initialization] FFTDocker using device: {self.device.type.upper()}")

    def dock(self, case) -> List[DockingResult]:
        print(f"\n[{case.complex_id}] Starting GPU-Accelerated FFT Docking...")
        start_time = time.time()

        # ── Collect coordinates ──────────────────────────────────────────
        pro_atoms  = self.grid_manager.builder._collect_atoms(case.protein_struct, "protein")
        pro_coords = np.array([[a.x, a.y, a.z] for a in pro_atoms])

        rna_atoms  = self.grid_manager.builder._collect_atoms(case.rna_struct, "rna")
        rna_coords = np.array([[a.x, a.y, a.z] for a in rna_atoms])

        origin, dims = self.grid_manager.determine_common_shape(pro_coords, rna_coords)
        print(f"  Common Grid Shape : {dims}")

        # ── Build protein grid (once, stays fixed) ───────────────────────
        print("  Building fixed protein grid...")
        pro_grid   = self.grid_manager.build_protein_grid(case.protein_struct, origin, dims)
        pro_tensor = torch.tensor(pro_grid, dtype=torch.float32, device=self.device)
        fft_pro    = torch.fft.fftn(pro_tensor)

        # ── Build RNA grid ONCE at native orientation (Issue 2 fix) ─────
        print("  Building native RNA grid (once)...")
        rna_grid_native, _ = self.grid_manager.build_rna_grid_native(
            case.rna_struct, origin, dims
        )

        # ── Rotation loop ────────────────────────────────────────────────
        results  = []
        n_rots   = self.sampler.n_rotations
        print(f"  Evaluating {n_rots} rotations...")

        for i, R in enumerate(self.sampler):
            if i > 0 and i % 50 == 0:
                print(f"    Processed {i}/{n_rots} rotations...")

            # Rotate the grid itself via affine_transform (Issue 2 fix)
            # — no re-voxelisation, no binary erosion, no atom loop
            rna_grid_rotated = rotate_grid(rna_grid_native, R)

            # ── GPU block ────────────────────────────────────────────────
            rna_tensor = torch.tensor(rna_grid_rotated, dtype=torch.float32, device=self.device)
            fft_rna    = torch.fft.fftn(rna_tensor)

            # Cross-correlation: IFFT( FFT(P) * conj(FFT(R)) )
            corr_tensor  = torch.fft.ifftn(fft_pro * torch.conj(fft_rna)).real

            best_idx_flat = torch.argmax(corr_tensor).item()
            best_score    = corr_tensor.flatten()[best_idx_flat].item()
            # ── end GPU block ─────────────────────────────────────────────

            best_idx = np.unravel_index(best_idx_flat, dims)

            shift_x = best_idx[0] if best_idx[0] < dims[0]/2 else best_idx[0] - dims[0]
            shift_y = best_idx[1] if best_idx[1] < dims[1]/2 else best_idx[1] - dims[1]
            shift_z = best_idx[2] if best_idx[2] < dims[2]/2 else best_idx[2] - dims[2]

            translation_vector = np.array([shift_x, shift_y, shift_z]) * self.resolution

            results.append(DockingResult(
                score=float(best_score),
                rotation_matrix=R,
                translation_vector=translation_vector,
            ))

        results.sort(key=lambda x: x.score, reverse=True)

        elapsed = time.time() - start_time
        print(f"[{case.complex_id}] Docking complete in {elapsed:.2f} seconds.")
        print(f"  Top Score: {results[0].score:.2f}")

        return results


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 4: FFT Correlation")

    parser.add_argument("--json",       default=r"D:\BTP Files\PRDBv3.0\PRDBv3_info.json")
    parser.add_argument("--pdb_root",   default=r"D:\BTP Files\PRDBv3.0")
    parser.add_argument("--step",       type=float, default=30.0)
    parser.add_argument("--resolution", type=float, default=1.0)
    args = parser.parse_args()

    cases, skipped = load_uu_cases(args.json, args.pdb_root)
    if not cases:
        print("No cases loaded. Check paths.")
        exit()

    test_case = cases[0]
    docker    = FFTDocker(angular_step=args.step, resolution=args.resolution)
    top_results = docker.dock(test_case)

    print("\nTop 5 Poses Found:")
    for idx, res in enumerate(top_results[:5]):
        print(
            f"  Pose {idx+1}: Score = {res.score:>8.2f} | "
            f"Translation = [{res.translation_vector[0]:>6.1f}, "
            f"{res.translation_vector[1]:>6.1f}, "
            f"{res.translation_vector[2]:>6.1f}] Å"
        )

    import pickle

    results_dict = {test_case.complex_id: top_results}

    with open("../results.pkl", "wb") as f:
        pickle.dump(results_dict, f)

    print("Results saved to results.pkl")