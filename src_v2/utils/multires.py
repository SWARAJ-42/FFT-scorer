"""
multires.py — Multi-Resolution Hierarchical Docking Search
===========================================================
Wraps the FFT correlation engine in a coarse-to-fine two-pass search:

    Pass 1 — Coarse  (default 2.0 Å):
        Build protein and RNA grids at coarse_res.
        Evaluate ALL rotations from the SO(3) sampler.
        Keep the top_k poses by raw FFT shape score.

    Pass 2 — Fine  (default 1.0 Å):
        Build protein and RNA grids at fine_res.
        Re-evaluate ONLY the top_k rotation matrices from Pass 1.
        Final ranked list is returned.

Theoretical speedup
-------------------
Full fine search cost   ≈ N_rots   FFT evaluations at fine_res
MultiRes cost           ≈ N_rots   at coarse (8× cheaper each)
                          + top_k  at fine
Speedup ≈ N_rots / (N_rots / 8 + top_k)

Example: N_rots = 10,000  top_k = 500  resolution ratio = 2×
    Speedup ≈ 10,000 / (1,250 + 500) ≈ 5.7×

Integration
-----------
phase4.py — MultiResDocker is imported and used instead of FFTDocker when
            the --multires flag is passed via run.py.
run.py    — Selects MultiResDocker when --multires is set (default: off).
            See run.py integration section at the bottom of this file.
"""

import math
import time
import numpy as np
import torch
from typing import List, Tuple

from phase1 import Structure
from phase2 import GridBuilder, _next_power_of_two, get_vdw_radius
from phase3 import SO3Sampler
from phase4 import (
    DockingResult,
    _build_base_flat_grid,
    build_sampling_grid_chunk,
    rotate_grid_gpu,
    ROTATION_BATCH_SIZE,
    MAX_GRID_DIM,
)


# ═══════════════════════════════════════════════════════════════════════════
# Resolution-parameterised Grid Manager
# ═══════════════════════════════════════════════════════════════════════════

class ResolutionGridManager:
    """
    Like CommonGridManager (phase4.py) but built at an arbitrary resolution.
    Used by the coarse pass to construct lower-resolution grids.
    """

    def __init__(self, resolution: float = 2.0, padding: float = 10.0):
        self.resolution = resolution
        self.padding    = padding
        self.builder    = GridBuilder(resolution=resolution, padding=padding)

    def determine_common_shape(
        self,
        pro_coords: np.ndarray,
        rna_coords: np.ndarray,
    ) -> Tuple[np.ndarray, Tuple[int, int, int]]:
        p_min = pro_coords.min(axis=0)
        p_max = pro_coords.max(axis=0)

        rna_center   = rna_coords.mean(axis=0)
        rna_max_dist = np.max(np.linalg.norm(rna_coords - rna_center, axis=1))

        lo  = p_min - rna_max_dist - self.padding
        hi  = p_max + rna_max_dist + self.padding
        dims = tuple(
            min(_next_power_of_two(math.ceil(s / self.resolution)), MAX_GRID_DIM)
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
    ) -> np.ndarray:
        atoms  = self.builder._collect_atoms(struct, "rna")
        coords = np.array([[a.x, a.y, a.z] for a in atoms], dtype=np.float64)
        radii  = np.array([get_vdw_radius(a) for a in atoms], dtype=np.float64)
        return self.builder._build_shape_grid(coords, radii, origin, dims)


# ═══════════════════════════════════════════════════════════════════════════
# Core FFT scan — single pass over a given rotation subset
# ═══════════════════════════════════════════════════════════════════════════

def _fft_scan(
    pro_grid:         np.ndarray,
    rna_grid:         np.ndarray,
    rotation_subset:  List[np.ndarray],
    resolution:       float,
    device:           torch.device,
    batch_size:       int = ROTATION_BATCH_SIZE,
    progress_label:   str = "",
) -> List[DockingResult]:
    """
    Run FFT cross-correlation over `rotation_subset` using pre-built grids.

    This is a self-contained scan that mirrors the inner loop of
    FFTDocker.dock() (phase4.py) but operates on externally supplied grids
    and a rotation sub-list.  It is called twice by MultiResDocker:
        once with coarse grids + all rotations,
        once with fine grids   + top_k rotations.

    Parameters
    ----------
    pro_grid        : (Nx, Ny, Nz)  float32  protein shape grid (CPU numpy)
    rna_grid        : (Nx, Ny, Nz)  float32  RNA native shape grid (CPU numpy)
    rotation_subset : list of (3,3) rotation matrices to evaluate
    resolution      : Å/voxel (used to convert voxel shift → Angstrom)
    device          : torch.device
    batch_size      : number of rotations per GPU chunk
    progress_label  : optional prefix for progress print

    Returns
    -------
    List[DockingResult] sorted by score descending.
    """
    dims       = pro_grid.shape
    Nx, Ny, Nz = dims
    n_rots     = len(rotation_subset)

    # ── Push protein to GPU, precompute its FFT (done once) ─────────────
    pro_tensor = torch.tensor(pro_grid, dtype=torch.float32, device=device)
    fft_pro    = torch.fft.fftn(pro_tensor)

    # ── Push native RNA to GPU in grid_sample-compatible format ──────────
    rna_native_gpu = (
        torch.tensor(rna_grid, dtype=torch.float32, device=device)
        .unsqueeze(0).unsqueeze(0)          # (1, 1, Nx, Ny, Nz)
    )

    # ── Build base coordinate grid once — reused across every chunk ───────
    flat_grid = _build_base_flat_grid(dims, device)   # (M, 3)

    results = []
    t_start = time.time()

    for chunk_start in range(0, n_rots, batch_size):
        chunk_end  = min(chunk_start + batch_size, n_rots)
        chunk_rots = rotation_subset[chunk_start:chunk_end]
        B          = len(chunk_rots)

        # Progress report every 10 chunks
        if chunk_start > 0 and chunk_start % (batch_size * 10) == 0:
            elapsed = time.time() - t_start
            eta     = elapsed / chunk_start * (n_rots - chunk_start)
            tag     = f"[{progress_label}] " if progress_label else ""
            print(f"    {tag}{chunk_start:>6}/{n_rots}  "
                  f"elapsed {elapsed:.0f}s  ETA {eta:.0f}s")

        sg_chunk = build_sampling_grid_chunk(
            chunk_rots, flat_grid, dims, device
        )   # (B, Nx, Ny, Nz, 3)

        for b in range(B):
            sg          = sg_chunk[b].unsqueeze(0)          # (1, Nx, Ny, Nz, 3)
            rna_rotated = rotate_grid_gpu(rna_native_gpu, sg)  # (Nx, Ny, Nz)

            fft_rna    = torch.fft.fftn(rna_rotated)
            corr       = torch.fft.ifftn(fft_pro * torch.conj(fft_rna)).real

            best_flat  = torch.argmax(corr).item()
            best_score = corr.flatten()[best_flat].item()
            best_idx   = np.unravel_index(best_flat, dims)

            shift_x = best_idx[0] if best_idx[0] < Nx / 2 else best_idx[0] - Nx
            shift_y = best_idx[1] if best_idx[1] < Ny / 2 else best_idx[1] - Ny
            shift_z = best_idx[2] if best_idx[2] < Nz / 2 else best_idx[2] - Nz
            translation = np.array([shift_x, shift_y, shift_z]) * resolution

            results.append(DockingResult(
                score              = float(best_score),
                rotation_matrix    = chunk_rots[b].copy(),
                translation_vector = translation,
            ))

        del sg_chunk   # free chunk grids immediately — keeps VRAM usage flat

    results.sort(key=lambda x: x.score, reverse=True)
    return results


# ═══════════════════════════════════════════════════════════════════════════
# MultiResDocker — the main class
# ═══════════════════════════════════════════════════════════════════════════

class MultiResDocker:
    """
    Two-pass hierarchical docking search.

    Pass 1 (coarse):
        Build protein and RNA grids at coarse_res (default 2.0 Å).
        Evaluate ALL rotation matrices from the SO(3) sampler.
        Keep the top_k poses by FFT shape score.

    Pass 2 (fine):
        Build grids at fine_res (default 1.0 Å) on the same spatial domain.
        Re-evaluate only the top_k rotation matrices from Pass 1.
        Return the re-sorted final ranked list.

    The interface is drop-in compatible with FFTDocker: call .dock(case)
    and receive a sorted List[DockingResult].
    """

    def __init__(
        self,
        angular_step: float = 30.0,
        coarse_res:   float = 2.0,
        fine_res:     float = 1.0,
        top_k:        int   = 500,
    ):
        if coarse_res <= fine_res:
            raise ValueError(
                f"coarse_res ({coarse_res} Å) must be larger than "
                f"fine_res ({fine_res} Å)."
            )
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        self.coarse_res = coarse_res
        self.fine_res   = fine_res
        self.top_k      = top_k
        self.sampler    = SO3Sampler(angular_step_deg=angular_step)
        self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        speedup_estimate = self.sampler.n_rotations / (
            self.sampler.n_rotations / (coarse_res / fine_res) ** 3 + top_k
        )
        print(
            f"\n[MultiResDocker]"
            f"  coarse={coarse_res} Å  fine={fine_res} Å  top_k={top_k}\n"
            f"  device={self.device.type.upper()}  "
            f"  estimated speedup ≈ {speedup_estimate:.1f}×"
        )

    # ── Coordinate helpers ───────────────────────────────────────────────

    @staticmethod
    def _coords(struct: Structure, mol_type: str) -> np.ndarray:
        chains = (struct.protein_chains() if mol_type == "protein"
                  else struct.rna_chains())
        return np.array(
            [[a.x, a.y, a.z] for chain in chains for a in chain.atoms],
            dtype=np.float64,
        )

    # ── Main entry point ─────────────────────────────────────────────────

    def dock(self, case) -> List[DockingResult]:
        """
        Run the two-pass hierarchical search for one DockingCase.

        Returns
        -------
        List[DockingResult] sorted by fine-resolution score (descending).
        The .score field reflects the fine-resolution FFT shape score.
        """
        cid = case.complex_id
        print(f"\n[{cid}] MultiRes docking — "
              f"{self.sampler.n_rotations:,} rotations total")
        t0 = time.time()

        pro_coords = self._coords(case.protein_struct, "protein")
        rna_coords = self._coords(case.rna_struct,     "rna")

        # ──────────────────────────────────────────────────────────────────
        # Pass 1: Coarse scan — ALL rotations, coarse grid
        # ──────────────────────────────────────────────────────────────────
        print(f"  ┌─ Pass 1 [coarse {self.coarse_res} Å]  "
              f"building grids …", flush=True)

        c_mgr              = ResolutionGridManager(self.coarse_res, padding=10.0)
        c_origin, c_dims   = c_mgr.determine_common_shape(pro_coords, rna_coords)

        print(f"  │  grid shape {c_dims}  "
              f"({c_dims[0]*c_dims[1]*c_dims[2]:,} voxels)")

        c_pro = c_mgr.build_protein_grid(case.protein_struct, c_origin, c_dims)
        c_rna = c_mgr.build_rna_grid_native(case.rna_struct,  c_origin, c_dims)

        t1 = time.time()
        coarse_results = _fft_scan(
            c_pro, c_rna,
            self.sampler.rotations,
            self.coarse_res,
            self.device,
            progress_label=f"{cid} coarse",
        )
        dt1 = time.time() - t1
        print(f"  │  done in {dt1:.1f}s  "
              f"— top coarse score: {coarse_results[0].score:.1f}")

        # ──────────────────────────────────────────────────────────────────
        # Select top_k unique rotation matrices for fine refinement
        # ──────────────────────────────────────────────────────────────────
        top_rots = [r.rotation_matrix for r in coarse_results[:self.top_k]]
        n_sel    = len(top_rots)
        print(f"  │  {n_sel} rotations selected for fine refinement  "
              f"({100*n_sel/self.sampler.n_rotations:.1f}% of total)")

        # Free coarse GPU resources (handled by Python GC but good hygiene)
        del c_pro, c_rna, coarse_results

        # ──────────────────────────────────────────────────────────────────
        # Pass 2: Fine scan — top_k rotations, fine grid
        # ──────────────────────────────────────────────────────────────────
        print(f"  └─ Pass 2 [fine {self.fine_res} Å]  "
              f"building grids …", flush=True)

        f_mgr            = ResolutionGridManager(self.fine_res, padding=10.0)
        f_origin, f_dims = f_mgr.determine_common_shape(pro_coords, rna_coords)

        print(f"     grid shape {f_dims}  "
              f"({f_dims[0]*f_dims[1]*f_dims[2]:,} voxels)")

        f_pro = f_mgr.build_protein_grid(case.protein_struct, f_origin, f_dims)
        f_rna = f_mgr.build_rna_grid_native(case.rna_struct,  f_origin, f_dims)

        t2 = time.time()
        fine_results = _fft_scan(
            f_pro, f_rna,
            top_rots,
            self.fine_res,
            self.device,
            progress_label=f"{cid} fine",
        )
        dt2 = time.time() - t2
        print(f"     done in {dt2:.1f}s  "
              f"— top fine score: {fine_results[0].score:.1f}")

        # ── Summary ──────────────────────────────────────────────────────
        elapsed = time.time() - t0
        # Cost of a full fine scan (hypothetical)
        full_fine_cost = dt1 * (self.fine_res / self.coarse_res) ** 3 \
                       + dt2 * (self.sampler.n_rotations / n_sel)
        actual_speedup = full_fine_cost / elapsed if elapsed > 0 else float("inf")

        print(
            f"\n[{cid}] MultiRes complete  "
            f"total={elapsed:.1f}s  "
            f"coarse={dt1:.1f}s  fine={dt2:.1f}s  "
            f"actual speedup ≈ {actual_speedup:.1f}×"
        )

        return fine_results

    # ── Summary string ───────────────────────────────────────────────────

    def summary(self) -> str:
        return (
            f"MultiResDocker\n"
            f"  SO(3) sampler  : {self.sampler.n_rotations:,} rotations "
            f"@ {self.sampler.angular_step}°\n"
            f"  coarse pass    : {self.coarse_res} Å  (all rotations)\n"
            f"  fine pass      : {self.fine_res} Å  (top {self.top_k} rotations)\n"
            f"  device         : {self.device.type.upper()}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    from phase1 import load_cases

    parser = argparse.ArgumentParser(
        description="multires.py — test multi-resolution hierarchical docking"
    )
    parser.add_argument("--json",       default="../assets/PRDBv3.json")
    parser.add_argument("--pdb_root",   default="../assets/ALL_PDBs")
    parser.add_argument("--step",       type=float, default=30.0,
                        help="SO(3) angular step in degrees (default 30)")
    parser.add_argument("--coarse_res", type=float, default=2.0,
                        help="Coarse resolution in Å (default 2.0)")
    parser.add_argument("--fine_res",   type=float, default=1.0,
                        help="Fine resolution in Å (default 1.0)")
    parser.add_argument("--top_k",      type=int,   default=500,
                        help="Rotations to carry to fine pass (default 500)")
    args = parser.parse_args()

    cases, skipped = load_cases(args.json, args.pdb_root)
    if not cases:
        print("No cases loaded. Check --json and --pdb_root.")
        raise SystemExit(1)

    docker  = MultiResDocker(
        angular_step = args.step,
        coarse_res   = args.coarse_res,
        fine_res     = args.fine_res,
        top_k        = args.top_k,
    )
    print(docker.summary())

    results = docker.dock(cases[0])

    print("\nTop 5 Poses:")
    for i, r in enumerate(results[:5]):
        t = r.translation_vector
        print(
            f"  Rank {i+1}: score={r.score:>10.2f}  "
            f"t=[{t[0]:>6.1f}, {t[1]:>6.1f}, {t[2]:>6.1f}] Å"
        )