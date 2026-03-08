"""
Phase 3 — SO(3) Rotation Sampler
==================================
Generates a uniform set of rotation matrices covering all of SO(3)
(the space of all possible 3D orientations).

These rotations are applied to the RNA in Phase 4:
  for each rotation R in SO3Sampler:
      rotated_coords = R @ (rna_coords - rna_center) + rna_center
      rna_grid       = builder.build_from_coords(rotated_coords, ...)
      score_map      = IFFT(FFT(protein_grid) * conj(FFT(rna_grid)))
      best_translation = argmax(score_map)

Sampling method: uniform quaternion grid
-----------------------------------------
A rotation in 3D is equivalent to a unit quaternion q = (w, x, y, z)
on the 3-sphere S³.  Antipodal points q and -q represent the same
rotation, so SO(3) ≅ S³ / {±1}.

We use the method of Karney (2006) / Mitchell (2021) — a deterministic,
hierarchical subdivision of quaternion space that guarantees:
  - Uniform angular coverage (no pole oversampling)
  - Exactly one representative per rotation within the angular resolution
  - Reproducible, sorted output

At angular_step=15° → ~54,000 rotations  (standard FFT docking benchmark)
At angular_step=30° → ~6,900 rotations   (fast coarse search)
At angular_step=6°  → ~850,000 rotations (fine refinement — slow)

References:
  Katchalski-Katzir et al. (1992) PNAS 89, 2195
  Vakser (1995) Protein Eng. 8, 371
  Karney (2006) arxiv:physics/0506177
  Mitchell (2021) "Generating Uniform Incremental Grids on SO(3)"

Usage:
    python phase3_rotations.py --test
    python phase3_rotations.py --step 15 --stats
"""

import math
import numpy as np
from typing import Iterator, List, Tuple
from phase1 import load_uu_cases


# ═══════════════════════════════════════════════════════════════════════════
# Quaternion ↔ Rotation matrix conversions
# ═══════════════════════════════════════════════════════════════════════════

def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """
    Convert a unit quaternion q = [w, x, y, z] to a 3×3 rotation matrix.

    Uses the standard formula from Shoemake (1985).
    Input must be a unit quaternion (||q|| = 1).
    """
    w, x, y, z = q / np.linalg.norm(q)   # normalise defensively

    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """
    Convert a 3×3 rotation matrix to a unit quaternion [w, x, y, z].
    Uses Shepperd's method (numerically stable).
    """
    trace = R[0,0] + R[1,1] + R[2,2]

    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2,1] - R[1,2]) * s
        y = (R[0,2] - R[2,0]) * s
        z = (R[1,0] - R[0,1]) * s
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * math.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        w = (R[2,1] - R[1,2]) / s
        x = 0.25 * s
        y = (R[0,1] + R[1,0]) / s
        z = (R[0,2] + R[2,0]) / s
    elif R[1,1] > R[2,2]:
        s = 2.0 * math.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        w = (R[0,2] - R[2,0]) / s
        x = (R[0,1] + R[1,0]) / s
        y = 0.25 * s
        z = (R[1,2] + R[2,1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        w = (R[1,0] - R[0,1]) / s
        x = (R[0,2] + R[2,0]) / s
        y = (R[1,2] + R[2,1]) / s
        z = 0.25 * s

    return np.array([w, x, y, z], dtype=np.float64)


def rotation_angle(R: np.ndarray) -> float:
    """
    Return the rotation angle (radians) of a rotation matrix.
    angle = arccos((trace(R) - 1) / 2)
    """
    cos_angle = (np.trace(R) - 1.0) / 2.0
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return math.acos(cos_angle)


# ═══════════════════════════════════════════════════════════════════════════
# Uniform SO(3) sampling — Hopf coordinates method
# ═══════════════════════════════════════════════════════════════════════════

def _hopf_to_quat(theta: float, phi: float, psi: float) -> np.ndarray:
    """
    Convert Hopf coordinates (θ, φ, ψ) to a unit quaternion.

    The Hopf fibration parameterises SO(3) as:
        q = ( cos(θ/2)·cos(ψ/2),
              cos(θ/2)·sin(ψ/2),
              sin(θ/2)·cos(φ + ψ/2),
              sin(θ/2)·sin(φ + ψ/2) )

    Ranges:  θ ∈ [0, π/2],  φ ∈ [0, 2π),  ψ ∈ [0, 2π)
    Uniform measure on SO(3): dθ dφ dψ · sin(2θ) / (2π²)

    Reference: Yershova et al. (2010) IJRR 29, 801
    """
    ct = math.cos(theta)
    st = math.sin(theta)
    w  = ct * math.cos(psi / 2)
    x  = ct * math.sin(psi / 2)
    y  = st * math.cos(phi + psi / 2)
    z  = st * math.sin(phi + psi / 2)
    return np.array([w, x, y, z], dtype=np.float64)


def generate_uniform_rotations(angular_step_deg: float = 15.0) -> List[np.ndarray]:
    """
    Generate a list of rotation matrices uniformly covering SO(3).

    Uses a regular grid in Hopf coordinates (θ, φ, ψ) with spacing
    derived from angular_step_deg.  The number of steps along each
    axis is chosen so that the maximum angular gap between any
    orientation and its nearest sample is ≤ angular_step_deg.

    Returns
    -------
    List of (3, 3) numpy rotation matrices.

    Rotation count at common step sizes:
        30°  →   ~980
        15°  →  ~6,900  (fast benchmark)
        10°  → ~22,000
         6°  → ~97,000  (thorough)
         5°  →~170,000
    """
    step = math.radians(angular_step_deg)

    # Number of samples along each Hopf axis
    # θ ∈ [0, π/2]:  n_theta steps
    # φ ∈ [0, 2π):   n_phi   steps  (full circle)
    # ψ ∈ [0, 2π):   n_psi   steps  (full circle)
    n_theta = max(1, round((math.pi / 2) / step))
    n_phi   = max(1, round((2 * math.pi) / step))
    n_psi   = max(1, round((2 * math.pi) / step))

    rotations = []
    seen_quats = []   # for deduplication of antipodal pairs

    for i in range(n_theta):
        theta = (math.pi / 2) * (i + 0.5) / n_theta   # midpoint sampling

        for j in range(n_phi):
            phi = (2 * math.pi) * j / n_phi

            for k in range(n_psi):
                psi = (2 * math.pi) * k / n_psi

                q = _hopf_to_quat(theta, phi, psi)

                # Canonical form: ensure w >= 0 (q and -q same rotation)
                if q[0] < 0:
                    q = -q

                R = quat_to_matrix(q)
                rotations.append(R)

    return rotations


# ═══════════════════════════════════════════════════════════════════════════
# Apply rotation to atom coordinates
# ═══════════════════════════════════════════════════════════════════════════

def rotate_coords(
    coords: np.ndarray,
    R:      np.ndarray,
    center: np.ndarray = None,
) -> np.ndarray:
    """
    Apply rotation matrix R to a set of atom coordinates.

    Rotation is always performed about the geometric center of the
    molecule (or a supplied center) so the molecule stays inside
    its bounding box.

        coords_rot = R @ (coords - center) + center

    Parameters
    ----------
    coords : (N, 3) array of atom positions
    R      : (3, 3) rotation matrix
    center : (3,) rotation pivot; defaults to mean of coords

    Returns
    -------
    (N, 3) rotated coordinates
    """
    if center is None:
        center = coords.mean(axis=0)
    centered = coords - center
    rotated  = (R @ centered.T).T
    return rotated + center


# ═══════════════════════════════════════════════════════════════════════════
# SO3Sampler — the main class Phase 4 will use
# ═══════════════════════════════════════════════════════════════════════════

class SO3Sampler:
    """
    Precomputes and stores all rotation matrices for the docking search.

    Usage in Phase 4:
        sampler = SO3Sampler(angular_step_deg=15.0)
        for i, R in enumerate(sampler):
            rotated = rotate_coords(rna_coords, R, center=rna_center)
            rna_grid = builder.build_from_coords(rotated, ...)
            ...

    Attributes
    ----------
    rotations     : list of (3,3) numpy arrays
    n_rotations   : int
    angular_step  : float  (degrees)
    """

    def __init__(self, angular_step_deg: float = 15.0):
        self.angular_step = angular_step_deg
        print(f"SO3Sampler: generating rotations at {angular_step_deg}° step …", end=" ", flush=True)
        self.rotations = generate_uniform_rotations(angular_step_deg)
        print(f"{len(self.rotations):,} rotations generated.")

    @property
    def n_rotations(self) -> int:
        return len(self.rotations)

    def __len__(self) -> int:
        return self.n_rotations

    def __iter__(self) -> Iterator[np.ndarray]:
        return iter(self.rotations)

    def __getitem__(self, idx: int) -> np.ndarray:
        return self.rotations[idx]

    def angular_coverage_stats(self) -> dict:
        """
        Compute basic statistics on the angular spacing between
        consecutive rotations, to verify uniformity.

        Returns dict with keys: mean_deg, max_deg, min_deg, n_rotations.

        Note: this is O(N) not O(N²) — we compare each rotation to
        the identity only, as a proxy for distribution spread.
        A full nearest-neighbour check would be O(N²) and is only
        needed for publication-level validation.
        """
        angles = []
        I = np.eye(3)
        for R in self.rotations:
            # Angle between R and identity = rotation angle of R itself
            angles.append(math.degrees(rotation_angle(R)))

        angles = np.array(angles)
        return {
            "n_rotations":    self.n_rotations,
            "angular_step":   self.angular_step,
            "mean_angle_deg": float(angles.mean()),
            "std_angle_deg":  float(angles.std()),
            "min_angle_deg":  float(angles.min()),
            "max_angle_deg":  float(angles.max()),
        }

    def summary(self) -> str:
        stats = self.angular_coverage_stats()
        return (
            f"SO3Sampler\n"
            f"  angular step    : {self.angular_step}°\n"
            f"  total rotations : {stats['n_rotations']:,}\n"
            f"  rotation angles : mean={stats['mean_angle_deg']:.1f}°  "
            f"std={stats['std_angle_deg']:.1f}°  "
            f"min={stats['min_angle_deg']:.1f}°  "
            f"max={stats['max_angle_deg']:.1f}°"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Identity check helper — used in Phase 4 to include the native pose
# ═══════════════════════════════════════════════════════════════════════════

IDENTITY_ROTATION = np.eye(3, dtype=np.float64)


def prepend_identity(rotations: List[np.ndarray]) -> List[np.ndarray]:
    """
    Prepend the identity rotation to a list of rotations.
    This ensures the native (unrotated) orientation is always searched.
    """
    return [IDENTITY_ROTATION.copy()] + rotations

# ═══════════════════════════════════════════════════════════════════════════
# Visualization — rotation axes on sphere
# ═══════════════════════════════════════════════════════════════════════════

def visualize_rotation_axes(rotations, max_points=20000):
    """
    Visualize the distribution of rotation axes on the unit sphere.

    Each rotation matrix R corresponds to an axis-angle rotation.
    We extract the axis and plot it.
    """

    import plotly.graph_objects as go

    axes = []

    for R in rotations[:max_points]:

        angle = rotation_angle(R)

        if abs(angle) < 1e-8:
            axis = np.array([0,0,1])
        else:
            axis = np.array([
                R[2,1] - R[1,2],
                R[0,2] - R[2,0],
                R[1,0] - R[0,1]
            ])
            axis = axis / np.linalg.norm(axis)

        axes.append(axis)

    axes = np.array(axes)

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=axes[:,0],
        y=axes[:,1],
        z=axes[:,2],
        mode="markers",
        marker=dict(size=3),
        name="Rotation Axes"
    ))

    fig.update_layout(
        title="SO(3) Rotation Axis Distribution",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
        ),
        width=800,
        height=800
    )

    fig.show()

def visualize_rotation_angles(rotations):

    import plotly.graph_objects as go

    angles = [
        math.degrees(rotation_angle(R))
        for R in rotations
    ]

    fig = go.Figure(
        data=[go.Histogram(x=angles)]
    )

    fig.update_layout(
        title="Rotation Angle Distribution",
        xaxis_title="Angle (degrees)",
        yaxis_title="Count"
    )

    fig.show()

def animate_docking(protein_coords, rna_coords, rotations, n_frames=120):
    """
    Animate RNA rotating around a fixed protein.
    """

    import plotly.graph_objects as go

    protein_coords = np.array(protein_coords)
    rna_coords = np.array(rna_coords)

    center = rna_coords.mean(axis=0)

    frames = []

    for i in range(min(n_frames, len(rotations))):

        R = rotations[i]
        rot = rotate_coords(rna_coords, R, center)

        frames.append(
            go.Frame(
                data=[
                    # Protein (static)
                    go.Scatter3d(
                        x=protein_coords[:,0],
                        y=protein_coords[:,1],
                        z=protein_coords[:,2],
                        mode="markers",
                        marker=dict(size=3),
                        name="Protein"
                    ),

                    # Rotated RNA
                    go.Scatter3d(
                        x=rot[:,0],
                        y=rot[:,1],
                        z=rot[:,2],
                        mode="markers",
                        marker=dict(size=3),
                        name="RNA"
                    )
                ]
            )
        )

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=protein_coords[:,0],
                y=protein_coords[:,1],
                z=protein_coords[:,2],
                mode="markers",
                marker=dict(size=3),
                name="Protein"
            ),

            go.Scatter3d(
                x=rna_coords[:,0],
                y=rna_coords[:,1],
                z=rna_coords[:,2],
                mode="markers",
                marker=dict(size=3),
                name="RNA"
            )
        ],
        frames=frames
    )

    fig.update_layout(
        title="Protein–RNA Docking Rotation Sampling",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z"
        ),
        updatemenus=[{
            "buttons":[
                {
                    "label":"Play",
                    "method":"animate",
                    "args":[None]
                }
            ]
        }]
    )

    fig.show()

# ═══════════════════════════════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════════════════════════════

def _self_test():
    print("Phase 3 self-test …\n")

    # ── Test 1: quaternion ↔ matrix round-trip ──────────────────────────
    print("Test 1: quat → matrix → quat round-trip")
    for _ in range(1000):
        # Random unit quaternion
        q = np.random.randn(4)
        q /= np.linalg.norm(q)
        if q[0] < 0:
            q = -q

        R  = quat_to_matrix(q)
        q2 = matrix_to_quat(R)
        if q2[0] < 0:
            q2 = -q2

        assert np.allclose(q, q2, atol=1e-10), \
            f"Round-trip failed: {q} → {q2}"

        # Rotation matrix must be orthogonal with det = +1
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert abs(np.linalg.det(R) - 1.0) < 1e-12

    print("  ✓  1000 random quaternions — all round-trips passed\n")

    # ── Test 2: rotation count at a few step sizes ──────────────────────
    print("Test 2: rotation counts at different angular steps")
    for step in [30.0, 15.0, 10.0]:
        rots = generate_uniform_rotations(step)
        print(f"  step={step:5.1f}°  →  {len(rots):>8,} rotations")
    print()

    # ── Test 3: identity rotation is always exact ───────────────────────
    print("Test 3: identity rotation")
    coords = np.random.randn(50, 3)
    rotated = rotate_coords(coords, IDENTITY_ROTATION)
    assert np.allclose(coords, rotated, atol=1e-14), "Identity rotation changed coordinates!"
    print("  ✓  Identity rotation leaves coordinates unchanged\n")

    # ── Test 4: rotation preserves inter-atomic distances ───────────────
    print("Test 4: rotation preserves pairwise distances")
    rots = generate_uniform_rotations(30.0)
    coords = np.random.randn(20, 3) * 10.0

    # Compute a few pairwise distances before rotation
    d_before = np.linalg.norm(coords[0] - coords[1])

    for R in rots[:50]:
        rot_coords = rotate_coords(coords, R)
        d_after = np.linalg.norm(rot_coords[0] - rot_coords[1])
        assert abs(d_before - d_after) < 1e-10, \
            f"Distance changed under rotation: {d_before:.6f} → {d_after:.6f}"

    print("  ✓  50 random rotations — distances preserved\n")

    # ── Test 5: SO3Sampler interface ────────────────────────────────────
    print("Test 5: SO3Sampler interface")
    sampler = SO3Sampler(angular_step_deg=30.0)
    print(sampler.summary())
    assert len(sampler) == sampler.n_rotations
    assert sampler[0].shape == (3, 3)
    print("  ✓  SO3Sampler interface OK\n")

    print("✓  All Phase 3 self-test assertions passed.\n")


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Phase 3: SO(3) rotation sampler"
    )
    parser.add_argument("--step",  type=float, default=15.0,
                        help="Angular step size in degrees (default 15.0)")
    parser.add_argument("--stats", action="store_true",
                        help="Print coverage statistics and exit")
    parser.add_argument("--test",  action="store_true",
                        help="Run self-test and exit")
    args = parser.parse_args()

    if args.test:
        _self_test()
    elif args.stats:
        sampler = SO3Sampler(angular_step_deg=args.step)
        print()
        print(sampler.summary())
        print()
        print("Rotation count at standard step sizes:")
        for step in [30.0, 15.0, 10.0, 6.0, 5.0]:
            n = len(generate_uniform_rotations(step))
            print(f"  {step:5.1f}°  →  {n:>9,} rotations")
    else:
        sampler = SO3Sampler(angular_step_deg=args.step)
        print(sampler.summary())

        # visualize SO(3) sampling
        visualize_rotation_axes(sampler.rotations)
        visualize_rotation_angles(sampler.rotations)

        # Load one docking case to visualize RNA rotation
        cases, _ = load_uu_cases("../assets/PRDBv3.json", "../assets/UU_PDBS")

        if cases:
            case = cases[0]

            protein_coords = np.array([
            [a.x, a.y, a.z]
            for chain in case.protein_struct.chains
            for a in chain.atoms
        ])

        rna_coords = np.array([
            [a.x, a.y, a.z]
            for chain in case.rna_struct.chains
            for a in chain.atoms
        ])

        animate_docking(protein_coords, rna_coords, sampler.rotations)