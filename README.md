# FFT-Scorer — Documentation

**Goal:**
Predict the docking pose of an RNA molecule binding to a protein using a **Fast Fourier Transform (FFT)–based rigid docking algorithm**.

The pipeline reads structural data from PDB files, converts molecules into 3D voxel grids, samples molecular rotations, evaluates docking scores using FFT correlation, and outputs the best predicted complexes.

---

## Pipeline Overview

```
PDB files
   ↓
Parse structures
   ↓
Voxel grid encoding
   ↓
Rotation sampling
   ↓
FFT correlation scoring
   ↓
Top docking poses
   ↓
Clustering + RMSD evaluation
```

---

## Phase 1 — PDB Parser & Preprocessor

**Purpose:**
Load and parse the protein–RNA structures required for docking.

**Steps**

1. Read docking dataset metadata (`PRDBv3.json`)
2. Filter **UU docking cases** (unbound protein + unbound RNA).
3. Locate PDB files from folder structure.
4. Parse PDB atom records.

**Chain identification**

* **Protein chains**

  * Residues: standard amino acids (`ALA`, `ARG`, etc.)
  * Records: `ATOM`

* **RNA chains**

  * Residues: `A`, `U`, `G`, `C` and variants (`RA`, `RG`, etc.)
  * Records: `ATOM` or `HETATM`

The parser automatically determines whether each chain is **protein, RNA, or unknown**.

**Output**

```
DockingCase
 ├── protein structure
 ├── RNA structure
 └── complex structure (ground truth)
```

---

## Phase 2 — Grid Builder (FFT Core Representation)

**Purpose:**
Convert atomic coordinates into a **3D voxel grid representation** suitable for FFT correlation.

**Steps**

1. Center molecules at the origin.
2. Define a **3D grid** (typically `128 × 128 × 128`).
3. Use a **grid spacing** (commonly `1 Å`).
4. Assign grid values based on atoms.

**Encodings**

| Feature                   | Description                   |
| ------------------------- | ----------------------------- |
| Shape                     | interior vs surface occupancy |
| Electrostatics (optional) | partial charge distribution   |

The result is a **3D numerical grid** for both molecules.

---

## Phase 3 — Rotation Sampling

**Purpose:**
Explore different orientations of the ligand (RNA).

Because rigid docking requires testing many orientations, the RNA molecule is rotated using **uniform SO(3) sampling**.

Typical sampling:

```
~54,000 rotations
```

Methods:

* Euler angle grid
* Quaternion-based uniform sampling

Each rotation produces a **rotated RNA grid**.

---

## Phase 4 — FFT Scorer (Core Algorithm)

**Purpose:**
Efficiently evaluate **all translations simultaneously** using FFT correlation.

For each sampled rotation:

```
1. Rotate RNA grid
2. Compute FFT(protein grid)
3. Compute FFT(RNA grid)
4. Multiply in frequency space
5. Inverse FFT
```

Mathematically:

```
Score = IFFT( FFT(protein) * conj(FFT(RNA)) )
```

Result:

A **3D score volume** representing docking scores for every translation.

The algorithm stores the **Top-K highest scoring poses**.

Each pose includes:

```
(rotation, translation, score)
```

---

## Phase 5 — Clustering & Output

**Purpose:**
Remove redundant poses and produce final docking predictions.

**Steps**

1. Collect top scoring poses.
2. Perform **RMSD-based clustering**.
3. Select cluster centers as final predictions.
4. Generate predicted **protein–RNA complex PDB files**.

**Evaluation**

Predicted structures are compared with the known bound complex using:

```
RMSD (Root Mean Square Deviation)
```

Lower RMSD indicates better docking accuracy.

---

## Final Output

For each docking case:

```
Top predicted docking poses
Clustered predictions
Predicted complex PDB files
RMSD vs ground truth
```

---

## Typical Parameters

| Parameter      | Typical Value |
| -------------- | ------------- |
| Grid size      | 128³          |
| Grid spacing   | 1 Å           |
| Rotations      | ~54,000       |
| Top poses kept | 100–1000      |

---
