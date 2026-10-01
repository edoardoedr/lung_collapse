# Collapsed lung pipeline

Workflow for building a GNN surrogate model of collapsed lung anatomy in VATS surgery, from CT segmentation through inverse FEM pressure fitting.

## Overview

![Pipeline overview](collapsed_lung_project_workflow.svg)

The pipeline is being rewritten step by step in [`pipeline/`](pipeline/). The original scripts are kept in [`pipeline_codes_v1/`](pipeline_codes_v1/) until every step has been ported.

| Step | Status |
|---|---|
| 1 Segmentation | manual, 3D Slicer (nnInteractive + manual correction) |
| 2 Surface meshing | ✅ `pipeline/surface_mesh.py` |
| 3 Hilum anchor | ✅ `pipeline/hilum.py` (computed from the airway / vessel segmentations) |
| 4 Registration | ✅ `pipeline/registration.py` (replaces the Slicer Elastix workflow) |
| 5–8 Correspondence, inverse FEM, sequence, visualisation | still in `pipeline_codes_v1/` |

## Repository layout

```
main.py               entry point, runs the steps listed in a config
configs/
  patient_<N>.json    one config per patient, one section per step
  elastix/            elastix parameter files (SlicerElastix default preset)
pipeline/
  config.py           config loading (dataclass per step, path resolution)
  data_io.py          surfaces (LPS/RAS aware), labelmaps, Slicer markups
  surface_mesh.py     step 2
  hilum.py            step 3
  registration.py     step 4
Input_Data/           patient data (not in git)
results/              pipeline outputs (not in git)
pipeline_codes_v1/    original scripts
```

## Setup

```bash
pip install -r requirements.txt
```

Requires Python ≥ 3.10.

## Running

```bash
python main.py --config configs/patient_10.json                      # steps listed in the config
python main.py --config configs/patient_10.json --steps hilum        # only some steps
```

Steps always run in pipeline order: `surface_mesh` → `hilum` → `registration`. The hilum step runs before the registration because the registration uses it as landmarks. If a step fails, the error is written to `logs/pipeline.log` and the run stops.

## Input data

One folder per patient in `Input_Data/`. All files are surface models exported from 3D Slicer (`.vtk`, LPS), from the same scan:

| File | Used by |
|---|---|
| `lung_left_collapsed.vtk` | collapsed left lung, steps 2 and 3 |
| `lung_left.vtk` | assumed inflated left lung, step 4 |
| `lung_airways.vtk`, `lung_arteries.vtk`, `lung_veins.vtk` | step 3 |

## Configuration

`configs/patient_<N>.json` has the top-level keys `patient`, `data_dir`, `output_dir` and `steps`, plus one section per step. Each key is documented in the step's dataclass in [`pipeline/config.py`](pipeline/config.py).

Path rules:

- `data_dir`, `output_dir` and the elastix parameter files are relative to the JSON file.
- Step inputs are relative to `data_dir`, step outputs relative to `output_dir`.
- An input set to `null` is taken from the previous step's output.

An unknown key raises an error, so a typo cannot be silently ignored. Keys starting with `_` are comments.

## Steps

### 2 · Surface meshing (`surface_mesh`)

- **Input:** dense collapsed-lung surface from Slicer (`lung_left_collapsed.vtk`).
- **Output:** `lung_collapsed_mesh.vtp`.
- **Method:**
  - clean the surface: weld duplicate points, keep the largest component, Taubin smoothing, fill holes;
  - remesh uniformly with ACVD (pyacvd) to ~480 nodes / 956 triangles;
  - light post-smoothing and a watertightness check.
- **Result:** the node set used by every later step.

### 3 · Hilum anchor (`hilum`)

- **Inputs:** collapsed lung surface + airway, artery and vein surfaces.
- **Output:** `hilum_anchor.mrk.json` (LPS) with 4 point lists of one point each:
  - `airways`, `arteries`, `veins`: centre and radius of the ring where that tree enters the lung;
  - `hilum`: centroid of the three ring centres, with radius = mean of the three ring radii.

  Slicer gives every point of a list the same glyph size, so each point is its own list. In Slicer each one shows as a sphere with its diameter, and the radius is also in the point description (`radius_mm=…`).
- **Method:** exact surface–surface intersection of each tree with the lung surface. This gives closed rings, and per structure the largest ring (by perimeter) is the hilar one. The others are small peripheral branches.

### 4 · Registration (`registration`)

- **Inputs:**
  - the inflated lung (`lung_left.vtk`), rasterised to a mask;
  - the step-2 mesh, rasterised as the collapsed mask. It is also the mesh that gets warped.
- **Output:** `lung_inflated_mesh.vtp`. It has the same nodes and faces as `lung_collapsed_mesh.vtp`, moved to the inflated shape, so node *i* of the two meshes corresponds. The point array `RegistrationDisplacement_mm` holds the per-node displacement.
- **Method:**
  - elastix rigid + B-spline with the SlicerElastix default preset (`configs/elastix/`), fixed = inflated, moving = collapsed, as in the Slicer workflow;
  - elastix returns the fixed → moving (resampling) transform, so every node is mapped through its inverse. The inverse is solved per node: lookup start point + damped Gauss-Newton. This is what Slicer does when hardening a transform on a model.
  - `direction: "forward"` (fixed = collapsed, no inversion) is available but gave worse registrations on the test patients.
  - **Hilum landmarks.** Collapsed and inflated lung come from the same scan, and the hilum is assumed fixed, so the four points of `hilum_anchor.mrk.json` should map onto themselves. They are added to elastix through the `CorrespondingPointsEuclideanDistanceMetric` (keys `landmarks`, `landmark_points`, `landmark_weight`; weight 0 = QA only).
    - On patient_10, weight 0.001 keeps the shape match unchanged (Dice 0.977) while the hilum moves 0.4 mm instead of 12.9 mm.
    - Weights ≥ 0.01 pin the points but spoil the shape match (Dice 0.94 → 0.70), because the landmark term dominates the optimiser.
  - **Non-converged nodes.** Where the transform is near-singular, the inverse may not converge at some nodes. Up to `max_interpolated_nodes` (default 10) of them get their displacement by harmonic interpolation from the neighbouring nodes on the mesh. They are listed in the log and flagged in the point array `Interpolated`. With more of them, the step fails.
- **QA in the log:** image and mesh Dice, volume, displacement, flipped triangles, inversion residual, landmark error, and mean displacement of the nodes near the hilum.

## Outputs

```
results/patient_<N>/
├── lung_collapsed_mesh.vtp     collapsed lung, remeshed
├── lung_inflated_mesh.vtp      inflated lung, same nodes (registration)
├── hilum_anchor.mrk.json       3 ring centres + hilum, each a sphere with its radius
├── logs/                       pipeline.log, config_used.json
├── registration/               masks, elastix log and transforms (TransformParameters.*-Composite.h5 loads in Slicer)
└── hilum/                      rings.vtp (all rings, cell data Structure / Hilar), hilum_anchor.json
```

All surfaces are in LPS, with the space stored in the file.

To view in 3D Slicer, drag and drop `lung_collapsed_mesh.vtp`, `lung_inflated_mesh.vtp`, `hilum_anchor.mrk.json` (loads as 4 point lists) and `hilum/rings.vtp` (colour by `Hilar`).

## Known issues

- **patient_2 registration is unreliable.** The lung goes from 712 to 5351 mL (×7.5).
  - The inflated → collapsed transform is near-singular at 4 of 480 nodes; these are now interpolated.
  - The warped mesh still has flipped triangles (7 with landmarks, 2 without), mostly within 30 mm of the hilum.
  - The nodes near the hilum move ~90–100 mm.
  - Bending-energy regularisation, the forward direction and signed distance maps did not help. Still open.
- **Node correspondence is not anatomical.** Mask registration only matches the boundaries, so nodes can slide tangentially along the surface. Keep this in mind for the point-to-point loss of the inverse FEM.

## Inverse FEM pressure fitting (step 6 detail, legacy code)

![Inverse FEM detail](inverse_fem_pressure_fit_detail.svg)
