# Collapsed lung pipeline

Workflow for building a GNN surrogate model of collapsed lung anatomy in VATS surgery, from CT segmentation through inverse FEM pressure fitting.

## Overview

![Pipeline overview](collapsed_lung_project_workflow.svg)

The pipeline is being rewritten step by step in [`pipeline/`](pipeline/). The original scripts are kept in [`pipeline_codes_v1/`](pipeline_codes_v1/) until every step has been ported.

| Step | Status |
|---|---|
| 1 Segmentation | manual, 3D Slicer (nnInteractive + manual correction) |
| 1b Input check | ✅ `pipeline/check_inputs.py` (sanity check of the input surfaces) |
| 2 Surface meshing | ✅ `pipeline/surface_mesh.py` |
| 3 Hilum anchor | ✅ `pipeline/hilum.py` (computed from the airway / vessel segmentations) |
| 4 Registration | ✅ `pipeline/registration.py` (replaces the Slicer Elastix workflow) |
| 5 Inverse FEM | `pipeline/collapse/`: `fem_setup` ✅, `fem_fit` ported, GetFEM core not yet run in the new pipeline |
| 6–7 Sequence, visualisation | still in `pipeline_codes_v1/` |

## Repository layout

```
main.py               entry point, runs the steps listed in a config
configs/
  patient_<N>.json    one config per patient, one section per step
  README.md           every config key: meaning and range
  elastix/            elastix parameter files (SlicerElastix default preset + an affine stage)
pipeline/
  config.py           config loading (dataclass per step, path resolution)
  data_io.py          surfaces (LPS/RAS aware), labelmaps, Slicer markups
  check_inputs.py     step 1b
  surface_mesh.py     step 2
  hilum.py            step 3
  registration.py     step 4
  collapse/           step 5, inverse FEM
    setup.py          fem_setup: clamp, alignment, regions, volume mesh (no FEM library)
    fit.py            fem_fit: coarse-to-fine pressure fit, talks only to solvers/base.py
    problem.py        CollapseProblem, the arrays passed from setup to fit and to the core
    anchor.py, regions.py, volume_mesh.py, geometry.py, run_control.py
    solvers/          FEM cores: base.py (interface), common.py (options, material constants),
                      getfem_solver.py; nodal.py (Newton, load path, wall, linear solvers,
                      shared by) warp_solver.py and torch_solver.py; materials.py (torch materials)
Input_Data/           patient data (not in git)
results/              pipeline outputs (not in git)
pipeline_codes_v1/    original scripts
```

## Setup

```bash
pip install -r requirements.txt
```

Requires Python ≥ 3.10.

`fem_fit` also needs the library of the FEM core selected in the config, which is not on PyPI and is imported only by that step:

- **GetFEM:** `conda install -c conda-forge getfem` (linux-64, osx-64, win-64), or on Ubuntu `apt install python3-getfem` with the system Python (built against `numpy<2`).
- **Warp:** `pip install warp-lang`, optionally `"nvmath-python[cu12]"` (cuDSS on GPU) and `pypardiso`.
- **PyTorch:** `pip install torch` (≥ 2.0, for `torch.func`), same optional linear solvers as Warp.

## Running

```bash
python main.py --config configs/patient_10.json                      # steps listed in the config
python main.py --config configs/patient_10.json --steps hilum        # only some steps
python main.py --config configs/patient_10.json --steps fem_fit      # e.g. on the machine with GetFEM
```

Steps always run in pipeline order: `check_inputs` → `surface_mesh` → `hilum` → `registration` → `fem_setup` → `fem_fit`. The hilum step runs before the registration because the registration uses it as landmarks. If a step fails, the error is written to `logs/pipeline.log` and the run stops.

## Input data

One folder per patient in `Input_Data/`. All files are surface models exported from 3D Slicer (`.vtk`, LPS), from the same scan:

| File | Used by |
|---|---|
| `lung_left_collapsed.vtk` | collapsed left lung, steps 2 and 3 |
| `lung_left.vtk` | assumed inflated left lung, step 4 |
| `lung_airways.vtk`, `lung_arteries.vtk`, `lung_veins.vtk` | step 3 |

## Configuration

`configs/patient_<N>.json` has the top-level keys `patient`, `data_dir`, `output_dir` and `steps`, plus one section per step. **Every key, with its purpose and range, is explained in [`configs/README.md`](configs/README.md).**

Path rules:

- `data_dir`, `output_dir` and the elastix parameter files are relative to the JSON file.
- Step inputs are relative to `data_dir`, step outputs relative to `output_dir`.
- An input set to `null` is taken from the previous step's output.

An unknown key raises an error, so a typo cannot be silently ignored. Keys starting with `_` are comments.

## Steps

### 1b · Input check (`check_inputs`)

Runs first, on the raw surfaces (by default the files named in the other sections), so segmentation problems show up before registration and Gmsh fail on them. Findings are warnings with the position in LPS and RAS (as shown in Slicer) and a hint on how to fix them; with `fail_on_warning: true` they stop the pipeline.

- **Integrity:** readable, space tag, closed, number of pieces and volume. Extra pieces of a lung (islands, internal holes) are a finding; the following checks use the largest piece. Airway and vessel trees are normally in many pieces, so for them this is only logged.
- **Same scan:** every structure overlaps the collapsed lung.
- **Containment:** the collapsed lung must lie inside the inflated one. Points more than `outside_tol_mm` outside are flagged, warning above `max_outside_fraction`. Fix in Segment Editor: *Logical operators → Add* the collapsed segment to the inflated one, after smoothing.
- **Narrow notches** (folds, open fissures), in both lungs: the mask is closed with a ball of `notch_radius_mm`; points deeper than `notch_depth_mm` inside the closed volume are flagged, warning above `max_notch_fraction`. They cause the flipped triangles in the registration and the self-intersections Gmsh refuses. Fix: *Smoothing → Closing*.
- **Volume ratio** inflated / collapsed above `max_volume_ratio`.
- **Output:** `checks/collapsed_check.vtp` (point data `DistanceToInflated_mm`, `NotchDepth_mm`), `checks/inflated_check.vtp` (`NotchDepth_mm`), `checks/check_summary.json`. Colour them in Slicer to find the spots to fix.

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
  - for a large volume ratio (patient_2, ×7.2) an affine stage goes between them (`Parameters_Affine.txt`, the rigid file with `AffineTransform`): the rigid stage cannot scale, and a B-spline doing the whole expansion folds. On patient_10 (×1.9) it is not needed and slightly worse;
  - both lungs are rasterised on one grid covering both (as the CT grid in Slicer); with separate tight grids the initial alignment can move samples outside the smaller image and elastix stops;
  - elastix returns the fixed → moving (resampling) transform, so every node is mapped through its inverse. The inverse is solved per node: lookup start point + damped Gauss-Newton. This is what Slicer does when hardening a transform on a model.
  - `direction: "forward"` (fixed = collapsed, no inversion) is available but gave worse registrations on the test patients.
  - **Hilum landmarks.** Collapsed and inflated lung come from the same scan, and the hilum is assumed fixed, so the four points of `hilum_anchor.mrk.json` should map onto themselves. They are added to elastix through the `CorrespondingPointsEuclideanDistanceMetric` (keys `landmarks`, `landmark_points`, `landmark_weight`; weight 0 = QA only).
    - On patient_10, weight 0.001 keeps the shape match unchanged (Dice 0.977) while the hilum moves 0.4 mm instead of 12.9 mm.
    - Weights ≥ 0.01 pin the points but spoil the shape match (Dice 0.94 → 0.70), because the landmark term dominates the optimiser.
  - **Non-converged nodes.** Where the transform is near-singular, the inverse may not converge at some nodes. Up to `max_interpolated_nodes` (default 10) of them get their displacement by harmonic interpolation from the neighbouring nodes on the mesh. They are listed in the log and flagged in the point array `Interpolated`. With more of them, the step fails.
  - **Flipped triangles.** Where the transform folds, some warped triangles turn over (normal rotated by more than 90°) and the surface intersects itself, which Gmsh refuses. The displacement of their nodes is re-interpolated from the neighbours in the same way, growing the region by one ring of nodes until no triangle is flipped. Up to `max_unflip_nodes` (default 40) nodes; with more, the registration folds over a large region and the step fails. These nodes have `Interpolated` = 2 (1 = inversion not converged, 0 = registration).
- **QA in the log:** image and mesh Dice, volume, displacement, flipped triangles, inversion residual, landmark error, and mean displacement of the nodes near the hilum.

### 5 · Inverse FEM (`fem_setup`, `fem_fit`)

Finds the regional pleural pressures that deform the inflated lung (reference) onto the collapsed one (target), with the hilum clamped. Port of `pipeline_codes_v1/6.1_lung_inverse_fem_fit.py` and of its wall variant `6.2_…_wall.py`, split in two steps so the FEM core can be exchanged.

**`fem_setup`** (seconds, no FEM library):

- **Inputs:** `lung_inflated_mesh.vtp` (reference), `lung_collapsed_mesh.vtp` (target) and `hilum_anchor.mrk.json`. Reference and target must have the same nodes and triangles, i.e. the registration output.
- **Clamped region:** surface triangles within `hilum sphere radius × anchor_radius_factor` of the hilum (default 2.0 → ~30 mm on patient_10). The radius grows while the region has fewer than `anchor_min_points` points or is nearly flat, otherwise rigid motions would stay free.
- **Alignment:** `align: "hilum_rigid"` moves the target rigidly so that its clamped points best match the reference ones (consistent with u = 0 there); `rigid` uses all points, `none` keeps it as is.
- **Cavity wall (optional, `wall`):** closed surface the lung may not leave during the collapse (the chest cavity). `"reference"` (used in the configs) takes the registered inflated surface itself, so every point starts on it; a file name (e.g. `lung_left.vtk`) takes that surface. A point may go at most `wall_tol_mm` (2 mm) beyond it, plus however far it is already outside in the reference. The log reports how many target points lie beyond: those are an error the fit cannot remove. `wall: null` = no wall (as 6.1).
- **Pressure regions:** connectivity-constrained Ward clustering of the displacement with the rigid part removed (`cluster_field`), for every level in `levels` (1 → 40 regions). The regions do not depend on the pressures, so they are all computed here.
- **Volume mesh:** Gmsh tetrahedra (`mesh_size_mm` inside), with the surface points and triangles unchanged (checked).
- **Output:** `fem/setup/problem.npz`, read by `fem_fit` and passed to the core, plus files to inspect: `reference.vtp` (cell data `Clamped`, `Regions_K*`; point data `WallDistance_mm`), `target_aligned.vtp`, `volume.vtu`, `setup_summary.json`.

**`fem_fit`** (hours with GetFEM):

- **FEM core:** `solver` names a class in `pipeline/collapse/solvers/` (`"getfem"`, `"warp"`, `"torch"`) or any `"module:Class"`; `solver_options` go to that class. Every core implements `ForwardSolver` (`solvers/base.py`): hyperelastic material with E = 1 (compressible Neo-Hookean; `torch` also others) (so the unknowns are p/E per region and ν), u = 0 on the clamp, follower pressure on the other triangles, positive pressure pushing inward, and the wall as rigid contact if the problem has one; `solve(q, nu)` returns the surface displacement. A core without wall support refuses a problem with a wall. To add a core, write one file in `solvers/` and register it in `solvers/__init__.py`; a P1 core that assembles the system itself only needs `_assemble` on top of `nodal.NodalSolver`.
- **Wall contact, `wall_update`:** `"outer"` (both cores) alternates Newton solves with the wall linearisation fixed and re-linearisations, up to `wall_max_updates` rounds until the gap changes less than `wall_settle_mm`; each solve is several Newton runs, which is why the fit with a wall is much slower. `"newton"` (Warp and torch) re-linearises the wall at every residual evaluation, so the contact is part of a single Newton solve; at convergence it solves the same equations. With a wall the configs also set `warm_substeps: true`, so a failed step is retried in sub-steps before restarting the ramp from zero.
- **GetFEM wall contact:** penalty force on the pressure faces, zero inside the wall and growing smoothly beyond it (`wall_stiffness`, `wall_eps`); the gap is re-linearised between Newton solves until it settles (`wall_max_updates`, `wall_settle_mm`). Warm start as in 6.1: one direct step from the last converged state, then a ramp from zero. `warm_substeps: true` adds, before the ramp, `load_steps` and 3×`load_steps` sub-steps from the last state (as 6.2): useful with contact, but each failed attempt costs Newton runs up to `newton_maxit`.
- **Method:** one sign check (q > 0 must collapse), then the levels coarse → fine, each warm-started from the best so far. Per level: bounded least squares (`lsq`; Jacobian from the core when `jacobian: "analytic"` and the core provides one, i.e. one linear solve per region instead of one nonlinear solve; `"2-point"` = finite differences as before) on the point-to-point error plus a smoothness penalty between adjacent regions (`reg`), or Nelder-Mead (`nm`); ν optionally free.
- **Stops:**
  - *a level* ends when the optimiser converges (`lsq_*` tolerances) or when its best error improved by less than `level_min_improve` (relative, default 0.5 %) over the last `level_patience` optimiser iterations (default 3, i.e. 3 × (parameters + 1) solves); the fit then moves to the next K. On karl04 this would have cut 84 to ~32 min with the same error up to K=14 (the lsq tolerances alone kept going 50–200 solves past the last improvement);
  - an *iteration* is one Jacobian for lsq, `parameters + 1` solves for Nelder-Mead;
  - *the whole fit* ends on `target_error_mm`, `time_budget_min`, `fit_patience` levels in a row improving by less than `fit_min_improve`, the end of the levels, or Ctrl+C / SIGTERM. The best result is checkpointed on improvement, and a watchdog kills the run at budget + `hard_grace_min`, so a forced stop keeps the best solution.
- **Warp core (`"solver": "warp"`):** same physics, options and load path as the GetFEM core, P1 only, assembled with NVIDIA Warp kernels in float64 on GPU (or CPU); extra options `device` and `linear_solver` (`auto`, `cudss`, `pardiso`, `scipy`). Newton reproduces GetFEM's classical Newton with the "simplest" line search, and the wall term uses the IM_TRIANGLE(3) face rule; both are to be confirmed against GetFEM with `python scripts/compare_warp_getfem.py --config configs/<p>.json [--timing] [--fit]` (forward solves with and without wall, Jacobian, full fit, timing).
- **PyTorch core (`"solver": "torch"`):** the Warp core with the assembly written in PyTorch. Everything except the assembly (Newton, load path, wall rounds, linear solvers, Jacobian) is shared with Warp in `solvers/nodal.py`. The material is only a strain-energy function W(F, ν) in `solvers/materials.py`; stress and tangent come from automatic differentiation, so a new material is a new function (see the docstring of `materials.py`). Extra options `device` (`cpu`, `cuda`; not `mps`), `material`, `material_params`. Validation: on a machine without GetFEM (e.g. the Mac, CPU) `python scripts/check_jacobian.py --config configs/<p>.json --solver torch`, which also tests the autodiff tangent; against GetFEM `python scripts/compare_warp_getfem.py --config configs/<p>.json --core torch`. `torch_solver.TorchFEM` is the assembly alone, differentiable and on any device and dtype (also `mps`, float32): the residual of a predicted displacement can serve as a physics loss for the GNN.
- **Analytic Jacobian (GetFEM):** at the converged state, `K_t dU/dq = d(rhs)/dq` on the free dofs, with one factorisation of the tangent matrix (pypardiso if installed, else SuperLU via scipy). The rhs is linear in each pressure, so each column is the pressure term assembled with q = 1; d/dν by a central difference of the rhs. With a wall the contact linearisation is held fixed (Gauss-Newton Jacobian). Check it against finite differences, and compare full fits, with `python scripts/check_jacobian.py --config configs/<p>.json [--compare-fit]`.
- **Output:** `lung_fem_fit.vtp` (reference triangles at the fitted positions; point data `Displacement_mm`, `Error_mm`; cell data `PressureRegion`, `Pressure_Pa`, `Clamped`; with a wall, point data `WallPenetration_mm`, > 0 = beyond the allowed position) and `fem/fit/`: `result_summary.json`, `history.csv`, `best_state.npz`, `best_params.json`, `volume_best.vtk` (if the core can export it).

### Collapse sequence (separate script)

To look at one patient's collapse, after `fem_fit`. The script re-solves the FEM with the fitted pressures ramped 0 → 100 % (mode `fem`, same core and options as the fit), because the fit only stores the final state and the intermediate shapes are not a linear scaling of it. Mode `linear` is a straight morph from the inflated to the fitted shape: instant, no FEM library needed, but not the physical path.

1. **Create the sequence** (where the FEM core is installed):
   ```bash
   python scripts/collapse_sequence.py --config configs/<p>.json      # [--frames 30] [--mode fem|linear] [--volume] [--fps 8]
   ```
   It writes `results/<p>/sequence/`: one `.vtp` per frame, `collapse.pvd` for ParaView, the target, the hilum spheres and `load_collapse_in_slicer.py`. The folder is self-contained.
2. **Copy the folder to the machine with Slicer**, e.g. from the Jupyter server: `cd results/<p> && zip -r sequence_<p>.zip sequence`, then right click on the zip in the Jupyter file browser → Download, and unzip it.
3. **Open it in 3D Slicer**, either way:
   - Slicer closed, from a terminal (macOS path; adapt it if Slicer is installed elsewhere):
     ```bash
     /Applications/Slicer.app/Contents/MacOS/Slicer --python-script <folder>/sequence/load_collapse_in_slicer.py
     ```
   - Slicer open: View → Python Console, then
     ```python
     exec(open('<folder>/sequence/load_collapse_in_slicer.py').read())
     ```

   The lung collapse plays in a loop, coloured by displacement (blue = still, red = largest displacement); the grey wireframe is the target (collapsed lung) and the spheres are the hilum. Pause or scrub the load with the Sequences toolbar.

## Outputs

```
results/patient_<N>/
├── lung_collapsed_mesh.vtp     collapsed lung, remeshed
├── lung_inflated_mesh.vtp      inflated lung, same nodes (registration)
├── hilum_anchor.mrk.json       3 ring centres + hilum, each a sphere with its radius
├── lung_fem_fit.vtp            inflated lung deformed by the fitted pressures (fem_fit)
├── logs/                       pipeline.log, config_used.json
├── checks/                     input check: collapsed_check.vtp, inflated_check.vtp, check_summary.json
├── registration/               masks, elastix log and transforms (TransformParameters.*-Composite.h5 loads in Slicer)
├── hilum/                      rings.vtp (all rings, cell data Structure / Hilar), hilum_anchor.json
└── fem/                        setup/ (problem.npz, reference.vtp, target_aligned.vtp, volume.vtu, gmsh/), fit/
```

All surfaces are in LPS, with the space stored in the file.

To view in 3D Slicer, drag and drop `lung_collapsed_mesh.vtp`, `lung_inflated_mesh.vtp`, `hilum_anchor.mrk.json` (loads as 4 point lists) and `hilum/rings.vtp` (colour by `Hilar`).

## Known issues

- **patient_2 registration is still approximate** (segmentations corrected on 2026-10-01; ×7.2 in volume, rigid + affine + B-spline):
  - Dice 0.953 on the mesh, volume 5342 / 5350 mL; 6 nodes where the inverse does not converge are interpolated.
  - 7 flipped triangles, removed by re-interpolating 25 nodes 5–38 mm from the hilum (moved up to 138 mm: the registration had folded them far out). The surface then meshes cleanly, with small tetrahedra (min 4 mm³ against ~20 mm³ on patient_10).
  - The hilum region is poorly matched: nodes within 30 mm of the hilum move ~88 mm; after `hilum_rigid` alignment the clamped points are ~23 mm from their target, and 49 target points lie beyond the wall (up to 15 mm).
  - Bending-energy regularisation, the forward direction and signed distance maps did not help on the old segmentations. Next to try: a larger `landmark_weight` for this patient only, `align: "none"`.
- **Node correspondence is not anatomical.** Mask registration only matches the boundaries, so nodes can slide tangentially along the surface. Keep this in mind for the point-to-point loss of the inverse FEM.

## Inverse FEM pressure fitting (step 5 detail, legacy code)

![Inverse FEM detail](inverse_fem_pressure_fit_detail.svg)
