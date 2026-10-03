# Configuration reference

One JSON file per patient (`configs/<patient>.json`) drives the whole pipeline. This page lists every key: what it is, what it is for, and the values it can take. The defaults are those in [`pipeline/config.py`](../pipeline/config.py); a key left out of the JSON takes its default.

**Range** gives the allowed values (enumerations), or the sensible interval with the reason. Values outside a sensible interval are accepted by the code but usually make a step slow, fail or meaningless.

## General rules

- **Unknown keys are an error**, so a typo cannot be silently ignored. Keys starting with `_` (e.g. `_comment`) are comments and are skipped.
- **Paths:**
  - `data_dir` and `output_dir` are relative to the JSON file;
  - step inputs are relative to `data_dir`, step outputs to `output_dir`;
  - elastix `parameter_files` are relative to the JSON file;
  - absolute paths are used as given.
- **`null` for an input** = take it from the step that produces it (e.g. `fem_setup.reference: null` = the registration output). This is the normal setting.
- **Coordinate spaces:** all surfaces inside the pipeline are LPS (3D Slicer's file convention).

## Top level

| Key | Default | What it is | Range |
|---|---|---|---|
| `patient` | – (required) | Name used in the logs. | any string |
| `data_dir` | – (required) | Folder with the input surfaces of this patient. | path |
| `output_dir` | – (required) | Folder for all the results. Two configs of the same patient with different settings (e.g. with / without wall) need different `output_dir`s, or they overwrite each other. | path |
| `steps` | – | Steps run by `python main.py --config ...`; always executed in pipeline order. `--steps` on the command line overrides it. | subset of `check_inputs`, `surface_mesh`, `hilum`, `registration`, `fem_setup`, `fem_fit` |

## `check_inputs` – sanity check of the input surfaces

Only warns (unless `fail_on_warning`); nothing it computes is used by later steps.

| Key | Default | What it is | Range |
|---|---|---|---|
| `collapsed` | `null` | Collapsed lung to check. `null` = `surface_mesh.input`. | path or `null` |
| `inflated` | `null` | Inflated lung to check. `null` = `registration.inflated`. | path or `null` |
| `structures` | `null` | `{name: file}` of the airway / vessel trees. `null` = `hilum.structures`. | dict or `null` |
| `outside_tol_mm` | 1.0 | A collapsed point farther than this outside the inflated lung is flagged. | 0.5–3 mm (about the voxel size: smaller flags segmentation noise) |
| `max_outside_fraction` | 0.01 | Warn when more than this fraction of the collapsed points is flagged. | 0–0.05 |
| `notch_radius_mm` | 5.0 | Radius of the closing ball: notches narrower than about twice this are detected. | 2–10 mm |
| `notch_depth_mm` | 3.0 | A point deeper than this inside a notch is flagged. | 1–5 mm |
| `max_notch_fraction` | 0.005 | Warn when more than this fraction of the points is in notches. | 0–0.02 |
| `max_volume_ratio` | 5.0 | Warn when inflated / collapsed volume is larger (large ratios make the registration hard). | 3–10 |
| `raster_spacing_mm` | 1.0 | Voxel size of the notch check. | 0.5–2 mm (smaller = more precise, slower) |
| `fail_on_warning` | `false` | Stop the pipeline when any check fails. | `true` / `false` |

## `surface_mesh` – remeshing of the collapsed lung (step 2)

Turns the dense Slicer surface into the uniform mesh whose nodes are followed through all later steps.

| Key | Default | What it is | Range |
|---|---|---|---|
| `input` | – (required) | Collapsed lung surface from Slicer. | file in `data_dir` |
| `output` | – (required) | Remeshed surface. | file name, `.vtp` |
| `input_space` | `auto` | Space of the input file. `auto` reads the tag Slicer writes (LPS if none). | `auto`, `LPS`, `RAS` |
| `output_space` | `LPS` | Space of the output. Keep LPS: later steps expect it. | `LPS`, `RAS` |
| `merge_tolerance` | 1e-5 | Distance [mm] under which duplicate points are welded. | 1e-6–1e-3 |
| `pre_smooth_iters` | 15 | Taubin smoothing before remeshing (removes the segmentation staircase). | 0–50 (more = smoother, slightly smaller details) |
| `post_smooth_iters` | 5 | Taubin smoothing after remeshing. | 0–20 |
| `smooth_pass_band` | 0.1 | Taubin pass band: lower = stronger smoothing. | 0.01–0.5 |
| `target_nodes` | 480 | Number of nodes of the output mesh (faces ≈ 2 × nodes). Sets the resolution of the whole FEM surface and of the registration correspondence. | 300–2000 (more = finer but slower FEM) |
| `max_faces` | 1000 | Hard ceiling on the faces: if exceeded, `target_nodes` is reduced. | ≥ 2 × `target_nodes` |
| `shrink_factor` | 0.9 | Factor applied to `target_nodes` at each retry when `max_faces` is exceeded. | 0.8–0.95 |
| `max_remesh_attempts` | 6 | Retries before the step fails. | 3–10 |
| `min_points_per_cluster` | 12 | The input is subdivided until it has at least this many points per output node (ACVD needs a dense input). | 5–20 |
| `max_subdivisions` | 4 | Maximum subdivisions for the rule above. | 2–5 |
| `hole_size` | 1e4 | Largest hole [mm] that is filled. The default fills any hole. | keep large |

## `hilum` – hilum anchor (step 3)

| Key | Default | What it is | Range |
|---|---|---|---|
| `lung` | – (required) | Lung surface the trees enter; same scan as the trees (the collapsed lung). | file in `data_dir` |
| `structures` | – (required) | `{name: file}` of the trees, usually `airways`, `arteries`, `veins`. The names become the point labels in the `.mrk.json`. | dict |
| `output` | – (required) | Slicer markups file with one sphere per ring plus the hilum. | file name, `.mrk.json` |
| `max_ring_distance_mm` | 50.0 | Warn if a ring centre is farther than this from the hilum (wrong ring picked). | 30–80 mm |

## `registration` – inflated lung onto the collapsed mesh nodes (step 4)

Computed on masks with elastix, then applied to the nodes of the step-2 mesh. Output: the same nodes in the inflated shape.

| Key | Default | What it is | Range |
|---|---|---|---|
| `inflated` | – (required) | Inflated lung: surface (rasterised) or labelmap. | file in `data_dir` |
| `inflated_label` | `null` | Label value if `inflated` is a labelmap. `null` = any voxel > 0. | integer or `null` |
| `collapsed` | `null` | Collapsed lung mask. `null` = rasterise the step-2 mesh (recommended: the mask matches the mesh exactly). | file or `null` |
| `collapsed_label` | `null` | As `inflated_label`. | integer or `null` |
| `surface` | `null` | Mesh to warp. `null` = `surface_mesh.output`. | file or `null` |
| `surface_space` | `auto` | Space of `surface`. | `auto`, `LPS`, `RAS` |
| `direction` | `inverse` | `inverse`: fixed = inflated, as in Slicer, the nodes are mapped through the inverse transform. `forward`: fixed = collapsed, no inversion. `inverse` gave better results. | `inverse`, `forward` |
| `landmarks` | `null` | `.mrk.json` with landmarks. `null` = `hilum.output`. | file or `null` |
| `landmark_points` | `["hilum"]` | Labels of the landmarks used (the hilum must not move). | any of `airways`, `arteries`, `veins`, `hilum` |
| `landmark_weight` | 0.0 | Weight of the landmark term in elastix. 0 = landmarks only for the QA log. | 0–0.005: 0.001 keeps the shape match and pins the hilum; ≥ 0.01 spoils the shape match |
| `hilum_region_mm` | 30.0 | QA only: the log reports the displacement of the nodes within this distance of the hilum. | 20–50 mm |
| `raster_spacing_mm` | 1.0 | Voxel size for rasterised surfaces. | 0.5–2 mm (smaller = more precise, much slower) |
| `crop_margin_mm` | 20.0 | Empty margin around the lungs in the masks. | 10–40 mm |
| `parameter_maps` | `["rigid", "bspline"]` | elastix built-in presets, used only if `parameter_files` is empty. | `rigid`, `affine`, `bspline` |
| `parameter_files` | `[]` | elastix parameter files, in order (they replace `parameter_maps`). The configs use the SlicerElastix preset `Rigid` + `BSpline`; add `Affine` in between when the inflated / collapsed volume ratio is large (e.g. ×7). | files in `configs/elastix/` |
| `parameter_overrides` | `{}` | elastix parameters forced in every stage, e.g. `{"MaximumNumberOfIterations": "1000"}`. | elastix keys |
| `random_seed` | 42 | Seed of elastix's random sampling: same seed = same result. | any integer |
| `inversion_tol_mm` | 1e-3 | Accuracy of the per-node inverse transform. | 1e-4–1e-2 mm |
| `inversion_max_iter` | 100 | Iterations of the per-node inversion. | 50–200 |
| `inversion_samples` | 200000 | Lookup points used for the starting guess of the inversion. | 1e5–1e6 |
| `max_interpolated_nodes` | 10 | Nodes where the inversion does not converge are interpolated from their neighbours; above this number the step fails. | 0–20 |
| `max_unflip_nodes` | 40 | Nodes re-interpolated to remove flipped triangles; above this number the step fails (the registration folds a large region). | 0–60 |
| `output` | – (required) | Warped mesh (the inflated lung with the collapsed mesh's nodes). | file name, `.vtp` |
| `output_space` | `LPS` | Keep LPS. | `LPS`, `RAS` |

## `fem_setup` – preparation of the inverse FEM (step 5a)

| Key | Default | What it is | Range |
|---|---|---|---|
| `reference` | `null` | Inflated surface = undeformed FEM configuration. `null` = registration output. | file or `null` |
| `target` | `null` | Collapsed surface to reach, same nodes. `null` = step-2 mesh. | file or `null` |
| `anchor` | `null` | Markups with the hilum. `null` = hilum output. | file or `null` |
| `anchor_point` | `hilum` | Point of `anchor` that centres the clamped region (u = 0). | `hilum`, `airways`, `arteries`, `veins` |
| `anchor_radius_factor` | 2.0 | Clamp radius = sphere radius of `anchor_point` × this (about 30 mm for a 15 mm hilum). Larger = more of the lung held fixed. | 1.5–3 |
| `anchor_min_points` | 20 | The clamp radius grows until it holds at least this many surface points (and is not flat), otherwise the FEM has free rigid motions. | 10–40 |
| `anchor_growth` | 1.1 | Factor of each radius growth step. | 1.05–1.3 |
| `anchor_max_growth_steps` | 10 | Maximum growth steps (with 1.1: up to ×2.6). | 5–20 |
| `wall` | `null` | Cavity the lung may not leave while collapsing. `"reference"` = the registered inflated surface (recommended); a file name = that surface; `null` = no wall. | `"reference"`, file, `null` |
| `wall_tol_mm` | 2.0 | How far a point may move beyond the wall (registration noise margin). | 0.5–5 mm |
| `align` | `hilum_rigid` | Rigid alignment of the target before the fit. `hilum_rigid`: so that the clamped points match (consistent with u = 0 there); `rigid`: all points; `none`: as is (try it when the segmentations of the same scan are consistent). | `hilum_rigid`, `rigid`, `none` |
| `cluster_field` | `rigid_residual` | Displacement used to group the triangles into pressure regions: with the full rigid part removed, with the clamp rigid part removed, or raw. | `rigid_residual`, `hilum_rigid`, `raw` |
| `feature` | `normal` | Ward feature: normal displacement (inward / outward) or full 3D vector. | `normal`, `vector` |
| `pos_weight` | 0.3 | Weight of the position in the clustering: higher = rounder, more compact regions. | 0–1 |
| `levels` | `[1, 4, 8, 14, 25, 40]` | Number of pressure regions per level, coarse → fine. Each level starts from the previous one's best. More / finer levels = better match, longer fit, less identifiable pressures. | increasing integers, 1–100 |
| `mesh_size_mm` | 8.0 | Maximum tetrahedron size inside the lung (the surface keeps the step-2 triangles). Larger = fewer tetrahedra = faster FEM. | 8–15 mm (14 ≈ 6500 tets, 8 ≈ 30000 tets) |
| `gmsh_timeout_s` | 300 | Gmsh is killed after this time. | 60–600 s |

## `fem_fit` – regional pressure fit (step 5b)

### Model and solver

| Key | Default | What it is | Range |
|---|---|---|---|
| `output` | – (required) | Fitted surface (main result). | file name, `.vtp` |
| `solver` | `getfem` | FEM core. `warp` and `torch` share the same Newton, load path and wall code (`solvers/nodal.py`) and differ only in the assembly; `torch` also accepts other materials. | `getfem`, `warp`, `torch`, or `module:Class` |
| `solver_options` | `{}` | Options passed to the core, see [below](#solver_options). | dict |
| `E_Pa` | 3000 | Young's modulus. Only scales the reported pressures: from shapes alone only p / E can be found. | any > 0 (literature for lung: ~1–5 kPa) |
| `nu` | 0.30 | Poisson's ratio: fixed value, or start value if `free_nu`. | 0.05–0.45 (0.5 = incompressible, not allowed) |
| `free_nu` | `true` | Optimise ν together with the pressures. | `true` / `false` |
| `nu_bounds` | `[0.05, 0.40]` | Bounds of ν when free. With `order: 2` the upper bound can go to 0.45 (P1 elements lock near 0.5). | within 0.01–0.45 |
| `levels` | `null` | Subset of the `fem_setup` levels to fit (e.g. `[1, 4, 8, 14]` for a quick run). `null` = all. | values of `fem_setup.levels` |

### Optimiser

| Key | Default | What it is | Range |
|---|---|---|---|
| `optimizer` | `lsq` | `lsq`: bounded least squares (trust region), uses the Jacobian. `nm`: Nelder-Mead, derivative-free, much slower for many regions. | `lsq`, `nm` |
| `jacobian` | `analytic` | `analytic`: Jacobian from the core, one linear solve per region (≈14× faster fit on karl04, same error). `2-point`: finite differences, one nonlinear solve per region. | `analytic`, `2-point` |
| `lsq_diff_step` | 5e-3 | Relative finite-difference step (only `2-point`, or a core without Jacobian). | 1e-3–1e-2 |
| `lsq_ftol` | 1e-6 | lsq stops when the cost changes less than this (relative). | 1e-8–1e-4 |
| `lsq_xtol` | 1e-6 | lsq stops when the parameters change less than this (relative). | 1e-8–1e-4 |
| `lsq_gtol` | 1e-8 | lsq stops when the gradient is smaller than this. | 1e-10–1e-6 |
| `q0` | 0.5 | Initial pressure p / E of all regions (0.5 × 3000 Pa = 1500 Pa). | 0.1–1 |
| `q_bounds` | `[-1.0, 3.0]` | Bounds of p / E per region. Negative = pushing outward. | lower −1–0, upper 2–5 |
| `reg` | 1.0 | Smoothness between adjacent regions: penalises pressure jumps [mm of error per unit p / E]. Higher = smoother, more identifiable pressures, slightly worse match. | 0–10 |
| `sign_check_q` | 0.2 | p / E of the start-up check that positive pressure collapses the lung. | 0.05–0.5 |

### Stops

A **level** ends on the first of: lsq convergence (`lsq_*`) or level plateau. The **whole fit** ends on the first of: target error, time budget, fit plateau, end of the levels, Ctrl+C.

| Key | Default | What it is | Range |
|---|---|---|---|
| `level_patience` | 3 | Level plateau: number of optimiser iterations (lsq: one Jacobian each) looked back. | 2–10 |
| `level_min_improve` | 0.005 | Level plateau: move to the next K when the level's best error improved by less than this fraction over the last `level_patience` iterations. With the analytic Jacobian iterations are cheap: a lower value (0.001–0.002) gains accuracy for a few minutes. | 0.001–0.02 |
| `fit_patience` | 2 | Fit plateau: number of consecutive levels looked at. | 1–3 |
| `fit_min_improve` | 0.02 | Fit plateau: stop when `fit_patience` levels in a row improve the best error by less than this fraction. | 0.01–0.05 |
| `target_error_mm` | 2.5 | Stop when the mean error reaches this. | 1–5 mm (registration and discretisation alone give ~2–3 mm) |
| `time_budget_min` | 240 | Stop after this many minutes. The best result so far is kept. | any |
| `hard_grace_min` | 10 | A watchdog kills the process at budget + grace (if a solve hangs). | 5–30 min |

### `solver_options`

Common to `getfem`, `warp` and `torch` unless noted. The defaults are in [`pipeline/collapse/solvers/common.py`](../pipeline/collapse/solvers/common.py).

| Key | Default | What it is | Range |
|---|---|---|---|
| `order` | 1 | Finite element order. 2 = more accurate, no volumetric locking, much slower; `warp` and `torch` support only 1. | 1, 2 |
| `load_steps` | 4 | Load ramp from zero when a solve cannot start from the previous one (retried with 3 × steps). More = more robust, slower restarts. | 4–16 |
| `newton_tol` | 1e-7 | Newton tolerance. | 1e-9–1e-6 |
| `newton_maxit` | 30 | Newton iterations per load step. | 20–50 |
| `pressure_sign` | 1.0 | Sign of the pressure term. The start-up check corrects it if wrong. | +1, −1 |
| `warm_substeps` | `false` | If the direct step from the previous solution fails, first retry in `load_steps` and 3 × `load_steps` sub-steps from it, before restarting from zero. Useful with a wall; without wall it can cost failed attempts. | `true` with a wall, `false` without |
| `wall_stiffness` | 20.0 | Wall penalty stiffness [E per mm of penetration]. Higher = less penetration, harder Newton. | 5–100 |
| `wall_eps` | 0.5 | Width [mm] of the smooth start of the penalty. Larger = smoother contact, easier Newton, more penetration. | 0.1–2 |
| `wall_update` | `outer` | How the contact is solved. `outer`: Newton with the wall linearisation fixed, alternated with re-linearisations (both cores). `newton`: the wall is re-linearised at every Newton iteration, a single Newton solve (`warp` and `torch`; much faster with a wall). | `outer`; `newton` (warp, torch) |
| `wall_max_updates` | 6 | `outer` only: maximum Newton / re-linearisation rounds per solve. | 2–10 |
| `wall_settle_mm` | 0.05 | `outer` only: rounds stop when the contact data change less than this. | 0.01–0.5 mm |
| `linear_solver` | `null` | Sparse linear solver. `getfem`: `null` = MUMPS if available, or a GetFEM name (`mumps`, `superlu`). `warp` and `torch`: `null`/`auto`, `cudss` (GPU), `pardiso` (CPU, `pypardiso`), `scipy`. Do not copy a GetFEM value into a `warp` / `torch` config. | see left |
| `device` | `null` | `warp` and `torch`: device, e.g. `cuda:0` or `cpu`. `null` = GPU if available. `torch` refuses `mps` (Apple GPU): float32 only and no sparse solver, Newton would not converge. | Warp / torch device name |
| `material` | `neo_hookean` | `torch` only: material, a name in [`solvers/materials.py`](../pipeline/collapse/solvers/materials.py). `neo_hookean` = the GetFEM / Warp material. | `neo_hookean`, `mooney_rivlin`, or one you add |
| `material_params` | `{}` | `torch` only: the material's fixed parameters, e.g. `{"c01_fraction": 0.3}` for `mooney_rivlin` (share of the shear stiffness in the I2 term; 0 = Neo-Hookean). | depends on the material |

## Quick recipes

- **Quick test fit:** `fem_fit.levels: [1, 4, 8]`, `time_budget_min: 30`.
- **Faster FEM:** `fem_setup.mesh_size_mm: 14`, `fem_fit.jacobian: "analytic"`, `solver: "warp"` on a GPU machine.
- **With the wall, fast:** `fem_setup.wall: "reference"`, `solver: "warp"`, `solver_options.wall_update: "newton"`, `warm_substeps: true`.
- **Smoother, more identifiable pressures:** raise `reg` (e.g. 3) or stop at a coarser level (`levels` up to 14 or 25).
- **Registration of a strongly collapsed lung (volume ratio > ~4):** add `elastix/Parameters_Affine.txt` between the rigid and B-spline files.
