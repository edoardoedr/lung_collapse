# Configuration reference

One JSON file per patient (`configs/<patient>.json`) describes the whole model and what to run. This page lists every key. The defaults are those in [`eurelaps_lung/config.py`](../eurelaps_lung/config.py); a key left out of the JSON takes its default. The configs of the earlier schema (regions fit, registration) are in git on `v1_version` (with `scripts/migrate_config.py`, which converted them; removed 2026-11-19).

## General rules

- **Paths:** `data_dir` and `output_dir` are relative to the JSON file; inputs are relative to `data_dir`, outputs to `output_dir`; absolute paths are used as given.
- **Comments:** keys starting with `_` are ignored.
- **Typos:** an unknown key is an error.
- **Overrides:** `python main_fit.py --config ... --set key=value ...` (same for `main_predict.py`), with the value parsed as JSON, e.g. `physics.E_Pa=1000`, `'physics.collapse={"volume_ratio": 0.5}'`, `solver.newton_maxit=50`, `run_name=test`.
- **Physical parameters** (`physics.E_Pa`, `physics.material.nu`, `physics.collapse.q0`) are written in one of two ways:

  ```json
  "E_Pa": 1360
  "E_Pa": {"fit": true, "init": 1360, "bounds": [300, 10000]}
  ```

  A number is fixed. The second form is fitted by the `fit` step, starting from `init`, inside `bounds` (optional; defaults in the tables). The `collapse` (forward) step refuses a config with anything marked `fit`.

## Top level

| Key | Default | Meaning | Values |
|---|---|---|---|
| `patient` | – (required) | Name used in the logs and summaries. | any string |
| `data_dir` | – (required) | Folder with the input surfaces of this patient. | path |
| `output_dir` | – (required) | Folder for all the results of this patient. | path |
| `steps` | `[]` | Steps run by `python main_fit.py --config ...`, always in order; `--steps` on the command line replaces it. `main_predict.py` ignores it (it builds the geometry when `problem.npz` is missing, then runs `collapse`). | `check_inputs`, `geometry`, `hilum`, `fem_setup`, `fit` |
| `run_name` | `run` | Folder of the forward or fit run: `output_dir/runs/<run_name>/`. Give each variant its own name. | plain folder name |

## `inputs` – the segmented surfaces

Surface models exported from 3D Slicer (`.vtk`), all from the same scan.

| Key | Default | Meaning |
|---|---|---|
| `inflated` | – (required) | The inflated lung: the model's undeformed shape, and by default the chest wall. |
| `collapsed` | `null` | The collapsed lung: the target of the `fit` step and of the reported errors. `null` = none (prediction only: `collapse` step, no `check_inputs`, `hilum.lung: "inflated"`). |
| `structures` | `{}` | `{name: file}` of the airway, artery and vein trees (`airways`, `arteries`, `veins`): they locate the hilum. |
| `ct` | `null` | The CT image (`.nii[.gz]`, `.nrrd`, `.mha`), relative to `data_dir`: the `ct` step measures the lung density and the emphysema volume from it (`physics.ct_density`, `ct_emphysema`, written to `output_dir/ct/ct_values.json`). With `collapsed` (the pneumothorax CT): mass of the collapsed lung / volume of the cavity, air HU from the pleural air. Without: the inflated lung's own density, air HU from the trachea. Needs SimpleITK or itk. In the cohort configs the file is `<patient>.nii[.gz]`: copy each CT into its `data_dir`. |
| `contralateral` | `[]` | Surfaces of the other lung (the lobes), relative to `data_dir`: with `collapsed`, the emphysema volume is measured there (the operated lung is seen only collapsed; emphysema is bilateral). Without it, on the collapsed lung. |

## `check_inputs` – sanity check of the surfaces

Needs the collapsed lung. Findings are warnings with the position in LPS and RAS; the flagged surfaces go to `checks/`.

| Key | Default | Meaning | Values |
|---|---|---|---|
| `outside_tol_mm` | 1.0 | A collapsed point farther than this outside the inflated lung is flagged. | 0.5–3 mm |
| `max_outside_fraction` | 0.01 | Warn when more than this fraction of the collapsed points is flagged. | 0–0.05 |
| `notch_radius_mm` | 5.0 | Radius of the closing ball: notches narrower than about twice this are detected. | 3–8 mm |
| `notch_depth_mm` | 3.0 | A point deeper than this inside a notch is flagged. | 1–5 mm |
| `max_notch_fraction` | 0.005 | Warn when more than this fraction of the points is in notches. | 0–0.02 |
| `max_volume_ratio` | 5.0 | Warn when inflated / collapsed volume is larger. | 3–8 |
| `raster_spacing_mm` | 1.0 | Voxel size of the notch check. | 0.5–2 mm |
| `fail_on_warning` | `false` | Stop the pipeline when any check fails. | `true` / `false` |

## `geometry` – surface meshes

The inflated lung and, if given, the collapsed lung are cleaned and remeshed uniformly with the same settings.

| Key | Default | Meaning | Values |
|---|---|---|---|
| `target_nodes` | 480 | Nodes of each output mesh (faces ≈ 2 × nodes). Sets the resolution of the model. | 300–2000 |
| `max_faces` | 1000 | Hard ceiling on the faces: if exceeded, `target_nodes` is reduced. | ≥ 2 × `target_nodes` |
| `input_space` | `auto` | Space of the input files. `auto` reads the tag Slicer writes (LPS if none). | `auto`, `LPS`, `RAS` |
| `merge_tolerance` | 1e-5 | Distance [mm] under which duplicate points are welded. | 1e-6–1e-3 |
| `pre_smooth_iters` | 15 | Taubin smoothing before remeshing (removes the segmentation staircase). | 0–50 |
| `post_smooth_iters` | 5 | Taubin smoothing after remeshing. | 0–20 |
| `smooth_pass_band` | 0.1 | Taubin pass band: lower = stronger smoothing. | 0.01–0.5 |
| `shrink_factor` | 0.9 | Factor applied to `target_nodes` at each retry when `max_faces` is exceeded. | 0.8–0.95 |
| `max_remesh_attempts` | 6 | Retries before the step fails (too many faces, or not watertight). | 3–10 |
| `min_points_per_cluster` | 12 | The input is subdivided until it has at least this many points per output node. | 8–20 |
| `max_subdivisions` | 4 | Maximum subdivisions for the rule above. | 2–5 |
| `hole_size` | 1e4 | Largest hole [mm] that is filled. | keep large |
| `inflated_output` | `lung_inflated_mesh.vtp` | File of the inflated mesh in `output_dir`. | file name |
| `collapsed_output` | `lung_collapsed_mesh.vtp` | File of the collapsed mesh in `output_dir`. | file name |

## `hilum` – hilum anchor

Where each tree enters the lung (exact surface intersection, largest ring); the hilum is the centroid of the ring centres. Written as spheres for Slicer.

| Key | Default | Meaning | Values |
|---|---|---|---|
| `lung` | `collapsed` if there is one, else `inflated` | The lung surface the trees are intersected with. | `collapsed`, `inflated` |
| `output` | `hilum_anchor.mrk.json` | Slicer markups file with one sphere per ring plus the hilum. | file name |
| `max_ring_distance_mm` | 50.0 | Warn if a ring centre is farther than this from the hilum (wrong ring picked). | 30–80 mm |

## `fem_setup` – the model's geometry

Writes `fem/setup/problem.npz`, read by the `collapse` and `fit` steps. Rerun it after changing anything here.

| Key | Default | Meaning | Values |
|---|---|---|---|
| `clamp.points` | `["hilum", "airways", "arteries", "veins"]` | Points of the hilum file whose spheres make the clamped (fixed) region: the union of the balls. | names in the hilum file |
| `clamp.radius_factor` | 2.0 | Radius of each ball = sphere radius × this (about 30 mm for the hilum). | 1–3 |
| `clamp.min_points` | 20 | The first ball grows until the clamp holds at least this many surface points, not all in a plane (otherwise rigid motions stay free). | 10–50 |
| `clamp.growth` | 1.1 | Factor of each growth step. | 1.05–1.3 |
| `clamp.max_growth_steps` | 10 | Maximum growth steps. | 5–20 |
| `mesh_size_mm` | 10.0 | Netgen `maxh`: maximum tetrahedron size inside the lung (the surface keeps the `geometry` triangles). The mesher is Netgen (LGPL, `pip install netgen-mesher`); Gmsh was removed on 2026-11-18 (analysis/results.md 7d: same fits and leave-one-out, better tetrahedra). | 8–14 mm |
| `mesh_optsteps3d` | 10 | Netgen 3D optimisation steps (Netgen's default 3 left slivers on karl04). | 5–20 |
| `mesh_timeout_s` | 300 | Netgen is killed after this time. | 60–600 s |
| `wall.surface` | `reference` | The cavity the lung may not leave. `reference` = the inflated surface itself; a file name (in `data_dir`) = that closed surface; `null` = no wall. With gravity the wall carries the lung's weight, so the physical model needs it. | `reference`, file, `null` |
| `wall.tol_mm` | 5.0 | How far a point may go beyond the wall (segmentation margin), on top of how far it already lies outside. | 2–10 mm |
| `wall.grid_mm` | 1.5 | Grid spacing of the wall stored as a smooth field (signed distance, smoothed), needed by `solver.wall_update: "potential"`. `null` = not built. Tens of MB in `problem.npz`, ~30 s. | `null`, 1–2 mm |
| `wall.grid_margin_mm` | 30.0 | The grid covers the wall's bounding box plus this margin; it must exceed any outward motion. | 20–50 mm |
| `wall.smoothing_mm` | 3.0 | Gaussian σ of the smoothing: rounds the wall's corners by a few mm, which lets Newton converge there. Cost: the lung can leave the true wall by ~2.5 mm more than the margin. | 0–5 mm |
| `regions.levels` | none | Only for `physics.perturbation` regions (needs the collapsed lung): numbers of Ward pressure regions, coarse → fine. Filled in automatically from `physics.perturbation.regions`. | e.g. `[1, 4, 8, 14, 25, 40]` |
| `regions.pos_weight` | 0.3 | Weight of the position in the clustering: higher = more compact regions. | 0–1 |

For a `fem_setup` output made before the wall field existed: `python scripts/add_wall_field.py --config ...` adds it without touching the mesh.

The configs' `output_dir` is `.../FEM_v2_{HD,CZ}/<p>` (the current code's folders). `python scripts/init_output_dir.py --configs configs/*.json --apply` fills them from the old `.../FEM_fit_{HD,CZ}/<p>_noreg_hilum_wall` (setup and the runs in the new layout), so the setup steps need not be rerun.

## `physics` – the collapse model

Pressure on the free surface, relative to E (METHODOLOGY.md):

```
q = q0 + γ (h − h_mean) [+ perturbation],     γ = tissue_density · g / E_Pa
```

with h the height against gravity in the reference position.

| Key | Default | Meaning | Values |
|---|---|---|---|
| `E_Pa` | 1360 | Effective Young's modulus of the lung tissue [Pa] (tissue, surfactant and recoil together). It sets the gravity slope and scales the reported pressures; without gravity the shape does not depend on it. **Parameter** (bounds 300–10000); fitting it needs gravity. Third form, forward only (not with `collapse.p0_Pa` / `q0`, where the collapse % is unknown: give E as a number): `{"rule": "collapse"}` = E from the collapse % of `collapse.volume_ratio`, log₁₀ E = a + b × collapse % (the prediction rule fitted on the cohort, Netgen meshes, `a` 3.5380, `b` −0.012883 in the rule file `eurelaps_lung/data/rules.json`, written by `main_calibrate.py --make-default`; another file for a whole process: environment variable `EURELAPS_RULES`; both coefficients can also be given: `{"rule": "collapse", "a": .., "b": ..}`). | 500–5000 Pa |
| `material.nu` | 0.30 | Poisson's ratio: how compressible the tissue is (the lung loses air). **Parameter** (bounds 0.05–0.45). | 0.2–0.45 |
| `material.name` | `neo_hookean` | Strain-energy function, a name in [`solvers/materials.py`](../eurelaps_lung/fem/solvers/materials.py). Other than `neo_hookean`: `torch` core only. | `neo_hookean`, `mooney_rivlin` |
| `material.params` | `{}` | The material's fixed extra parameters. | dict |
| `tissue_density` | 200 | Mean density of the inflated lung in the model [kg/m³]: the measured pleural gradient, 0.2 cmH₂O/cm (Milic-Emili et al. 1966), divided by g (250 until 2026-11-17; E 1360 = 1700 × 0.8 keeps the same load). Only density / E matters for the shape. | 150–400 |
| `ct_density` | from the `ct` step | **Derived, do not set** (a value here overrides the `ct` step): the same measured on the patient's CT (`ct` step, `inputs.ct`; `scripts/ct_density.py` for the cohort table: mass of the collapsed lung / cavity volume; for a new patient with the CT of the inflated lung, its mean density, `--config configs/<p>.json --ct <file> --write-config` sets it) [kg/m³]. **Report only**: the summary adds `ct_density_kg_m3`, `E_Pa_ct` = E × ct_density / tissue_density and `p0_Pa_ct`; the shape does not change. Set in every config (2026-11-17). | 85–300 |
| `emphysema` | none | **The user's choice**: consider emphysema (`true`) or not (`false`), from reading the CT (the index below does not separate the mild cases: a few bullae are evident to the eye but 1–3 % of the volume; `analysis/emphysema_review.md`). Needed by `barrier` `{"phi": "ct"}`: with `false` the trapped air is 0. Set in every config from the visual check (2026-11-19). | `true`, `false` |
| `ct_emphysema` | from the `ct` step | **Derived, do not set**: emphysema volume of the patient's CT, LAA%−950 [%] (`scripts/ct_density.py`: of the inflated lung itself for a new patient, of the contralateral lung with a collapsed lung; `--write-config` sets it with `ct_density`). Its volume: report only (`ct_emphysema_percent` in the summary), used by `barrier` `{"phi": "ct"}` only when `emphysema` is `true`. Set in every config (2026-11-19). | 0–60 |
| `barrier` | `null` | Limit of compression (METHODOLOGY.md 3.4.1): no tetrahedron may go below a volume fraction φ of its reference volume (the tissue plus the air trapped by emphysema), J ≥ φ. `null` = pure Neo-Hookean (no limit: a uniform q above K/E = 0.83 at ν 0.30 has no equilibrium). `{"phi": "ct", "trapped_fraction": 0.54, "tissue_density": 1050, "stiffness": 1.0, "onset": 0.1}`: φ = ct_density / tissue_density + trapped_fraction × ct_emphysema / 100, the second term only if `emphysema` is `true` (needs `ct_density`, `emphysema`, and `ct_emphysema` when true; 0.54 = the cohort's trapped share of the emphysematous air, median of the 7 patients with LAA%−950 ≥ 5 %, results 5j); or `{"phi": 0.2}` given. Switched on by the user (default off). The barrier energy κ⟨J_on − J⟩₊³ / (J − φ), J_on = φ + onset, is zero above J_on: states that stay above it are unchanged. Needed for collapses driven by a surgical pressure (`collapse.p0_Pa` 1600 = 12 mmHg). torch and warp cores only. The summary adds `compression` (J min, tets below J_on, mean / max density) and `volume.vtk` the cell field `VolumeRatio` (density = reference density / J). | `null`, `{"phi": "ct"}` |
| `gravity` | `supine` | Patient position: `{"reference": ..., "collapse": ...}` = position in the inflated CT and during the collapse; one name = the same for both; `null` = no gravity term. When they differ, the change of the weight's direction is added as a body force (not available with GetFEM). | `supine`, `prone`, `left_lateral`, `right_lateral`, or a vector `[x, y, z]` in LPS (direction of gravity) |
| `collapse` | `{"q0": {"fit": true, "init": 0.3}}` | The uniform part of the pressure, exactly one of: `q0` = p/E (**parameter**, bounds −1–3); `p0_Pa` = the same in Pa (forward only); `volume_ratio` = V_collapsed / V_inflated to reach (forward only: q0 is searched, 3–6 solves). | `q0` 0.1–1.5 |
| `wall.stiffness` | 20.0 | Wall penalty [E per mm of penetration]. Higher = less penetration, harder Newton. | 5–100 |
| `wall.eps` | 0.5 | Width [mm] of the smooth start of the penalty. | 0.1–2 |
| `perturbation` | `null` | Diagnostic term added to the load, **fit step only**: `{"gradient": ["si", "ml"]}` = free linear slopes along anatomical axes (`ap` + = anterior, `si` + = superior, `ml` + = lateral; `ap` only with `gravity: null`, since it is the gravity axis of a supine patient); `{"regions": [1, 4, 8], "reg": 1.0}` = one p/E per Ward region, levels coarse → fine (it replaces q0; `reg` = smoothness between adjacent regions). | `null`, gradient, regions |

Examples:

| Goal | `physics` |
|---|---|
| fit the collapse pressure, E given (default) | `"E_Pa": 1360, "collapse": {"q0": {"fit": true, "init": 0.3}}` |
| fit E too | `"E_Pa": {"fit": true, "init": 1360}` |
| predict from the collapse % only (forward) | `"E_Pa": {"rule": "collapse"}, "collapse": {"volume_ratio": 0.5}` (E from the rule file in use: `eurelaps_lung/data/rules.json`, or `$EURELAPS_RULES`) |
| predict from the pressure (forward: the collapse % is the result) | `"E_Pa": 1300, "collapse": {"p0_Pa": 330}` (`main_predict.py --p0 330 --E 1300`) |
| predict a collapse to 45 % of the volume | `"collapse": {"volume_ratio": 0.45}` (step `collapse`) |
| predict in lateral decubitus from a supine CT | `"gravity": {"reference": "supine", "collapse": "left_lateral"}` |
| the old regions fit (best shape the FEM can give) | `"gravity": null, "perturbation": {"regions": [1, 4, 8, 14, 25, 40]}` |
| where the physics is missing | `"perturbation": {"regions": [4, 8]}` on top of gravity |

## `solver` – FEM core and numerical options

`core` picks the core; every other key is an option of that core (an unknown one is an error). `torch` and `warp` share Newton, load path and wall (`solvers/nodal.py`) and differ in the assembly; `getfem` is kept for validation only (no smooth-field wall, no body force, no other material).

| Key | Default | Meaning | Values |
|---|---|---|---|
| `core` | `torch` | FEM core. | `torch`, `warp`, `getfem`, or `module:Class` |
| `wall_update` | `outer` | How the contact is solved. `potential` (the configs; `torch` / `warp`): the wall is the smooth field of `fem_setup.wall.grid_mm`, evaluated at the current position inside Newton with its exact tangent: no rounds, exact Jacobian. `outer`: Newton with the wall linearised and fixed, alternated with re-linearisations until the gap settles. | `potential`, `outer` |
| `load_steps` | 4 | First load step = 1 / this of the path from rest (the configs use 8). | 4–16 |
| `adaptive_steps` | `false` | `true` (the configs): a failed load step is retried at half size, a successful one doubles the next. | `true` / `false` |
| `min_load_step` | 1/64 | Smallest adaptive step, as a fraction of the path: below it the solve fails. | 1/256–1/16 |
| `warm_substeps` | `false` | Without `adaptive_steps`: retry a failed warm start in sub-steps before ramping from rest. | `true` / `false` |
| `newton_tol` | 1e-7 | Newton tolerance. | 1e-9–1e-6 |
| `newton_maxit` | 30 | Newton iterations per load step. The configs use 50: with the `potential` wall some steps reached 30 and were redone smaller. | 20–60 |
| `max_solve_s` | `null` | Seconds after which one forward solve gives up (a failed solve is a rejected trial of the fit). The configs use 120. | `null`, 60–600 |
| `wall_max_updates` | 6 | `outer` only: maximum rounds per solve; not settled by then = failed. The configs use 15. | 4–15 |
| `wall_settle_mm` | 0.05 | `outer` only: the rounds stop when the linearised gap is within this of the true one near the wall. The configs use 0.5. | 0.01–1 mm |
| `linear_solver` | `null` | Sparse solver. `torch` / `warp`: `null` = cuDSS on GPU, else pardiso, else scipy. | `cudss`, `pardiso`, `scipy`, `null` |
| `device` | `null` | `torch` / `warp`: e.g. `cuda:0` or `cpu`. `null` = GPU if available. | device name |
| `order` | 1 | Element order (1 only for `torch` / `warp`). | 1 |
| `pressure_sign` | 1.0 | Sign of the pressure term; the start-up check corrects it if wrong. | +1, −1 |

Wall stiffness and material are set in `physics`, not here.

## `collapse` – forward run

| Key | Default | Meaning | Values |
|---|---|---|---|
| `sequence_frames` | 0 | > 0: also write the collapse sequence for Slicer / ParaView (`runs/<run_name>/sequence/`), the load ramped 0 → 100 % in this many frames. | 0, 20–40 |
| `volume_tol` | 0.005 | `physics.collapse.volume_ratio`: accepted error of the reached ratio. | 0.001–0.02 |
| `max_search_solves` | 15 | ... and the most solves spent searching q0. | 8–30 |

## `fit` – inverse run

Bounded least squares (trust region) with the analytic Jacobian, on the distance between the fitted and the collapsed surface in both directions.

| Key | Default | Meaning | Values |
|---|---|---|---|
| `symmetric_weight` | 1.0 | Weight of the target → fitted distance next to the fitted → target one. 0 = one-sided: a lung that shrinks onto part of the target scores well although the rest stays uncovered. | 0–2 |
| `best_by` | `rms` | The result is the state with the lowest `rms` distance (what the optimiser minimises) or the lowest `mean`. With 1–2 parameters `mean` can keep the starting point. | `rms`, `mean` |
| `slope_bound` | 4.0 | Bound on the slopes of `physics.perturbation.gradient` [p/E across the lung]. | 1–6 |
| `target_error_mm` | 2.5 | Stop when the error reaches this. | 1–5 mm |
| `time_budget_min` | 240 | Stop after this many minutes; the best state so far is the result. | any |
| `hard_grace_min` | 10 | A watchdog kills the process at budget + grace (if a solve hangs). | 5–30 min |
| `level_patience` | 3 | A level stops when its best error improved by less than `level_min_improve` over this many optimiser iterations. | 2–5 |
| `level_min_improve` | 0.005 | ... relative improvement. | 0.001–0.02 |
| `fit_patience` | 2 | Regions levels: stop when this many levels in a row improve by less than `fit_min_improve`. | 1–3 |
| `fit_min_improve` | 0.02 | ... relative improvement. | 0.01–0.05 |
| `lsq_ftol`, `lsq_xtol` | 1e-6 | The optimiser stops when the cost / the parameters change less than this (relative). | 1e-8–1e-4 |
| `lsq_gtol` | 1e-8 | ... or when the gradient is smaller than this. | 1e-10–1e-6 |
| `sign_check_q` | 0.2 | p/E of the start-up check that positive pressure collapses the lung. | 0.05–0.5 |
