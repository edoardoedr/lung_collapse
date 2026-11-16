# CLAUDE.md

Guidance for Claude Code in this repository. What lives where:

| File | Content |
|---|---|
| [README.md](README.md) | user quick-start: steps, running, viewing |
| [configs/README.md](configs/README.md) | every config key with meaning and range |
| [METHODOLOGY.md](METHODOLOGY.md) | methods: pipeline, model, equations, pressure models, inverse problem, contact |
| [analysis/results.md](analysis/results.md) | results, evidence behind the design choices, problems found (by hand until a script writes it) |
| [NEXT.md](NEXT.md) | plan (agreed 2026-10-15) and to-do list |
| this file | how to work here, environments, commands, configs, code architecture, implementation notes, step details |

Read them instead of re-deriving, and keep them current: a new result goes in `analysis/results.md`, a method change in METHODOLOGY.md, a decision or task in NEXT.md. Older history (experiments up to 2026-11-07, the earlier versions of this file) is in git, e.g. `git show 3260c03:NEXT.md`, `git show 3260c03:CLAUDE.md`.

## Project

Collapsed-lung pipeline for a GNN surrogate of lung collapse in VATS surgery: CT segmentations (3D Slicer) → surface mesh → hilum anchor → (registration, now off) → inverse FEM that fits regional pleural pressures so that the inflated lung collapses onto the collapsed one. The fitted FEM results are the source of the GNN's training data (planned in PyTorch, possibly PyG).

The original scripts (`pipeline_codes_v1/`) were removed on 2026-10-05 (in git; their data in `results/old/pipeline_codes_v1/`). Their Slicer multi-sequence loader lives on as `scripts/slicer_compare_sequences.py`.

## Working with the user

- The user writes in Italian: answer in Italian. Code, comments, docstrings, READMEs, configs and the docs above are in English.
- **Do not install anything on the Mac** (no getfem, warp, torch, pyvista...). The heavy runs happen on the server; give the user the commands to run there and read the logs they paste back.
- **No commits unless asked.** The user commits themself.
- Inspection tools (e.g. the collapse sequence for Slicer) are **standalone scripts in `scripts/`**, not pipeline steps.
- Before changing physics or defaults, explain the trade-off and let the user decide; implementation details can be decided directly.
- A new or changed config key goes in `pipeline/config.py`, `configs/README.md` and, if user-visible, `README.md`.

## Environments

| Where | What is there | Use |
|---|---|---|
| Mac (this machine) | base Python with numpy/scipy; **no** pyvista, vtk, itk, gmsh, getfem, warp, torch | editing, `py_compile`, `load_config` checks, numpy-only tests, reading `results/` |
| Server (Jupyter, conda env `lungfem`, `/home/jovyan/Lung_project/lung_collapse`) | GetFEM (MUMPS), warp-lang, torch, nvmath-python (cuDSS), pypardiso; Xeon CPUs (slow) + CUDA GPU (driver CUDA 12; a cuda-bindings 13 warning is harmless) | all pipeline runs and validation scripts |

- Local testing without the heavy libraries: `pipeline/collapse/geometry.py` imports pyvista/vtk at module level, so tests that import `nodal.py` need those modules stubbed (`sys.modules[...] = MagicMock()`). A numpy mirror of the torch assembly on `NodalSolver`, or a fake linear core, on a local `problem.npz` was used this way to test the shared code.
- `results/` on the Mac is a downloaded copy of the server outputs (`/home/jovyan/Dataset_Lungs/{HD_dataset/FEM_fit_HD, CZ_dataset/FEM_fit_CZ}/<p>_noreg_hilum[_wall]`): `fem/setup/problem.npz` and every `fem/<run>/result_summary.json`, `best_state.npz`, `history.csv`. Check the file dates against the git log before saying which code a run used.

## Commands

```bash
python main.py --config configs/<p>_torch.json              # whole pipeline (or <p>_wall_torch.json)
python main.py --config configs/<p>_torch.json --steps fem_setup fem_fit    # some steps only
python main.py --config ... --set fem_fit.run_name=fit_x fem_fit.output=lung_fem_fit_x.vtp ...   # overrides
JOBS=4 bash scripts/run_all.sh [case ...] [-- main.py args]    # configs/<case>_torch.json, JOBS in parallel (default all); logs in logs_run_all/
python scripts/check_jacobian.py --config configs/<p>_torch.json [--solver getfem] [--options '{...}'] [--compare-fit]
python scripts/compare_warp_getfem.py --config configs/<p>_torch.json [--core torch] [--timing] [--fit] [--wall-only] \
       [--jac-q 0.1] [--core-options '{"wall_update": "newton"}']
python scripts/wall_diagnose.py --config configs/<p>_wall_torch.json [--q 0.2] [--steps 10] [--options '{...}'] \
       [--from-fit RUN [--fit-state]] [--perturb r ...] [--trials 3]      # Newton / wall-round trace
python scripts/collapse_sequence.py --config configs/<p>_torch.json [--frames 30] [--mode fem|linear] [--volume]
python scripts/summarize_fits.py <results folder> [...] [--out analysis]   # Mac: runs.csv, compare.csv, report.md, plots
# in Slicer's console: exec(open('scripts/slicer_compare_sequences.py').read()); load('<seq folder>', ...)
python -m py_compile pipeline/collapse/solvers/*.py scripts/*.py    # the only check possible on the Mac for FEM code
```

Pass `2>&1 | tee <name>.txt` on the server so the user can paste the output. Many parallel jobs slow every solve (~15× with 28 jobs): use `JOBS` ≤ 4 when times or wall failures matter.

## Configs

- **Current set (since 2026-10-27, user's decision: no more experiments with the registration):** `configs/<p>_torch.json` and `<p>_wall_torch.json` for the 16 patients, whole pipeline (`check_inputs` → `fem_fit`), torch core, absolute `data_dir` / `output_dir` on the server, output `<p>_noreg_hilum` / `<p>_noreg_hilum_wall`, fit in `fem/fit_torch`. Settings in METHODOLOGY.md section 2; in short: `registration.method: "none"`, `align: "none"`, clamp `["hilum", "airways", "arteries", "veins"]` with `anchor_radius_factor` 2 and `clamp_still_mm` 0, `mesh_size_mm` 14, ν 0.40, `loss: "plane"`, `loss_symmetric_weight` 1, `jacobian: "analytic"`, `target_error_mm` 2.5. Wall configs add `wall: "reference"`, `wall_tol_mm` 5, `outer`, `nodal` quadrature, `adaptive_steps`, `warm_substeps`, `wall_settle_mm` 0.5, `wall_max_updates` 15, `slow_ramp` false, `max_solve_s` 120. `support_k` is not set (0).
- The runs in `results/` beyond `fit_torch` (`fit_sym`, `fit_sym_k*`, `fit_grad_*`) were made with `--set` overrides of `run_name`, `output` and the changed keys; the list is in analysis/results.md section 1.
- **Patients** (16): left lung karl01, karl10, karl17, karl18, karl20, karl23, karl24, patient_1, patient_2, patient_3, patient_10; right lung karl02, karl04, karl06, karl08, karl19. karl = HD dataset, patient_* = CZ dataset. karl19 stops on its input (NEXT.md). patient_2 is excluded from the wall runs (user, 2026-10-05).
- **Archive:** every earlier config (registered, and `_noreg` with still points; 64 files) is in `configs/archive/`, still loadable (elastix paths `../elastix/...`); their `output_dir` still points to the old place, so rerunning one writes a fresh folder there. Their results are in `results/archive/` (server: `FEM_fit_HD/archive`, `FEM_fit_CZ/archive`), kept as the evidence that the registration is not needed; `summarize_fits.py` still finds them (it skips only `old`).
- To run another core set `fem_fit.solver`; the validation scripts drop the options the other core does not accept (`solvers.options_for`: torch's `device` / `material*` and `support_k` for GetFEM, `linear_solver` to / from GetFEM).
- Unknown keys raise an error; `_`-prefixed keys are comments; paths: `data_dir`/`output_dir` relative to the JSON, inputs to `data_dir`, outputs to `output_dir`, `null` inputs come from the previous step.

## Pipeline essentials

- Steps in order: `check_inputs` (1b) → `surface_mesh` (2) → `hilum` (3) → `registration` (4) → `fem_setup` (5a) → `fem_fit` (5b). All surfaces inside the pipeline are **LPS**.
- `registration` with `method: "none"` remeshes the inflated surface with the `surface_mesh` settings (`surface_mesh.remesh`) into the same `output`, own nodes. With elastix (archived configs) it registers rasterised masks and maps the collapsed mesh's nodes by per-node inversion (details below).
- `fem_setup` is FEM-library-free: clamp, alignment (`none` now), optional wall, Ward pressure regions per level, Gmsh tet mesh → `fem/setup/problem.npz` (`CollapseProblem`). It detects a reference with different topology from the target and switches to closest-point matching: `align` must be `none`, the per-point target is the closest point (still points, clustering field `u = cp − X_ref`), `problem.npz` gains `target_points` / `target_tris` (`CollapseProblem.closest_point`, `target_surface`; old files load unchanged).
- `fem_fit`: sign check, coarse-to-fine levels, scipy `least_squares` (TRF) with the core's analytic Jacobian; regions or gradient pressure model (METHODOLOGY.md sections 4–5). Checkpoint `best_state.npz` (q / tri_labels as the core sees them, plus `params`), watchdog at budget + `hard_grace_min`.

## FEM cores (`pipeline/collapse/solvers/`)

Model and equations in METHODOLOGY.md section 3.

| File | Role |
|---|---|
| `base.py` | `ForwardSolver` interface: `set_regions`, `solve(q, nu)` → surface displacement, `jacobian(q, nu, with_nu)`, `get_state` / `set_state(state, load)`, `export_volume`. `fit.py` talks only to this. |
| `__init__.py` | registry `getfem`, `warp`, `torch` (lazy import) or `"module:Class"`; `options_for` |
| `common.py` | `DEFAULTS` solver options, `mat_params`, `dmat_params_dnu`, `adaptive_path`, `wall_gap_change` |
| `getfem_solver.py` | GetFEM core (P1/P2, MUMPS); Jacobian via `K_t dU/dq = d(rhs)/dq`, d/dν by central difference; wall only `"outer"`, re-linearises at every warm start; no `support_k` |
| `nodal.py` | `NodalSolver`: everything except assembly for P1 cores that assemble themselves — mesh prep, sparsity, `_csr(Kb, Fb)`, `LinearSolver` (cudss / pardiso / scipy), GetFEM-replica Newton, load path on face pressures (`fq_last`) with `adaptive_steps` / `warm_substeps`, wall `"outer"` (equilibrium linearisation kept for the last 256 converged states) or `"newton"`, `wall_quadrature` `face` / `nodal`, `support_k` block in `_assemble_at`, Jacobian |
| `warp_solver.py` | Warp kernels: hand-derived `nh_P` / `nh_dP`, pressure, wall; `_dfint_dnu` uses linearity in (c1, d1) |
| `torch_solver.py` | `TorchFEM` (assembly only, differentiable, any device/dtype incl. mps float32: physics loss for the GNN) + `TorchSolver` (cpu/cuda float64; refuses mps). Stress and tangent by `torch.func` (vmap(grad), vmap(jacfwd(grad))), d/dν by `jvp` |
| `materials.py` | torch materials as energy W(F, nu, **params) with E = 1: `neo_hookean` (= GetFEM/Warp), `mooney_rivlin` (example, unvalidated). New material = new function + entry in `MATERIALS`; options `material`, `material_params` |

- Block layout for `_csr`: `Kb[t, a, b, i, j] = dR[node a, i] / dU[node b, j]` (n_tets, 4, 4, 3, 3); `Fb` the same per face (n_faces, 3, 3, 3, 3), pressure + wall summed.
- Changing material elsewhere: Warp needs new `nh_P`/`nh_dP` and `_dfint_dnu`; GetFEM another law name in `add_finite_strain_elasticity_brick` and matching `mat_params`. A fitted material parameter (beyond ν) needs `fit.py` changes too.
- Wall data (`geometry.WallDistance`): interpolated area-weighted vertex normal at the closest point, φ measured along it (the raw VTK distance gradient jumps across edges / vertices). Shared by all cores.

## Implementation notes

Not obvious from the code; the reasons and numbers are in analysis/results.md.

- `fit.Loss`: clamped points are left out of the residual and of every reported error (summary `metric_points`, `clamped_points`, `clamped_mismatch_mm`). With closest-point matching the forward plane term is only Gauss-Newton on a triangulated target (interpolated vertex normal ≠ distance gradient: cost gradient vs FD 7e-2 on karl01, 1.4e-8 on an analytic sphere); the symmetric term's gradient is exact (vs FD 2e-10).
- `fit.run_level` re-anchors the solver to the lowest-cost state of the level before each solve (`set_state(state, load)` with the p/E per triangle; GetFEM ignores the load), and stops the level if its first solve fails.
- 2-point Jacobian step = `lsq_diff_step` × max(|x|, 0.1) (relative to |x| it was ~1e-19 for slopes starting at 0).
- `PressureModel`: regions = indicator Φ; gradient = one core region per free triangle and Jacobian J_tri · Φ, so no core changed. Tested with a fake linear core (synthetic gradient recovered to 1e-15).
- `gravity_E_Pa`: `PressureModel.offset` = fixed p/E γ (h − h̄) added to q_tri and to the core pressures (`fit.gravity_offset`), not a parameter, so Jacobians are unchanged; `ap` is dropped from the fitted axes and `E_Pa` is overwritten. Checked on karl01 (Mac, stubbed imports): q_solver = q_tri, x0_from recovers x, regression slope = γ L. Prefit levels (`gravity_prefit`, attribute `prefit` on the model) warm-start the gravity level and are then dropped from the tracker (`trk.best` reset, `warm` kept). `best_by` picks the result by mean or rms. The whole loop was tested with a fake linear core that fails cold starts with a non-uniform load (`fit.run` with fakes for WallDistance / closest_on_surface / assd / polydata).
- Outer wall rounds: unsettled after `wall_max_updates` = failure (`fail_unsettled`); the settle measure is `common.wall_gap_change` (previous linearised gap vs true gap, points with gap > −1 mm). A settled state keeps the linearisation it was solved with (Jacobian too); states loaded from a file (`best_state.npz`, `wall_diagnose --fit-state`) have none and re-linearise.
- `support_k`: constant block k a (I − n nᵀ) per surface node (reference area / normal), added to R and the free-dof tangent, so Newton, load path and Jacobian see it; `support_share` in the summary = Σ|support force| / Σ|pressure force|.
- Summary keys added over time: `mean_normal_err_mm`, `best_loss_err_mm`, `assd_mm`, `target_to_fitted_mm`, `rigid_rotation_deg` / `rigid_shift_mm` (target, fitted), `baseline_loss_mm`, `pressure_model`, `gradient`, `support_k`, `support_share`, `wall_max_penetration_mm`, `wall_points_beyond`; the fitted surface has `NormalError_mm`.

## Validation status (karl04, 1310 nodes, 6509 tets, K = 4)

- GetFEM analytic Jacobian vs FD ~1e-8; full fit 2-point vs analytic 8.873 vs 8.824 mm, 1416 vs 71 solves, 111 vs 7.9 min.
- Warp vs GetFEM, no wall: rel ~1e-15; Jacobian vs FD 1e-8, vs GetFEM 1e-13.
- Torch vs GetFEM, no wall: rel 7e-16–2e-14 up to p/E 2; Jacobian vs FD 2e-9–1e-8, vs GetFEM 1e-13 (ν 1e-10).
- Torch / Warp vs GetFEM with wall (`outer`): rel ≤ 1e-11, identical penetrations. With a wall the Jacobian vs FD check is meaningless under `outer` (FD dominated by `wall_settle_mm`); torch vs GetFEM Jacobian 1e-14.
- Full no-wall fits, Warp and torch: identical (same evaluations, Newton iterations, errors to 3 decimals).
- Forward solve: GetFEM 70–107 s (Xeon), Warp + cuDSS 3.2 s, torch + cuDSS 5.7 s (assembly 3.2 s), torch + scipy 11.4 s, torch + pardiso 27 s (pardiso slow on this server: prefer scipy on CPU-only machines). Jacobian: torch 0.1 s, GetFEM 0.46 s.
- Not validated against GetFEM: `support_k`, the kept equilibrium linearisation (GetFEM has neither), `mooney_rivlin`.

## Open technical points

Open tasks and decisions are in NEXT.md, open problems in analysis/results.md section 9. Code-level ones only:

- Wall: outer rounds still alternate for points in corners of the wall (analysis/results.md section 7); every wall fit in `results/` predates the 2026-10-30 fix.
- `wall_update: "newton"` oscillates (residual 2-cycle; the tangent misses the normal's rotation): kept but unused.
- Newton `maxit` failures with `support_k` and no wall, cause unknown.
- `level_min_improve` 0.005 may stop levels early with the cheap analytic Jacobian (0.001–0.002 suggested).
- Possible speed-ups: torch assembly (`torch.compile`, closed-form Neo-Hookean tangent); `max_solve_s` ~20 s with a wall. Contact-in-Newton for GetFEM would need a custom Newton loop.
- The 6.2 script also had `--warm-start` from a previous run's pressures (not ported) and a separate wall file, so lung nodes were not on wall vertices.

## Step details

The full step descriptions (moved from the README on 2026-10-05). Headings are one level down. Registration (step 4) and the target alignment are not used by the current configs.

### Steps

#### 1b · Input check (`check_inputs`)

Runs first, on the raw surfaces (by default the files named in the other sections), so segmentation problems show up before registration and Gmsh fail on them. Findings are warnings with the position in LPS and RAS (as shown in Slicer) and a hint on how to fix them; with `fail_on_warning: true` they stop the pipeline.

- **Integrity:** readable, space tag, closed, number of pieces and volume. Extra pieces of a lung (islands, internal holes) are a finding; the following checks use the largest piece. Airway and vessel trees are normally in many pieces, so for them this is only logged.
- **Same scan:** every structure overlaps the collapsed lung.
- **Containment:** the collapsed lung must lie inside the inflated one. Points more than `outside_tol_mm` outside are flagged, warning above `max_outside_fraction`. Fix in Segment Editor: *Logical operators → Add* the collapsed segment to the inflated one, after smoothing.
- **Narrow notches** (folds, open fissures), in both lungs: the mask is closed with a ball of `notch_radius_mm`; points deeper than `notch_depth_mm` inside the closed volume are flagged, warning above `max_notch_fraction`. They cause the flipped triangles in the registration and the self-intersections Gmsh refuses. Fix: *Smoothing → Closing*.
- **Volume ratio** inflated / collapsed above `max_volume_ratio`.
- **Output:** `checks/collapsed_check.vtp` (point data `DistanceToInflated_mm`, `NotchDepth_mm`), `checks/inflated_check.vtp` (`NotchDepth_mm`), `checks/check_summary.json`. Colour them in Slicer to find the spots to fix.

#### 2 · Surface meshing (`surface_mesh`)

- **Input:** dense collapsed-lung surface from Slicer (`lung_left_collapsed.vtk`).
- **Output:** `lung_collapsed_mesh.vtp`.
- **Method:**
  - clean the surface: weld duplicate points, keep the largest component, Taubin smoothing, fill holes;
  - remesh uniformly with ACVD (pyacvd) to ~480 nodes / 956 triangles;
  - light post-smoothing and a watertightness check.
- **Result:** the node set used by every later step.

#### 3 · Hilum anchor (`hilum`)

- **Inputs:** collapsed lung surface + airway, artery and vein surfaces.
- **Output:** `hilum_anchor.mrk.json` (LPS) with 4 point lists of one point each:
  - `airways`, `arteries`, `veins`: centre and radius of the ring where that tree enters the lung;
  - `hilum`: centroid of the three ring centres, with radius = mean of the three ring radii.

  Slicer gives every point of a list the same glyph size, so each point is its own list. In Slicer each one shows as a sphere with its diameter, and the radius is also in the point description (`radius_mm=…`).
- **Method:** exact surface–surface intersection of each tree with the lung surface. This gives closed rings, and per structure the largest ring (by perimeter) is the hilar one. The others are small peripheral branches.

#### 4 · Registration (`registration`)

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

#### 5 · Inverse FEM (`fem_setup`, `fem_fit`)

Finds the regional pleural pressures that deform the inflated lung (reference) onto the collapsed one (target), with the hilum clamped. Port of `pipeline_codes_v1/6.1_lung_inverse_fem_fit.py` and of its wall variant `6.2_…_wall.py`, split in two steps so the FEM core can be exchanged.

**`fem_setup`** (seconds, no FEM library):

- **Inputs:** `lung_inflated_mesh.vtp` (reference), `lung_collapsed_mesh.vtp` (target) and `hilum_anchor.mrk.json`. With a registered reference they have the same nodes and triangles (node-to-node target); otherwise each reference point is matched to its closest point on the target (`registration.method: "none"`, current configs).
- **Clamped region:** surface triangles within `hilum sphere radius × anchor_radius_factor` of the hilum (default 2.0 → ~30 mm on patient_10). The radius grows while the region has fewer than `anchor_min_points` points or is nearly flat, otherwise rigid motions would stay free.
- **Alignment:** `align: "hilum_rigid"` moves the target rigidly so that its clamped points best match the reference ones (consistent with u = 0 there); `rigid` uses all points, `none` keeps it as is.
- **Cavity wall (optional, `wall`):** closed surface the lung may not leave during the collapse (the chest cavity). `"reference"` (used in the configs) takes the registered inflated surface itself, so every point starts on it; a file name (e.g. `lung_left.vtk`) takes that surface. A point may go at most `wall_tol_mm` (default 2 mm, configs 5) beyond it, plus however far it is already outside in the reference. The log reports how many target points lie beyond: those are an error the fit cannot remove. `wall: null` = no wall (as 6.1).
- **Pressure regions:** connectivity-constrained Ward clustering of the displacement with the rigid part removed (`cluster_field`), for every level in `levels` (1 → 40 regions). The regions do not depend on the pressures, so they are all computed here.
- **Volume mesh:** Gmsh tetrahedra (`mesh_size_mm` inside), with the surface points and triangles unchanged (checked).
- **Output:** `fem/setup/problem.npz`, read by `fem_fit` and passed to the core, plus files to inspect: `reference.vtp` (cell data `Clamped`, `Regions_K*`; point data `WallDistance_mm`), `target_aligned.vtp`, `volume.vtu`, `setup_summary.json`.

**`fem_fit`** (hours with GetFEM):

- **FEM core:** `solver` names a class in `pipeline/collapse/solvers/` (`"getfem"`, `"warp"`, `"torch"`) or any `"module:Class"`; `solver_options` go to that class. Every core implements `ForwardSolver` (`solvers/base.py`): hyperelastic material with E = 1 (compressible Neo-Hookean; `torch` also others) (so the unknowns are p/E per region and ν), u = 0 on the clamp, follower pressure on the other triangles, positive pressure pushing inward, and the wall as rigid contact if the problem has one; `solve(q, nu)` returns the surface displacement. A core without wall support refuses a problem with a wall. To add a core, write one file in `solvers/` and register it in `solvers/__init__.py`; a P1 core that assembles the system itself only needs `_assemble` on top of `nodal.NodalSolver`.
- **Wall contact, `wall_update`:** `"outer"` (both cores) alternates Newton solves with the wall linearisation fixed and re-linearisations, up to `wall_max_updates` rounds until the gap changes less than `wall_settle_mm`; each solve is several Newton runs, which is why the fit with a wall is much slower. `"newton"` (Warp and torch) re-linearises the wall at every residual evaluation, so the contact is part of a single Newton solve; at convergence it solves the same equations. The contact normal is continuous: area-weighted vertex normals of the wall interpolated at the closest point, and the gap measured along it (`WallDistance`); the raw distance gradient (direction to the closest point) jumps across edges and vertices, and the `"reference"` wall has the lung's nodes on its vertices, so Newton could not converge on it. With a wall the configs also set `warm_substeps: true`, so a failed step is retried in sub-steps before restarting the ramp from zero.
- **GetFEM wall contact:** penalty force on the pressure faces, zero inside the wall and growing smoothly beyond it (`wall_stiffness`, `wall_eps`); the gap is re-linearised between Newton solves until it settles (`wall_max_updates`, `wall_settle_mm`). Warm start as in 6.1: one direct step from the last converged state, then a ramp from zero. `warm_substeps: true` adds, before the ramp, `load_steps` and 3×`load_steps` sub-steps from the last state (as 6.2): useful with contact, but each failed attempt costs Newton runs up to `newton_maxit`.
- **Method:** one sign check (q > 0 must collapse), then the levels coarse → fine, each warm-started from the best so far. Per level: bounded least squares (`lsq`; Jacobian from the core when `jacobian: "analytic"` and the core provides one, i.e. one linear solve per region instead of one nonlinear solve; `"2-point"` = finite differences as before) on the loss (`point`, or `plane` along the target normal, plus the symmetric term without registration) plus a smoothness penalty between adjacent regions (`reg`), or Nelder-Mead (`nm`); ν optionally free.
- **Stops:**
  - *a level* ends when the optimiser converges (`lsq_*` tolerances) or when its best error improved by less than `level_min_improve` (relative, default 0.5 %) over the last `level_patience` optimiser iterations (default 3, i.e. 3 × (parameters + 1) solves); the fit then moves to the next K. On karl04 this would have cut 84 to ~32 min with the same error up to K=14 (the lsq tolerances alone kept going 50–200 solves past the last improvement);
  - an *iteration* is one Jacobian for lsq, `parameters + 1` solves for Nelder-Mead;
  - *the whole fit* ends on `target_error_mm`, `time_budget_min`, `fit_patience` levels in a row improving by less than `fit_min_improve`, the end of the levels, or Ctrl+C / SIGTERM. The best result is checkpointed on improvement, and a watchdog kills the run at budget + `hard_grace_min`, so a forced stop keeps the best solution.
- **Warp core (`"solver": "warp"`):** same physics, options and load path as the GetFEM core, P1 only, assembled with NVIDIA Warp kernels in float64 on GPU (or CPU); extra options `device` and `linear_solver` (`auto`, `cudss`, `pardiso`, `scipy`). Newton reproduces GetFEM's classical Newton with the "simplest" line search, and the wall term uses the IM_TRIANGLE(3) face rule by default (`wall_quadrature: "nodal"` in the configs); both confirmed against GetFEM with `python scripts/compare_warp_getfem.py` (forward solves with and without wall, Jacobian, full fit, timing).
- **PyTorch core (`"solver": "torch"`):** the Warp core with the assembly written in PyTorch. Everything except the assembly (Newton, load path, wall rounds, linear solvers, Jacobian) is shared with Warp in `solvers/nodal.py`. The material is only a strain-energy function W(F, ν) in `solvers/materials.py`; stress and tangent come from automatic differentiation, so a new material is a new function (see the docstring of `materials.py`). Extra options `device` (`cpu`, `cuda`; not `mps`), `material`, `material_params`. Validation: on a machine without GetFEM (e.g. the Mac, CPU) `python scripts/check_jacobian.py --config configs/<p>.json --solver torch`, which also tests the autodiff tangent; against GetFEM `python scripts/compare_warp_getfem.py --config configs/<p>.json --core torch`. `torch_solver.TorchFEM` is the assembly alone, differentiable and on any device and dtype (also `mps`, float32): the residual of a predicted displacement can serve as a physics loss for the GNN.
- **Failed solves with a wall:** `slow_ramp: false` and `max_solve_s` (set in the wall configs, as in the 6.2 script) make a hopeless solve fail fast: the optimiser treats it as a rejected trial. To see why solves fail, `python scripts/wall_diagnose.py --config configs/<p>_torch.json [--q 0.2] [--options '{...}']` (Warp / torch only, no GetFEM) ramps the load and prints per increment Newton runs, iterations, wall rounds, failure reasons (inverted element, iteration limit, linear solve, non-finite, time), contact points, penetration and the smallest element J, plus the Newton iterations of the first failed attempt; the full trace goes to `fem/<run_name>/wall_diagnose.csv`.
- **Analytic Jacobian (GetFEM):** at the converged state, `K_t dU/dq = d(rhs)/dq` on the free dofs, with one factorisation of the tangent matrix (pypardiso if installed, else SuperLU via scipy). The rhs is linear in each pressure, so each column is the pressure term assembled with q = 1; d/dν by a central difference of the rhs. With a wall the contact linearisation is held fixed (Gauss-Newton Jacobian). Check it against finite differences, and compare full fits, with `python scripts/check_jacobian.py --config configs/<p>.json [--compare-fit]`.
- **Output:** `lung_fem_fit.vtp` (reference triangles at the fitted positions; point data `Displacement_mm`, `Error_mm`; cell data `PressureRegion`, `Pressure_Pa`, `Clamped`; with a wall, point data `WallPenetration_mm`, > 0 = beyond the allowed position) and `fem/fit/`: `result_summary.json`, `history.csv`, `best_state.npz`, `best_params.json`, `volume_best.vtk` (if the core can export it).

#### Collapse sequence (separate script)

To look at one patient's collapse, after `fem_fit`. The script re-solves the FEM with the fitted pressures ramped 0 → 100 % (mode `fem`, same core and options as the fit), because the fit only stores the final state and the intermediate shapes are not a linear scaling of it. Mode `linear` is a straight morph from the inflated to the fitted shape: instant, no FEM library needed, but not the physical path.

1. **Create the sequence** (where the FEM core is installed):
   ```bash
   python scripts/collapse_sequence.py --config configs/<p>_torch.json      # [--frames 30] [--mode fem|linear] [--volume] [--fps 8]
   ```
   It writes `results/<p>/sequence/`: one `.vtp` per frame, `collapse.pvd` for ParaView, the target, the hilum spheres and `load_collapse_in_slicer.py`. The folder is self-contained.
2. **Copy the folder to the machine with Slicer**, e.g. from the Jupyter server: `cd results && tar -czf sequences.tar.gz */sequence*`, then right click on the archive in the Jupyter file browser → Download, and extract it into `results/` with `tar -xzf sequences.tar.gz`. The folder is `sequence/` for a fit with `run_name` `fit`, `sequence_<run_name>/` otherwise (e.g. `sequence_fit_torch/`).
3. **Open it in 3D Slicer**, either way:
   - Slicer closed, from a terminal (macOS path; adapt it if Slicer is installed elsewhere):
     ```bash
     /Applications/Slicer.app/Contents/MacOS/Slicer --python-script <folder>/sequence/load_collapse_in_slicer.py
     ```
   - Slicer open: View → Python Console, then
     ```python
     p = '<folder>/sequence/load_collapse_in_slicer.py'
     exec(open(p).read(), {'__file__': p})        # passing __file__ lets the script find its frames
     ```

   The lung collapse plays in a loop, coloured by displacement (blue = still, red = largest displacement); the grey wireframe is the target (collapsed lung) and the spheres are the hilum. Pause or scrub the load with the Sequences toolbar.

### Known issues

- **Node correspondence of the mask registration is not anatomical:** it only matches the boundaries, so nodes slide tangentially along the surface. That is why the current configs use no registration, no alignment and the `plane` loss.
- **patient_2 registration** (archived configs) stayed approximate (×7.2 in volume; hilum region poorly matched, clamped points ~23 mm from their target after alignment). Not an issue without registration.

### Input data

One folder per patient in `Input_Data/`. All files are surface models exported from 3D Slicer (`.vtk`, LPS), from the same scan:

| File | Used by |
|---|---|
| `lung_left_collapsed.vtk` | collapsed left lung, steps 2 and 3 |
| `lung_left.vtk` | assumed inflated left lung, step 4 |
| `lung_airways.vtk`, `lung_arteries.vtk`, `lung_veins.vtk` | step 3 |
