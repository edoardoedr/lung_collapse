# CLAUDE.md

Guidance for Claude Code in this repository. What lives where:

| File | Content |
|---|---|
| [README.md](README.md) | user quick-start: the model, steps, running, viewing |
| [configs/README.md](configs/README.md) | every config key with meaning and range |
| [METHODOLOGY.md](METHODOLOGY.md) | paper-style methods: geometry, physical model, equations, solver, forward and inverse problem, assumptions and what is not modelled, approaches abandoned |
| `*.svg` (root), `scripts/figures/` | the three figures (workflow, physical model, forward / inverse) and the Python that draws them: `cd scripts/figures && python3 workflow.py && python3 model.py && python3 fit.py`, then copy the `.svg` to the root |
| [analysis/results.md](analysis/results.md) | results, evidence behind the design choices, problems found |
| [NEXT.md](NEXT.md) | plan and to-do list |
| [REWRITE_PLAN.md](REWRITE_PLAN.md) | the plan of the rewrite on branch `refactoring_to_v2` (2026-11-16) and its status |
| this file | how to work here, environments, commands, configs, code architecture, implementation notes |

Read them instead of re-deriving, and keep them current: a new result goes in `analysis/results.md`, a method change in METHODOLOGY.md, a decision or task in NEXT.md. The code before the rewrite (regions fit, registration, the earlier versions of every doc including the long step descriptions of this file) is on `main`: `git show main:CLAUDE.md`, `git show main:pipeline/collapse/fit.py`.

## Project

Lung-collapse pipeline for a GNN surrogate of lung collapse in VATS surgery. CT segmentations (3D Slicer) → surface meshes → hilum anchor → FEM model of the collapse: an elastic lung clamped at the hilum, loaded by the pleural pressure it loses in the pneumothorax (uniform part + the gradient that carried its weight), inside the chest wall. The model runs **forward** (parameters given → collapsed lung) or **inverse** (parameters fitted to the collapsed CT). The forward runs are the source of the GNN's training data (planned in PyTorch, possibly PyG).

## Working with the user

- The user writes in Italian: answer in Italian. Code, comments, docstrings, READMEs, configs and the docs above are in English.
- **Do not install anything on the Mac** (no getfem, warp, torch, pyvista, pytest...). The heavy runs happen on the server; give the user the commands to run there and read the logs they paste back.
- **No commits unless asked.** The user commits themself.
- Before changing physics or defaults, explain the trade-off and let the user decide; implementation details can be decided directly.
- A new or changed config key goes in `pipeline/config.py`, `configs/README.md` and, if user-visible, `README.md`.
- Only what worked is in this code. The experiments that led nowhere (registration, alignment, still points, point loss, Nelder-Mead, finite-difference fit Jacobian, `wall_update: "newton"`, face wall quadrature, `support_k`, the gravity continuation) were left on `main` on purpose: do not bring them back without asking.

## Environments

| Where | What is there | Use |
|---|---|---|
| Mac (this machine) | base Python with numpy / scipy; **no** pyvista, vtk, itk, gmsh, getfem, warp, torch, pytest | editing, `py_compile`, the tests in `tests/`, reading `results/` |
| Server (Jupyter, conda env `lungfem`, `/home/jovyan/Lung_project/lung_collapse`) | GetFEM (MUMPS), warp-lang, torch, nvmath-python (cuDSS), pypardiso; Xeon CPUs (slow) + CUDA GPU (driver CUDA 12; a cuda-bindings 13 warning is harmless) | all pipeline runs and validation scripts |

- **Tests on the Mac:** `python3 tests/<file>.py` (pytest-compatible files, run without it).
  - `tests/stubs.py`: `MagicMock` stand-ins for the heavy libraries, and the test runner.
  - `tests/numpy_core.py`: a numpy Neo-Hookean core on `NodalSolver` (same physics, complex-step tangent). Use it as `"core": "numpy_core:NumpySolver"`. Slow (seconds per solve): give it `max_solve_s`.
  - `tests/sdf_numpy.py`, `tests/fakes.py`: numpy closest point / signed distance, patched over the VTK helpers of `collapse/geometry.py`, so that `fit.run` / `forward.run` work end to end.
  - They run on the local `results/karl04_noreg_hilum_wall/fem/setup/problem.npz` (skipped without it); the wall grid (4 mm) is cached in `tests/.cache/`.
- **Server output folders.** Current code: `/home/jovyan/Dataset_Lungs/{HD_dataset/FEM_v2_HD, CZ_dataset/FEM_v2_CZ}/<p>/` (the configs' `output_dir`; filled by `scripts/init_output_dir.py` from the old folders: setup + the v2 runs). Old code (frozen, not written any more): `.../FEM_fit_{HD,CZ}/<p>_noreg_hilum[_wall]`.
- `results/` on the Mac is a downloaded copy of the old server folders. Old layout: `fem/<run>/result_summary.json`, `best_state.npz`; new layout: `runs/<run_name>/summary.json`, `state.npz`. Check the file dates against the git log before saying which code a run used.

## Commands

```bash
python main.py --config configs/<p>.json                       # the steps of the config
python main.py --config configs/<p>.json --steps fit --set physics.E_Pa=1000 run_name=E1000
python main.py --config configs/<p>.json --steps collapse --set 'physics.collapse={"volume_ratio": 0.5}' run_name=half
JOBS=4 bash scripts/run_all.sh [case ...] [-- main.py args]    # configs/<case>.json in parallel; logs_run_all/<case>[_$LOG].txt
python scripts/fit_cohort.py --configs configs/karl*.json --param physics.E_Pa --values 1000 1700 3000 [--jobs 4] [--collect]
python scripts/summarize_runs.py <results folder> [--runs fit E1000] [--out analysis/runs_v2.csv]   # Mac or server
python scripts/leave_one_out.py --configs configs/*.json [--fit-run fitE] [--run loo] [--jobs 4] [--frames 30] [--collect]   # E from the others' E(collapse %), forward to the own collapse %
python scripts/add_wall_field.py --config configs/<p>.json [--grid 1.5] [--smoothing 3]   # wall field into an existing problem.npz
python scripts/init_output_dir.py --configs configs/*.json [--apply]   # new output_dir <- setup + v2 runs of the old one (dry run without --apply)
python scripts/check_jacobian.py --config configs/<p>.json [--set ...] [--solver getfem] [--options '{...}']
python scripts/compare_warp_getfem.py --config configs/<p>.json [--core torch] [--timing] [--fit] [--wall-only]
python scripts/wall_diagnose.py --config configs/<p>.json [--q 0.3] [--from-fit RUN] [--perturb r ...] [--set ...]
python scripts/collapse_sequence.py --config configs/<p>.json [--frames 30] [--mode fem|linear] [--set run_name=...]
python scripts/migrate_config.py OLD_wall_torch.json --out configs/      # old schema (main) -> new
python scripts/summarize_fits.py <results folder> [--out analysis]       # OLD layout only (fem/<run>/)
python -m py_compile pipeline/*.py pipeline/collapse/*.py pipeline/collapse/solvers/*.py scripts/*.py
python3 tests/test_wall_field.py; python3 tests/test_wall_potential.py; python3 tests/test_steps.py   # Mac
```

Pass `2>&1 | tee <name>.txt` on the server so the user can paste the output. Many parallel jobs slow every solve (~15× with 28 jobs): use `JOBS` ≤ 4 when times or failures matter.

## Configs

- **One file per patient**, `configs/<p>.json` (16), made by `scripts/migrate_config.py` from the old `<p>_wall_torch.json`: gravity supine → supine, E 1.7 kPa, q0 fitted (start 0.3, 0.5 before), ν 0.30 (0.40 until 2026-11-17), wall = the inflated surface with 5 mm margin, smooth-field wall (`solver.wall_update: "potential"`, grid 1.5 mm, σ 3 mm), torch core, steps `check_inputs` → `fit`. `output_dir` = `FEM_v2_{HD,CZ}/<p>` (2026-11-17): the geometry, hilum and `fem/setup/problem.npz` (with wall field) are copied there from the old `<p>_noreg_hilum_wall` by `init_output_dir.py`, so run `--steps fit` / `collapse` only; a patient whose old setup has no wall field needs `add_wall_field.py` first (patient_2 at least).
- **Patients:** left lung karl01, karl10, karl17, karl18, karl20, karl23, karl24, patient_1, patient_2, patient_3, patient_10; right lung karl02, karl04, karl06, karl08, karl19. karl = HD dataset, patient_* = CZ dataset. karl19 stops on its input (NEXT.md). patient_2 is included again (user, 2026-11-17; it had been left out of the old wall runs).
- The old runs are reproduced with overrides: regions fit = `physics.gravity=null 'physics.perturbation={"regions":[1,4,8,14,25,40]}' fit.best_by=mean` (the existing `problem.npz` has those regions); free ap slope = `physics.gravity=null 'physics.perturbation={"gradient":["ap"]}'`; outer wall = `solver.wall_update=outer`.
- Unknown keys raise an error; `_`-prefixed keys are comments; `data_dir` / `output_dir` relative to the JSON, inputs to `data_dir`, outputs to `output_dir`.

## Code architecture

Steps (`main.py`): `check_inputs` → `geometry` → `hilum` → `fem_setup` → `collapse` (forward) / `fit` (inverse). All surfaces are **LPS**. The first four get their config section; `collapse` and `fit` get the whole `PipelineConfig`.

| File | Role |
|---|---|
| `pipeline/config.py` | schema and loader: `Param` (number = fixed, `{fit, init, bounds}` = fitted), `PhysicsConfig`, `PipelineConfig.solver_options()` (solver options + wall stiffness / material from physics) |
| `pipeline/check_inputs.py`, `raster.py` | input checks (need the collapsed lung) |
| `pipeline/surface_mesh.py` | `geometry` step: ACVD remesh of the inflated and the collapsed lung, same settings |
| `pipeline/hilum.py` | hilum rings and spheres |
| `pipeline/collapse/setup.py` | `fem_setup`: clamp, wall (surface + smooth field), optional Ward regions, Gmsh → `problem.npz`. FEM-library-free |
| `pipeline/collapse/problem.py` | `CollapseProblem` (arrays of `problem.npz`; old files load: `has_target`, `has_wall`, `has_wall_field`) |
| `pipeline/collapse/load.py` | `LoadModel` (builds the levels from `PhysicsConfig`) and `Level` (x → q per triangle, body magnitude, ν; `chain` = Jacobian chain rule; `x0_from` = warm start; `describe` → physical units) |
| `pipeline/collapse/fit.py` | `Loss` (plane + symmetric), `run_level` (lsq with analytic Jacobian, anchoring), `run`; also `make_solver`, `sign_check` |
| `pipeline/collapse/forward.py` | forward run, `find_q0` (volume-ratio search) |
| `pipeline/collapse/results.py` | `write_run`: `surface.vtp`, `state.npz`, `summary.json`; `target_errors` |
| `pipeline/collapse/sequence.py` | collapse sequence + the Slicer loader script |
| `pipeline/collapse/run_control.py` | `Tracker` (history, best state, checkpoints, stops), watchdog |
| `pipeline/collapse/wall_field.py` | `signed_distance_grid` (VTK), `WallField` (numpy tricubic B-spline: value, gradient, Hessian), `penalty` |
| `pipeline/collapse/geometry.py` | surface helpers, `WallDistance` (closest point + interpolated normal; VTK) |
| `pipeline/collapse/regions.py`, `anchor.py`, `volume_mesh.py` | Ward regions, clamp, Gmsh |

FEM cores (`pipeline/collapse/solvers/`):

| File | Role |
|---|---|
| `base.py` | `ForwardSolver`: `set_regions`, `set_body(direction)`, `solve(q, nu, body=0)` → surface displacement, `jacobian(q, nu, with_nu, with_body)`, `get_state` / `set_state(state, load, body)`, `export_volume`; `supports_wall`, `supports_body` |
| `__init__.py` | registry `getfem`, `warp`, `torch` (lazy import) or `"module:Class"`; `options_for` |
| `common.py` | `DEFAULTS` solver options, `mat_params`, `adaptive_path`, `wall_gap_change` |
| `nodal.py` | `NodalSolver`: everything except assembly for the P1 cores: mesh prep, sparsity, `_csr(Kb, Fb)`, `LinearSolver`, GetFEM-replica Newton, load path on the vector [face pressures, body magnitude], wall `potential` (field force + exact tangent in `_assemble_at`) or `outer` (rounds, linearisation cache), body force, Jacobian |
| `torch_solver.py` | `TorchFEM` (assembly only, differentiable, any device / dtype: physics loss for the GNN) + `TorchSolver` (cpu / cuda float64). Stress and tangent by `torch.func` |
| `warp_solver.py` | Warp kernels: hand-derived `nh_P` / `nh_dP`, pressure, linearised wall |
| `getfem_solver.py` | GetFEM core (MUMPS), validation only: `outer` wall, no smooth-field wall, no body force, no other material |
| `materials.py` | torch materials as energy W(F, nu, **params) with E = 1: `neo_hookean`, `mooney_rivlin` (unvalidated). New material = new function + entry in `MATERIALS` |

- Block layout for `_csr`: `Kb[t, a, b, i, j] = dR[node a, i] / dU[node b, j]` (n_tets, 4, 4, 3, 3); `Fb` the same per face (n_faces, 3, 3, 3, 3).
- New features go in `nodal.py` when they can (torch and Warp both get them). A new material is torch only unless asked; Warp needs new `nh_P` / `nh_dP` and `_dfint_dnu`.

## Implementation notes

Not obvious from the code; the reasons and numbers are in analysis/results.md.

- **Load parametrisation:** the gravity term is `s_g · hn`, `hn = (h − h̄) / L` per triangle (`h` = −e_A · centroid, `L` its extent over the free triangles), `s_g = ρ̄ g L / E`. The fitted parameter is `gravity_slope` = s_g (same scale as q0), reported as E; the E bounds become slope bounds. The body force direction passed to the solver is `(e_B − e_A) / L`, its magnitude `s_g`, so one number scales both.
- **Core regions:** one per free triangle when the load varies in space (gravity or slopes), else the Ward regions, else one. `Level` checks that the core regions resolve the field. GetFEM builds one brick per region: slow with gravity.
- **Regions perturbation replaces q0:** the region values carry the mean; their start is the projection of the starting field. `describe` then reports q0 as the area-weighted mean.
- **State files** (`state.npz`, `best_state.npz`): `q` and `tri_labels` as the core sees them (unsigned q; `pressure_sign` stored), `body`, `body_direction`, `nu`, `params`, `solver_state`: enough to re-solve (sequence, `wall_diagnose --from-fit`).
- `fit.run_level` re-anchors the solver to the lowest-cost state of the level before each solve (`set_state(state, load, body)`), and stops the level if its first solve fails. Cores without the body force are called with the two-argument forms.
- **Wall `potential`:** `problem.npz` holds `wall_grid` / `wall_origin` / `wall_spacing` / `wall_smoothing_mm` / `wall_field_allow`. A Newton trial with a surface point outside the grid is rejected like an inverted element (`fail_inverted`). The cores' own (linearised) wall term is skipped (`_assemble(..., wall=False)`). The margin is measured with the smoothed field, so against the exact wall the points go ~2.5 mm beyond it (summary: `wall_max_penetration_mm` exact, `wall_field_max_penetration_mm` smoothed).
- **Wall `outer`:** unsettled after `wall_max_updates` = failure; a settled state keeps the linearisation it was solved with (Jacobian too, Gauss-Newton); states loaded from a file have none and re-linearise.
- `Loss`: the forward plane term is only Gauss-Newton on a triangulated target (interpolated vertex normal ≠ distance gradient); the symmetric term's gradient is exact.
- `forward.find_q0`: first guess `(1 − ratio) · K/E`, extrapolation at most ×2 until bracketed, then secant kept 10 % away from the bracket ends; a failed solve halves the step from the last good q0.
- Summary keys (`runs/<run>/summary.json`): `physics` (E_Pa, q0, p0_Pa, nu, gravity {positions, extent_mm, slope_pE, gamma_pE_per_mm, body_force}, fitted), `volume_ratio`, `collapse_percent`, `assd_mm`, `mean_err_mm`, `mean_normal_err_mm`, `target_to_fitted_mm`, `target_volume_ratio`, `rigid_rotation_deg`, wall penetrations, `solver_stats`; fit: `stop_reason`, `levels`, `evals`, `failed_solves`, `parameters`; forward: `solves`, `volume_search`.

## Validation status

- Torch / Warp vs GetFEM (karl04, before the rewrite): no wall rel ~1e-15, `outer` wall ≤ 1e-11; Jacobian vs FD ~1e-8. Not rerun after the rewrite (the cores' assembly did not change; `nodal.py` did: load vector with the body magnitude, options removed).
- Smooth-field wall on the server (torch, 4 patients, 2026-11-16): Jacobian vs FD 9e-7; E 1.7 and 1 kPa fits from rest with no failed solve (analysis/results.md 7b).
- On the Mac (numpy core, karl04): wall tangent vs FD 1e-7, Jacobian with contact 3e-8, body-force column 1e-7; the rewritten `fit` reproduces the server result of the previous code (ASSD 9.00 mm, p0 1283 vs 1282 Pa, 11 solves); a fit of (q0, E) on a synthetic target recovers them; forward to a volume ratio in 3 solves.
- After the rewrite, on the server (torch, karl04, 2026-11-16): the new `fit` reproduces the previous code evaluation by evaluation (ASSD 9.001 mm, 11 solves); `check_jacobian.py` core 2.6e-8, load model 2.2e-9; `add_wall_field.py` on 12 setups.
- **Not yet run on the server after the rewrite:** the `collapse` step, a body-force run, the regions / gradient perturbations, `geometry` / `fem_setup` (their output must equal the existing files), `fit_cohort.py`, `summarize_runs.py`, `compare_warp_getfem.py`, `wall_diagnose.py`, `collapse_sequence.py`. Not validated against GetFEM: the smooth-field wall and the body force (GetFEM has neither), `mooney_rivlin`.

## Open technical points

Open tasks and decisions are in NEXT.md, open problems in analysis/results.md section 9. Code-level ones only:

- Decided (user, 2026-11-17): `newton_maxit` 50 in the configs (30 was reached inside some solves with the `potential` wall), wall smoothing σ 3 mm kept (points leave the true wall by ~2.5 mm beyond the margin).
- `level_min_improve` 0.005 may stop levels early (0.001–0.002 suggested).
- `scripts/summarize_fits.py` reads only the old layout; `summarize_runs.py` is the simple table for the new one (no pressure-pattern analysis, no plots).
- Possible speed-ups: torch assembly (`torch.compile`, closed-form Neo-Hookean tangent).
