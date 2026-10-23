# CLAUDE.md

Guidance for Claude Code in this repository. The user-facing documentation is [README.md](README.md) (a short quick-start: steps, running, viewing; the full step details moved to the end of this file) and [configs/README.md](configs/README.md) (every config key with meaning and range): read them for details instead of re-deriving, and keep them up to date when behaviour or keys change. This file holds what those do not: how to work here, the FEM architecture, validation status and open problems.

## Project

Collapsed-lung pipeline for a GNN surrogate of lung collapse in VATS surgery: CT segmentations (3D Slicer) → surface mesh → hilum anchor → registration of the inflated lung onto the collapsed mesh → inverse FEM that fits regional pleural pressures so that the inflated lung collapses onto the collapsed one. The fitted FEM results are the training data of the GNN (planned in PyTorch, possibly PyG).

The original scripts (`pipeline_codes_v1/`) were removed on 2026-10-05 (in the git history; their untracked data are in `results/old/pipeline_codes_v1/`). Their Slicer multi-sequence loader lives on as `scripts/slicer_compare_sequences.py`.

## Working with the user

- The user writes in Italian: answer in Italian. Code, comments, docstrings, READMEs and configs are in English.
- **Do not install anything on the Mac** (no getfem, warp, torch, pyvista...). The heavy runs happen on the server; give the user the commands to run there and read the logs they paste back.
- **No commits unless asked.** The user commits themself.
- Inspection tools (e.g. the collapse sequence for Slicer) are **standalone scripts in `scripts/`**, not pipeline steps.
- Before changing physics or defaults, explain the trade-off and let the user decide; implementation details can be decided directly.
- Keep the docs in sync: a new or changed config key goes in `pipeline/config.py`, `configs/README.md` and, if user-visible, `README.md`.

## Environments

| Where | What is there | Use |
|---|---|---|
| Mac (this machine) | base Python with numpy/scipy; **no** pyvista, vtk, itk, gmsh, getfem, warp, torch | editing, `py_compile`, `load_config` checks, numpy-only tests |
| Server (Jupyter, conda env `lungfem`, `/home/jovyan/Lung_project/lung_collapse`) | GetFEM (MUMPS), warp-lang, torch, nvmath-python (cuDSS), pypardiso; Xeon CPUs (slow) + CUDA GPU (driver CUDA 12; a cuda-bindings 13 warning is harmless, cuDSS works) | all pipeline runs and validation scripts |

Local testing without the heavy libraries: `pipeline/collapse/geometry.py` imports pyvista/vtk at module level, so tests that import `nodal.py` need those modules stubbed (`sys.modules[...] = MagicMock()`); a numpy mirror of the torch assembly on `NodalSolver` was used this way to test the shared code on `results/karl04/fem/setup/problem.npz` (Newton converged, Jacobian vs FD ~1e-8). `results/<case>/fem/setup/problem.npz` files exist locally.

## Commands

```bash
python main.py --config configs/<p>_torch.json              # whole pipeline (or <p>_wall_torch.json)
python main.py --config configs/<p>_torch.json --steps fem_setup fem_fit    # some steps only
python scripts/check_jacobian.py --config configs/<p>_torch.json [--solver getfem] [--options '{...}'] [--compare-fit]
python scripts/compare_warp_getfem.py --config configs/<p>_torch.json [--core torch] [--timing] [--fit] [--wall-only] \
       [--jac-q 0.1] [--core-options '{"wall_update": "newton"}']
python scripts/wall_diagnose.py --config configs/<p>_torch.json [--q 0.2] [--steps 10] [--options '{...}']   # Newton trace with a wall
python scripts/collapse_sequence.py --config configs/<p>_torch.json [--frames 30] [--mode fem|linear] [--volume]
# in Slicer's console: exec(open('scripts/slicer_compare_sequences.py').read()); load('<seq folder>', ...)   # sequences side by side
python -m py_compile pipeline/collapse/solvers/*.py scripts/*.py    # the only check possible on the Mac for FEM code
```

Pass `2>&1 | tee <name>.txt` on the server so the user can paste the output.

## Configs

- Since 2026-10-13 only torch configs (the GetFEM / Warp ones are in the git history): per patient `configs/<p>_torch.json` (no wall, `results/<p>`) and `configs/<p>_wall_torch.json` (`wall: "reference"`, `results/<p>_wall`), each running the whole pipeline (`check_inputs` → `fem_fit`), fit in `fem/fit_torch`, `lung_fem_fit_torch.vtp`, sequence `sequence_fit_torch/`. Redo only the FEM with `--steps fem_setup fem_fit`. To run another core set `fem_fit.solver`; the validation scripts (`check_jacobian.py --solver`, `compare_warp_getfem.py`) drop the options the other core does not accept (`solvers.options_for`: torch's `device` / `material*` for GetFEM, `linear_solver` to / from GetFEM).
- Identical FEM settings everywhere (best found on karl04): `mesh_size_mm` 14, ν = 0.40 fixed, `load_steps` 8, `jacobian: "analytic"`, `time_budget_min` 600, clamp `["hilum", "airways", "arteries", "veins"]` + `clamp_still_mm` 5, `fem_setup.align: "none"` + `fem_fit.loss: "plane"` (~2–3 mm from the real collapsed surface instead of 5–15 mm aligned). Wall configs add `wall_tol_mm` 5, `outer`, `nodal`, adaptive steps, `wall_settle_mm` 0.5, `wall_max_updates` 15, `slow_ramp` false, `max_solve_s` 120.
- Patients: karl01, karl10, karl17 (left lung), karl02, karl04, karl06, karl08 (right; karl06 added by the user, karl01 / 02 / 08 on 2026-10-12, copied from karl04; karl10 / 17 on 2026-10-14, copied from karl01; karl18 / 23 / 24 on 2026-10-21, copied from karl01 (left-lung file names), not run yet), patient_2, patient_10 (left). Registration rigid + affine + B-spline everywhere except patient_10 (rigid + B-spline).
- `scripts/run_all.sh [case ...] [-- main.py args]` runs `configs/<case>_torch.json` for each case in parallel (default: all), logs in `logs_run_all/`.
- Results kept after the 2026-10-13 cleanup (torch, current configs; mean point-to-point / along-the-normal error on the free points, K reached): karl01 5.65 / 2.43 (K 25, target error reached), karl01_wall 5.49 / 2.26; karl06 6.87 / 2.29 (K 14), karl06_wall 6.88 / 2.26; karl10 8.07 / 2.35, **karl10_wall 11.51 / 4.82 stopped at K = 8** (15 failed, 31 min); karl17 7.53 / 2.27 (K 25), karl17_wall 8.17 / 2.55 (K 14, 15 failed, 32 min); karl02 7.90 / 2.49, **karl02_wall 13.26 / 5.06 stopped at K = 8** (9 of 31 solves failed, "first solve failed" at K = 14 and 25, 19 min: to look at); karl04 10.09 / 2.75, karl04_wall 9.65 / 2.80 (35 min, 14 of 79 failed); karl08 6.29 / 2.19, karl08_wall 6.24 / 2.12; patient_10 9.97 / 2.10, patient_10_wall 9.79 / 2.12. karl04 / patient_10 no-wall results are the former `_noalign` test runs moved in place (same settings as the current configs; their log is appended to `logs/pipeline.log`). patient_2 / patient_2_wall: only meshes, hilum and registration kept (old aligned fits deleted); karl06: no results downloaded yet. All earlier experiment fits (point, plane01, vessels, still, warp, aligned) were deleted.
- With a wall all variants use `wall_update: "outer"` (GetFEM can only do that); `"newton"` (Warp/torch) is under test, see below.
- Unknown keys raise an error; `_`-prefixed keys are comments; paths: `data_dir`/`output_dir` relative to the JSON, inputs to `data_dir`, outputs to `output_dir`, `null` inputs come from the previous step.

## Pipeline essentials

- Steps in order: `check_inputs` (1b) → `surface_mesh` (2) → `hilum` (3, before registration: it provides the landmarks) → `registration` (4) → `fem_setup` (5a) → `fem_fit` (5b).
- All surfaces inside the pipeline are **LPS**.
- Registration is done on **labels**: surfaces are rasterised on a common grid, elastix registers the masks, the transform is applied to the collapsed mesh's nodes by per-node inversion. Non-converged nodes (Interpolated = 1) and flipped-triangle nodes (`unflip`, Interpolated = 2) are re-interpolated harmonically.
- `fem_setup` is FEM-library-free: clamp at the hilum (radius = hilum sphere × `anchor_radius_factor`), rigid alignment of the target, optional wall (`"reference"` = the registered inflated surface), Ward pressure regions per level K = 1, 4, 8, 14, 25, 40, Gmsh tet mesh → `fem/setup/problem.npz` (`CollapseProblem`).
- `fem_fit`: coarse-to-fine levels, scipy `least_squares` (TRF) on p/E per region (+ ν if `free_nu`), with the analytic Jacobian from the core. Stops: per level (lsq tolerances, or `level_patience`/`level_min_improve` plateau counted in optimiser iterations), whole fit (`target_error_mm`, `time_budget_min`, `fit_patience`/`fit_min_improve`). Checkpoint `best_state.npz`, watchdog at budget + `hard_grace_min`.

## FEM cores (`pipeline/collapse/solvers/`)

Model: compressible Neo-Hookean, non-dimensional with E = 1 (`[c1, d1] = [mu/2, K/2]`, `common.mat_params`), unknowns q = p/E; follower pressure (Nanson, current area vector / 3 per node on P1 faces), positive q pushes inward (`fit.sign_check` verifies); u = 0 on clamped faces; wall = penalty `k/(2e)(pos(g)^2 - pos(g-e)^2)` with linearised gap g = u·n − (n·Us − φ + allow).

| File | Role |
|---|---|
| `base.py` | `ForwardSolver` interface: `set_regions`, `solve(q, nu)` → surface displacement, `jacobian(q, nu, with_nu)`, state, `export_volume`. `fit.py` talks only to this. |
| `__init__.py` | registry `getfem`, `warp`, `torch` (lazy import) or `"module:Class"` |
| `common.py` | `DEFAULTS` solver options, `mat_params`, `dmat_params_dnu` |
| `getfem_solver.py` | GetFEM core (P1/P2, MUMPS); Jacobian via `K_t dU/dq = d(rhs)/dq`, d/dν by central difference; wall only `"outer"` |
| `nodal.py` | `NodalSolver`: everything except assembly for P1 cores that assemble themselves — mesh prep, sparsity, `_csr(Kb, Fb)`, `LinearSolver` (cudss / pardiso / scipy), GetFEM-replica Newton (criterion `min(|R|_1, |dU|_1/|U|_1)`, "simplest" line search), load path with `warm_substeps`, wall `"outer"` or `"newton"`, Jacobian, face quadrature IM_TRIANGLE(3) |
| `warp_solver.py` | Warp kernels: hand-derived `nh_P` / `nh_dP`, pressure, wall; `_dfint_dnu` uses linearity in (c1, d1) |
| `torch_solver.py` | `TorchFEM` (assembly only, differentiable, any device/dtype incl. mps float32: physics loss for the GNN) + `TorchSolver` (cpu/cuda float64; refuses mps). Stress and tangent by `torch.func` (vmap(grad), vmap(jacfwd(grad))), d/dν by `jvp` |
| `materials.py` | torch materials as energy W(F, nu, **params) with E = 1: `neo_hookean` (= GetFEM/Warp), `mooney_rivlin` (example, unvalidated). New material = new function + entry in `MATERIALS`; options `material`, `material_params` |

Block layout for `_csr`: `Kb[t, a, b, i, j] = dR[node a, i] / dU[node b, j]` (n_tets, 4, 4, 3, 3); `Fb` the same per face (n_faces, 3, 3, 3, 3), pressure + wall summed.

Changing material elsewhere: Warp needs new `nh_P`/`nh_dP` and `_dfint_dnu`; GetFEM needs another law name in `add_finite_strain_elasticity_brick` and matching `mat_params`. A fitted material parameter (beyond ν) needs `fit.py` changes too.

## Validation status (karl04, 1310 nodes, 6509 tets, K = 4)

- GetFEM analytic Jacobian vs FD: ~1e-8. Full fit 2-point vs analytic: 8.873 vs 8.824 mm mean error, 1416 vs 71 solves, 111 vs 7.9 min.
- Warp vs GetFEM, no wall: rel ~1e-15; Jacobian vs FD 1e-8, vs GetFEM 1e-13.
- Torch vs GetFEM, no wall: rel 7e-16–2e-14 up to p/E 2; Jacobian vs FD 2e-9–1e-8, vs GetFEM 1e-13 (ν 1e-10, GetFEM's FD).
- Torch / Warp vs GetFEM with wall (`outer`): rel ≤ 1e-11, identical penetrations; all cores stop at the same load (50 % at p/E 0.2, 10 % at 0.5, none at 1.0) → limitation of `outer`, not of the cores.
- Forward solve times: GetFEM 70–107 s (Xeon), Warp + cuDSS 3.2 s, torch + cuDSS 5.7 s (assembly 3.2 s of it), torch + scipy 11.4 s, torch + pardiso 27 s (pardiso slow on this server: prefer scipy on CPU-only machines). Jacobian: torch 0.1 s, GetFEM 0.46 s.

- Full no-wall fits, Warp and torch (2026-10-04, `run_all.sh`, results in `results/<p>/fem/fit_{warp,torch}`): the two cores give identical fits (same evaluations, Newton iterations, errors to 3 decimals); karl04 8.824 mm = the earlier GetFEM analytic fit. Whole fit 1–2 min (GetFEM analytic 7.9 min, 2-point 111 min); torch 0.59–1.19 s/solve, Warp 0.67–1.66 s. Mean error: karl04 8.82 mm (baseline 27.3), patient_10 7.12 (28.6), patient_2 23.0 (78.6; 13 of 40 regions at the p/E bound 3, clamped points 32.8 mm from their target after hilum alignment → registration-limited). Every level stopped on the plateau after few evaluations and the error still fell at every level (fits end on "levels exhausted").

- Rotation seen in the collapse sequences: the target itself is rotated vs the inflated lung after hilum alignment (rigid part karl04 5.5°/15 mm, patient_10 8.1°/22 mm, patient_2 15.9°/53 mm) and the fit reproduces it (fitted 6.2 / 8.2 / 18.8°). Possibly tangential slip of the mask-based correspondence. Added `fem_fit.loss: "plane"` (residual (n nᵀ + w(I − n nᵀ)) d with target vertex normals, `loss_tangent_weight` w; Jacobian projected the same way, core-independent; checked vs FD 1e-10 and end-to-end with a fake linear core) and `main.py --set key=value` overrides. Summary now has `mean_normal_err_mm`, `best_loss_err_mm`, `rigid_rotation_deg` / `rigid_shift_mm` (target, fitted), `baseline_loss_mm`; the fitted surface has `NormalError_mm`. To run: no-wall cases with `--set fem_fit.loss=plane fem_fit.run_name=fit_torch_plane fem_fit.output=lung_fem_fit_torch_plane.vtp`, compare with `fit_torch`.

- Point vs plane on the no-wall cases (torch): `plane` matches the shape better (normal error karl04 2.8 vs 5.0 mm, patient_10 2.5 vs 3.5, patient_2 7.9 vs 11.3) but the lung rotates more (14–25° vs 6–19°; point-to-point error 11–30 vs 7–23 mm): without the point correspondence the optimiser uses the cheap rigid rotation about the hilum. So the rotation comes mostly from the model's weak support, not from the registration. Next: larger clamp. `fem_setup.anchor_point` now takes a list (union of balls; only the first grows), configs use `["hilum", "airways", "arteries", "veins"]`: karl04 25 → 62 clamped triangles, patient_10 23 → 53, patient_2 unchanged (22, vessel balls inside the grown hilum ball). Hilum-only is identical to before.

- Vessel clamp results (torch, 2026-10-04): rotation lower (karl04 point 6.2 → 4.1°, plane 16.4 → 9.7°; patient_10 point 8.2 → 6.6°), point-to-point error slightly higher (karl04 8.82 → 9.16, patient_10 7.12 → 7.95) because the larger clamp keeps more points away from their targets (clamped-point mismatch karl04 3.2 → 7.8 mm, patient_10 2.6 → 5.5 mm); patient_2 unchanged. The user's view: clamped points must not move, that mismatch is expected and should not count in the error; result "quasi decente", kept. Ideas the user wants to try later: (1) exclude clamped points from the loss and the reported errors; (2) also clamp surface points that barely move between inflated and collapsed (|target − inflated| < ~5 mm: karl04 40 points / 50 triangles, patient_10 83 / 117, patient_2 none — its registration leaves no point within 8 mm).

- Implemented (2026-10-05): clamped points left out of the fit residual and of all reported errors (`fit.Loss` with a free mask; it does not change the optimum, the clamped residual being constant; summary `metric_points`, `clamped_points`, `clamped_mismatch_mm`), and `fem_setup.clamp_still_mm` (triangles whose 3 points move < threshold after the target alignment are clamped too; `Clamped` cell data 1 = anchor, 2 = still). All configs: `clamp_still_mm: 5.0`. Not yet run on the server; old fits' `mean_err_mm` included the clamped points, so compare with care.

- `clamp_still_mm: 5` results (torch, point; compared on the same free points): patient_10 +28 triangles / 26 points, mean 8.33 → 7.58 mm, p95 14.8 → 13.7, max 19.3 → 17.0, rotation unchanged (6.7° vs target 5.7°); karl04 and patient_2 got no still triangle (karl04 none below 5 mm with the vessel-based alignment; patient_2 registration) so their fit is unchanged. Possible: `clamp_still_mm` 8 for karl04. The user judges the result good.

## Open problems / next steps

The plan from the fits to inflated-only collapses and the GNN (agreed 2026-10-15) is in [NEXT.md](NEXT.md); keep its to-do list current.

- Wall configs: `wall_tol_mm` 2 → 5 mm (2026-10-05, user's choice of a global enlargement; a per-point allowance that just contains the target was the alternative). To be run with the new clamps (vessels + still).
- Wall fits, torch, vessel clamp + still points + `wall_tol_mm` 5 (2026-10-05, `results/<p>_wall/fem/fit_torch`; no-wall references on the same setup = `fit_torch_still`, same free points):
  - patient_10: 7.58 mm, identical to no wall, K = 40, 2.4 min, 0 failed solves; the wall barely acts (1 target point beyond the 5 mm allowance). The large clamp removed the rigid-mode failures here.
  - karl04: 10.23 mm (no wall 9.29), stopped at K = 14 on `fit_patience`, 76 min, 38 of 87 solves failed (2 min each: `max_solve_s`). 22 target points lie beyond the allowance (up to 20.6 mm, setup: 21 up to 15.3 mm), unreachable.
  - patient_2: 24.8 mm (no wall 23.0), K = 14, 57 min, 27 of 63 failed.
  - Cause of the failures and of the early stop (from `history.csv` and the log), fixed the same day:
    - the warm start lost its load: `set_regions` and `set_state` cleared `q_last`, so a solve at a new level jumped from the best state to its load in one step, and `adaptive_path` halved a zero-length path (same jump every time) before the ramp from zero, which fails at fit loads (rigid mode) → 2 min lost per failure. Now the load path runs on the face pressures (`fq_last`, partition-independent) and `set_state(state, load)` takes the p/E per triangle (nodal cores; GetFEM ignores it). Checked with the numpy core: a simulated "large step fails" now halves 0.083 → 0.01 and converges; before it retried 0.083 seven times and failed;
    - every trial started from the last converged solve, often a rejected trial: now `fit.run_level` re-anchors the solver to the state of the lowest lsq cost in the level (the optimiser's iterate) before each solve, so a shrinking trust region means a shrinking load step;
    - when a level's first solve failed, the Jacobian fell back to finite differences (K more failing solves: 30 min at K = 14): now the level stops ("first solve failed") and the fit goes on.

    To re-run: the three `_wall_torch` configs (`CORES=torch bash scripts/run_all.sh karl04_wall patient_10_wall patient_2_wall`).
  - The unreachable target points come from the target alignment, not from the data: as in the scan no karl04 / patient_10 target point is beyond the 5 mm margin (max 2.2 / 1.3 mm), after `hilum_rigid` 22 / 1 (max 20.6 / 6.4 mm); the alignment rotates karl04's target ~16° and the clamped points gain nothing (7.84 → 7.83 mm). It exists to remove the rigid part of the node correspondence (karl04 14° → 4.5°), which tangential slip of the registration creates. Test (user, 2026-10-05): `configs/<p>_wall_noalign_torch.json` for karl04 / patient_10 = `align: "none"` + `loss: "plane"` (the plane loss ignores the tangential slip, clamp and wall limit the rotation), own `fem_setup` in `results/<p>_wall_noalign` from the `<p>_wall` meshes; point-loss variant by `--set`.
  - Result of that test (torch; all fits measured against the collapsed lung as in the scan, surface distance fitted → target / point-to-point / fitted points > 5.5 mm beyond the wall): karl04 aligned no wall 14.6 / 27.2 / 15, aligned wall 14.8 / 27.3 / 0, unaligned wall point 7.7 / 13.8 / 4 (stopped at K = 4, solves failing), unaligned wall plane **3.4** / 9.3 / 1 (K = 25, 20 of 73 solves failed, `unsettled`); patient_10 aligned 5.2 / 9.9, unaligned point 3.1 / **7.0**, unaligned plane **2.3** / 8.6, no failures, < 1 min. The alignment was making the fit chase an artefact of the registration. Without it `clamp_still_mm` also clamps far more (karl04 62 → 148 triangles), since near the hilum the unaligned target coincides with the reference. Adopted (user, 2026-10-05) in all wall configs: `align: "none"`, `loss: "plane"`, `wall_settle_mm` 0.5 and `wall_max_updates` 15 against the unsettled rounds (to be checked). The `_wall_noalign` test configs were removed (= the wall configs now; their results stay in `results/<p>_wall_noalign`). Next test: the same without wall, `configs/<p>_noalign_torch.json` (karl04, patient_10) → `results/<p>_noalign`; the no-wall standard configs are still aligned + point and are 14.6 mm (karl04) from the real collapsed surface.
  - Results with the adopted wall configs (torch, 2026-10-05; surface distance to the collapsed lung as in the scan / point-to-point / fitted points > 5.5 mm beyond the wall, max): karl04 wall **3.08** / 8.74 / 3 (6.0 mm), K = 40 reached ("levels exhausted"), 14 of 79 solves failed (`unsettled` 153, `time` 61), 35 min (settle 0.2: 3.37 mm, K = 25, 20 of 73 failed, 47 min); patient_10 wall 2.27 / 8.55 / 3 (5.8 mm), 0 failed, 0.5 min. No-wall unaligned plane (`<p>_noalign_torch.json`): karl04 3.28 / 9.09 but 9 points out of the cavity up to 22.2 mm, 0.5 min; patient_10 2.26 / 8.69, 3 out up to 14.1 mm. So without alignment the wall barely changes the shape error but keeps the lung inside the cavity; aligned no-wall fits were 14.6 (karl04) / 5.2 mm away. karl04 with wall still has failing solves (rounds not settling, 2 min each) → most of its 35 min. Adopted in the no-wall configs too (user, 2026-10-06); the `<p>_noalign_torch.json` test configs were removed (results stay in `results/<p>_noalign`).
- **Idea (user, 2026-10-05): predict the collapse of a new patient's normal lung with the fitted pressures.** Inputs: only the inflated CT (lung surface + airways / vessels for the hilum). Pipeline: surface mesh of the inflated lung (step 2 on the inflated input), hilum + vessel clamp from the same scan, `fem_setup` without target (no registration, alignment or still points; optional wall = the inflated lung), then a new step `fem_predict`: one forward solve with transferred pressures → predicted collapsed lung + Slicer sequence. The real work is transferring the pressures (defined on the source patients' regions): (1) uniform mean p/E (~0.9) as a baseline; (2) pressure as a function of anatomical coordinates (apex–base, anterior–posterior, distance from the hilum, lobe) learnt from the fitted patients, no inter-patient registration; (3) atlas: register all inflated lungs to a template with elastix, average the fitted maps there, map back. Mirror left/right (karl04 right, patient_10 / patient_2 left). Validation: leave-one-out, e.g. predict patient_10 from karl04's mirrored pressures and compare with its real collapsed lung (fit error 7–9 mm is the reference). This is the GNN's task; the FEM prediction is the physical baseline and the training-data generator.
  - Summary of the 8 no-wall fits (2026-10-14; p/E area-weighted over the free triangles, collapse = 1 − V_collapsed / V_inflated of the meshes): collapse 24–58 %, mean p/E 0.44–1.05, and **mean p/E ≈ 0.06 + 1.65 × collapse** (residual ≤ 0.06), so the collapse fraction sets the pressure level. The pattern relative to the mean is consistent only anterior → posterior (anterior third 1.13–1.31, posterior 0.73–0.92, all patients); the apex is lower in 6 of 8 (0.77–0.88); medial / lateral and distance from the hilum are inconsistent (the latter is affected by the clamp). The regions themselves are per-patient Ward clusters of the target displacement, so they do not exist for a new lung. A uniform p/E (K = 1) gives 6–15 mm normal error vs 2–3 mm fitted. The clamped fraction of the surface varies 5–37 % (mostly still points, which need the target), and that changes the fitted p/E too. A linear fit of the K = 40 p/E on the three axes explains only 12–23 % of their variance (similar leave-one-out), so how much shape the gradient captures has to be measured with the FEM.
  - Implemented (2026-10-14, user's choice: patient supine, anterior = up): `fem_fit.pressure_model: "gradient"` (`fit.PressureModel`, `axis_coords`): q_tri = mean + Σ slope_a ξ_a, ξ normalised by the lung's extent, levels = uniform then mean + slopes (`gradient_axes`, default ap + si); each free triangle is one core region and the Jacobian is J_tri · Φ (chain rule), so no core changed. `"regions"` unchanged (same code path, Φ = indicator). `best_state.npz` keeps q / tri_labels as the core sees them (collapse_sequence works unchanged) plus `params`; summary `pressure_model`, `gradient`. Tested on the Mac with a fake linear core on karl10's setup: a synthetic gradient (0.8, ap 0.5, si −0.3, ml 0.2) recovered to 1e-15 with analytic and 2-point Jacobian, and with free ν; the regions model gives the same levels as before. Fixed on the way: the 2-point step was relative to |x|, ~1e-19 for a slope starting at 0 (zero columns); now `lsq_diff_step` × max(|x|, 0.1). First run (2026-10-14, no wall, ap + si, `fem/fit_gradient`, `lung_fem_fit_gradient.vtp`), normal error uniform → gradient (regions): karl01 5.72 → 5.01 (2.43), karl02 14.90 → 8.56 (2.49), karl06 5.99 → 5.14 (2.29), karl08 8.84 → 7.11 (2.19), karl10 10.31 → 6.84 (2.35), karl17 9.08 → 7.23 (2.27): the gradient closes 22–51 % of the uniform → regions gap, < 1 min each. Slopes: ap > 0 in all (absolute 0.17–0.43 p/E, mean 0.31 ± 0.09; relative +19 to +68 %), si < 0 in 4 of 6 (karl02 ≈ 0, karl10 +0.16). The relative slopes follow the collapse (corr −0.91 ap, +0.97 si: 24–30 % collapse → ap +47–68 %, si −45 to −61 %; 54–58 % → ap +19–31 %, si −2 to +16 %), but the low-collapse patients are also the heavily clamped ones (20–37 % of the surface vs 5–10 %), so this may be the clamp. karl04 / patient_10 ran on the server's old aligned `fem_setup` (the noalign setups were moved in place only locally): to redo with `--steps fem_setup fem_fit` first. On karl04 level 2 lowered the rms (13.52 → 13.08) but not the mean error, which picks the best, so the uniform state was kept. Next: leave-one-out with mean + slopes from the other patients (needs a forward-only mode: fixed parameters, no fit).
- **Registration optional (user, 2026-10-23): `registration.method: "none"`.** The step then remeshes the inflated surface with the `surface_mesh` settings (`surface_mesh.remesh`, the step-2 code without I/O) into the same `output`, own nodes. `fem_setup` detects the different topology and switches to closest-point matching (user's choice over "stop with an error" / "target-free setup"): `align` must be `none`, `target` = closest point of each reference point (still points, clustering field `u = cp − X_ref`, `cluster_field` ignored), `problem.npz` gains `target_points` / `target_tris` (`CollapseProblem.closest_point`, `target_surface`; old files load unchanged). `fit.Loss.residual` re-matches every fitted point to its closest point (`WallDistance.closest`: interpolated vertex normal) at every evaluation; the Jacobian holds cp / n fixed (Gauss-Newton of point-to-plane ICP). One-sided (fitted → target); a symmetric term (target → fitted, Jacobian through the barycentric weights) is possible if the fit leaves parts of the target uncovered. Checked on the Mac with a fake linear core on karl04's setup and an analytic sphere target: cost gradient vs FD 1.4e-8 (plane and point); node mode bit-identical. Not yet run on the server.
- patient_2 excluded from the wall runs (user, 2026-10-05).
- **Idea (user, 2026-10-05): vessels / airways in the same simulation.** Preferred route: embedded elements. Each vessel-surface node lies in a lung tet; its displacement = the P1 interpolation of that tet's nodes. Level A (passive, post-processing, all cores, needs only the vessel surfaces): deformed vessel trees after the fit and per sequence frame — likely the most useful output for VATS. Level B (vessels stiffen the lung): label the lung tets the vessels cross (dilated by 2–3 mm) as stiffer (per-element material: easy in torch, an array in the Warp kernel, a P0 field in GetFEM), rather than tetrahedralising the vessel trees (fragile in Gmsh). Conforming multi-body meshes or separate bodies with contact: not worth it. Open question before starting: which scan are `lung_airways/arteries/veins.vtk` from? The `hilum` step assumes the collapsed one; then they must be pulled back through the fitted map (exact per tet, the P1 map is linear) to embed them in the inflated reference. If they (also) exist in the inflated scan, the FEM predicts their collapsed position, and comparing with the collapsed-scan vessels validates the model on points not used in the fit.

- **Wall convergence.** First test of `wall_update: "newton"` (torch, raw distance gradient as normal): worse than `outer` (10 % of p/E 0.2 vs 50 %). Cause found: `vtkImplicitPolyDataDistance` gradient = direction to the closest point, which jumps across edges/vertices, and with `wall: "reference"` every lung node starts on a wall vertex. Fixed in `WallDistance.__call__`: interpolated vertex normal at the closest point, phi measured along it (sphere test: normal jump per 0.02 mm step 0.07° vs 8.8° before). This changes the wall data of all cores (GetFEM too). Re-test with the continuous normal: torch = GetFEM still (1e-14), but the reach barely changed (`outer`: 50 / 20 / 0 / 10 / 0 % of the five cases for GetFEM; `newton`: 20 / 10 / 0 / 0 / 0 %) → the normal was not the main cause. With a wall the Jacobian vs FD check is meaningless under `outer` (FD dominated by the `wall_settle_mm` tolerance); torch vs GetFEM Jacobian agrees to 1e-14. Next: `scripts/wall_diagnose.py` on `karl04_wall_torch.json` to see the failure reason (inverted elements, maxit, line search). Meanwhile, from the colleague's 6.2 script: wall configs use `slow_ramp: false`, `max_solve_s: 120`, `wall_settle_mm: 0.2`, `wall_max_updates: 4` (the default ladder could spend ~65 Newton runs on one hopeless solve; a compare run took > 3 h). `wall_diagnose.py` on karl04_wall torch (p/E 0.2): no inverted elements (min J 0.81), every failure is `maxit` from oscillation (`newton`: residual 2-cycle 20.4 ↔ 22.0; `outer`: fine until a wall re-linearisation, then line-search stalls); penetration grows with load to 2.3 mm beyond the margin. Suspected cause: the wall term uses the IM_TRIANGLE(3) face rule with a negative centre weight (−27/48), which can pull nodes outward and make the contact tangent indefinite. New option `wall_quadrature: "nodal"` (vertex rule, weights 1/3; GetFEM `IM_NC(3,1)`), default still `"face"`. Tested (torch, p/E 0.2): penetration 0.04–0.25 mm instead of 0.4–2.3 mm up to 80 % load; `outer` then reached 100 % but only by accepting unsettled rounds (penetration 12 and 98 mm, max|U| 203 mm): the round loop accepted the state after `wall_max_updates` even if not settled. Fixed: unsettled = failure (`fail_unsettled`) in nodal.py and getfem_solver.py. `newton` + nodal still fails at 50 % with a residual 2-cycle (tangent misses the normal's rotation / closest-point changes) → use `outer`. Next test (all 3 loads, torch): every Newton converged quadratically and the state stopped changing (penetration 0.04–0.08 mm), yet every solve failed `unsettled`: the settle measure |ΔwallG| jumped by a constant 1.5 / 7.9 mm per round because G = n·u − φ + allow moves with the normal at points far from the wall (closest point toggling near the cavity's medial axis). Fixed: `common.wall_gap_change` = |previous linearised gap − true gap| over points with gap > −1 mm. With the gap-error measure (v4): p/E 0.2 in 10 increments reaches 100 % with 1–2 rounds and ~0.7 s per increment, penetration ≤ 0.18 mm; p/E 0.5 / 1.0 fail at the same load (p/E ≈ 0.2) when the increment is 0.05 / 0.1: points slide far along the tangent-plane linearisation, the rounds diverge (gap error 0.7 → 4 → 62 → 188 mm) and the lung swings about the hilum. Fix: `adaptive_steps` (`common.adaptive_path`: halve on failure, double on success, down to `min_load_step`), on in the wall configs. v5 (adaptive steps): p/E 0.2 fine (0.7 s/increment, pen ≤ 0.18 mm); p/E 0.5 / 1.0 still fail around p/E ≈ 0.22–0.3 even at steps of 1/64: after 2–3 wall rounds the lung swings about the hilum (max|U| 24 → 75 → 180 mm, gap error 0.2 → 2 → 60 → 188 mm). Diagnosis: near-mechanism — only a small hilum clamp and frictionless contact, so an unbalanced pressure field rotates/slides the lung along the wall almost for free (same mode as max|U| ≈ 380 mm without wall at p/E 0.2). Fitted pressures need p/E ≈ 0.1–3 (karl04 no wall: mean 0.94, max at the bound 3.0), so limiting q is not an option. Options put to the user: larger / anatomical clamp (hilum + pulmonary ligament), weak elastic support springs, or dynamic relaxation. Wall configs now: `adaptive_steps` true, `outer`, `nodal`, `wall_max_updates` 10, `wall_settle_mm` 0.2, `slow_ramp` false, `max_solve_s` 120. To check: `wall_diagnose.py` at p/E 0.2 / 0.5 / 1.0, then `compare_warp_getfem.py --core torch --wall-only --jac-q 0.1` (GetFEM with `IM_NC(3,1)`), then full fits. 6.2 also has `--warm-start` from a previous run's pressures (not ported) and uses a separate wall file (original inflated segmentation), so lung nodes are not on wall vertices.
- Full fits with torch/Warp (`karl04_torch.json`, then wall and patients) compared with GetFEM's `fem/fit/result_summary.json`.
- max|U| ≈ 380–400 mm without wall at p/E 0.2–1 (likely non-physical: the lung is unconstrained except at the hilum).
- Wall penetration ~1.7 mm beyond the allowed position (penalty stiffness).
- Pressures of fine levels (K = 40) poorly identifiable; `reg` / stopping at K ≤ 25 are options.
- `level_min_improve` 0.005 may stop levels early with the cheap analytic Jacobian (0.001–0.002 suggested).
- patient_2 registration still approximate (see README Known issues).
- Possible: speed up torch assembly (`torch.compile`, closed-form tangent for Neo-Hookean); contact-in-Newton for GetFEM would need a custom Newton loop.

## Pipeline details (moved from the README on 2026-10-05)

The README was cut to a quick-start; the full step descriptions it had are kept here. Headings are one level down.

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

- **Inputs:** `lung_inflated_mesh.vtp` (reference), `lung_collapsed_mesh.vtp` (target) and `hilum_anchor.mrk.json`. Reference and target must have the same nodes and triangles, i.e. the registration output.
- **Clamped region:** surface triangles within `hilum sphere radius × anchor_radius_factor` of the hilum (default 2.0 → ~30 mm on patient_10). The radius grows while the region has fewer than `anchor_min_points` points or is nearly flat, otherwise rigid motions would stay free.
- **Alignment:** `align: "hilum_rigid"` moves the target rigidly so that its clamped points best match the reference ones (consistent with u = 0 there); `rigid` uses all points, `none` keeps it as is.
- **Cavity wall (optional, `wall`):** closed surface the lung may not leave during the collapse (the chest cavity). `"reference"` (used in the configs) takes the registered inflated surface itself, so every point starts on it; a file name (e.g. `lung_left.vtk`) takes that surface. A point may go at most `wall_tol_mm` (2 mm) beyond it, plus however far it is already outside in the reference. The log reports how many target points lie beyond: those are an error the fit cannot remove. `wall: null` = no wall (as 6.1).
- **Pressure regions:** connectivity-constrained Ward clustering of the displacement with the rigid part removed (`cluster_field`), for every level in `levels` (1 → 40 regions). The regions do not depend on the pressures, so they are all computed here.
- **Volume mesh:** Gmsh tetrahedra (`mesh_size_mm` inside), with the surface points and triangles unchanged (checked).
- **Output:** `fem/setup/problem.npz`, read by `fem_fit` and passed to the core, plus files to inspect: `reference.vtp` (cell data `Clamped`, `Regions_K*`; point data `WallDistance_mm`), `target_aligned.vtp`, `volume.vtu`, `setup_summary.json`.

**`fem_fit`** (hours with GetFEM):

- **FEM core:** `solver` names a class in `pipeline/collapse/solvers/` (`"getfem"`, `"warp"`, `"torch"`) or any `"module:Class"`; `solver_options` go to that class. Every core implements `ForwardSolver` (`solvers/base.py`): hyperelastic material with E = 1 (compressible Neo-Hookean; `torch` also others) (so the unknowns are p/E per region and ν), u = 0 on the clamp, follower pressure on the other triangles, positive pressure pushing inward, and the wall as rigid contact if the problem has one; `solve(q, nu)` returns the surface displacement. A core without wall support refuses a problem with a wall. To add a core, write one file in `solvers/` and register it in `solvers/__init__.py`; a P1 core that assembles the system itself only needs `_assemble` on top of `nodal.NodalSolver`.
- **Wall contact, `wall_update`:** `"outer"` (both cores) alternates Newton solves with the wall linearisation fixed and re-linearisations, up to `wall_max_updates` rounds until the gap changes less than `wall_settle_mm`; each solve is several Newton runs, which is why the fit with a wall is much slower. `"newton"` (Warp and torch) re-linearises the wall at every residual evaluation, so the contact is part of a single Newton solve; at convergence it solves the same equations. The contact normal is continuous: area-weighted vertex normals of the wall interpolated at the closest point, and the gap measured along it (`WallDistance`); the raw distance gradient (direction to the closest point) jumps across edges and vertices, and the `"reference"` wall has the lung's nodes on its vertices, so Newton could not converge on it. With a wall the configs also set `warm_substeps: true`, so a failed step is retried in sub-steps before restarting the ramp from zero.
- **GetFEM wall contact:** penalty force on the pressure faces, zero inside the wall and growing smoothly beyond it (`wall_stiffness`, `wall_eps`); the gap is re-linearised between Newton solves until it settles (`wall_max_updates`, `wall_settle_mm`). Warm start as in 6.1: one direct step from the last converged state, then a ramp from zero. `warm_substeps: true` adds, before the ramp, `load_steps` and 3×`load_steps` sub-steps from the last state (as 6.2): useful with contact, but each failed attempt costs Newton runs up to `newton_maxit`.
- **Method:** one sign check (q > 0 must collapse), then the levels coarse → fine, each warm-started from the best so far. Per level: bounded least squares (`lsq`; Jacobian from the core when `jacobian: "analytic"` and the core provides one, i.e. one linear solve per region instead of one nonlinear solve; `"2-point"` = finite differences as before) on the point-to-point error plus a smoothness penalty between adjacent regions (`reg`), or Nelder-Mead (`nm`); ν optionally free.
- **Stops:**
  - *a level* ends when the optimiser converges (`lsq_*` tolerances) or when its best error improved by less than `level_min_improve` (relative, default 0.5 %) over the last `level_patience` optimiser iterations (default 3, i.e. 3 × (parameters + 1) solves); the fit then moves to the next K. On karl04 this would have cut 84 to ~32 min with the same error up to K=14 (the lsq tolerances alone kept going 50–200 solves past the last improvement);
  - an *iteration* is one Jacobian for lsq, `parameters + 1` solves for Nelder-Mead;
  - *the whole fit* ends on `target_error_mm`, `time_budget_min`, `fit_patience` levels in a row improving by less than `fit_min_improve`, the end of the levels, or Ctrl+C / SIGTERM. The best result is checkpointed on improvement, and a watchdog kills the run at budget + `hard_grace_min`, so a forced stop keeps the best solution.
- **Warp core (`"solver": "warp"`):** same physics, options and load path as the GetFEM core, P1 only, assembled with NVIDIA Warp kernels in float64 on GPU (or CPU); extra options `device` and `linear_solver` (`auto`, `cudss`, `pardiso`, `scipy`). Newton reproduces GetFEM's classical Newton with the "simplest" line search, and the wall term uses the IM_TRIANGLE(3) face rule; both are to be confirmed against GetFEM with `python scripts/compare_warp_getfem.py --config configs/<p>.json [--timing] [--fit]` (forward solves with and without wall, Jacobian, full fit, timing).
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

- **patient_2 registration is still approximate** (segmentations corrected on 2026-10-01; ×7.2 in volume, rigid + affine + B-spline):
  - Dice 0.953 on the mesh, volume 5342 / 5350 mL; 6 nodes where the inverse does not converge are interpolated.
  - 7 flipped triangles, removed by re-interpolating 25 nodes 5–38 mm from the hilum (moved up to 138 mm: the registration had folded them far out). The surface then meshes cleanly, with small tetrahedra (min 4 mm³ against ~20 mm³ on patient_10).
  - The hilum region is poorly matched: nodes within 30 mm of the hilum move ~88 mm; after `hilum_rigid` alignment the clamped points are ~23 mm from their target, and 49 target points lie beyond the wall (up to 15 mm).
  - Bending-energy regularisation, the forward direction and signed distance maps did not help on the old segmentations. Next to try: a larger `landmark_weight` for this patient only, `align: "none"`.
- **Node correspondence is not anatomical.** Mask registration only matches the boundaries, so nodes can slide tangentially along the surface. Keep this in mind for the point-to-point loss of the inverse FEM.


### Config sets

Until 2026-10-13 each case had three configs: `<p>.json` (GetFEM, whole pipeline), `<p>_warp.json` and `<p>_torch.json` (only `fem_fit`, on the base `fem_setup`, fits in `fem/fit_warp`, `fem/fit_torch`), for `karl04`, `patient_2`, `patient_10` with and without wall. Now only the self-contained torch configs remain (see "Configs" above); the others are in the git history.

### Input data

One folder per patient in `Input_Data/`. All files are surface models exported from 3D Slicer (`.vtk`, LPS), from the same scan:

| File | Used by |
|---|---|
| `lung_left_collapsed.vtk` | collapsed left lung, steps 2 and 3 |
| `lung_left.vtk` | assumed inflated left lung, step 4 |
| `lung_airways.vtk`, `lung_arteries.vtk`, `lung_veins.vtk` | step 3 |
