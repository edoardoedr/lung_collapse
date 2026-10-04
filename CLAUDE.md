# CLAUDE.md

Guidance for Claude Code in this repository. The user-facing documentation is [README.md](README.md) (pipeline, steps, outputs, known issues) and [configs/README.md](configs/README.md) (every config key with meaning and range): read them for details instead of re-deriving, and keep them up to date when behaviour or keys change. This file holds what those do not: how to work here, the FEM architecture, validation status and open problems.

## Project

Collapsed-lung pipeline for a GNN surrogate of lung collapse in VATS surgery: CT segmentations (3D Slicer) → surface mesh → hilum anchor → registration of the inflated lung onto the collapsed mesh → inverse FEM that fits regional pleural pressures so that the inflated lung collapses onto the collapsed one. The fitted FEM results are the training data of the GNN (planned in PyTorch, possibly PyG).

`pipeline_codes_v1/` holds the original scripts (reference only, being replaced). `pipeline/` is the rewrite; `6.1_lung_inverse_fem_fit.py` there was split into `pipeline/collapse/`.

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
python main.py --config configs/<p>.json                    # steps listed in the config
python main.py --config configs/<p>.json --steps fem_fit    # some steps only
python scripts/check_jacobian.py --config configs/<p>.json [--solver torch] [--options '{...}'] [--compare-fit]
python scripts/compare_warp_getfem.py --config configs/<p>.json [--core torch] [--timing] [--fit] [--wall-only] \
       [--jac-q 0.1] [--core-options '{"wall_update": "newton"}']
python scripts/wall_diagnose.py --config configs/<p>_torch.json [--q 0.2] [--steps 10] [--options '{...}']   # Newton trace with a wall
python scripts/collapse_sequence.py --config configs/<p>.json [--frames 30] [--mode fem|linear] [--volume]
python -m py_compile pipeline/collapse/solvers/*.py scripts/*.py    # the only check possible on the Mac for FEM code
```

Pass `2>&1 | tee <name>.txt` on the server so the user can paste the output.

## Configs

- Per case three configs with identical FEM parameters (best found on karl04: `mesh_size_mm` 14, ν = 0.40 fixed, `load_steps` 8, `jacobian: "analytic"`, `time_budget_min` 600):
  - `<p>.json`: GetFEM, whole pipeline, fit in `fem/fit/`;
  - `<p>_warp.json`, `<p>_torch.json`: steps `["fem_fit"]` only, same `output_dir`, `fem_fit.run_name` = `fit_warp` / `fit_torch`, so they reuse the base config's `fem_setup`.
- Cases: `karl04` (no wall), `karl04_wall`, `patient_2`, `patient_10` (both with wall). patient_2 registration uses rigid + affine + B-spline; patient_10 rigid + B-spline.
- With a wall all variants use `wall_update: "outer"` (GetFEM can only do that); `"newton"` (Warp/torch) is under test, see below.
- Patients' `fem_setup` must be re-run after the mesh-size change (8 → 14 mm) before their fits.
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

## Open problems / next steps

- **Wall convergence.** First test of `wall_update: "newton"` (torch, raw distance gradient as normal): worse than `outer` (10 % of p/E 0.2 vs 50 %). Cause found: `vtkImplicitPolyDataDistance` gradient = direction to the closest point, which jumps across edges/vertices, and with `wall: "reference"` every lung node starts on a wall vertex. Fixed in `WallDistance.__call__`: interpolated vertex normal at the closest point, phi measured along it (sphere test: normal jump per 0.02 mm step 0.07° vs 8.8° before). This changes the wall data of all cores (GetFEM too). Re-test with the continuous normal: torch = GetFEM still (1e-14), but the reach barely changed (`outer`: 50 / 20 / 0 / 10 / 0 % of the five cases for GetFEM; `newton`: 20 / 10 / 0 / 0 / 0 %) → the normal was not the main cause. With a wall the Jacobian vs FD check is meaningless under `outer` (FD dominated by the `wall_settle_mm` tolerance); torch vs GetFEM Jacobian agrees to 1e-14. Next: `scripts/wall_diagnose.py` on `karl04_wall_torch.json` to see the failure reason (inverted elements, maxit, line search). Meanwhile, from the colleague's 6.2 script: wall configs use `slow_ramp: false`, `max_solve_s: 120`, `wall_settle_mm: 0.2`, `wall_max_updates: 4` (the default ladder could spend ~65 Newton runs on one hopeless solve; a compare run took > 3 h). `wall_diagnose.py` on karl04_wall torch (p/E 0.2): no inverted elements (min J 0.81), every failure is `maxit` from oscillation (`newton`: residual 2-cycle 20.4 ↔ 22.0; `outer`: fine until a wall re-linearisation, then line-search stalls); penetration grows with load to 2.3 mm beyond the margin. Suspected cause: the wall term uses the IM_TRIANGLE(3) face rule with a negative centre weight (−27/48), which can pull nodes outward and make the contact tangent indefinite. New option `wall_quadrature: "nodal"` (vertex rule, weights 1/3; GetFEM `IM_NC(3,1)`), default still `"face"`. Tested (torch, p/E 0.2): penetration 0.04–0.25 mm instead of 0.4–2.3 mm up to 80 % load; `outer` then reached 100 % but only by accepting unsettled rounds (penetration 12 and 98 mm, max|U| 203 mm): the round loop accepted the state after `wall_max_updates` even if not settled. Fixed: unsettled = failure (`fail_unsettled`) in nodal.py and getfem_solver.py. `newton` + nodal still fails at 50 % with a residual 2-cycle (tangent misses the normal's rotation / closest-point changes) → use `outer`. Wall configs now: `outer`, `nodal`, `wall_max_updates` 10, `wall_settle_mm` 0.2, `slow_ramp` false, `max_solve_s` 120. To check: `wall_diagnose.py` at p/E 0.2 / 0.5 / 1.0, then `compare_warp_getfem.py --core torch --wall-only --jac-q 0.1` (GetFEM with `IM_NC(3,1)`), then full fits. 6.2 also has `--warm-start` from a previous run's pressures (not ported) and uses a separate wall file (original inflated segmentation), so lung nodes are not on wall vertices.
- Full fits with torch/Warp (`karl04_torch.json`, then wall and patients) compared with GetFEM's `fem/fit/result_summary.json`.
- max|U| ≈ 380–400 mm without wall at p/E 0.2–1 (likely non-physical: the lung is unconstrained except at the hilum).
- Wall penetration ~1.7 mm beyond the allowed position (penalty stiffness).
- Pressures of fine levels (K = 40) poorly identifiable; `reg` / stopping at K ≤ 25 are options.
- `level_min_improve` 0.005 may stop levels early with the cheap analytic Jacobian (0.001–0.002 suggested).
- patient_2 registration still approximate (see README Known issues).
- Possible: speed up torch assembly (`torch.compile`, closed-form tangent for Neo-Hookean); contact-in-Newton for GetFEM would need a custom Newton loop.
