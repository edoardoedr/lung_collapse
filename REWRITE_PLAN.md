# Rewrite plan: the physical collapse model

Plan for a new branch (`physical-model`) that rewrites the collapse part of the pipeline around the physical load model (METHODOLOGY.md section 8). Written 2026-11-16 for the agent that implements it. Read CLAUDE.md first (environments, rules, commands), then METHODOLOGY.md sections 3–8 and analysis/results.md sections 4–5b.

## 0. Why, and what stays

The code grew around the **regions fit**: Ward regions of the patient's own displacement, fitted p/E per region. Gravity was then added on top as an offset to a gradient model (`fit.gravity_offset`, `fem_fit.gravity_E_Pa`, the ap axis written in the code). The physical parameters are spread over `fem_setup`, `fem_fit` and `solver_options`, mixed with numerical settings. The patient configs still say `pressure_model: "regions"`, so every gravity run so far was made with `--set` overrides.

The model is now physical:

```
find u (u = 0 on the hilum clamp) with, for every δu,
  ∫ P(F) : ∇δu dV  +  s ∫_Γ q(X) J F⁻ᵀN · δu dA  +  wall(u)  =  0
  q(X) = q0 + γ (h(X) − h̄) + δq(X)
  q0 = p0/E,  γ = ρ̄ g / E,  h = −e_g · X (reference position),  δq = optional diagnostic perturbation
```

Goal of the rewrite: **one config per patient where every physical quantity is a named parameter that is either given or fitted**; the same model runs forward (choose the parameters, or the collapse %, get the collapsed lung) or inverse (fit the marked parameters to the collapsed CT). The setup exists only to build this model.

**Decided by the user (do not reopen):**

- Physical load model is the default. Regions and free gradients stay only as diagnostics (δq on top of the physical load, or alone for the old reference fits).
- No registration, no target alignment, no still points (2026-10-27). The collapsed lung, when present, is a closest-point target only.
- ρ̄ is assumed (250 kg/m³): there are no inflated CT images, so no HU and no per-element density.
- The wall becomes a smooth potential inside Newton (Phase 1 below, first).
- E common to all patients ≈ 1.7 kPa from the scan (analysis/results.md 5b): the default value, but E must stay fittable.
- Cores: **torch** (production), **Warp** (kept, maintained alongside torch), **GetFEM** (validation only: what it cannot do, e.g. the potential wall or new materials, it simply refuses; no effort to port features to it).
- **Only what worked is carried over** to the new code; the experiments that led nowhere or were wrong stay on `main` (list in section 0b).
- The patient position is an option, and it can differ between the reference CT and the collapse (section 0c).

## 0b. What to carry over, what not

Carry over (validated, in use; details in CLAUDE.md "Implementation notes" and analysis/results.md):

| Feature | Where now |
|---|---|
| Closest-point target (no registration), interpolated vertex normals | `setup.py`, `fit.Loss`, `geometry.WallDistance.closest` |
| Loss `plane` (fitted → target along the target normal) **plus the symmetric term** (target → fitted, barycentric, 0 on clamped triangles), weight 1 | `fit.Loss` |
| Clamped points out of the loss and of every reported error | `fit.Loss` |
| Analytic Jacobian (one factorisation, columns per load parameter), bounded lsq (TRF) | `fit.run_level`, `nodal.jacobian` |
| Levels coarse → fine with warm start (area-weighted projection), re-anchoring the solver to the level's best state, failed solve = rejected trial, level stops if its first solve fails | `fit.run`, `fit.run_level` |
| Stops: lsq tolerances, `level_patience` / `level_min_improve`, `fit_patience`, `target_error_mm`, time budget, signals; checkpoint on improvement; watchdog | `run_control.py` |
| Hilum clamp from the 4 anchor spheres (`anchor_radius_factor` 2, growth to `anchor_min_points`) | `anchor.py` |
| Newton = GetFEM replica, `adaptive_steps`, `warm_substeps`, fast fail (`slow_ramp: false`, `max_solve_s`) | `nodal.py` |
| Wall: `reference` surface, `wall_tol_mm` 5, nodal quadrature, interpolated normal; `outer` rounds with the settled-state linearisation cache only until the potential wall replaces them | `nodal.py`, `geometry.WallDistance` |
| Regions (Ward on the closest-point field, `reg` smoothness), gradient slopes, gravity term — as load model and diagnostic perturbation | `regions.py`, `fit.PressureModel`, `fit.gravity_offset` |
| `best_by` (mean / rms), lsq bounds | `fit.py` |
| Summary keys (ASSD, mean normal error, rotation, penetration, collapse %, ...) and the result files | `fit.py` |
| `check_inputs`, surface meshing (ACVD), hilum rings, Gmsh volume mesh | as now |
| `TorchFEM` (differentiable assembly, physics loss for the GNN), `materials.py` | `torch_solver.py` |

Do not carry over:

| Feature | Why |
|---|---|
| Registration (elastix), target alignment (`align`), still points (`clamp_still_mm`), node-to-node target, `cluster_field` / `feature` variants | not used since 2026-10-27; tangential sliding, not anatomical |
| Loss `point`, `loss_tangent_weight` | for the registered node-to-node target only |
| Nelder-Mead (`optimizer: nm`), 2-point fit Jacobian (`jacobian: "2-point"`, `lsq_diff_step`) | slower, superseded by the analytic Jacobian; finite differences stay only inside `check_jacobian.py` |
| `wall_update: "newton"` | oscillates (residual 2-cycle) |
| `wall_quadrature: "face"` | negative centre weight, indefinite contact tangent |
| `slow_ramp: true` | hopeless solves run for minutes |
| `support_k` | artificial substitute for friction, mixed results, unexplained Newton `maxit` failures; gravity + wall should make it unnecessary |
| `gravity_prefit` / `gravity_bisect` continuation | numerical workaround for the outer wall; reintroduce only if Phase 1 shows the potential wall still needs it |
| GetFEM P2 (`order: 2`) | never used in the fits |

If something in neither list turns up, ask the user.

## 0c. Patient position: reference and collapse

The inflated CT (the reference) and the collapse may be in different positions, e.g. supine CT and lateral decubitus surgery. The config names both:

```json
"gravity": {"reference": "supine", "collapse": "left_lateral"}     // or null = no gravity
```

Positions as directions of gravity e_g in LPS: supine = +y, prone = −y, left lateral decubitus (lying on the left side) = +x, right lateral = −x, or any vector [x, y, z] (normalised).

Physics (extends METHODOLOGY.md section 8; update it). The reference is stress-free under its own loads: the weight along e_A plus the pleural gradient along e_A that balances it. The collapsed state carries the weight along e_B, a uniform pleural (atmospheric) pressure and the wall. The load from the reference is the difference:

```
surface:  q(X) = q0 + γ (h_A(X) − h̄_A),      h_A = −e_A · X      (release of the pleural gradient of position A)
volume:   b = γ (e_B − e_A)                    (non-dimensional body force, dead load, per unit reference volume)
```

- Same position (e_A = e_B): b = 0, which is exactly the current model ("no body force", METHODOLOGY 8).
- Check: the resultant of the two terms on the reference is γ V e_B, i.e. the lung's weight along the collapse position, whatever A is.
- b is constant: in `nodal.py` (core-independent, so torch and Warp both get it) a fixed nodal vector (each tet gives b V_t / 4 to its nodes), added to R in `_assemble_at`; no tangent. It is linear in γ, so the Jacobian column for γ is the pressure column (dU/dq · (h_A − h̄_A)) plus −K_t⁻¹ (dR/dγ of the body term). This needs a small extension of `ForwardSolver`: a body force in `solve` / `set_state` and its column in `jacobian`. GetFEM may refuse e_A ≠ e_B.
- Turning the patient with the chest closed also moves the lung a little before the pneumothorax; the model above ignores that intermediate state (it goes straight from the reference to the collapse). Say so in METHODOLOGY.
- Tests (Mac): the resultant check above; q and b for each named position; the Jacobian column vs finite differences with a fake linear core.

**Rules (from CLAUDE.md):** code, comments and docs in English. Nothing is installed on the Mac; the Mac only has numpy / scipy, so heavy runs happen on the server and you give the user the commands to run there. No commits unless the user asks. Physics or default changes are explained and left to the user. A config key goes in `pipeline/config.py`, `configs/README.md` and, if user-visible, `README.md`.

**Prerequisite:** the current work on `main` (gravity offset, `gravity_prefit`, `gravity_bisect`, `best_by`, docs) is uncommitted. Ask the user to commit it on `main`, then create `physical-model` from it. `main` stays as it is: it is the reference for the regression checks.

## Phase 1: wall as a smooth potential

METHODOLOGY.md section 8 (a). Do this first, before the restructuring: it only touches the solver side, and its results decide the wall defaults of the new configs.

**Problem now.** The wall is a penalty whose normal `wallN` and gap offset `wallG` are frozen during a Newton solve (`nodal._update_wall`, `geometry.WallDistance`). Outer rounds alternate Newton and re-linearisation (`wall_max_updates`, `wall_settle_mm`). The closest-point normal jumps at the corners of the cavity, so rounds alternate there and solves fail (`fail_unsettled`, `fail_time`). Also the Jacobian holds the linearisation fixed (Gauss-Newton), so a finite-difference check with a wall is meaningless. `wall_update: "newton"` (re-linearise at every residual) oscillated because it misses the curvature term and the normal jumps.

**Model.** A smoothed signed distance φ̃ (> 0 outside the cavity), C², evaluated at the current position. Per free surface node i, with reference nodal area a_i (the `nodal` vertex rule: 1/3 of each adjacent non-clamped reference face) and allowance A_i:

```
g_i = φ̃(x_i) − A_i
E_wall = Σ_i a_i Ψ(g_i),     Ψ' = w(g) = (k / 2e) (⟨g⟩₊² − ⟨g − e⟩₊²)      (the current force law, same k, e)
f_i = a_i w(g_i) ∇φ̃(x_i)
H_i = a_i [ w'(g_i) ∇φ̃ ∇φ̃ᵀ + w(g_i) ∇²φ̃ ]        (3×3, symmetric, only where g_i > 0)
```

Keep the force law w(g) of the current penalty (C¹ force, smooth onset over e = 0.5 mm, linear beyond), so k, e and the results stay comparable with the outer rounds. Do not use the plain ⟨g⟩₊² of METHODOLOGY 8 (a): its tangent jumps at contact.

**Field.** Built once in `fem_setup` (needs vtk/pyvista, server only):

1. Signed distance to the wall surface on a regular grid: spacing `wall_grid_mm` (default 1.5), box = wall bounding box + margin `wall_grid_margin_mm` (default 30, larger than any outward motion), with pyvista `ImageData.compute_implicit_distance` (vectorised VTK; flip the sign if needed so outside > 0).
2. Gaussian smoothing with σ = `wall_smoothing_mm` (default 3, try 2 / 3 / 5): rounds the corners (paravertebral gutter, costophrenic sulcus) and removes the kinks of the exterior medial axis. Log the max | |∇φ̃| − 1 | within 10 mm of the surface (the gap stays in mm only approximately).
3. Store the smoothed grid (float32), origin, spacing and smoothing in `problem.npz`: new optional `CollapseProblem` fields `wall_grid`, `wall_origin`, `wall_spacing`, `wall_smoothing_mm` (empty = none; old files load unchanged).
4. The allowance uses φ̃, not the exact distance: `A_i = wall_tol_mm + max(φ̃(X_i), 0)`, as now with φ0. The `reference` wall has the lung's nodes on it, and smoothing moves the zero level inward where the cavity is pointed, so some reference points sit slightly outside φ̃ = 0. Log how many target points lie beyond the allowance (as now), with φ̃.

**Evaluation (numpy only, testable on the Mac).** A new class in `geometry.py`, e.g. `WallField`. Tricubic B-spline interpolation:

- coefficients from `scipy.ndimage.spline_filter(grid, order=3)` once at load time;
- for each point, the 4×4×4 coefficient stencil and the 1D cubic B-spline basis B, B', B'' per axis;
- returns φ̃ (N,), ∇φ̃ (N, 3) and ∇²φ̃ (N, 3, 3), vectorised over the ~1300 surface nodes;
- points outside the grid raise an error naming the margin key.

Trilinear interpolation is not enough: its gradient is discontinuous.

**Solver.** `wall_update: "potential"` in `nodal.py` (shared by the torch and Warp cores; GetFEM refuses it):

- In `_assemble_at`, add f_i to R and H_i to the free-dof tangent, as `support_k` is added now (a sparse block-diagonal matrix on the surface dofs). No `wallN` / `wallG`, no outer rounds: `_solve` is one Newton (plus the load path).
- The core's own wall term (`TorchSolver._wall`, the Warp kernel) is skipped in this mode: `_assemble(..., wall=False)`.
- `jacobian()` already factorises the tangent from `_assemble_at`, so it becomes the exact Jacobian with a wall.
- The equilibrium-linearisation cache (`_get_lin` / `_put_lin`, the 256 kept states) is not needed in this mode; `set_state` warm starts as without a wall.
- `wall_diagnose.py`: report the contact points, the max g and the Newton iterations as now; there are no rounds.
- Summary: `wall_max_penetration_mm`, `wall_points_beyond` computed with the exact distance (as now), plus the same with φ̃.

**Checks.**

| Where | Check | Pass |
|---|---|---|
| Mac | `WallField` on an analytic sphere SDF (no smoothing): φ̃, ∇φ̃, ∇²φ̃ vs analytic | ~1e-3 mm at 1 mm spacing, better on finer grids |
| Mac | ∇φ̃, ∇²φ̃ vs finite differences of φ̃ at random points, including points on cell boundaries | rel 1e-6 |
| Mac | f_i, H_i vs finite differences of E_wall | rel 1e-6 |
| Server | `check_jacobian.py --config configs/karl04_wall_torch.json --options '{"wall_update": "potential"}'` | Jacobian vs FD ~1e-6 (now meaningful with a wall) |
| Server | karl04 forward solves at the fitted q (`best_state.npz` of `fit_gE1700`), outer vs potential | displacement differences only near the corners, a few mm; same contact set elsewhere |
| Server | rerun `fit_gE1700` and `fit_gE1000` with the potential on karl20, karl23, karl01, karl04 (and smoothing 2 / 3 / 5 on karl20) | failed solves, time, ASSD vs analysis/results.md 5b |

Report the numbers to the user; they choose the default (`potential` vs `outer`, smoothing). Write the results in analysis/results.md, the method in METHODOLOGY.md sections 3 and 6.

## Phase 2: config schema

One file per patient, `configs/<p>.json`. Physical quantities in one `physics` section, numerical settings elsewhere.

**Parameter convention.** Every physical parameter is either a number (fixed) or an object (fitted):

```json
"E_Pa": 1700
"E_Pa": {"fit": true, "init": 1700, "bounds": [300, 10000]}
```

A small parser (`pipeline/collapse/params.py`) turns `physics` into a list of `Param(name, value, fit, init, bounds)`. Fitted parameters become the optimiser's vector; fixed ones are constants. The summary always reports every parameter in physical units and says which were fitted.

**Proposed schema** (the values are the current defaults; key names can change if something reads better, but keep the split; `//` marks explanations here only, real configs use `_`-prefixed comment keys):

```json
{
  "patient": "karl01",
  "side": "left",
  "data_dir": "...", "output_dir": "...",
  "steps": ["check_inputs", "geometry", "hilum", "fem_setup", "collapse"],

  "inputs": {
    "inflated": "lung_left.vtk",
    "collapsed": "lung_left_collapsed.vtk",          // null = no target (prediction only)
    "structures": {"airways": "lung_airways.vtk", "arteries": "lung_arteries.vtk", "veins": "lung_veins.vtk"}
  },

  "check_inputs": { ... as now ... },
  "geometry":     { "target_nodes": 480, ... the surface_mesh settings ... },
  "hilum":        { "lung": "collapsed", "max_ring_distance_mm": 50 },

  "fem_setup": {
    "clamp": {"structures": ["hilum", "airways", "arteries", "veins"], "radius_factor": 2.0, "min_points": 20},
    "mesh_size_mm": 14,
    "wall": {"surface": "reference", "tol_mm": 5, "grid_mm": 1.5, "grid_margin_mm": 30, "smoothing_mm": 3}
  },

  "physics": {
    "material": {"name": "neo_hookean", "nu": 0.40},
    "E_Pa": 1700,
    "tissue_density": 250,
    "gravity": {"reference": "supine", "collapse": "supine"},   // section 0c; null = no gravity
    "collapse": {"q0": {"fit": true, "init": 0.3, "bounds": [-1, 3]}},
    "wall": {"stiffness": 20, "eps": 0.5},
    "perturbation": null                             // null | {"gradient": ["si"]} | {"regions": [1, 4, 8]}
  },

  "solver": {"core": "torch", "device": "cuda", "linear_solver": "cudss", "wall_update": "potential",
             "load_steps": 4, "adaptive_steps": true, "warm_substeps": true, "max_solve_s": 120, ...},

  "fit": {"loss": "plane", "symmetric_weight": 1.0, "best_by": "rms", "target_error_mm": 2.5,
          "time_budget_min": 240, "levels": "...", "lsq": {...}, "stops": {...}},

  "run_name": "gE1700"
}
```

Notes for the implementation:

- **`collapse`** sets the uniform part in one of three ways: `{"q0": value or fit spec}`, `{"p0_Pa": value}` (q0 = p0 / E), or `{"volume_ratio": 0.45}` (V_collapsed / V_inflated: forward mode finds q0 by a monotone 1D search, 4–6 solves, warm-started, Phase 4).
- **`gravity`**: the positions of the reference CT and of the collapse (section 0c). h̄ = area-weighted mean over the free triangles (as `fit.gravity_offset`). `null` = no gravity term. Default supine → supine: the data say supine on 15 / 15 (NEXT.md); the position of the collapsed CTs is still to be confirmed by the user.
- **E** enters the shape only through γ = ρ̄g/E (q0 is p0/E). With gravity on, E is identifiable through γ; with gravity off, E only scales the reported pressures, so `fit: true` on E without gravity must be an error. Fit γ internally (bounds from the E bounds), report E = ρ̄g/γ and p0 = q0 E.
- **`nu`** can be fitted (it is now `free_nu`), default fixed 0.40.
- **`perturbation`** is the old pressure models, now as δq on top of the physical load: `gradient` adds free slopes along the listed axes; `regions` adds a p/E per Ward region with the levels coarse → fine (as the current regions fit, `reg` smoothness). The old pure regions fit = gravity `null`, q0 fitted, `regions` perturbation. Ward clustering in `fem_setup` runs only when a config asks for regions (it needs the target).
- **`wall`**: geometry (surface, tolerance, grid) in `fem_setup`, because it goes in `problem.npz`; stiffness in `physics`. `surface: null` = no wall (diagnostics only: with gravity the wall carries the weight, so the physical model needs it; log a warning when gravity is on and the wall is off).
- **`solver`** = the current `solver_options` plus the core name. Keep the option filtering of `solvers.options_for` for the validation scripts.
- **`fit`** = everything numerical from `fem_fit`. No continuation (`gravity_prefit`, `gravity_bisect`, section 0b): every solve uses the physical proportional ramp; reintroduce it only if the Phase 1 runs show that the potential wall still cannot reach the gravity load. `best_by`: default `rms` (what the optimiser minimises), but this changes a default, so confirm with the user.
- `--set` overrides must reach nested keys (`physics.E_Pa=1700`, `physics.collapse.q0={"fit":true,"init":0.3}`): `apply_overrides` already parses JSON values; check it with nested sections.
- Unknown keys raise an error, `_`-prefixed keys are comments, paths as now.
- Write `scripts/migrate_config.py`: old `<p>_torch.json` / `<p>_wall_torch.json` (+ the `--set` overrides of a run) → new schema. Use it to make the new configs and for the regression checks of Phase 6.

## Phase 3: code structure

Target layout (adapt names if needed; keep files small and the solver interface unchanged):

| Module | Content |
|---|---|
| `main.py` | steps `check_inputs`, `geometry`, `hilum`, `fem_setup`, `collapse` (forward), `fit` (inverse) |
| `pipeline/config.py` | new schema, parameter specs, validation |
| `pipeline/geometry.py` | today's `surface_mesh` + `registration.run_unregistered`: meshes the **inflated** lung (the model) and, if given, the collapsed lung (the target), same settings as now |
| `pipeline/hilum.py` | as now; `lung: inflated` must work for the prediction-only case (check on all patients that the hilum from the inflated and from the collapsed lung agree within a few mm, report the numbers) |
| `pipeline/collapse/setup.py` | clamp, wall field, Gmsh mesh, optional target, optional Ward regions → `problem.npz` |
| `pipeline/collapse/params.py` | parameter specs: fixed / fitted, bounds, packing to and from the optimiser vector |
| `pipeline/collapse/load.py` | the load model: q per triangle from (q0, γ, e_g, δq), its Jacobian dq/dθ (columns for q0, γ, δq; γ column = h − h̄), physical units out. Replaces `PressureModel` and `gravity_offset` |
| `pipeline/collapse/forward.py` | one solve at given parameters, or the q0 search for a `volume_ratio`; writes the result; optional collapse sequence |
| `pipeline/collapse/fit.py` | the inverse problem: loss, levels (only for the regions perturbation), lsq, stops, checkpoint. Talks to `load.py` and the solver only |
| `pipeline/collapse/run_control.py`, `loss` part of `fit.py` | as now |
| `pipeline/collapse/solvers/` | unchanged interface (`ForwardSolver`); `potential` wall from Phase 1 |

**Remove on the branch** (they stay in git and on `main`): everything in the "do not carry over" list of section 0b, `registration.py` (elastix), `configs/elastix/`, `configs/archive/`, and `PressureModel` / `gravity_offset` (replaced by `load.py`). `summarize_fits.py` must still read the old `results/` summaries.

**Cores:** torch and Warp both get every new feature that lives in `nodal.py` (potential wall, body force). A new material is torch only, unless the user asks for it in Warp too (hand-derived stress and tangent). GetFEM stays as the validation core for what it already does (no wall, outer wall, Jacobian); it refuses the rest with a clear message, and `compare_warp_getfem.py` / `check_jacobian.py` keep working on those cases.

**Outputs** per run, `output_dir/runs/<run_name>/`:

- `summary.json` with a `physics` block: every parameter in physical units (E_Pa, p0_Pa, q0, γ, ρ̄, e_g, ν, wall) and whether it was fixed or fitted; volume ratio; errors when there is a target (the current keys: `assd_mm`, `mean_normal_err_mm`, rotation, wall penetration...); solve counts and failures;
- the collapsed surface (`.vtp`, point data `Displacement_mm`, cell data `Pressure_Pa`, and `Error_mm` with a target), `state.npz`, optional `volume.vtk`;
- with a fit also `history.csv`, `best_state.npz`, `best_params.json` (as now).

`summarize_fits.py` reads both the old and the new summaries.

## Phase 4: forward mode

`collapse` step (`forward.py`):

1. All physics parameters fixed (no `fit: true` allowed here): one solve with the proportional load ramp (`load_steps` / `adaptive_steps`), write the result.
2. `collapse: {"volume_ratio": r}`: find q0 such that V(q0) / V_inflated = r. V is monotone in q0: bracket from the uniform solve, then secant / bisection to 0.5 % in volume, warm-starting each solve from the previous state. Report the q0 found and the number of solves.
3. Option `sequence: {"frames": 30}`: the collapse sequence for Slicer, as `scripts/collapse_sequence.py` (move the script's code here and keep the script as a thin wrapper).
4. Works without a collapsed CT (`inputs.collapsed: null`); with one, also report the errors against it.

Leave-one-out check (NEXT.md, the first use of this mode): for each patient, q0 from a collapse-% rule fitted on the other patients, E 1.7 kPa, one forward run, errors vs the real collapsed lung and vs the patient's own fit. Write `scripts/leave_one_out.py` once the rule is agreed with the user.

## Phase 5: fits

`fit` step (`fit.py`): fits the parameters marked `fit: true`.

- One level when there is no regions perturbation; levels coarse → fine for regions, warm-started as now.
- Jacobian columns from `load.py` (dq/dθ) times the core's dU/dq: q0 and γ columns are free (linear in q), so the analytic Jacobian is kept for every parameter. ν by the core's `with_nu`.
- Keep: the closest-point loss with the symmetric term, clamped points out of the loss, the re-anchoring of the solver to the level's best state, the checkpoint, the watchdog, the stop reasons, failed solve = rejected trial.
- Cohort fits with a shared parameter (the common-E scan): `scripts/fit_cohort.py --configs ... --shared E_Pa --values 1000 1700 3000` runs the per-patient fits for each value (or a golden-section search on the summed cost) and writes one table. Later, if needed, a joint fit.

## Phase 6: regression, configs, docs

**Regression on the server** (old = `main`, new = branch), one patient (karl04), then 3–4 more:

| Old run | New config | Must match |
|---|---|---|
| `fem_setup` of `karl04_torch.json` | `fem_setup` with gravity null, no wall | `problem.npz` arrays identical (surface, clamp, tets if Gmsh is deterministic; otherwise surface and clamp identical) |
| `fit_torch` (regions, no wall) | gravity null, q0 fit, `regions` perturbation, no wall | ASSD and errors to 1e-3 mm, same evaluations |
| `fit_grad_ap_sym` (gradient ap, wall, outer) | gravity null, `gradient: ["ap"]`, wall outer | same |
| `fit_gE1700` rerun on `main` with `--set fem_fit.gravity_prefit=[] fem_fit.gravity_bisect=0` (no continuation, as the new code) | gravity supine → supine, E 1700, q0 fit, wall outer | same |

Only after these match, switch the wall to `potential` and compare as in Phase 1.

**Configs:** one set, `configs/<p>.json`, made with `migrate_config.py`: gravity supine, E 1.7 kPa, q0 fitted, wall on (`reference`, 5 mm), torch core. patient_2 was excluded from the wall runs (user, 2026-10-05): ask whether it now gets the wall. karl19 stops on its input (NEXT.md).

**Docs:** README.md (steps, the two modes, how to choose or fit parameters), configs/README.md (every key), METHODOLOGY.md (section 8 becomes the model; sections 3–6 updated; the potential wall), CLAUDE.md (code architecture, commands, configs, implementation notes), NEXT.md (tasks done / open). Results only in analysis/results.md.

**Tests:** a `tests/` folder with pytest tests that run on the Mac (numpy / scipy only): the heavy modules stubbed in `conftest.py` (an importlib finder that returns `MagicMock` modules for pyvista, vtk, vtkmodules, sklearn, torch, warp, getfem, gmsh, itk, pyacvd). Cover: `WallField`, parameter parsing and packing, `load.py` (q and dq/dθ vs finite differences; γ slope over the extent; e_g directions), the config migration, and `fit.run` / `forward.run` with a fake linear core (the approach used for the gravity continuation test on 2026-11-16: fakes for `WallDistance`, `closest_on_surface`, `assd`, `polydata`, `write_surface`).

## Order and stop points

1. Phase 1 (wall potential) → **stop**: user chooses the wall default and the smoothing.
2. Phase 2 + 3 (schema, structure, positions of section 0c) with the regression of Phase 6 → **stop**: user confirms the `best_by` default and whether any continuation is still needed.
3. Phase 4 (forward, leave-one-out) → **stop**: user agrees the collapse-% rule.
4. Phase 5 (fits, cohort) and the rest of Phase 6 (configs, docs).

At each stop: what changed, the server commands to run (with `2>&1 | tee <name>.txt`), the numbers, and the decisions needed.
