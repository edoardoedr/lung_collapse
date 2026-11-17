# Context: parameter-estimation experiments (handoff, 2026-11-17)

For a new chat that designs and analyses experiments to estimate the model's parameters. Repo: `lung_collapse`, branch `refactoring_to_v2`. Read first: `CLAUDE.md` (how to work here, commands, architecture), `METHODOLOGY.md` (model and equations), `analysis/results.md` section 5c (the latest results), `NEXT.md` (to-do).

## Rules of the project

- **Do not install anything on the Mac.** All runs happen on the server (conda env `lungfem`, `/home/jovyan/Lung_project/lung_collapse`): give the commands, the user pastes back the logs / tables. Append `2>&1 | tee <name>.txt` to every server command.
- No commits unless asked. Before changing physics or defaults, explain the trade-off and let the user decide.
- A new config key goes in `pipeline/config.py`, `configs/README.md` and, if user-visible, `README.md`. A new result goes in `analysis/results.md`, a decision or task in `NEXT.md`.
- Do not bring back the abandoned approaches (registration, point loss, Nelder-Mead, FD Jacobian, `wall_update: "newton"`, `support_k`, gravity continuation) without asking.

## The model

The inflated lung (from CT) is a compressible Neo-Hookean body, clamped at the hilum, inside the chest wall. In the pneumothorax it loses the pleural pressure, so it shrinks and settles on the wall. Load on the surface, divided by E:

```
q = q0 + γ (h − h̄),     q0 = p0 / E,     γ = ρ̄ g / E,     h = −e_A · X (height in the CT position)
```

| Parameter | Config key | Notes |
|---|---|---|
| E | `physics.E_Pa` | default 1700 Pa; with gravity, only ρ̄ / E is identified |
| ν | `physics.material.nu` | 0.40 fixed (K/E = 1.67, collapse ≈ q0 / 1.67 if hydrostatic) |
| ρ̄ | `physics.tissue_density` | 250 kg/m³ |
| positions | `physics.gravity` | `{reference, collapse}` ∈ supine / prone / left_lateral / right_lateral / [x,y,z] LPS; different positions add a body force γ(e_B − e_A) |
| q0 | `physics.collapse` | `q0` or `p0_Pa` or `volume_ratio` (forward only: q0 searched to reach that volume) |
| wall | `fem_setup.wall`, `physics.wall` | smooth signed-distance field (grid 1.5 mm, σ 3 mm), penalty k 20, ε 0.5, margin 5 mm |

Each parameter is a number (fixed) or `{"fit": true, "init": ..., "bounds": [...]}`. The fit internally uses the slope s_g = γL (L = height extent) and reports E. Diagnostic only: `physics.perturbation` = `{"gradient": ["ap","si","ml"]}` or `{"regions": [K...]}`.

- **Forward (`collapse` step):** parameters given → collapsed lung (`runs/<run_name>/surface.vtp`, `summary.json`, `state.npz`, optional `sequence/`).
- **Inverse (`fit` step):** fits the `fit` parameters to the collapsed CT. Loss: plane distance + symmetric term, TRF least squares, analytic Jacobian. Summary: `assd_mm`, `collapse_percent`, `target_volume_ratio`, `physics.{E_Pa,q0,p0_Pa,...}`, `rigid_rotation_deg`, wall penetrations.

## Data and folders

- 16 configs `configs/<p>.json`. Left lung: karl01, karl10, karl17, karl18, karl20, karl23, karl24, patient_1, patient_2, patient_3, patient_10. Right lung: karl02, karl04, karl06, karl08, karl19. karl19 fails on its input; patient_2 has no result yet (probably no wall field: `python scripts/add_wall_field.py --config configs/patient_2.json`).
- **Output folders of the current code** (configs' `output_dir`): `/home/jovyan/Dataset_Lungs/{HD_dataset/FEM_v2_HD, CZ_dataset/FEM_v2_CZ}/<p>/`, with `fem/setup/problem.npz` and `runs/<run_name>/`. Filled once from the old folders by `python scripts/init_output_dir.py --configs configs/*.json --apply` (copies the setup + the gE scan + fitE runs). The old `FEM_fit_*/<p>_noreg_hilum[_wall]` folders are the previous code: do not write there.
- Every experiment = a `run_name`, so each lands in its own `runs/<run_name>/`.

## Tools

```bash
python main.py --config configs/<p>.json --steps fit --set physics.E_Pa=1000 run_name=E1000
python main.py --config configs/<p>.json --steps fit --set 'physics.E_Pa={"fit": true, "init": 1300, "bounds": [200, 20000]}' run_name=fitE
python main.py --config configs/<p>.json --steps collapse --set 'physics.collapse={"volume_ratio": 0.5}' physics.E_Pa=900 run_name=pred
JOBS=4 bash scripts/run_all.sh [case ...] -- --steps fit --set ... run_name=X      # several patients, logs_run_all/
python scripts/fit_cohort.py --configs configs/*.json --param physics.E_Pa --values 700 1000 1700 --jobs 4
python scripts/summarize_runs.py /home/jovyan/Dataset_Lungs/HD_dataset/FEM_v2_HD /home/jovyan/Dataset_Lungs/CZ_dataset/FEM_v2_CZ --runs fitE [--out analysis/runs_v2.csv]
```

Keep `JOBS` ≤ 4 (parallel jobs slow every solve). One fit takes 0.2–2 min on the GPU.

## Results so far (analysis/results.md 5c)

- **Common-E scan** (q0 fitted, E 0.7 / 1.0 / 1.3 / 1.7 / 3.0 kPa, 14 patients, 556 solves, 0 failed): cohort ASSD mean 8.42–8.94 mm, median 7.1–7.8; no common E (9 of 14 best at an end of the range, in opposite directions).
- **E fitted per patient (`fitE`, bounds 0.2–20 kPa):** ASSD median 6.41, mean 7.10 mm (best-of-scan 6.45 / 7.66). Per patient E (kPa): karl01 1.29, karl02 1.02, karl04 0.61, karl06 1.76, karl08 1.72, karl10 19.3 (upper bound), karl17 1.31, karl18 20 (upper bound), karl20 0.31, karl23 0.20 (lower bound; collapse 40 vs 51 % target), karl24 3.59, patient_1 0.64, patient_10 0.58, patient_3 1.63.
- **E falls with the collapse %** (11 in bounds): log₁₀ E = 3.69 − 0.0168 × collapse % (r −0.83, p 0.001): ≈1.8 kPa at 25 %, ≈0.7 kPa at 50 %.
- **Reading (open):** the fit identifies ρ̄/E, not E. Neo-Hookean has no compression stiffening (that would predict the opposite trend), so the gradient is probably standing in for a missing load that grows with the collapse (mediastinum / heart on the medial face, diaphragm) or a denser dependent lung. karl10 / karl18 want no gradient; karl23 is under-collapsed even at the lowest E.
- Before the rewrite: free Ward-regions pressure fit ~2.9 mm median (upper bound of what a pressure field can do); gradient fits ~7–8 mm.

## Open experiment ideas (from NEXT.md, not started)

1. **Leave-one-out of the transferable model:** for each patient, E from the regression of the others, q0 from its own collapse % (forward `volume_ratio`), one forward run; ASSD vs its fitE. Needs `scripts/leave_one_out.py`.
2. **Missing load:** on karl10, karl18, karl23 look where the fit misses (sequences in Slicer); candidate extra load terms (medial pressure, diaphragm), which are physics changes to discuss with the user first.
3. Other free gradients (`perturbation: {"gradient": [...]}`) together with E, to see whether a medial-lateral or cranio-caudal slope explains the trend.
4. Sensitivity to ν and to the wall margin / smoothing on a few patients.
5. Pending validation: regions fit with the new code; `geometry` + `fem_setup` from scratch vs the copied setup.
