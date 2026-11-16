# Fit results

Results of the fits, the evidence behind the design choices and the problems found. Methods and equations are in [METHODOLOGY.md](../METHODOLOGY.md), the plan in [NEXT.md](../NEXT.md). Written by hand until a script produces it.

Updated 2026-11-12. Shape errors, collapse, rotation and pressure pattern measured by `scripts/summarize_fits.py` (same measure for every fit; output in `analysis/runs.csv`, `compare.csv`, `report.md`, `*.png`):

    python scripts/summarize_fits.py results/*_noreg_hilum results/*_noreg_hilum_wall --out analysis

ASSD = mean of fitted → target and target → fitted surface distance, free points only; collapse = 1 − V_collapsed / V_inflated. Failed solves, K, times and stop reasons come from each fit's `result_summary.json`.

## 1. Status

- 16 patients, 15 with results. karl19 stops in `registration` (method none) at `surface_mesh.clean_surface` of `lung_right.vtk`: "triangulation left non-triangle cells".
- Runs (hilum-only clamp, no registration):

| Run | Loss | Wall | `support_k` | Patients | Notes |
|---|---|---|---|---|---|
| `fit_torch` | one-sided | both | 0 | 15 | superseded by `fit_sym` |
| `fit_sym` | symmetric | no / yes | 0 | 15 / 14 | |
| `fit_sym_k0.001` | symmetric | no / yes | 0.001 | 14 / 14 | wall: rerun one at a time |
| `fit_sym_k0.003` | symmetric | no / yes | 0.003 | 14 / 14 | all 28 run in parallel: times and wall failures not reliable |
| `fit_grad_ap`, `fit_grad_apsi` | one-sided | no | 0 | 15 | to redo with the symmetric loss |
| `fit_sym_k0.001_{t20,t300,q2}` | symmetric | yes | 0.001 | karl04 | tests of `max_solve_s` and of the q bound |
| `fit_sym_k0.001_fix` | symmetric | yes | 0.001 | karl04 | after the 2026-10-30 fix; stopped by hand at K = 25 (4 min): ASSD 5.88, 11 failed solves, same as before the fix |

- **Every wall fit except `fit_sym_k0.001_fix` (karl04, interrupted) predates the 2026-10-30 fix** of the outer rounds (section 7).

## 2. Regions without wall (`fit_sym`)

- 15 patients, no failed solve, 0.2–1.0 min per fit, K = 40 in 12. Stop: "target error reached" in 5, "levels exhausted" in the rest.
- **ASSD median 2.92 mm**, range 2.25–5.00 (patient_2).
- Fitted collapse matches the real one: median gap 0.6 points.
- Rotation (rigid part of the fitted displacement) median 11.0°, large where the collapse is large: karl23 73°, patient_2 44°, patient_1 34°, karl04 32°. Not yet checked in the collapse sequences.
- Some regions sit at the p/E bounds: −1 on karl10 / karl20, 3 on 6 patients.

## 3. Pressure–collapse rule and pattern (regions, 15 patients)

- **Mean p/E = −0.18 + 2.19 × collapse**, r 0.98, max residual 0.17 (collapse 23–86 %). With the support, on 14 patients (collapse ≤ 59 %): −0.03 + 1.82 c (k 0.001, r 0.96), −0.03 + 1.94 c (k 0.003, r 0.99).
- The material alone gives 1.67 × collapse in hydrostatic compression (METHODOLOGY.md, section 3): the rule largely reflects the chosen material (ν). A uniform pressure above p/E 1.67 has no equilibrium; the regions at p/E 3 hold only because they are local.
- **Anterior third > posterior third in 15 / 15; apex < base in 14 / 15.**
- A linear gradient (ap, si, ml) explains only **2–20 %** of the regional p/E variance (R² 0.02–0.20): most of the regional pattern is local and patient-specific.

## 4. Gradient fits (one-sided loss, no wall)

| Patient | Collapse | Regions ASSD | ap: ASSD | ap: slope | ap + si: ASSD | ap slope | si slope | Fitted collapse (ap + si) |
|---|---|---|---|---|---|---|---|---|
| karl01 | 25 % | 2.40 | 6.38 | 0.14 | 6.66 | 0.17 | −0.13 | 37 % |
| karl02 | 53 % | 3.20 | 8.44 | 0.12 | 8.50 | 0.12 | −0.04 | 53 % |
| karl04 | 51 % | 3.90 | 12.97 | 0.10 | 12.20 | 0.14 | −0.14 | 53 % |
| karl06 | 23 % | 2.31 | 5.61 | 0.09 | 5.20 | 0.08 | −0.04 | 28 % |
| karl08 | 28 % | 2.25 | 7.53 | 0.07 | 6.80 | 0.09 | −0.18 | 30 % |
| karl10 | 56 % | 2.75 | 12.60 | 0.47 | 7.37 | −0.05 | 0.39 | 63 % |
| karl17 | 24 % | 2.99 | 8.19 | 0.13 | 7.70 | 0.09 | −0.06 | 33 % |
| karl18 | 31 % | 2.42 | 6.02 | 0 (uniform kept) | 6.03 | 0.14 | 0.16 | 42 % |
| karl20 | 58 % | 3.94 | 12.60 | 1.02 | 11.06 | 0.83 | −0.59 | 72 % |
| karl24 | 24 % | 2.45 | 6.11 | 0.05 | 5.20 | 0.03 | −0.07 | 28 % |
| patient_3 | 37 % | 2.67 | 8.51 | 0.15 | 8.54 | 0.16 | −0.03 | 47 % |
| patient_10 | 47 % | 2.92 | 8.79 | 0.25 | 8.06 | 0.25 | −0.09 | 52 % |
| **median (12)** | | **2.71** | **8.32** | | **7.54** | | | |
| karl23 | 51 % | 3.98 | 28.27 | −3.49 | 27.57 | −4.00 | −2.71 | 94 % |
| patient_1 | 59 % | 2.93 | 29.74 | −0.70 | 29.42 | −1.11 | 2.66 | 97 % |
| patient_2 | 86 % | 5.00 | 27.15 | −1.11 | 27.09 | −0.86 | 2.99 | 99 % |

- Slopes in p/E across the lung. About 0.2 min per fit, no failed solve.
- The shape error is ~3× that of the regions. Adding si helps clearly only on karl10.
- Signs (ap + si, 12 patients): ap > 0 in 11, si < 0 in 10.
- **karl23, patient_1, patient_2 are broken:** mean p/E at the bound 3, collapse 94–99 %, ASSD 27–30 mm. It is the shrinkage of the one-sided loss that the symmetric term removed in the regions fits (section 8). The others over-collapse too (karl01 37 vs 25 %, karl17 33 vs 24 %, karl18 42 vs 31 %, karl20 72 vs 58 %, patient_3 47 vs 37 %), so p/E and slopes are biased upward. The pressure rule from these fits is meaningless (r 0.77).

## 5. Physical reading of the pressures

- p/E is an equivalent surface load lumping elastic recoil, weight and the support of the neighbouring organs.
- **Mean level:** mostly the volumetric response of the material (section 3).
- **Anterior > posterior:** probably gravity in the supine scan (now the proposed load model, METHODOLOGY.md section 8). The inflated lung carries its weight through the vertical gradient of pleural pressure, which the pneumothorax removes; with a stress-free reference the collapse load is uniform + linear in height, slope ρ g per unit height. A body force on top would count gravity twice. Hence E ≈ ρ g L / slope (L = AP depth).
- E estimate (ρ = 250 kg/m³, assumed): from the linear slope of the regional p/E (12 patients) median 2.3 kPa, range 0.4–8.4; from the gradient fits (ap only, one-sided loss) median 4.4 kPa, IQR 3.1–6.6, range 0.5 (karl20) to 9.1 (karl24). Plausible in sign and order of magnitude, not proven: E spreads ×18. To redo with the symmetric loss.

## 6. Wall and tangential support

**Medians over 14 patients** (patient_2 excluded from the wall runs):

| | ASSD | Range | No failed solve | Failed solves (total) | K = 40 reached | Rotation | Collapse gap (points) |
|---|---|---|---|---|---|---|---|
| No wall, k = 0 | 2.83 | 2.25–3.98 | 14 | 0 | 11 | 10.9° | 0.6 |
| No wall, k = 0.001 | 2.53 | 2.17–4.57 | 13 | 20 | 11 | 8.4° | 0.7 |
| No wall, k = 0.003 | 2.52 | 2.24–6.20 | 10 | 61 | 7 | 6.1° | 1.0 |
| Wall, k = 0 | 5.38 | 2.23–13.14 | 3 | 146 | 1 | 12.5° | 2.8 |
| Wall, k = 0.001 | 3.72 | 2.36–11.89 | 4 | 136 | 0 | 8.0° | 1.7 |
| Wall, k = 0.003 | 3.07 | 2.27–14.54 | 5 | 158 | 2 | 5.4° | 1.6 |

**Per patient**, ASSD mm (failed solves, K reached):

| Patient | No wall k 0 | k 0.001 | k 0.003 | Wall k 0 | k 0.001 | k 0.003 |
|---|---|---|---|---|---|---|
| karl01 | 2.40 (0, 40) | 2.30 (0, 40) | 3.30 (16, 25) | 3.55 (14, 8) | 3.70 (12, 8) | 6.27 (14, 4) |
| karl02 | 3.20 (0, 40) | 2.90 (0, 40) | 2.72 (0, 40) | 6.57 (12, 4) | 6.49 (11, 4) | 2.71 (11, 40) |
| karl04 | 3.90 (0, 40) | 3.21 (0, 40) | 3.26 (0, 40) | 8.33 (14, 4) | 5.58 (22, 14) | 5.03 (10, 14) |
| karl06 | 2.31 (0, 25) | 2.37 (0, 14) | 2.30 (0, 14) | 2.33 (0, 25) | 2.36 (0, 14) | 2.27 (0, 14) |
| karl08 | 2.25 (0, 40) | 2.17 (0, 40) | 2.45 (0, 25) | 2.23 (0, 40) | 2.46 (0, 25) | 2.44 (0, 25) |
| karl10 | 2.75 (0, 40) | 2.55 (0, 40) | 2.49 (0, 40) | 4.54 (11, 8) | 3.75 (12, 14) | 3.53 (11, 14) |
| karl17 | 2.99 (0, 40) | 2.52 (0, 40) | 2.35 (0, 40) | 4.81 (15, 4) | 3.52 (12, 14) | 2.47 (0, 25) |
| karl18 | 2.42 (0, 14) | 2.35 (0, 14) | 2.30 (0, 14) | 2.25 (0, 14) | 2.42 (0, 8) | 2.39 (0, 8) |
| karl20 | 3.94 (0, 40) | 3.64 (0, 40) | 3.18 (3, 40) | 7.35 (11, 4) | 6.02 (13, 8) | 11.56 (15, 4) |
| karl23 | 3.98 (0, 40) | 3.04 (0, 40) | 6.20 (32, 14) | 13.14 (26, 14) | 11.89 (17, 4) | 14.54 (17, 4) |
| karl24 | 2.45 (0, 25) | 2.33 (0, 25) | 2.24 (0, 25) | 4.24 (10, 4) | 2.43 (0, 25) | 2.29 (0, 25) |
| patient_1 | 2.93 (0, 40) | 4.57 (20, 40) | 3.78 (10, 40) | 11.37 (11, 4) | 7.52 (13, 4) | 12.57 (46, 14) |
| patient_3 | 2.67 (0, 40) | 2.59 (0, 40) | 2.54 (0, 25) | 6.25 (12, 4) | 3.61 (12, 14) | 2.75 (23, 40) |
| patient_10 | 2.92 (0, 40) | 2.50 (0, 40) | 2.49 (0, 40) | 5.95 (10, 4) | 3.98 (12, 14) | 3.39 (11, 25) |

- **Without wall** the support lowers ASSD by ~0.3 mm and the rotation (median 11.0 → 8.4 → 6.1°; karl23 73 → 42 → 11°), takes 5 / 12 % of the load (mean p/E up ~5 / 10 %) and keeps the collapse matched. It brings Newton `maxit` failures (patient_1 at k 0.001; karl01, karl20, karl23, patient_1 at k 0.003), cause unknown (possibly the fixed reference normal at large local rotations).
- **Points > 5 mm outside the inflated lung without wall:** 218 / 162 / 152 over 14 patients at k 0 / 0.001 / 0.003 (karl04 21 / 4 / 0; karl23 34 / 27 / 37, up to 12–14 mm; patient_1 2 / 10 / 19).
- **With wall** the support helps the median and some patients a lot (k 0.003: karl02 6.57 → 2.71, karl17 4.81 → 2.47, karl24 4.24 → 2.29, patient_3 6.25 → 2.75) but makes others worse (karl01 3.55 → 6.27, karl20 7.35 → 11.56, karl23, patient_1). Penetration ≤ 0.6 mm, support share 2–8 % at k 0.001.
- karl06, karl08, karl18 stay at 2.2–2.4 mm in every variant.
- Failures with wall at k 0.001 (one at a time) are `unsettled` and `maxit`; `time` is only their consequence. Only karl06 / 08 / 18 / 24 converge, in < 1 min.

## 7. Wall failures: diagnosis

Cause found in the end: a **near-mechanism**. The small clamp and frictionless contact let the lung rotate / slide along the wall almost for free, and the outer rounds diverge. Fitted pressures need p/E 0.1–3, so limiting q is not an option.

**Fixed** (now part of the method, METHODOLOGY.md section 6):

| Problem | Cause | Fix |
|---|---|---|
| Newton failing on the wall | the VTK distance gradient jumps across edges / vertices, and every lung node starts on a wall vertex | interpolated vertex normal (modest gain) |
| Indefinite contact tangent, penetration 0.4–2.3 mm | negative centre weight (−27/48) of the face quadrature | `nodal` quadrature: 0.04–0.25 mm |
| Unsettled states accepted (max\|U\| 203 mm) | the round loop accepted the state after `wall_max_updates` | unsettled = failure |
| False "unsettled" | the settle measure moved with the normal far from the wall | gap error over points near the wall |
| Large load steps diverging | points slide far along the tangent-plane linearisation | `adaptive_steps` |
| Warm start losing its load, trials restarting from rejected states | `set_regions` cleared the last load | load path on face pressures, re-anchoring to the best state |
| Slow fallback | a failed first solve switched the Jacobian to finite differences | the level stops |
| Unreachable targets (up to 20 mm beyond the wall) | rigid alignment of the target | no alignment |
| A settled state was not a fixed point: steps of relative size 1e-6 failing | every warm start re-linearised, out of equilibrium | keep the equilibrium linearisation |

**Hypotheses checked and rejected** (karl04 wall, k 0.001, one at a time):

- *Time limit:* `max_solve_s` 300 gives a bit-identical result to 120 (ASSD 5.58 mm, K = 14, 22 failed; 97 vs 44 min). Converged solves take ≤ 2.6 s, failed ones never converge, so ~20 s would do (karl04 ~44 → ~10 min).
- *Too high pressures:* q bound 2 is worse (ASSD 6.79, stuck at K = 4) and still fails.
- *Crushed elements:* J reaches 0.086 under the regions at p/E ≥ 2.5 (also without wall: 0.04–0.08 on karl04, karl23, patient_1; median J ~0.5), but the failures are almost all `unsettled`, with no inverted element.
- *Parallel slowdown:* rerun one at a time, the failures remain.

**After the equilibrium-linearisation fix:** the karl04 fit (`fit_sym_k0.001_fix`, interrupted at K = 25) still had 11 failed solves in 4 min, ASSD 5.88. `wall_diagnose.py --from-fit` on karl04: rounds per increment 3 → 2 up to 55 % of the load; from 60 % they still drift (gap error 0.8 → 5.2 mm), 65 % fails. **Remaining cause: points in corners of the wall.** Few points, always the same, 90–160 mm from the clamp: the most posterior ones at mid-height (paravertebral gutter, sliding 19–24 mm tangentially) and one at the base where the normal jumps 76°. With one tangent plane per point and round, a point in a corner goes through the other face, the next round takes that plane and the point goes back: the rounds alternate.

**Options** (parked by the user on 2026-11-02): smoothing only the wall copy in `fem_setup` (preferred), two-plane contact, anatomical clamp (hilum + pulmonary ligament). Rejected: accepting a state within a penetration tolerance (4–5 mm here, too much); Coulomb friction (lubricated pleura, path-dependent solution, noisy objective, non-smooth tangent).

## 8. Evidence behind the design choices

- **No alignment, `plane` loss** (2026-10-05, registered setup): with `hilum_rigid` alignment karl04 was 14.6 mm from the real collapsed surface, against 3.1–3.3 mm unaligned with the plane loss; the alignment rotated the target ~16° and put up to 22 target points beyond the wall (20.6 mm).
- **No registration** (2026-10-27, 15 patients, still-point clamp): median ASSD 2.20 (registered) vs 2.24 mm; less rotation (−2.9° median), 1–4 min faster. The registered reference volume was wrong on karl23 (1984 vs 2554 mL, Dice 0.85: collapse 37 instead of 51 %) and karl20 (−7 %); patient_2 9.3 vs 3.6 mm.
- **Hilum-only clamp** (2026-10-28): without registration the still-point clamp covered 19–58 % of the surface. Hilum only: clamped surface 43 → 5 % (median), pressure rule r 0.70 → 0.97, at the cost of ASSD 2.20 → 2.89 mm and rotation 4.5 → 10.3°.
- **Symmetric loss** (2026-10-28): with the one-sided loss karl23 / patient_1 / patient_2 shrank onto part of the target (collapse 93–98 % vs real 51–86 %, ASSD 24–27 mm). Symmetric: ASSD 4.0 / 2.9 / 5.0 mm, collapse 51 / 57 / 84 % (real 51 / 59 / 86 %); the other 12 within ±0.4 mm; pressure rule r 0.87 → 0.98.

## 9. Other open points

- Rotation of the large collapses (section 2), to check in the sequences.
- Regions at the p/E bounds; K = 40 pressures poorly identifiable.
- Neo-Hookean does not limit the local volume ratio near that of airless tissue (~0.1–0.2).
- Patient position in the collapsed CT (should be supine): it sets the direction of the gradient.
