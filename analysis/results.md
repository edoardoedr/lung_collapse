# Fit results

Results of the fits, the evidence behind the design choices and the problems found. Methods and equations are in [METHODOLOGY.md](../METHODOLOGY.md), the plan in [NEXT.md](../NEXT.md). Written by hand until a script produces it.

Updated 2026-11-16. Shape errors, collapse, rotation and pressure pattern measured by `scripts/summarize_fits.py` (same measure for every fit; output in `analysis/runs.csv`, `compare.csv`, `report.md`, `*.png`):

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
| `fit_grad_ap`, `fit_grad_apsi` | one-sided | no | 0 | 15 | superseded by the `_sym` runs |
| `fit_grad_ap_sym`, `fit_grad_apsi_sym` | symmetric | no | 0 | 15 | 2026-11-16 |
| `fit_grad_ap_sym` | symmetric | yes | 0 | 14 | 2026-11-16, after the 2026-10-30 fix, `JOBS=4` |
| `fit_sym_k0.001_{t20,t300,q2}` | symmetric | yes | 0.001 | karl04 | tests of `max_solve_s` and of the q bound |
| `fit_sym_k0.001_fix` | symmetric | yes | 0.001 | karl04 | after the 2026-10-30 fix; stopped by hand at K = 25 (4 min): ASSD 5.88, 11 failed solves, same as before the fix |

- **Every regions wall fit except `fit_sym_k0.001_fix` (karl04, interrupted) predates the 2026-10-30 fix** of the outer rounds (section 7).

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

## 4. Gradient fits (no wall)

Symmetric loss (`fit_grad_ap_sym`, `fit_grad_apsi_sym`, 2026-11-16) next to the one-sided runs (`fit_grad_ap`, `fit_grad_apsi`). ASSD in mm; slopes in p/E across the lung (ap + = anterior, si + = superior); collapse fitted vs real.

| Patient | Collapse | Regions ASSD | ap: one-sided → sym | ap slope | ap + si: one-sided → sym | ap slope | si slope | Fitted collapse ap / ap + si |
|---|---|---|---|---|---|---|---|---|
| karl01 | 25 % | 2.40 | 6.38 → 6.16 | 0.11 | 6.66 → 6.11 | 0.11 | −0.06 | 30 / 30 % |
| karl02 | 53 % | 3.20 | 8.44 → 8.31 | 0.10 | 8.50 → 8.30 | 0.10 | −0.01 | 49 / 49 % |
| karl04 | 51 % | 3.90 | 12.97 → 12.22 | 0.08 | 12.20 → 11.40 | 0.09 | −0.09 | 46 / 45 % |
| karl06 | 23 % | 2.31 | 5.61 → 5.29 | 0.07 | 5.20 → 4.91 | 0.07 | −0.04 | 25 / 24 % |
| karl08 | 28 % | 2.25 | 7.53 → 7.27 | 0.07 | 6.80 → 6.57 | 0.08 | −0.15 | 26 / 26 % |
| karl10 | 56 % | 2.75 | 12.60 → 10.24 | 0.01 | 7.37 → 7.04 | −0.02 | 0.30 | 54 / 56 % |
| karl17 | 24 % | 2.99 | 8.19 → 7.27 | 0.09 | 7.70 → 7.15 | 0.09 | −0.04 | 28 / 28 % |
| karl18 | 31 % | 2.42 | 6.02 → 5.75 | 0 (uniform kept) | 6.03 → 5.75 | 0 | 0 | 29 / 29 % |
| karl20 | 58 % | 3.94 | 12.60 → 12.00 | 0.69 | 11.06 → 10.52 | 0.61 | −0.45 | 69 / 66 % |
| karl23 | 51 % | 3.98 | 28.27 → 16.38 | 0.67 | 27.57 → 16.31 | 0.76 | 0.16 | 62 / 61 % |
| karl24 | 24 % | 2.45 | 6.11 → 5.70 | 0.03 | 5.20 → 5.00 | 0.03 | −0.04 | 26 / 25 % |
| patient_1 | 59 % | 2.93 | 29.74 → 13.79 | 0.34 | 29.42 → 12.48 | 0.23 | −0.24 | 67 / 61 % |
| patient_2 | 86 % | 5.00 | 27.15 → 12.03 | 0.57 | 27.09 → 9.80 | 0.51 | −0.28 | 87 / 85 % |
| patient_3 | 37 % | 2.67 | 8.51 → 8.06 | 0.10 | 8.54 → 8.07 | 0.10 | −0.02 | 39 / 39 % |
| patient_10 | 47 % | 2.92 | 8.79 → 8.56 | 0.21 | 8.06 → 7.84 | 0.20 | −0.08 | 48 / 48 % |
| **median (15)** | | **2.92** | 8.51 → **8.31** | | 8.06 → **7.84** | | | |

- 8–28 solves, ~0.2 min per fit, no failed solve, stop "levels exhausted" everywhere.
- **The symmetric loss repairs the 3 broken cases** (karl23 28 → 16 mm, patient_1 30 → 12–14, patient_2 27 → 10–12): mean p/E off the bound, collapse 61–87 % instead of 94–99 %, ap slope now positive. On the other 12 it gains 0.2–0.5 mm (karl10 ap 2.4).
- **Collapse:** fitted vs real median gap 2.5 points (ap) / 2.4 (ap + si), max 11 / 10 (karl23, karl20 over-collapse; karl04 under-collapses 5). The earlier systematic over-collapse is gone.
- **Mean p/E = −0.00 + 1.82 × collapse** (ap, r 0.96): close to the hydrostatic 1.67 c of the material.
- **Shape:** still 2.7× the regions (median 7.8–8.3 vs 2.9 mm); worst karl23 16, patient_1 12–14, karl04 / karl20 11–12.
- **ap slope > 0 (anterior more load) in 13 / 15**, 0 in karl18 (uniform best: the optimiser lowers the rms but not the mean error, which picks the best state) and in karl10 (it prefers si).
- **si slope < 0 (apex less load) in 12 / 15** (+0.30 karl10, +0.16 karl23, 0 karl18), as in the regions (apex < base 14 / 15). Not gravity in a supine scan. si gains > 1 mm only on karl10 (3.2), patient_2 (2.2), patient_1 (1.3); median gain 0.3 mm.
- Rotation of the fitted displacement lower than with the regions: karl23 23–28° (regions 73°), patient_2 35–38° (44°), patient_1 20–23° (34°).

### 4b. Gradient `ap` with the wall (`fit_grad_ap_sym` in `_wall`, 2026-11-16, k = 0, `JOBS=4`, after the 2026-10-30 fix)

14 patients (not patient_2). ASSD in mm; "regions + wall" = `fit_sym` / `fit_sym_k0.001` with the wall, **before the fix**, so their failures are not directly comparable.

| Patient | Collapse | ap no wall | **ap + wall** | ap slope (no wall → wall) | Fitted collapse | Failed / solves | min | Regions + wall (k 0 / 0.001) | Regions no wall |
|---|---|---|---|---|---|---|---|---|---|
| karl01 | 25 % | 6.16 | **5.76** | 0.11 → 0.26 | 28 % | 0 / 24 | 0.5 | 3.55 / 3.70 | 2.40 |
| karl02 | 53 % | 8.31 | **5.30** | 0.10 → 0.33 | 49 % | 4 / 27 | 8.4 | 6.57 / 6.49 | 3.20 |
| karl04 | 51 % | 12.22 | **8.62** | 0.08 → 0.45 | 46 % | 4 / 22 | 9.4 | 8.33 / 5.58 | 3.90 |
| karl06 | 23 % | 5.29 | **4.95** | 0.07 → 0.21 | 21 % | 0 / 26 | 0.5 | 2.33 / 2.36 | 2.31 |
| karl08 | 28 % | 7.27 | **6.76** | 0.07 → 0.19 | 25 % | 0 / 20 | 0.4 | 2.23 / 2.46 | 2.25 |
| karl10 | 56 % | 10.24 | **10.31** | 0.01 → 0.02 | 54 % | 0 / 18 | 0.2 | 4.54 / 3.75 | 2.75 |
| karl17 | 24 % | 7.27 | **6.10** | 0.09 → 0.27 | 24 % | 3 / 22 | 7.0 | 4.81 / 3.52 | 2.99 |
| karl18 | 31 % | 5.75 | **5.75** | 0 → 0 (uniform kept) | 29 % | 0 / 8 | 0.1 | 2.25 / 2.42 | 2.42 |
| karl20 | 58 % | 12.00 | **9.80** | 0.69 → 0.79 | 64 % | 8 / 25 | 16.6 | 7.35 / 6.02 | 3.94 |
| karl23 | 51 % | 16.38 | **14.24** | 0.67 → 1.09 | 61 % | 8 / 23 | 15.6 | 13.14 / 11.89 | 3.98 |
| karl24 | 24 % | 5.70 | **5.60** | 0.03 → 0.03 | 26 % | 0 / 16 | 0.2 | 4.24 / 2.43 | 2.45 |
| patient_1 | 59 % | 13.79 | **12.58** | 0.34 → 0.41 | 63 % | 5 / 21 | 11.3 | 11.37 / 7.52 | 2.93 |
| patient_3 | 37 % | 8.06 | **7.92** | 0.10 → 0.19 | 41 % | 2 / 21 | 4.5 | 6.25 / 3.61 | 2.67 |
| patient_10 | 47 % | 8.56 | **7.51** | 0.21 → 0.28 | 46 % | 6 / 21 | 12.5 | 5.95 / 3.98 | 2.92 |
| **median (14)** | | 8.19 | **7.14** | | | | 7.0 | 5.38 / 3.73 | 2.84 |

- **Shape:** the wall improves the gradient fit in 12 / 14 (equal on karl10, karl18), median 8.2 → 7.1 mm; most on karl04 (−3.6), karl02 (−3.0), karl20 (−2.2), karl23 (−2.1). karl02 with gradient + wall (5.3) beats the regions + wall fits (6.5–6.6). Still 2.5× the regions without wall.
- **Collapse** matched as without wall (median gap ~3 points; karl23 61 vs 51 %, karl20 64 vs 58 %). Wall penetration ≤ 0.52 mm (within `wall_settle_mm`), 0–54 points slightly beyond.
- **Failed solves 40 / 294 (14 %)** vs 146 / 496 (29 %) for the regions + wall (k 0, before the fix) and 25–26 % with k 0.001 / 0.003. Zero failures in 6 / 14 patients (karl01, 06, 08, 10, 18, 24: small collapse or small slope). Failures concentrate where collapse and slope are large: karl20 and karl23 8 each (32–35 %), patient_10 6, patient_1 5. Failed attempts by cause (summed): unsettled outer rounds 268, Newton `maxit` 160 (74 on karl23), time 147, inverted 3. Same causes as before (corners, section 7): the gradient halves the failure rate but does not remove it. Part of the gain may be the 2026-10-30 fix itself (karl04 regions after the fix: still 11 failures).
- Time: median 7 min (0.1–0.5 min without failures; 8–17 min with failures, ~20–40 s per solve).
- **The wall makes the ap slope larger, not smaller:** ×2–3 on karl01/02/06/08/17/patient_3, ×5 on karl04. Without the wall a strong anterior load pushes the lung backwards out of the cavity; with the wall the posterior side rests on it, so a larger gradient becomes usable. The hypothesis of section 5 (the no-wall slope stands in for the wall) was wrong.

## 5. Physical reading of the pressures

- p/E is an equivalent surface load lumping elastic recoil, weight and the support of the neighbouring organs.
- **Mean level:** mostly the volumetric response of the material (section 3).
- **Anterior > posterior:** probably gravity in the supine scan (now the proposed load model, METHODOLOGY.md section 8). The inflated lung carries its weight through the vertical gradient of pleural pressure, which the pneumothorax removes; with a stress-free reference the collapse load is uniform + linear in height, slope ρ g per unit height. A body force on top would count gravity twice. Hence E ≈ ρ g L / slope (L = AP depth).
- E estimate (ρ = 250 kg/m³ assumed, ρ g = 2.45 Pa/mm; L = AP extent of the free surface, 167–232 mm): E = ρ g L / θ_ap from the symmetric `ap` fits, 13 patients with θ_ap > 0.03 (not karl10, karl18): **median 4.4 kPa, range 0.74 (karl20) – 12.8 (karl24), ×17**. p0 = (mean p/E) E: 0.9–5.6 kPa. ap + si gives the same within 15 % except karl24 / patient_1.
- Low E (0.7–1.3 kPa) in the 4 largest-rotation / largest-collapse cases (karl20, karl23, patient_2, patient_1), where the slope is 0.3–0.7.
- **With the wall** (section 4b, 12 patients with θ_ap > 0.03): **median 1.7 kPa, range 0.46 (karl23) – 2.9 (karl08), ×6**, plus karl24 at 12.7 (slope 0.03, i.e. no gradient needed). Much tighter than without wall (×17): 8 of 12 between 1.1 and 2.9 kPa. Lowest still in karl23 (0.46), karl20 (0.64), patient_1 (1.07), karl04 (1.13). p0 = (mean p/E) E: 0.5–1.6 kPa (karl24 5.6). Order of magnitude plausible for lung tissue and for the transpulmonary pressure of an inflated lung; not proven.
- **E from no-wall fits is not reliable; the wall fits are the ones to use.** A common E (~1.7 kPa) is the next test: one parameter fewer per patient, with the slope then fixed by ρ̄ g L / E.
- **Common-E scan, first attempt (2026-11-16, `gravity_E_Pa` 1000 / 1700 / 3000, wall): not usable.** 27 of 42 fits stopped at the first solve (time limit 120 s): the fit started from rest with the whole gravity term, i.e. the lung's unbalanced weight on the wall from the first increment (more failures for lower E, larger slope: 13 / 14 at 1000, 10 / 14 at 1700, 4 / 14 at 3000). In several of the other 15 the result is just the starting q0 = 0.5 (karl20 E 3000: collapse 30 vs 58 %), because the best state is chosen by the mean distance while the optimiser lowers the rms (same as karl18 in section 4). Fixed in the code (continuation `gravity_prefit`, `best_by: "rms"`); scan to rerun.
- The relative slope θ_ap / mean p/E (= ρ g L / p0) is not constant either (0.07–0.59).

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
