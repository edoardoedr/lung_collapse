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

### 5b. Common-E scan with the wall (`fit_gE1000`, `fit_gE1700`, `fit_gE3000`, 2026-11-16)

Fixed ap slope = ρ̄ g L / E (ρ̄ 250 kg/m³), only the mean p/E fitted, wall k = 0, `best_by: "rms"`, continuation `gravity_prefit` [0, 0.5] + bisection. ASSD in mm; "free" = `fit_grad_ap_sym` with the wall (slope fitted, section 4b; picked by mean, not rms).

| Patient | free: ASSD (slope, E kPa) | E 1.0 | E 1.7 | E 3.0 | best |
|---|---|---|---|---|---|
| karl01 | 5.76 (0.26, 1.6) | not reached | **5.76** | 5.85 | 1.7 |
| karl02 | 5.30 (0.32, 1.6) | not reached | **5.36** | 6.67 | 1.7 |
| karl04 | 8.62 (0.45, 1.1) | not reached (0.75 of the gravity term) | **9.16** | 10.45 | 1.7 |
| karl06 | 4.95 (0.21, 2.6) | interrupted | 4.98 | **4.96** | flat |
| karl08 | 6.76 (0.18, 2.9) | not reached | **6.71** | 6.78 | flat |
| karl10 | 10.31 (0.02, –) | not reached | not reached | **10.59** | 3.0 (no gradient wanted) |
| karl17 | 6.10 (0.27, 2.1) | interrupted | not reached | **6.45** | – |
| karl18 | 5.75 (0, –) | 9.23 | 8.24 | **6.72** | 3.0 (no gradient wanted) |
| karl20 | 9.80 (0.79, 0.64) | **11.40** | 13.87 | 15.65 | ≤ 1.0 |
| karl23 | 14.24 (1.09, 0.46) | **15.52** | 17.21 | 20.83 | ≤ 1.0 |
| karl24 | 5.60 (0.03, 12.7) | interrupted | 6.03 | **5.64** | 3.0 |
| patient_1 | 12.58 (0.41, 1.1) | not reached | **12.98** | 14.05 | 1.7 |
| patient_3 | 7.92 (0.19, 2.3) | not reached | **7.85** | 7.89 | flat |
| patient_10 | 7.51 (0.28, 1.8) | not reached | **7.40** | 8.55 | 1.7 |
| **12 with E 1.7 and 3.0: median / mean** | 7.14 / 7.90 | | 7.62 / 8.80 | 7.33 / 9.50 | |

- **E = 1.7 kPa common to all:** within 0.5 mm of the patient's own free slope on 8 of 12 (karl01/02/06/08, patient_3/10 equal or better; karl24 +0.4, patient_1 +0.4), karl04 +0.5. Large loss only on karl18 (+2.5: wants no gradient), karl20 (+4.1) and karl23 (+3.0: want a much stronger one, E ~0.5 kPa).
- **Outliers at both ends:** karl10 / karl18 want no gradient at all; karl20 / karl23 (largest rotation, ~1/3 failed solves with the free slope) want 2–3× the gradient of 1.7 kPa. The E 1 kPa fits help them (11.4, 15.5) but stay worse than the free slope.
- **Failed solves almost gone:** at E 1.7 / 3.0, 0 failed solves in 22 of 26 finished fits (karl06 6, karl24 2), 0.3–2.4 min per fit, vs 40 / 294 and 7 min median with the free slope. Moving only q0 keeps the optimiser away from the trial loads that stall the wall rounds.
- **E 1 kPa is mostly not reachable:** 8 of 14 stopped in the continuation (karl04 at 0.75 of the gravity term: every further step hits the 120 s limit with unsettled wall rounds), 3 interrupted by hand; the interrupted ones (karl06, karl17, karl24, run with older code) exported a prefit state and are **not valid** (they are still in `runs.csv`; fixed in the code since). Valid at E 1 kPa: karl18, karl20, karl23.
- **Reading:** one E of ~1.7 kPa (ρ̄ / E = 0.15 s²/m², the quantity really fixed) describes the gravity gradient of most patients at almost no cost in shape; the transferable model is then p0 (from the collapse %) + this fixed slope. The remaining spread (karl10 / karl18 none, karl20 / karl23 much more) is patient-specific: stiffness, density, inflation level, or something the gradient stands in for (mediastinum / heart). E < 1.5 kPa is blocked by the wall solver, not shown to be unphysical.
- The relative slope θ_ap / mean p/E (= ρ g L / p0) is not constant either (0.07–0.59).

### 5c. Common-E scan with the rewritten code and the smooth-field wall (`runs/gE700` … `gE3000`, 2026-11-17)

14 patients (all but karl19 and patient_2), gravity supine, only q0 fitted, smooth-field wall, cold start, `newton_maxit` 30 (the configs were changed to 50 afterwards), `scripts/fit_cohort.py --jobs 4`. Table: `analysis/runs_v2.csv` (`scripts/summarize_runs.py`). ASSD in mm.

| Patient | E 0.7 | E 1.0 | E 1.3 | E 1.7 | E 3.0 | best E | target collapse | collapse at best | p0 at best |
|---|---|---|---|---|---|---|---|---|---|
| karl01 | 5.93 | 5.67 | 5.60 | **5.58** | **5.58** | 1.7–3.0 | 25 % | 28 % | 817 Pa |
| karl02 | 5.94 | 5.23 | **5.10** | 5.40 | 6.63 | 1.3 | 53 % | 49 % | 1049 Pa |
| karl04 | **8.36** | 8.45 | 8.65 | 9.00 | 10.17 | ≤ 0.7 | 51 % | 44 % | 491 Pa |
| karl06 | 6.15 | 5.50 | 5.11 | 4.95 | **4.88** | ≥ 3.0 | 23 % | 22 % | 1104 Pa |
| karl08 | 7.20 | 6.54 | **6.48** | 6.51 | 6.50 | 1.3–3.0 | 28 % | 25 % | 541 Pa |
| karl10 | 12.95 | 12.26 | 11.86 | 11.51 | **10.28** | ≥ 3.0 | 56 % | 54 % | 2778 Pa |
| karl17 | 6.22 | 6.15 | 6.00 | **5.99** | 6.29 | 1.7 | 24 % | 25 % | 683 Pa |
| karl18 | 9.29 | 8.87 | 8.49 | 8.06 | **6.41** | ≥ 3.0 | 31 % | 38 % | 1979 Pa |
| karl20 | **9.57** | 10.86 | 11.97 | 13.18 | 15.11 | ≤ 0.7 | 58 % | 60 % | 711 Pa |
| karl23 | **12.85** | 14.81 | 15.47 | 16.03 | 17.97 | ≤ 0.7 | 51 % | 46 % | 548 Pa |
| karl24 | 7.91 | 6.91 | 6.39 | 6.03 | **5.60** | ≥ 3.0 | 24 % | 25 % | 1259 Pa |
| patient_1 | **12.13** | 12.26 | 12.46 | 12.81 | 13.77 | ≤ 0.7 | 59 % | 57 % | 673 Pa |
| patient_10 | **6.40** | 6.62 | 6.87 | 7.24 | 8.27 | ≤ 0.7 | 47 % | 44 % | 468 Pa |
| patient_3 | 7.77 | 7.72 | 7.66 | **7.64** | 7.65 | flat | 37 % | 40 % | 1172 Pa |
| **median** | 7.84 | 7.32 | 7.26 | 7.44 | 7.14 | best per patient: 6.45 | | | |
| **mean** | 8.48 | 8.42 | 8.44 | 8.56 | 8.94 | best per patient: 7.66 | | | |

- **Solver:** 556 solves in 70 fits, **none failed**; 0.3–1.4 min per fit. Newton reached 30 iterations inside 12–42 solves per E value (more at low E), always recovered by smaller steps. Against the exact wall the points go up to 3.4–3.7 mm beyond the margin (smoothed field: within tolerance).
- **No E is common to the patients.** The cohort mean is flat between 0.7 and 1.7 kPa (8.42–8.56 mm) and the median between 1.0 and 3.0 (7.1–7.4): a common value of ~1–1.3 kPa costs ~0.8 mm against the best E per patient (median 6.45, mean 7.66). But 9 of 14 patients have their best at an end of the scanned range, in opposite directions: karl04, karl20, karl23, patient_1, patient_10 want E ≤ 0.7 kPa (a stronger gradient), karl06, karl10, karl18, karl24 want E ≥ 3 kPa (a weaker one).
- **The preferred E goes with the collapse:** the five that want a low E have collapsed by 47–59 %, three of the four that want a high E by 23–31 % (karl10, 56 %, is the exception; karl02, 53 %, sits at 1.3). A lung that has collapsed more behaves as softer against its weight. Possible readings: the tissue softens in compression more than the Neo-Hookean law does; the dependent lung is denser (ρ̄ / E is what is fitted); or the gradient stands in for something else (mediastinum, heart). To quantify it: fit E per patient (now possible) and plot E against the collapse %.
- **Sensitivity:** for 8 patients the ASSD changes by less than 1 mm over the whole range; the choice of E matters for karl20 (9.6 → 15.1), karl23 (12.9 → 18.0), karl18 (9.3 → 6.4), karl10, karl24, karl04.
- The fitted collapse is within 7 points of the target on all patients (karl04 44 vs 51 %, karl18 38 vs 31 %).

**E fitted per patient (`runs/fitE`, 2026-11-17).** q0 and E both fitted (E start 1.3 kPa, bounds 0.2–20 kPa), same settings as the scan, `newton_maxit` 50. 14 patients (patient_2 missing from the output), 151 solves, **none failed**, 0.2–1.7 min per fit.

| Patient | E fitted | q0 | p0 | collapse fit / target | ASSD fitE | best of the scan |
|---|---|---|---|---|---|---|
| karl01 | 1.29 kPa | 0.461 | 593 Pa | 27 / 26 % | 5.60 | 5.58 |
| karl02 | 1.02 kPa | 0.795 | 809 Pa | 49 / 53 % | 5.22 | 5.10 |
| karl04 | 0.61 kPa | 0.731 | 447 Pa | 47 / 51 % | 8.51 | 8.36 |
| karl06 | 1.76 kPa | 0.363 | 640 Pa | 22 / 23 % | 4.93 | 4.88 |
| karl08 | 1.72 kPa | 0.429 | 738 Pa | 25 / 28 % | 6.48 | 6.48 |
| karl10 | **19.3 kPa (upper bound)** | 0.931 | (18 kPa, not meaningful) | 54 / 56 % | 9.80 | 10.28 |
| karl17 | 1.31 kPa | 0.371 | 486 Pa | 24 / 24 % | 5.99 | 5.99 |
| karl18 | **20 kPa (upper bound)** | 0.586 | (12 kPa, not meaningful) | 34 / 31 % | 5.64 | 6.41 |
| karl20 | 0.31 kPa | 0.904 | 281 Pa | 58 / 58 % | 7.95 | 9.57 |
| karl23 | **0.20 kPa (lower bound)** | 0.519 | 104 Pa | **40 / 51 %** | 7.61 | 12.85 |
| karl24 | 3.59 kPa | 0.421 | 1511 Pa | 25 / 25 % | 5.55 | 5.60 |
| patient_1 | 0.64 kPa | 0.991 | 630 Pa | 59 / 59 % | 12.10 | 12.13 |
| patient_10 | 0.58 kPa | 0.653 | 377 Pa | 44 / 47 % | 6.35 | 6.40 |
| patient_3 | 1.63 kPa | 0.704 | 1148 Pa | 41 / 37 % | 7.66 | 7.64 |
| **median / mean** | 1.29 kPa (11 in bounds) | | | | **6.41 / 7.10** | 6.45 / 7.66 |

- **Shape:** median 6.41, mean 7.10 mm, better than any common E (mean 8.42–8.94) and than the best scanned value per patient (mean 7.66): the gain is on the patients whose optimum was outside 0.7–3 kPa (karl23 12.85 → 7.61, karl20 9.57 → 7.95, karl18 6.41 → 5.64).
- **Three at a bound.** karl10 and karl18 go to the upper bound: they want no gravity gradient at all, so E (and p0 = q0 E) is not identified there; only q0 means something. karl23 goes to the lower bound and still collapses 11 points less than its target (40 vs 51 %): to match the shape it gives up volume, so the model is missing something on this lung.
- **E decreases with the collapse.** On the 11 patients inside the bounds: log₁₀ E = 3.69 − 0.0168 × collapse % (r = −0.83, p = 0.001), i.e. E ≈ 1.8 kPa at 25 % collapse and ≈ 0.7 kPa at 50 % (Spearman −0.77, p = 0.005; with karl10 / karl18 / karl23 included −0.48, p = 0.09). The more a lung has collapsed, the stronger the effect of its weight needed to reproduce its shape.
- **Reading, not settled.** What is fitted is ρ̄ / E. The Neo-Hookean law has no stiffening in compression, and a real lung does stiffen, which would predict the opposite trend; so the trend is more likely a load the model lacks (mediastinum / heart pushing on the medial surface, diaphragm) that grows with the collapse and that the gravity gradient stands in for, or a dependent lung denser than the mean. karl10 (56 % collapse, no gradient wanted) does not fit the trend.
- **Use:** a prediction from the collapse % alone is now possible: q0 from the volume (forward `volume_ratio`) and E from the regression above. To test it: leave-one-out (NEXT.md).

**Forward runs (karl04, E 1.7 kPa, `volume_ratio` 0.55).** Supine (`v2_forward`): the first guess q0 = 0.75 from the hydrostatic response gave 55.2 % at once (1 solve, 21 s); ASSD vs the real collapsed lung 8.97 mm, the same as the fit. Supine CT → left lateral decubitus (`v2_lateral`, body force): converged in 1 solve (60 s), volume 54.7 %, rigid part 13° / 25.6 mm, 3.1 mm beyond the wall margin; ASSD 24.1 mm against the supine collapsed lung, as expected for another position (there is no lateral scan to compare with).

**ν fitted with q0 and E (`runs/fitEnu`, 2026-11-17).** karl04, karl06, karl20; E start 1.3 kPa, bounds 0.2–20 kPa; ν start 0.40, bounds 0.25–0.48; otherwise as `fitE`. 45 solves, none failed, ~0.5 min per fit.

| Patient | ν | E [kPa] fitEnu / fitE | q0 fitEnu / fitE | p0 [Pa] fitEnu / fitE | ASSD fitEnu / fitE | collapse fit / target |
|---|---|---|---|---|---|---|
| karl04 | 0.250 (lower bound) | 0.67 / 0.61 | 0.301 / 0.731 | 201 / 447 | 8.23 / 8.51 | 48 / 51 % |
| karl06 | 0.250 (lower bound) | 2.18 / 1.76 | 0.151 / 0.363 | 328 / 640 | 4.79 / 4.93 | 22 / 23 % |
| karl20 | 0.266 | 0.41 / 0.31 | 0.467 / 0.904 | 190 / 281 | 7.84 / 7.95 | 59 / 58 % |

- ν falls monotonically from 0.40 at every evaluation and stops at or near the lower bound; the shape gains only 0.1–0.3 mm ASSD, far below the residual error.
- ν trades off against q0: the volume is set by ~q0 / (K/E) (K/E 1.67 at ν 0.40, 0.67 at 0.25), and q0·E/K stays nearly constant (karl06 0.218 → 0.226, karl04 0.439 → 0.452). So **p0 alone is not identified, only p0 / K**: p0 changes by ×~2 with ν.
- E moves by only +10–30 %: the gravity gradient, and the E(collapse %) trend, do not depend on ν.
- Conclusion: ν is not identifiable from the collapsed shape; keep it fixed (from the literature) and always report p0 with the ν used.

### 5d. Is the pleural cavity larger than the lung? (2026-11-17)

There is one CT per patient, with the pneumothorax; the model's "inflated lung" is its pleural cavity (METHODOLOGY.md section 7, item 12). Hypothesis: the cavity is enlarged by the pneumothorax (chest wall, mediastinum, diaphragm), more so in the larger ones, and the low E of the large collapses compensates for it. Test from the input surfaces (`Input_Data/`, read with numpy on the Mac; same measures as `scripts/cavity_volumes.py`; table `analysis/cavity.csv`): cavity, collapsed and contralateral lung (sum of its lobes) volumes; enlargement = (cavity / contralateral) / normal ratio (right / left 1.15); E from `fitE` (ν 0.40) and `fitE_nu30`, patients with E at a bound left out. All surfaces closed; the collapse % against the cavity equals the fits' target.

| Patient | side | cavity mL | collapsed mL | contralateral mL | enlargement | collapse vs cavity | vs expected volume | E fitE_nu30 |
|---|---|---|---|---|---|---|---|---|
| karl01 | L | 2135 | 1584 | 2269 | 1.08 | 26 % | 20 % | 1468 |
| karl02 | R | 4976 | 2357 | 2671 | 1.62 | 53 % | 23 % | 1129 |
| karl04 | R | 5460 | 2658 | 4460 | 1.06 | 51 % | 48 % | 685 |
| karl06 | R | 3101 | 2384 | 2982 | 0.90 | 23 % | 31 % | 2089 |
| karl08 | R | 3695 | 2657 | 4026 | 0.80 | 28 % | 43 % | 2599 |
| karl10 | L | 2788 | 1231 | 2743 | 1.17 | 56 % | 48 % | (bound) |
| karl17 | L | 3088 | 2346 | 3773 | 0.94 | 24 % | 29 % | 1460 |
| karl18 | L | 1924 | 1338 | 3676 | 0.60 | 30 % | 58 % | (bound) |
| karl19 | R | 2182 | 751 | 2413 | 0.79 | 66 % | 73 % | — |
| karl20 | L | 2490 | 1036 | 2211 | 1.30 | 58 % | 46 % | 446 |
| karl23 | L | 2610 | 1270 | 2969 | 1.01 | 51 % | 51 % | (bound) |
| karl24 | L | 2185 | 1651 | 1904 | 1.32 | 24 % | 0 % | 3732 |
| patient_1 | L | 2800 | 1152 | 3161 | 1.02 | 59 % | 58 % | 719 |
| patient_10 | L | 4010 | 2122 | 3897 | 1.18 | 47 % | 37 % | 650 |
| patient_2 | L | 5350 | 740 | 2596 | 2.37 | 86 % | 67 % | 403 |
| patient_3 | L | 2758 | 1741 | 3451 | 0.92 | 37 % | 42 % | 1450 |

- **Enlargement:** median 1.06 (IQR 0.93–1.24), range 0.60–2.37. 11 of 16 are within 0.8–1.2, i.e. within the normal individual right / left variability: for most patients the cavity is close to the expected lung volume and is a fair reference; the outliers are patient_2 (2.37), karl02 (1.62), karl24 (1.32), karl20 (1.30) above and karl18 (0.60) below. The spread is far larger than any plausible enlargement (and below 1 in 6 patients): the contralateral lung is a noisy reference (individual right / left ratio, disease, segmentation). Outliers: patient_2 (2.37, cavity 5.3 L against a 2.6 L right lung), karl02 (1.62), karl18 (0.60).
- **It does not explain the E trend.** log₁₀ E against the collapse % measured on the cavity: r −0.87 (`fitE_nu30`, 12 patients), −0.84 (`fitE`, 11); against the enlargement: −0.51 / −0.21; against the collapse % measured on the expected volume: −0.75 / −0.69, weaker, not stronger. With both in one regression the enlargement coefficient is positive (+0.26 per unit), the opposite of the hypothesis.
- **E(collapse %) with ν 0.30** (`fitE_nu30`, 12 in bounds, patient_2 included): log₁₀ E = 3.63 − 0.0135 × collapse %, i.e. ≈ 2.0 kPa at 25 %, 0.9 kPa at 50 % (ν 0.40: 1.9 / 0.7). karl10, karl18 at the upper bound, karl23 at the lower bound, as with ν 0.40.
- **Trachea shift** (x of the top 15 mm of the airways against the middle of the two lateral extents): not usable as a mediastinal-shift proxy (karl19 99 mm, patient_3 −43 mm, karl08 −32 mm towards the pneumothorax).
- **patient_6** (CT before the pneumothorax 6_1, CT with it 6_3; no cavity segmented in 6_3 yet): left lung before 1972 mL (L / R 0.75), collapsed 1180 mL, i.e. a **true collapse of 40 %**. The right lung is **22 % smaller** in 6_3 (2646 → 2060 mL): different inspiration between the scans and / or compression by the mediastinal shift. The contralateral lung therefore changes a lot with the pneumothorax and the breathing state, which explains why it is a noisy reference above.
- **Reading:** with this measure the cavity enlargement is not what drives E down; the measure is too noisy to exclude a moderate enlargement. The direct test is patient_6 (CT before the pneumothorax, NEXT.md).

### 5e. Lung density from the CT of the pneumothorax (2026-11-17)

The CTs (`Input_CT/<p>.nii[.gz]`, one per patient, with the pneumothorax) are now available. `scripts/ct_density.py` (Mac env `lungct`; table `analysis/ct_density.csv`) voxelises the cavity, the collapsed lung and the contralateral lobes on the CT grid and converts HU to density, ρ = (HU − HU_air) / (−HU_air) g/mL, HU_air = median of the pleural air eroded by 2 mm. Only air leaves a collapsing lung, so mass = Σ ρ over the collapsed lung and **ρ̄ of the inflated lung = mass / V_cavity**.

- **Alignment of CT and surfaces is right** on all 16: pleural-air core −974 to −1038 HU, mask volumes equal to the surface volumes, no collapsed voxel outside the cavity. Mean HU of the whole pleural-air region is higher (−746 to −796) on karl17, karl18, karl24: something denser than air between lung and cavity (effusion, or a collapsed segmentation tighter than the lung): to look at in Slicer.
- **ρ̄ (inflated) median 163 kg/m³, range 85–294** (assumed 250). Collapsed lung 0.18–0.87 g/mL (patient_2 −130 HU: almost airless).
- **Mass check on patient_6** (two scans): left lung 446 g before the pneumothorax, 332 g collapsed (−26 %); right lung 548 → 598 g (+9 %). So the mass is reliable to ~±25 % only (partial volume at the border of a tight segmentation, blood leaving the collapsed lung by hypoxic vasoconstriction, different scans). Right-lung density 0.207 → 0.290 g/mL between the scans.
- **The density does not explain the E trend; it makes it steeper.** ρ̄ is lower where the collapse is larger (r −0.60, 12 patients with E in bounds) and goes with the fitted E (Spearman +0.81). Since only ρ̄ / E is identified, E = E_fit · ρ̄ / 250: the spread grows (sd log E 0.29 → 0.39) and log₁₀ E = 3.60 − 0.0181 × collapse % (r −0.86), ≈ 1.4 kPa at 25 %, 0.5 kPa at 50 %. Large-collapse lungs are lighter per cavity volume, yet the fit wants a stronger gravity load: the gradient stands in for something other than the weight.
- **Cavity enlargement from the mass** (V_expected = mass / ρ_contralateral, contralateral lung of the same CT): 0.49–3.79, r +0.47 with the volume-ratio measure of 5d; not reliable either, because the contralateral lung is itself denser during the pneumothorax (patient_6: +40 %) and differs between patients (karl19, patient_2 ~0.38 g/mL).
- **Density by AP thirds** of the collapsed lung (posterior / anterior): 1.1–2.4 in 12 of 16 (dependent lung denser), < 1 on karl04, karl20, patient_10 and **karl10 (0.43, anterior third 0.94 g/mL)**: karl10 is one of the two patients that want no gravity gradient. To look at in Slicer.
- **Does the measured ρ̄ help predict the load?** Prediction setting (user, 2026-11-17): inputs = inflated lung, collapse %, ρ̄; q0 from the volume, the gravity slope s_g = ρ̄ g L / E from a rule. Leave-one-out on the 12 `fitE_nu30` patients with E in bounds, error of the predicted log₁₀ s_g: rule E(collapse %) with ρ̄ = 250 for all rms 0.18 (×1.5); s_g(collapse %) 0.20; s_g(collapse %, ρ̄) 0.20 (coefficient of log ρ̄ −0.5, physics says +1); E(collapse %) with the measured ρ̄ 0.27 (×1.8), worse; no rule (constant s_g) 0.34 (×2.2). **The measured density makes the prediction worse**: the needed slope depends on the collapse %, not on the weight. log₁₀ s_g = −0.96 + 0.0141 × collapse % (0.25 at 25 %, 0.55 at 50 %).
- **Inflated lung of a new patient** (`ct_density.py --config ... ` without a collapsed lung: mean density, air calibrated in the trachea): patient_6_1, the only real inflated lung (CT before the pneumothorax), **208 kg/m³** (1971 mL, −775 HU, 410 g). Check: karl04's cavity treated the same way gives 89 kg/m³ against 85 from the mass method. The cohort's ρ̄ (mass / cavity, median 163) is lower than this real lung: larger cavity, or a different breathing state.
- **Use:** the shape fits do not change (only ρ̄ / E matters); the measured ρ̄ rescales the reported E and p0 = q0 E per patient, with ~±25 % uncertainty.

### 5f. Leave-one-out prediction from the collapse % (± the CT density) (2026-11-17)

`scripts/leave_one_out.py --fit-run fitE_nu30 --rule E | E_rho | sg_rho` (runs `loo`, `loo_rho`, `loo_sgrho`). Prediction setting: inputs = inflated lung, collapse %, optionally ρ̄ from the CT (5e); ν 0.30; q0 from the volume (forward `volume_ratio`, collapse reached within 0.5 points everywhere); the gravity slope from a rule fitted on the other patients (12 with E in bounds). ASSD of the prediction minus ASSD of the patient's own `fitE_nu30`, mm:

| Rule | inputs | slope error | median | mean | max | failed |
|---|---|---|---|---|---|---|
| `E`: E(collapse %), ρ̄ = 250 | collapse % | ×1.52 | +0.20 | +0.26 | +0.90 (karl02) | 0 |
| `E_rho`: material E(collapse %), weight from the measured ρ̄ | collapse %, ρ̄ | ×1.84 | +0.10 | +0.43 (11) | +1.99 (karl02) | patient_2 (E 59 Pa, slope 4.6) |
| `sg_rho`: slope(collapse %, log ρ̄) | collapse %, ρ̄ | ×1.60 | +0.09 | +0.19 | +0.66 (karl24) | 0 |

- **The prediction from the collapse % alone costs ~0.2 mm** against the patient's own fit (median 6.42 vs 6.22 mm, mean 7.32 vs 7.06): the error is the model's (5–12 mm), not the rule's. A ×1.5–2 error in the slope moves the shape by less than 1 mm on these 12.
- **The measured density does not help.** As a physical weight (`E_rho`) it is the worst (mean +0.43, karl02 +2.0 mm, patient_2 does not converge); as a free predictor (`sg_rho`) it gains 0.07 mm on the mean over `E`, not significant on 12 patients, with an unphysical sign (denser → smaller slope).
- **The three patients with E at a bound** (not in the rule) cost +1.8 to +5.6 mm with every rule: karl10 and karl18 (fit with no gradient) and karl23 (fit with the strongest gradient). That is where the model needs something else (5c, 5e).
- Decision proposed: the rule `E` (collapse % only, ν 0.30, ρ̄ 250 as a nominal value), log₁₀ E = a + b × collapse % on all 12; ρ̄ from the CT only to report E and p0 in physical units.

### 5g. Final fit and prediction, 16 patients (`runs/fit_final`, `runs/pred`, 2026-11-17)

Settings: ν 0.30, gravity supine, smooth-field wall. `fit_final`: q0 and E fitted (E 0.2–20 kPa), as `fitE_nu30` (identical results on the 15 shared patients) plus karl19, now set up (`as_triangles` fix). `pred`: `scripts/predict_cohort.py`, forward from the patient's collapse % with the E rule (fitted on 12 of these patients, so not a leave-one-out: see 5f for that). Table `analysis/runs_final.csv`. No failed solve (165 in the fits, 44 in the predictions; 1–8 solves per prediction).

| Patient | collapse | E fit | q0 | p0 | ρ̄ CT | E / p0 at ρ̄ CT | ASSD fit | E rule | ASSD pred | diff |
|---|---|---|---|---|---|---|---|---|---|---|
| karl01 | 25 % | 1468 | 0.240 | 352 | 294 | 1726 / 413 | 5.53 | 1937 | 5.39 | −0.13 |
| karl02 | 53 % | 1129 | 0.398 | 449 | 170 | 766 / 305 | 5.16 | 832 | 5.93 | +0.77 |
| karl04 | 51 % | 685 | 0.377 | 258 | 85 | 234 / 88 | 8.36 | 869 | 8.71 | +0.35 |
| karl06 | 23 % | 2089 | 0.186 | 389 | 173 | 1445 / 269 | 4.82 | 2084 | 4.84 | +0.01 |
| karl08 | 28 % | 2599 | 0.236 | 614 | 182 | 1893 / 447 | 6.39 | 1784 | 6.48 | +0.09 |
| karl10 | 56 % | (bound) | 0.466 | — | 285 | — | 10.31 | 751 | 13.08 | +2.77 |
| karl17 | 24 % | 1460 | 0.184 | 269 | 157 | 918 / 169 | 5.99 | 2029 | 5.91 | −0.08 |
| karl18 | 31 % | (bound) | 0.297 | — | 249 | — | 5.84 | 1653 | 8.47 | +2.62 |
| karl19 | 65 % | 728 | 0.655 | 477 | 98 | 286 / 187 | 8.66 | 564 | 8.59 | −0.07 |
| karl20 | 58 % | 446 | 0.547 | 244 | 120 | 215 / 118 | 8.02 | 700 | 8.50 | +0.48 |
| karl23 | 51 % | (bound) | 0.483 | — | 230 | — | 9.17 | 872 | 13.90 | +4.72 |
| karl24 | 25 % | 3732 | 0.212 | 792 | 191 | 2853 / 606 | 5.49 | 2000 | 5.78 | +0.29 |
| patient_1 | 59 % | 719 | 0.510 | 367 | 114 | 327 / 167 | 11.82 | 692 | 11.81 | −0.01 |
| patient_10 | 47 % | 650 | 0.310 | 202 | 118 | 308 / 96 | 6.05 | 992 | 6.32 | +0.27 |
| patient_2 | 86 % | 403 | 0.903 | 364 | 120 | 193 / 175 | 9.51 | 295 | 9.62 | +0.11 |
| patient_3 | 37 % | 1450 | 0.354 | 513 | 135 | 782 / 276 | 7.52 | 1365 | 7.49 | −0.03 |

E and p0 in Pa, ASSD in mm; "(bound)": E at the bound of the fit, E and p0 not identified (karl10 19 kPa, karl18 20 kPa, karl23 0.2 kPa).

- **13 patients with E in bounds:** fit median 6.39 / mean 7.18 mm, prediction 6.48 / 7.34; difference median +0.09, mean +0.16, max +0.77 mm (karl02). Predicting from the collapse % alone is as good as fitting, for these.
- **3 at a bound (karl10, karl18, karl23):** +2.6 to +4.7 mm; prediction 13.1, 8.5, 13.9 mm. All 16: prediction median 7.98, mean 8.18 mm (no deformation: median 14.8 mm).
- **karl19** (new): E 728 Pa, in line with the rule (564 predicted at 65 %); the fit over-collapses (76 vs 65 %: it trades volume for shape), the prediction at the exact volume has the same ASSD (8.59 vs 8.66) but a larger maximum error (60 vs 40 mm).
- **Rule with karl19** (13 patients): a 3.617, b −0.0130, within 5 % of the coded rule over 25–80 %: rule kept.
- The volume of the prediction matches the target within 0.2 points on all 16.

### 5h. Segmentation check from the CT (`scripts/qa_segmentation.py`, 2026-11-17)

Label 1: HU > −100 inside the collapsed lung (2 mm from its border, blobs ≥ 5 mL); label 2: HU > −500 between the collapsed lung and the cavity. Labelmaps `results/<p>/qa_segmentation.nrrd`.

- **Dense inside the collapsed lung:** 3–11 % (73–204 mL) in 13 patients, medial-posterior: the hilar vessels and dependent atelectasis, normal. **karl10: 437 mL (36 %)**, central-medial, anterior third at 0.94 g/mL: heart / mediastinum inside the collapsed segmentation (user, 2026-11-17: karl10 to be re-segmented). patient_2: 494 mL (67 %), posterior: the lung at 86 % collapse is almost airless (−130 HU), probably real; karl19: none.
- **Tissue between lung and cavity:** none or < 16 mL in 9 patients. **karl17 98 mL (15 HU), karl24 81 mL (84 HU)**: anterior-basal, not dependent, so not an effusion: diaphragm / abdomen / heart inside the cavity segmentation. **karl18 80 mL (27 HU)**, lateral-basal: the same or a small effusion. patient_1 33 mL and patient_2 98 mL, posterior (dependent), water density: pleural effusion (real; the lung cannot occupy it).
- karl10 and karl18 are two of the three patients the model fails on (5g): their segmentation is the first suspect.

### 5i. Fitted p0 against the physiological recoil pressure (2026-11-17)

p0 = q0 · E is the uniform transpulmonary pressure released by the pneumothorax, i.e. the mean elastic recoil pressure of the lung in the reference state (METHODOLOGY.md 3.1–3.2). Physiological reference, static recoil pressure Pst(l) (esophageal balloon, healthy adults, upright):
- **Turner, Mead & Wohl, J Appl Physiol 25:664–671, 1968** (doi:10.1152/jappl.1968.25.6.664; 44 subjects aged 13–61, standing, expiratory limb; Table 3 read from the PDF): Pst at 50 % TLC median 4.2 cmH₂O (1.2–8.7), at 60 % TLC 6.4 (2.9–12.4), at TLC 32 (19–69). FRC is 39–64 % of TLC (median 55 %, seated plethysmograph, rising +0.18 % TLC per year). Interpolated at each subject's FRC (our computation): **recoil at FRC median 4.8 cmH₂O (1.7–10.6)**, 4.7 (1.7–6.7) for the 17 subjects ≥ 40 years. Recoil falls with age at every volume, e.g. at 60 % TLC Pst = 9.92 − 0.0957 × age (≈ 4.2 cmH₂O at 60 years).
- **Milic-Emili et al., J Appl Physiol 21:749–759, 1966** (doi:10.1152/jappl.1966.21.3.749; 9 seated men): transpulmonary pressure at TLC mean 38 cmH₂O (26–68); pleural gradient ≈ 0.2 cmH₂O/cm (METHODOLOGY.md 3.1).
- "Pulmonary Function Testing", Murray & Nadel's Textbook of Respiratory Medicine (PMC7158317, doi:10.1016/B978-1-4557-3383-5.00025-7): about 12 cmH₂O to hold three fourths of TLC. From `analysis/runs_final.csv` (1 cmH₂O = 98.1 Pa):

| | n | p0 median | p0 range | p0 at ρ̄ CT median (range) |
|---|---|---|---|---|
| `fit_final`, E in bounds | 13 | 367 Pa (3.7 cmH₂O) | 202–792 Pa (2.1–8.1 cmH₂O) | 187 Pa (88–606) |
| `fit_final`, E at a bound | 3 | — | karl23 97, karl18 5943, karl10 8847 Pa | — |
| `pred` (E rule), all | 16 | 383 Pa (3.9 cmH₂O) | 297–438 Pa (3.0–4.5 cmH₂O) | 245 Pa (116–508) |

- **p0 matches the recoil at FRC**: fitted median 3.7 cmH₂O (2.1–8.1), Turner et al. at FRC median 4.8 (1.7–10.6), 4.7 above 40 years; predictions 3.0–4.5. It is far below the recoil at TLC (32–38 cmH₂O) and below the 12 cmH₂O at three fourths of TLC: the reference lung behaves as a lung at its resting volume. Turner's subjects were standing; supine the FRC, and with it the recoil, is lower, so the agreement holds within this uncertainty. Only the three patients at a bound of E are off (karl10 / karl18 60–90 cmH₂O, karl23 1 cmH₂O), and there p0 is not identified.
- **p0 hardly depends on the collapse %** (13 in bounds: r −0.31 against the collapse %, while q0 has r +0.96): the collapse % comes from E, not from p0. The E rule therefore amounts to a nearly constant released pressure, p0 ≈ 300–440 Pa in all 16 predictions. A rule "p0 fixed at ~370 Pa, E = p0 / q0" would be its physical restatement.
- **The pressure scale comes from ρ̄.** Only q0 and ρ̄ / E are identified by the shape, so p0 = q0 ρ̄ g L / s_g is proportional to the assumed density, i.e. to the pleural gradient ρ̄g = 2.45 Pa/mm: p0 / (ρ̄ g) ≈ 150 mm, the mean recoil equals the pleural gradient over ~15 cm of height. With the measured gradient of Milic-Emili et al. (0.2 cmH₂O/cm, ρ̄ ≈ 200 kg/m³) p0 would be 0.8 × these values (median ≈ 3.0 cmH₂O); with the CT density (median 163 kg/m³) a median 187 Pa (1.9 cmH₂O). Both are still the order of the resting recoil.
- **Change (user, 2026-11-17): ρ̄ = 200 kg/m³ in the model** (the measured gradient), E default 1360 Pa and E rule a = 3.5332: the shapes of every fit and prediction are unchanged, E and p0 of the runs above (all at ρ̄ 250) become × 0.8, i.e. p0 median 294 Pa = 3.0 cmH₂O (fit) and 306 Pa = 3.1 cmH₂O (prediction), still inside Turner's FRC range.
- Caveats: the reference is taken stress-free (METHODOLOGY.md 7, item 1), so p0 is the released pressure to first order, without the prestress stiffening; surface tension is in E and ν, not separate; the "inflated lung" is the pneumothorax cavity (5d), and the breathing state of the CT is not known.

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

### 7b. Wall as a smooth field (`wall_update: "potential"`), local test only (2026-11-16)

Not a server result: karl04 (`_wall` setup, 1355 nodes, 6777 tets) on the Mac with the numpy test core (`tests/numpy_core.py`, same Neo-Hookean model, ν 0.40) and a wall field built in numpy on a coarse grid (4 mm, σ 3 mm, tolerance 5 mm; the server default is 1.5 mm). Load = uniform p/E + ap gradient γ (h − h̄), **cold start from the reference with the whole load** (no continuation), `adaptive_steps`.

| Load | q0 | ap slope (p/E over the lung) | Result | Volume | Points in contact | Max gap beyond the margin | Newton runs / iterations |
|---|---|---|---|---|---|---|---|
| γ of E 1.7 kPa | 0.30 | 0.30 | converged | 81 % | 100 | 0.15 mm | 3 / 53 |
| γ of E 1 kPa | 0.30 | 0.51 | converged | 79 % | 139 | 0.19 mm | 4 / 97 |
| γ of E 1 kPa | 0.50 | 0.51 | converged | 68 % | 103 | 0.18 mm | 6 / 135 |
| γ of E 0.5 kPa | 0.40 | 1.02 | converged | 69 % | 160 | 0.23 mm | 6 / 141 |

- No wall rounds, no failed path. With the outer rounds on the server, karl04 at E 1 kPa needed the continuation and still stopped at 0.75 of the gravity term (section 5b).
- Checks: wall tangent vs finite differences 1e-7 (336 nodes in contact), Jacobian with contact vs finite differences 3e-8 (with the outer rounds this check was meaningless).
- Newton takes ~20–25 iterations per run, close to `newton_maxit` 30: watch `fail_maxit` on the server.
- To confirm on the server with the torch core and the 1.5 mm grid (commands in REWRITE_PLAN.md Phase 1 / NEXT.md).

**Server, torch core, grid 1.5 mm, σ 3 mm (2026-11-16, `fit_gE1700_pot`: E 1.7 kPa, only q0 fitted, cold start, no continuation, `JOBS=4`).** The E 1 kPa runs (`fit_gE1000_pot`) are not analysed yet (logs not received).

| Patient | ASSD outer (5b) | ASSD potential | Solves (failed) | Fit time | Newton `maxit` inside solves | Beyond the margin: exact wall | smoothed field |
|---|---|---|---|---|---|---|---|
| karl01 | 5.76 | 5.58 | 4 (0) | 20 s | 0 | 54 points, max 2.35 mm | 44, max 0.09 mm |
| karl04 | 9.16 | 9.00 | 11 (0) | 50 s | 4 | 54 points, max 2.47 mm | 50, max 0.15 mm |
| karl20 | 13.87 | 13.18 | 6 (0) | 34 s | 1 | 15 points, max 2.42 mm | 11, max 0.10 mm |
| karl23 | 17.21 | 16.03 | 8 (0) | 38 s | 2 | 11 points, max 1.90 mm | 6, max 0.09 mm |

**E 1 kPa with the potential wall (`fit_gE1000_pot`, same settings): reached on all 4, no failed solve, no continuation.** With the outer rounds karl01 and karl04 could not reach it (section 5b).

| Patient | free slope (4b) | E 1.0 outer | E 1.0 potential | E 1.7 potential | Solves (failed) at E 1.0 | Time | p0 at E 1.0 |
|---|---|---|---|---|---|---|---|
| karl01 | 5.76 | not reached | 5.67 | **5.58** | 6 (0) | 0.3 min | 442 Pa |
| karl04 | 8.62 | not reached | **8.45** | 9.00 | 10 (0) | 0.6 min | 723 Pa |
| karl20 | 9.80 | 11.40 | **10.86** | 13.18 | 11 (0) | 0.9 min | 1031 Pa |
| karl23 | 14.24 | 15.52 | **14.81** | 16.03 | 13 (0) | 0.9 min | 900 Pa |

- The continuation (`gravity_prefit`, `gravity_bisect`) is not needed with the potential wall: every fit starts from rest with the whole gravity term.
- Newton reaches `newton_maxit` (30) inside some solves (1–8 runs per fit, more at E 1 kPa; recovered by the sub-steps, no failed solve): try 50.
- E 1 kPa is now better than 1.7 on 3 of these 4 (karl04 even better than its free-slope fit with the outer wall), so the common-E scan of section 5b has to be redone on all patients with the potential wall before fixing E: the earlier preference for 1.7 was partly an artefact of the wall solver.

- Jacobian with the wall vs finite differences (karl04, K = 4, q 0.35–0.65): max rel 8.7e-7, all 5 columns.
- Wall field (4 patients): built in 23–37 s; smoothed distance of the reference points median 0.3, range −1.5 .. +3.3 mm (the smoothed wall lies up to ~3 mm inside the true one at pointed parts, covered by the per-point margin); |∇φ̃| at the reference points 0.37–1.00 (below 1 at sharp corners, where the gap is under-measured); no target point beyond its margin.
- Cost of the smoothing: against the exact wall the points go up to ~2.4 mm beyond the 5 mm margin (0.3 mm with the outer rounds on karl04).

### 7c. Rewritten code (branch `refactoring_to_v2`, now `main`), local checks only (2026-11-16)

karl04 on the Mac with the numpy test core and a 4 mm wall grid (`tests/test_steps.py`); the server runs of the new steps are still to do (REWRITE_PLAN.md, "Status").

| Check | Result |
|---|---|
| New `fit`, default physics (gravity supine, E 1.7 kPa, q0 fitted) | ASSD 9.00 mm, p0 1283 Pa, collapse 45 %, 11 solves, 0 failed: the same as the previous code on the server with torch (9.00 mm, 1282 Pa, 11 solves) |
| Fit of (q0, E) on a synthetic target (forward run at q0 0.35, E 1200 Pa; start 0.5, 2500) | q0 0.3500, E 1200 Pa, ASSD 0.0004 mm, 16 solves |
| Forward to a volume ratio 0.80 | q0 0.310 (p0 527 Pa) in 3 solves |
| Regions [1, 4] on top of gravity (diagnostic) | ASSD 9.00 → 7.63 mm with 4 regions, 21 solves, 0 failed |
| Regions [1, 4] alone, no gravity (the old regions fit) | ASSD 7.87 mm, 17 solves, 0 failed |
| Free ap slope, no gravity (the old gradient fit) | ASSD 8.57 mm, 19 solves, 0 failed (old code on the server, outer wall: 8.62) |
| si slope on top of gravity, ν fitted in [0.2, 0.45] | ASSD 8.53 mm, 18 solves, 0 failed; ν goes to its lower bound 0.2 (q0 0.27): ν trades with q0 for the volume, to look at on the server |
| Body force (supine → left lateral) | resultant = weight along the collapse position; Jacobian column vs FD 1e-7 |

**Server, torch core (2026-11-16, karl04, `runs/v2_check`):** the rewritten `fit` reproduces the previous code evaluation by evaluation (same mean / rms at every solve: 9.716 / 14.856, 8.620 / 13.011, ... 8.844 / 12.774): ASSD 9.001 mm, p0 1283 Pa, 11 solves, 0 failed, 39 s. `check_jacobian.py` with the smooth-field wall: core columns (4 bands + ν) vs finite differences max 2.6e-8, load-model column (q0) 2.2e-9. Wall field added to 8 more setups (16–28 s each, no target point beyond its margin). Still to run: forward, lateral position, regions, `geometry` / `fem_setup` from scratch (REWRITE_PLAN.md, "Status", steps 4–7).

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
