# Next steps

Plan agreed on 2026-10-15: from the patient fits to collapsing any inflated lung, and a GNN surrogate trained on synthetic collapses. Validation details and history are in [CLAUDE.md](CLAUDE.md).

## Where we are

| | Pressure field | Needs the collapsed CT | Transferable to a new lung | Normal error (no wall) |
|---|---|---|---|---|
| Regions fit (`fit_torch`) | up to 40 regional p/E, per patient | yes | no | 2.1–2.8 mm |
| Gradient fit (`fit_gradient`) | mean p/E + slope back → front + slope base → apex | yes (to fit it) | yes (3 numbers) | 5.0–8.6 mm |
| Uniform p/E | one value | no (only the collapse %) | yes | 5.7–14.9 mm |

Findings from the 8 regions fits and the 6 valid gradient fits:

- Collapse (1 − V_collapsed / V_inflated) 24–58 %. The mean p/E follows it: ≈ 0.24 + 1.27 × collapse (gradient fits; residual ≤ 0.1).
- Anterior pushed more than posterior in every patient (absolute slope 0.17–0.43 p/E, mean 0.31): consistent with the supine position.
- Apex pushed less than base in 4 of 6, with a large spread.
- Relative slopes are strong in the low-collapse patients and weak in the high-collapse ones. The low-collapse patients are also the heavily clamped ones (20–37 % of the surface vs 5–10 %), so the cause may be the clamp.
- The gradient closes only 22–51 % of the gap between uniform and regions. The rest of the deformation is local and patient-specific.

## The pipeline

1. **Regions fit (done).** For each patient with inflated + collapsed CT: the regional pressures that collapse the inflated lung onto the real collapsed one, 2–3 mm.
   - Only p/E is fitted: Neo-Hookean, ν = 0.40 fixed.
   - The regions are clusters of the patient's own displacement, so the pressures do not transfer.
   - Role: they show the physical model can reproduce real collapses, they are the source of the common rules (step 2) and they are the final test set for everything else.
2. **Gradient fit: the common rules (done on 6 patients).** Not a copy of the regions fit but a simplification of it: 3 numbers instead of 40. The cost is the shape error, 5–8.6 mm (and more on an unseen patient).
3. **Collapse of an inflated-only lung (to implement).** From the inflated CT alone (with airways and vessels for the hilum):
   - choose the collapse %;
   - find the mean p/E that gives exactly that volume (monotonic, 4–6 solves), with the slopes from the other patients;
   - one forward solve → collapsed surface + Slicer sequence.

   The volume comes out exact; the shape is approximate (~5–9 mm).
4. **Synthetic dataset + GNN (to do).** Same mechanism as step 3, on many inflated lungs × many pressure fields. The GNN learns **geometry + pressure field → displacement**, i.e. an instant, differentiable FEM valid for any lung. It does **not** learn which pressure a real patient has: that still comes from the pressure rule (step 2), or from a fit (fast, through the GNN) when a collapsed shape is available.

## GNN

**What it learns.**
- Input per node: reference position, pressure on the surface nodes, clamped / free, possibly ν.
- Edges: the mesh.
- Output: displacement per node.

It replaces the FEM solve: milliseconds instead of seconds, no Newton failures, differentiable.

**Training data.**
- *Geometries:*
  - the patients' inflated lungs;
  - the contralateral lungs: `Input_Data/<p>/lung_*_lobe_*` are the other lung's lobes; their union is another inflated lung, mirrored;
  - public CT datasets (TotalSegmentator, LIDC-IDRI): airways / vessels are needed for the hilum, or another hilum estimate;
  - augmentation: scaling, smooth deformations, left/right mirroring.
- *Pressures*, broader than the fitted gradients so that real fits are in distribution:
  - gradients with random slopes (also medial / lateral);
  - quadratic terms;
  - smooth random fields on the surface;
  - random region combinations.
- *Other parameters:* ν, clamp size, collapse %.
- *Physics loss:* `torch_solver.TorchFEM` (differentiable assembly, any device) gives the equilibrium residual of a predicted displacement, usable as a physics-informed loss term.
- *Test:* the real patients. The GNN must reproduce their regions fits and, in prediction, their real collapsed lung.

**Can the GNN find the pressures by itself?**
- In principle yes: a second network proposes the pressure field from the inflated geometry.
- Training: real patients → proposed pressure → frozen GNN-FEM → shape, with the loss on the real collapsed surface. The loss is on the shape, not on the poorly identified fitted pressures. This needs the GNN-FEM, being differentiable and fast.
- Limits:
  - Synthetic data cannot teach this (gradient-generated data only teaches the gradient). It must come from real inflated + collapsed pairs.
  - With 8–9 patients it must have few degrees of freedom (gradient + quadratic + a few modes, driven by features such as distance from the hilum, curvature, lobe), or it memorises the patients. A free per-triangle field needs tens of patients.
  - The pressure probably depends on the surroundings more than on the lung's shape: heart, diaphragm, mediastinum, gravity. These can be segmented with TotalSegmentator and given as input.
- Validation: leave-one-out on the real patients.

**What it adds over the classical fits.** It will not be more accurate than the FEM it learns from. It adds:

| | Classical FEM fit | GNN |
|---|---|---|
| Collapsed CT needed | yes, after the fact | no (with a pressure rule) |
| Time | minutes per fit, seconds per solve | milliseconds |
| Reliability | solves can fail (20–40 % with the wall) | always answers |
| Differentiable | only via dedicated Jacobians | natively |

Concrete uses:
1. **Intra-operative update:** fit the pressures in real time to partial data (thoracoscope surface, a few landmarks) and show hidden vessels / lobes. Not possible with minute-long fits; likely the strongest clinical contribution.
2. **Interactive planning:** change the collapse % or the pressures and see the shape and the vessels at once.
3. **Many more fits:** seconds per inverse fit → more real patients fitted (also public data with both scans) → a better pressure rule.
4. **Robustness:** no failing solves.

Accuracy on real patients stays limited by the pressure rule. Reaching the 2–3 mm of the regions fits needs more real patients.

## To do

- [ ] Rerun karl04 and patient_10 on the server: their `fem_setup` there is still the old aligned one.
  ```bash
  bash scripts/run_all.sh karl04 patient_10 -- --steps fem_setup fem_fit
  bash scripts/run_all.sh karl04 patient_10 -- --steps fem_fit --set fem_fit.pressure_model=gradient fem_fit.run_name=fit_gradient fem_fit.output=lung_fem_fit_gradient.vtp
  ```
- [ ] Optional: gradient with the third axis (`gradient_axes: ["ap", "si", "ml"]`) and/or quadratic terms, to see how much of the gap to the regions they close.
- [x] `scripts/summarize_fits.py` (2026-10-27): every fit under the given folders → `runs.csv` (collapse %, volumes, shape errors to the collapsed surface measured the same way for reg / noreg, rotation, mean p/E, linear ap / si / ml slopes and R² also for regions fits, third ratios, wall, failures), `compare.csv` (paired reg ↔ noreg, wall ↔ no wall, regions ↔ gradient), `report.md`, `pressure.png`, `assd.png`. numpy / matplotlib only, runs on the Mac.
- [x] First analysis of the reg + noreg runs (2026-10-27, 15 patients, regions fits, `analysis/`; karl19 has only noreg):
  - Shape: reg and noreg equal (median ASSD 2.20 / 2.24 mm no wall; target → fitted 1.82 / 1.54, noreg covers the target slightly better); noreg rotates less (−2.9° median) and is 1–4 min faster. The wall costs nothing in shape for reg (2.25) but brings the failing / early-stopped fits (karl04, 10, 17, patient_1, 2, 3 wall: 2–27 failed solves, up to 57 min, ASSD up to 14.6 mm); in noreg the wall is never touched (0 points beyond), so noreg wall = noreg no wall.
  - Registration quality shows in the reference volume: karl23 registered 1984 mL vs 2554 mL real inflated (Dice 0.85), so its reg fit collapses a wrong reference (collapse 37 % instead of 51 %); karl20 −7 % (Dice 0.93). patient_2 reg 9.3 mm ASSD vs noreg 3.6.
  - **Noreg clamps far more surface** (19–58 % vs 3–37 %): `clamp_still_mm` with the closest-point distance counts every point near the collapsed surface as still, even if it slid along it. Hence higher mean p/E (+0.42 median: less free surface for the same collapse), larger slopes and a worse p/E-vs-collapse rule (max residual 0.50 vs 0.10). To decide: `clamp_still_mm` 0 (or smaller) for noreg, and a reg vs noreg comparison at the same clamp.
  - Pressure rule (reg, wall, 15 patients): mean p/E ≈ 0.04 + 1.61 × collapse, max residual 0.10. Anterior > posterior in 15 / 15 (reg), 12 / 15 (noreg); apex < base 13 / 15. A linear gradient explains only 4–47 % of the regional p/E variance; many fits have regions at the p/E bound 3. With 15 patients the slope ↔ clamp correlation drops (rel ap ~ clamp 0.38, was 0.82 on 8).
  - Gradient fits (8, reg): 4.1 mm worse ASSD than regions in every patient.
- [ ] Decided (user, 2026-10-27): no registration from now on. Run the new configs (`configs/<p>[_wall]_torch.json`: no registration, clamp = hilum region only, `clamp_still_mm` 0) on all 16 patients, then `summarize_fits.py` (clamp comparison "still 5 → hilum only" on the noreg runs): shape error, rotation, wall activity and the p/E-vs-collapse rule without the still-point clamp.
  - First results (2026-10-28, 11 patients; karl19, patient_1 / 2 / 3 / 10 missing). Medians without karl23, still 5 → hilum only:
    - clamped surface 43 % → 5 %;
    - no wall: ASSD 2.20 → 2.69 mm, rotation 4.1 → 9.1°;
    - **pressure rule much cleaner:** mean p/E ≈ −0.09 + 2.11 × collapse, r 0.97, max residual 0.17 (still 5: r 0.74, max residual 0.33); anterior > posterior in 10 / 10.
    - **karl23 hilum only broke down:** fitted volume 177 mL vs 1245 collapsed, ASSD 24 mm, rotation 47°. The closest-point loss only measures fitted → target, so the lung shrank into one part of the target (target → fitted 40 mm). The fix: a symmetric loss (also target → fitted) or a volume term.
    - **The wall now acts** (3–73 points beyond it) and its failures come back: 8 of 11 fits stopped at K = 4–14 with 11–15 failed solves, 21–31 min, ASSD 4.5 mm. Only karl06, 08 and 18 are fine.
  - Update (2026-10-28, 15 patients, karl19 missing):
    - **Shrinkage also in patient_1 and patient_2:** fitted collapse 95 / 98 % vs real 59 / 86 %, ASSD 25 / 27 mm. patient_2 stopped at K = 1, already at the p/E bound 3.
    - **Medians over the 12 good fits, no wall:** ASSD 2.89 mm (still 5: 2.20), rotation 10.3° (4.5°). Mean p/E ≈ −0.10 + 2.09 × collapse, r 0.97, max residual 0.19 (still 5: r 0.70, max residual 0.33). Anterior > posterior in 12 / 12, apex < base in 10 / 12.
    - **Wall:** 12 failed solves per fit (median), 24 min, ASSD 4.62 mm. patient_3 wall: 32 failed solves, 65 min.
- [x] Symmetric loss for the no-registration fits (2026-10-28): `fem_fit.loss_symmetric_weight` (1 in all current configs). It adds target → fitted to the closest-point residual, against the shrinkage of karl23 / patient_1 / patient_2. Checked on the Mac (cost gradient vs FD 2e-10). Fake linear core on karl01: volume 1346 → 1467 mL (target 1553), target → fitted 2.78 → 2.07 mm.
- [ ] Rerun on the server with the symmetric loss (plan, user 2026-10-28). Old results moved to `archive/`, kept for the reg vs noreg comparison.
  1. [x] Regions, hilum-only clamp, no wall (`fem/fit_sym`), all 15 patients, results 2026-10-28:
     - **The over-collapse is gone.** ASSD went from 24.5 / 25.1 / 27.1 to 4.0 / 2.9 / 5.0 mm on karl23 / patient_1 / patient_2, and Hausdorff from ~100 to 24–30 mm. The fitted collapse is now 51 / 57 / 84 % (real 51 / 59 / 86 %; one-sided: 93 / 95 / 98 %).
     - **On the 12 good patients it is about the same.** ASSD stays within ±0.4 mm. Target → fitted gets better (e.g. karl02 4.2 → 3.5 mm), fitted → target a bit worse (+0.1–1.1 mm). The fitted collapse matches the real one better: median gap 2.1 → 0.6 points; the one-sided fits collapsed 1–3 points too much.
     - **The fits still converge.** No solve failed, 0.2–1.0 min each, K = 40 in 12 of 15. The fits stopped on "target error reached" in 5 and "levels exhausted" in the rest.
     - **The pressure–collapse rule now holds for all 15:** mean p/E = −0.18 + 2.19 × collapse, r 0.98, max residual 0.17 (12 good: −0.08 + 1.89c, r 0.99, max residual 0.08). With the one-sided loss, all 15 gave r 0.87 and max residual 0.97.
     - **Anterior > posterior in 15 / 15**, but patient_1 only barely (1.03 vs 1.03; its linear ap slope is −0.13). In the per-patient linear fit, the relative si slope is < 0 in 13 / 15.
     - **Open issues:**
       - Rotation is still large where the collapse is large: karl23 73°, patient_2 44°, patient_1 34°, karl04 32° (median 11°). Check it in the collapse sequences.
       - Some regions sit at the p/E bounds: −1 on karl10 / karl20, 3 on 6 patients.
  2. Then gradient ap and ap + si: is E = ρgL / slope more consistent? Also, from the regions fits, the zones for the gradient.
- [ ] karl19: the pipeline stops in `registration` (method none) at `surface_mesh.clean_surface` of the inflated `lung_right.vtk`, "triangulation left non-triangle cells". The old `karl19_noreg` failed the same way. Check the input file (cell types) on the server.
- [ ] Wall failures with the hilum-only clamp (discussed 2026-10-28). Cause: near-mechanism. A small clamp and frictionless contact let the lung rotate / slide along the wall almost for free, and the `outer` rounds diverge. Ideas, in order:
  1. Measure how far the no-wall hilum-only fits leave the cavity (Mac, from the existing results). If only a few mm, the wall may not be needed.
  2. Weak **tangential elastic support**: springs k_t (I − n nᵀ) u on the free surface nodes, n = reference normal. Possibly only where in contact. A path-independent "elastic friction": smooth, linear, analytic Jacobian unchanged, a few lines in `nodal.py`. Option off by default; tune k_t on patient_3_wall / karl04_wall with `wall_diagnose.py`. Cost: the springs take part of the load (p/E shifts), so report their energy share.
     - [x] Implemented (2026-10-28, user's go): `solver_options.support_k` (warp / torch, not GetFEM), on every surface node, off by default. The summary has `support_k` and `support_share` (sum of support force magnitudes / sum of pressure force magnitudes). Checked on the Mac (numpy small-strain core on karl01_noreg_hilum, p/E 0.02–0.04): Jacobian vs FD 5–8e-8; mean tangential displacement 0.51 → 0.20 / 0.12 / 0.03 mm at k 0.003 / 0.01 / 0.1, normal 0.43 → 0.35 / 0.34 / 0.32 mm, share 2 / 4 / 10 %.
     - [x] `wall_diagnose.py`, p/E 0.5, K = 4, 10 increments (2026-10-28):
       - k = 0 fails on all three cases (karl04, karl23, patient_3): `unsettled`, then `time`.
       - Every k ≥ 0.001 reaches 100 % with no failure, 0.3–2.3 s per increment, 2–4 wall rounds, penetration ≤ 0.41 mm. A larger k means fewer rounds and Newton iterations; patient_3 at k 0.01 needs 4 iterations per increment.
       - The support also takes load: max|U| at 100 % goes 40.6 / 29.4 / 22.0 mm on karl04 (k 0.001 / 0.003 / 0.01), 29.6 / 19.8 / 16.9 on karl23, 22.6 / 20.1 / 18.6 on patient_3. The fit will raise p/E to compensate; `support_share` tells by how much.
     - [x] First full fits with k 0.001 / 0.003, symmetric loss, 14 patients, with and without wall (2026-10-29). All 28 cases of each k ran in parallel, which made every solve ~15× slower: Jacobian 0.85–1.85 s vs 0.1, same Newton iterations. Times are not meaningful, and with a wall most failures are `fail_time` (120 s limit). `run_all.sh` now takes `JOBS=n`.
       - **No wall, k = 0 / 0.001 / 0.003 (medians):**
         - ASSD 2.92 / 2.53 / 2.52 mm; rotation 11.0 / 8.4 / 6.1°.
         - Rotation per patient: karl23 73 / 42 / 11°, karl04 32 / 17 / 9°, patient_1 34 / 20 / 18°, patient_10 20 / 14 / 9°.
         - Support share 4.9 / 11.9 %; mean p/E up ~5 / ~10 %; collapse still matched (median gap ≤ 1 point).
         - Failed solves 0 / 20 / 61. At k 0.001 only patient_1 (20, ASSD 2.9 → 4.6). At k 0.003: karl23 32 (maxit, ASSD 6.2), karl01 16, patient_1 10, karl20 3.
         - These failures are Newton maxit, not time: real, cause not yet known. Possibly the fixed reference normal of the springs at large local rotations.
       - **Wall:** still 10–22 failed solves on 10 / 14 and early stops (K 4–14), but confounded by the slowdown. The diagnosis run alone took 0.3–2.3 s per increment. To redo with `JOBS=4`.
     - [ ] Wall fits with k 0.001 (and 0.003) rerun with `JOBS=4` with the chosen k (with and without wall), comparing failed solves, time, rotation, ASSD, share.
  3. Anatomical clamp: hilum + pulmonary ligament (medial band from the hilum to the base); needs an automatic per-patient definition.
  - Coulomb friction rejected: the pleura is lubricated (very low physical μ); the solution would depend on the load path, which makes the fit objective noisy and the Jacobian ill-defined; the tangent would be non-smooth and non-symmetric.
- [ ] Physical meaning of the fitted pressures (discussed 2026-10-28): p/E is an equivalent surface load lumping elastic recoil, weight and the support of the neighbours.
  - The mean level is mostly the material's bulk response. Small-strain hydrostatic: p/E = c / (3(1 − 2ν)) = 1.67 c at ν = 0.4; fitted ≈ 2.09 c. So it scales with the chosen ν.
  - The anterior > posterior gradient is probably gravity in the supine scan, as the removal of the pleural pressure gradient. The inflated lung is in equilibrium: its weight is carried by the vertical gradient of pleural pressure. The pneumothorax makes the pleural pressure uniform, so with a stress-free inflated reference the collapse load is uniform + linear in height. The slope is ρ g per unit height, and the weight then goes to the dependent wall.
    - A body force on top of the stress-free reference would count gravity twice.
    - Hence: fitted ap slope (across the AP depth L) = ρ g L / E, so E ≈ ρ g L / slope.
    - With ρ = 250 kg/m³ (assumed, inflated lung) and the hilum-only noreg regions fits (linear slope of the regional p/E, 12 good patients): median E 2.3 kPa, range 0.4 kPa (karl20, slope 1.32) to 8.4 kPa (karl24, 0.05).
  - Possible tests:
    - impose the slope from ρ and a common E (one parameter fewer);
    - or fit E per patient through the slope, and check that it is plausible and consistent across patients;
    - regions as δp around that fixed gradient. The Ward clusters do not change (computed from the target in `fem_setup`), only their values.
    - The linear gradient explains only 4–47 % of the regional p/E variance, so the δp stay patchy.
  - Check the patient position of the collapsed CT (supine vs lateral decubitus): it sets the direction of the gradient. The user says the patients should be supine (2026-10-28).
  - **Gradient fits, hilum-only noreg, no wall (2026-10-28, 11 karl patients):** `fit_grad_ap` (uniform + ap) and `fit_grad_apsi` (+ si).
    - **Shape:** median ASSD 7.9 / 7.1 mm without karl23, against 2.7 mm for regions. The slope reduces the fit error by 0–25 % of the uniform level (karl18 ap: no gain, uniform kept).
    - **Signs:** ap > 0 in 9 / 10 (ap only); with si added, karl10 flips (ap −0.05, si +0.39). si < 0 in 8 / 10.
    - **E = ρgL / slope** (ap only, ρ = 250 kg/m³): median 4.4 kPa, IQR 3.1–6.6, range 0.5 (karl20) to 9.1 kPa (karl24). Not one common E.
    - **Over-collapse:** fitted collapse > real in most patients (karl01 33 vs 25 %, karl10 74 vs 56 %, karl17 36 vs 24 %, karl18 38 vs 31 %, karl20 77 vs 58 %). The one-sided closest-point loss biases p/E upward, and hence the slopes and E. karl23 is broken here too (95 %, ASSD 28 mm, slope at the bound −4).
    - **Conclusion:** the gravity reading is plausible (sign, order of magnitude of E) but not proven: E spread ×18, shape ~8 mm. Redo with the symmetric loss before drawing conclusions.
- [ ] Target-free setup: surface mesh, hilum + clamp and volume mesh from the inflated CT only (no registration, alignment or still points).
- [ ] Forward-only mode: fixed pressure model (gradient parameters, exact collapse % by a search on the mean p/E, arbitrary fields) → one solve → surface + sequence.
- [ ] Leave-one-out prediction on the real patients: mean p/E from the collapse %, slopes from the others, compared with the real collapsed lung and with the patient's own fits.
- [ ] Dataset generator: N pressure fields per inflated lung, saved as nodes / tets / pressures / displacements in a format PyTorch / PyG reads.
- [ ] Collect more inflated lungs (contralateral, public datasets) and, above all, more real inflated + collapsed pairs.
- [ ] GNN surrogate (geometry + pressure → displacement), tested on the real fits; then the pressure network, trained end-to-end through it.

Still open from the FEM side (see CLAUDE.md):
- karl02_wall / karl10_wall / karl17_wall stopped early with failing solves (~30 min each);
- karl04_wall speed (`max_solve_s` 40 or no ramp from zero with a wall);
- patient_2 rerun with the current configs.
