# Next steps

Plan agreed on 2026-10-15: from the patient fits to collapsing any inflated lung, and a GNN surrogate trained on synthetic collapses. Methods in [METHODOLOGY.md](METHODOLOGY.md), results and problems in [analysis/results.md](analysis/results.md), code history in [CLAUDE.md](CLAUDE.md). The old results log is in git: `git show 3260c03:NEXT.md`.

## Where we are (2026-11-16)

- **The code is now the physical model** (branch `refactoring_to_v2`, [REWRITE_PLAN.md](REWRITE_PLAN.md)): one config per patient, every physical parameter (E, ν, density, patient position, uniform pressure / collapse %) given or fitted; forward run (`collapse`) and inverse run (`fit`); wall as a smooth field inside Newton; body force for a collapse position different from the CT's. Tested on the Mac with a numpy core; **the new steps still have to run on the server** (first item of the to-do).
- **Smooth-field wall (server, 4 patients):** fits from rest at E 1.7 and 1 kPa with no failed solve, < 1 min, ASSD equal or better than with the outer rounds; E 1 kPa, not reachable before, is now better than 1.7 on 3 of 4 (analysis/results.md 7b).
- **Before the rewrite** (old code, on `main`): regions fit without wall reliable on 15 of 16 patients (median ASSD 2.9 mm); gradient fits ~7–8 mm; common E 1.7 kPa within 0.5 mm of the free slope on 8 / 12 with the outer wall (analysis/results.md sections 2, 4, 5).
- **Nothing yet towards the GNN** (steps 3–4 below), but the forward mode it needs now exists.

## The pipeline

1. **Regions fit (done).** For each patient with inflated + collapsed CT: the regional pressures that collapse the inflated lung onto the real collapsed one, ~3 mm.
   - Only p/E is fitted: Neo-Hookean, ν = 0.40 fixed.
   - The regions are clusters of the patient's own displacement, so the pressures do not transfer.
   - Role: they show the physical model can reproduce real collapses, they are the source of the common rules (step 2) and they are the final test set for everything else.
2. **Gradient fit: the common rules (becoming the physical load model, METHODOLOGY.md section 8).** Not a copy of the regions fit but a simplification of it: 2–3 numbers instead of 40. The cost is the shape error, ~7–8 mm (and more on an unseen patient).
3. **Collapse of an inflated-only lung (to implement).** From the inflated CT alone (with airways and vessels for the hilum):
   - choose the collapse %;
   - find the mean p/E that gives exactly that volume (monotonic, 4–6 solves), with the slopes from the other patients;
   - one forward solve → collapsed surface + Slicer sequence.

   The volume comes out exact; the shape is approximate (~7–9 mm).
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
  - With ~15 patients it must have few degrees of freedom (gradient + quadratic + a few modes, driven by features such as distance from the hilum, curvature, lobe), or it memorises the patients. A free per-triangle field needs tens of patients.
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

**Now**

- [ ] New output folders for the current code (2026-11-17): configs point to `FEM_v2_{HD,CZ}/<p>`; fill them on the server with `scripts/init_output_dir.py --configs configs/*.json --apply` (setup + the gE scan and fitE runs; the old `FEM_fit_*` folders stay untouched, archive them by hand). Every new experiment goes there.
- [ ] Run the rewritten code on the server (REWRITE_PLAN.md, "Status"). Done 2026-11-16: wall field in the setups, `fit` on karl04 identical to the previous code, `check_jacobian.py` OK. Forward run with `volume_ratio` and the lateral position: done 2026-11-17 (results 5c). To do: the regions fit, `geometry` + `fem_setup` in a fresh `output_dir` against the existing files.
- [x] Common-E scan with the smooth-field wall (2026-11-17, analysis/results.md 5c): no failed solve; no common E (9 of 14 patients best at an end of 0.7–3 kPa, in opposite directions; the preferred E goes down with the collapse %); a common ~1–1.3 kPa costs ~0.8 mm.
- [x] E fitted per patient (`fitE`, 2026-11-17, analysis/results.md 5c): ASSD median 6.41 / mean 7.10 mm, no failed solve; E falls with the collapse (log₁₀ E = 3.69 − 0.0168 × collapse %, r −0.83 on 11 patients); karl10 / karl18 at the upper bound (no gradient wanted), karl23 at the lower bound with 11 points too little collapse. patient_2 missing from the output: check its wall field and log.
- [x] ν fitted with q0 and E on karl04, karl06, karl20 (`fitEnu`, 2026-11-17, analysis/results.md 5c): ν goes to the lower bound 0.25 for only 0.1–0.3 mm, and trades off against q0 (only p0 / K identified). ν is not identifiable: decided (user, 2026-11-17) ν = 0.30 fixed in all configs (q0 start 0.3); report p0 with it. Rerun fitE at ν 0.30 (`fitE_nu30`).
- [ ] Leave-one-out of the transferable model (script ready: `scripts/leave_one_out.py`): for each patient, E from the E(collapse %) regression of the other patients, q0 from its collapse % (forward `volume_ratio`), one forward run; ASSD vs its own fitE. Needs a small script (`scripts/leave_one_out.py`).
- [ ] **The reference is the pleural cavity of the pneumothorax CT** (user, 2026-11-17: one CT per patient; METHODOLOGY.md section 7, item 12). The cavity is enlarged (chest wall out, mediastinum shifted, diaphragm down), so the collapse % is overestimated and the reference shape is wrong medially / caudally. Test whether this explains the E(collapse %) trend and the "missing load": cavity volume against the contralateral lung (normal right / left ≈ 1.1–1.2), mediastinal shift if measurable, then correlate with the fitted E. Needs the contralateral lung surfaces (check what is in the data folders).
- [x] Gravity direction: one CT per patient, so the inflated and collapsed surfaces share frame and position; gravity is the table vertical (+P for a supine scan) whatever the patient's anatomical tilt, so no registration to an "ideal supine". Optional check: fitE with gravity tilted ±10° (roll / pitch) on karl06, karl20, karl23.
- [ ] Look for the missing load on karl10, karl18, karl23 (collapse sequences in Slicer: where the fit misses, medial / diaphragmatic / dorsal).
- [x] Decided (user, 2026-11-17): `newton_maxit` 50, wall smoothing 3 mm kept, patient_2 included with the wall. Still open: `fit.best_by` default (`rms` now).

- [ ] karl19: the pipeline stops in `registration` (method none) at `surface_mesh.clean_surface` of the inflated `lung_right.vtk`, "triangulation left non-triangle cells". Check the input file (cell types) on the server.
- [ ] Check the rotation in the collapse sequences where the collapse is large: karl23 73°, patient_2 44°, patient_1 34°, karl04 32° (median 11°).

**Physical load model (agreed 2026-11-12, METHODOLOGY.md section 8)**

Uniform pleural pressure + loss of the pleural gradient that carried the lung's weight (Archimedes), wall in the model, later heterogeneous tissue. Decided (user, 2026-11-13): the physical model becomes the default; the regions model stays in the code as an option, for the reference fits (best shape this FEM can give, ~2.9 mm), for diagnosis (physical load + regional δp: where δp stays large, physics is missing) and for comparing with the existing fits, with few levels (e.g. K = 1, 4, 8). The GNN's synthetic pressures come from the physical model plus smooth perturbations, not from Ward regions. Tests, cheapest first:

- [x] Gradient fits `ap` and `ap + si` with the symmetric loss, no wall (`fit_grad_ap_sym`, `fit_grad_apsi_sym`, 2026-11-16; analysis/results.md sections 4–5): broken cases repaired, collapse matched, ap > 0 in 13 / 15, si < 0 in 12 / 15 (not gravity), E not consistent across patients (lowest where the collapse is largest: the slope probably replaces the missing wall).
- [x] Gradient `ap` with the wall (k = 0) (2026-11-16, analysis/results.md 4b): shape 8.2 → 7.1 mm, failures halved (14 %) but not gone (karl20 / karl23 ~1/3), slope ×2–3 larger than without wall, E 0.5–2.9 kPa (×6 instead of ×17).
- [x] Common E with the wall (scan 1 / 1.7 / 3 kPa, 2026-11-16, analysis/results.md 5b): E 1.7 kPa within 0.5 mm of the free slope on 8 / 12, outliers karl10 / karl18 (no gradient) and karl20 / karl23 (E ~0.5); almost no failed solves; E 1 kPa mostly not reachable (wall). To redo: E 1 kPa for karl06, karl17, karl24 (interrupted, invalid).
- [ ] Leave-one-out check of the transferable model: q0 from the collapse % (rule from the other patients) + slope from E 1.7 kPa, one forward solve per patient, vs the fitted ones.
- [x] Wall as a smooth potential (`solver.wall_update: "potential"`; METHODOLOGY.md section 3): done 2026-11-16, checked on the server on 4 patients (analysis/results.md 7b). Open: smoothing 2 / 3 / 5 mm.
- [ ] Later: deformable wall (section 8 (b)): stiffness per part of the cavity (ribs high, diaphragm / mediastinum low, from position and normal on the inflated surface).
- [x] `support_k` removed with the rewrite (gravity + wall instead).
- [x] Physical parametrisation (`pipeline/collapse/load.py`): q = q0 + γ (h − h̄), E and p0 reported, E fittable; a common E by `scripts/fit_cohort.py`.
- No inflated CT images (user, 2026-11-13): only the segmented surfaces, so no HU, no per-element tissue fraction and no measured density. ρ̄ is assumed (~250 kg/m³).
- [ ] Barrier material (J ≥ φ_t) in `materials.py` (torch only) with a **uniform** φ_t, or φ_t(h) linear in height with an assumed gradient. Bound from the data: φ_t ≤ V_collapsed / V_inflated of every patient (patient_2: 0.14). Physics change: test on a few patients against Neo-Hookean.
- [ ] Check the patient position of the collapsed CT (supine vs lateral decubitus): it sets e_g. The data say supine (anterior > posterior in 15 / 15).
- [ ] Predict the collapse in lateral decubitus (the VATS position): implemented (`physics.gravity: {"reference": "supine", "collapse": "left_lateral"}`, body force), tested only on the Mac; run it on a few patients and look at the sequences.

**Wall (decide)**

- [ ] Rerun the wall fits with the 2026-10-30 fix, `JOBS=1` or small, k 0.001 and 0.003 (the k 0.003 wall fits have only the parallel run: failures and times not reliable), comparing failed solves, K reached, ASSD, rotation, share. `max_solve_s` 20 gives the same result as 120 on karl04 (~44 → ~10 min per fit). Decide the items below after this.
- [ ] Is the wall needed? Without it the fits leave the cavity: 218 points > 5 mm over 14 patients at k = 0 (max 12–14 mm on karl23).
- [ ] Choose `support_k` and set it in the configs, or leave it off. k 0.001 / 0.003 lower the rotation and the points outside, but take 5 / 12 % of the load.
- [ ] Find the cause of the Newton `maxit` failures that `support_k` brings without wall (0 / 20 / 61 failed solves at k 0 / 0.001 / 0.003). Possibly the fixed reference normal at large local rotations.
- [ ] Parked (user, 2026-11-02, keep in mind, not now): wall smoothing in `fem_setup` (preferred), two-plane contact. Details in analysis/results.md, section 7.
- [ ] Anatomical clamp: hilum + pulmonary ligament (medial band from the hilum to the base); needs an automatic per-patient definition.

**Towards the GNN**

- [x] Target-free setup: `geometry`, `hilum` (`lung: "inflated"`) and `fem_setup` work from the inflated lung only. To check on the patients: the hilum from the inflated lung against the one from the collapsed lung.
- [x] Forward-only mode (`collapse` step): physics given, or the volume ratio to reach (search on q0) → surface + sequence; works without the collapsed lung (`inputs.collapsed: null`).
- [ ] Leave-one-out prediction on the real patients: mean p/E from the collapse %, slopes from the others, compared with the real collapsed lung and with the patient's own fits.
- [ ] Dataset generator: N pressure fields per inflated lung, saved as nodes / tets / pressures / displacements in a format PyTorch / PyG reads.
- [ ] Collect more inflated lungs (contralateral, public datasets) and, above all, more real inflated + collapsed pairs.
- [ ] GNN surrogate (geometry + pressure → displacement), tested on the real fits; then the pressure network, trained end-to-end through it.

**Optional**

- [ ] Gradient with the third axis (`gradient_axes: ["ap", "si", "ml"]`) and/or quadratic terms, to see how much of the gap to the regions they close.
- [ ] Physics tests of the gravity reading: impose the slope from ρ and a common E (one parameter fewer), or fit E per patient through the slope; regions as δp around that fixed gradient.
