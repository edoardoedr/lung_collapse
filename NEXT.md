# Next steps

Plan agreed on 2026-10-15: from the patient fits to collapsing any inflated lung, and a GNN surrogate trained on synthetic collapses. Methods in [METHODOLOGY.md](METHODOLOGY.md), results and problems in [analysis/results.md](analysis/results.md), code history in [CLAUDE.md](CLAUDE.md). The old results log is in git: `git show 3260c03:NEXT.md`.

## Where we are (2026-11-16)

- **The code is now the physical model** (now `main`, formerly branch `refactoring_to_v2`, [REWRITE_PLAN.md](REWRITE_PLAN.md)): one config per patient, every physical parameter (E, ν, density, patient position, uniform pressure / collapse %) given or fitted; forward run (`collapse`) and inverse run (`fit`); wall as a smooth field inside Newton; body force for a collapse position different from the CT's. Tested on the Mac with a numpy core; **the new steps still have to run on the server** (first item of the to-do).
- **Smooth-field wall (server, 4 patients):** fits from rest at E 1.7 and 1 kPa with no failed solve, < 1 min, ASSD equal or better than with the outer rounds; E 1 kPa, not reachable before, is now better than 1.7 on 3 of 4 (analysis/results.md 7b).
- **Before the rewrite** (old code, on `v1_version`): regions fit without wall reliable on 15 of 16 patients (median ASSD 2.9 mm); gradient fits ~7–8 mm; common E 1.7 kPa within 0.5 mm of the free slope on 8 / 12 with the outer wall (analysis/results.md sections 2, 4, 5).
- **Package + three entry points (2026-11-17):** `pipeline/` is now the package `eurelaps_lung/` (`collapse/` → `fem/`); `main_fit.py` (one patient), `main_calibrate.py` (fits of the cohort → rule file → leave-one-out), `main_predict.py` (inflated lung + collapse % + parameters → collapsed lung). Tested on the Mac (tests, the rule of the 12 `fitE_nu30` fits reproduced); **to run on the server** (to-do).
- **Nothing yet towards the GNN** (steps 3–4 below), but the forward mode it needs now exists.

## The pipeline

1. **Regions fit (done).** For each patient with inflated + collapsed CT: the regional pressures that collapse the inflated lung onto the real collapsed one, ~3 mm.
   - Only p/E is fitted: Neo-Hookean, ν = 0.40 fixed.
   - The regions are clusters of the patient's own displacement, so the pressures do not transfer.
   - Role: they show the physical model can reproduce real collapses, they are the source of the common rules (step 2) and they are the final test set for everything else.
2. **Gradient fit: the common rules (becoming the physical load model, METHODOLOGY.md section 8).** Not a copy of the regions fit but a simplification of it: 2–3 numbers instead of 40. The cost is the shape error, ~7–8 mm (and more on an unseen patient).
3. **Collapse of an inflated-only lung (`main_predict.py`, 2026-11-17).** From the inflated CT alone (with airways and vessels for the hilum):
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
- [ ] 3D Slicer extension (prediction only): design in [SLICER_EXTENSION.md](SLICER_EXTENSION.md) (rewritten 2026-11-18). **Package side done (2026-11-18):** `pyproject.toml` (`eurelaps-lung` 0.1.0; default install = the prediction path without vtk / torch; extras `vtk`, `torch`, `fit`, `gpu`, `warp`, `standalone`, `server`), `eurelaps_lung/predict.py` (API `predict.run`, `python -m eurelaps_lung.predict`, `--collapse target`), packaged defaults `eurelaps_lung/data/predict_default.json`, `load_config` from a dict, lazy scikit-learn in `fem/regions.py`, `EURELAPS_PYTHON` for Netgen, `tests/test_predict.py`. Decided (user, 2026-11-18): torch from the Slicer PyTorch extension (no numpy core). Open: airways-only clamp vs required vessels (SLICER_EXTENSION.md 8.3).
  - [ ] **User: commit, then tag `v0.1.0` and push the tag** (SLICER_EXTENSION.md 4); decide whether the repository is public (pip from a private GitHub repository needs a token) and its license.
  - [ ] Extension Part 1 (interface, Claude Code in the extension repository, SLICER_EXTENSION.md 7 and 9), then Part 2 (installation, logic, outputs).
  - [ ] **Server: cohort fits and leave-one-out with the hilum on the inflated lung** (`--set hilum.lung=inflated`, `geometry`/`hilum`/`fem_setup` rebuilt in a separate output folder): the prediction computes the hilum there, the calibration on the collapsed lung. Locally (analysis/results.md 7e) it moves 0–3.6 mm on 14 / 16 patients, 39 mm on karl17 (wrong airways ring), 12 mm on karl20. If the rule changes, it becomes the built-in one.
  - [x] **Netgen volume mesh against gmsh on the cohort** (server, 2026-11-18, analysis/results.md 7d): same ASSD (≤ 0.08 mm except karl23 at the E bound), E +2.4 % median, same rule quality and leave-one-out (median 6.48 vs 6.45 mm). **Decided (user, 2026-11-18): Netgen is the only mesher, gmsh removed** (`fem_setup.mesh_size_mm` = Netgen maxh 10, `mesh_optsteps3d`, `mesh_timeout_s`); built-in E rule = `E_netgen.json` (3.5380, −0.012883).
  - [ ] **Server: make `FEM_v2_{HD,CZ}/<p>/fem/setup/` the Netgen setups** (copy from `FEM_v2_netgen_*`, or rerun `fem_setup`), so that the runs after 2026-11-18 (fits, predictions, GNN data) use Netgen meshes; the earlier v2 runs there are on gmsh meshes.
  - [ ] Test (future, user 2026-11-18; gmsh is GPL and no longer a dependency: Netgen's STL remesh is the other candidate): surface remesh with gmsh instead of pyacvd, ~480 nodes from the area (SLICER_EXTENSION.md 10.5): meshes, ASSD of the fits, E rule against the ACVD ones.
  - [ ] Test (future, user 2026-11-18): forward in float32 (MPS on the Mac, or CUDA / CPU) against float64 (SLICER_EXTENSION.md 10.1): needs a `dtype` option in `TorchSolver` and a looser `newton_tol`; compare displacement, solves, failures, time.
  - [ ] Test (future): Mac sparse LU options against SciPy SuperLU (SLICER_EXTENSION.md 10.2), only if the linear solve turns out to be the bottleneck.
- [ ] Run the three entry points on the server (2026-11-17): `main_fit.py --steps fit` on karl04 = the previous `fit` (ASSD 9.00 mm); `main_calibrate.py --configs configs/*.json --fit E --fit-run fitE_nu30 --skip-fit` = the previous `leave_one_out.py` table (the rule is now at ρ̄ 200: fits at 250 enter as E × 0.8, same slopes; `leave_one_out.py` used E at 250 with the config's density); `main_predict.py --collapse` on a config with `inputs.collapsed: null` in a fresh `output_dir`.
- [ ] Run the rewritten code on the server (REWRITE_PLAN.md, "Status"). Done 2026-11-16: wall field in the setups, `fit` on karl04 identical to the previous code, `check_jacobian.py` OK. Forward run with `volume_ratio` and the lateral position: done 2026-11-17 (results 5c). To do: the regions fit, `geometry` + `fem_setup` in a fresh `output_dir` against the existing files.
- [x] Common-E scan with the smooth-field wall (2026-11-17, analysis/results.md 5c): no failed solve; no common E (9 of 14 patients best at an end of 0.7–3 kPa, in opposite directions; the preferred E goes down with the collapse %); a common ~1–1.3 kPa costs ~0.8 mm.
- [x] E fitted per patient (`fitE`, 2026-11-17, analysis/results.md 5c): ASSD median 6.41 / mean 7.10 mm, no failed solve; E falls with the collapse (log₁₀ E = 3.69 − 0.0168 × collapse %, r −0.83 on 11 patients); karl10 / karl18 at the upper bound (no gradient wanted), karl23 at the lower bound with 11 points too little collapse. patient_2 missing from the output: check its wall field and log.
- [x] ν fitted with q0 and E on karl04, karl06, karl20 (`fitEnu`, 2026-11-17, analysis/results.md 5c): ν goes to the lower bound 0.25 for only 0.1–0.3 mm, and trades off against q0 (only p0 / K identified). ν is not identifiable: decided (user, 2026-11-17) ν = 0.30 fixed in all configs (q0 start 0.3); report p0 with it. Rerun fitE at ν 0.30 (`fitE_nu30`).
- [ ] **Prediction setting (user, 2026-11-17): inputs = inflated lung + collapse % + ρ̄ (CT); ν 0.30 fixed; q0 from the volume (`volume_ratio`), the gravity slope from a rule fitted on the cohort.** The measured ρ̄ does not improve the rule (analysis/results.md 5e): the slope is a function of the collapse %. Done (analysis/results.md 5f): from the collapse % alone the prediction costs +0.20 mm median / +0.26 mean / +0.90 max against the own fit; the CT density adds nothing (`sg_rho` +0.19 mean) or makes it worse (`E_rho` +0.43, patient_2 fails). Decided (user, 2026-11-17): rule `E` is the model's prediction rule, in the code as `physics.E_Pa {"rule": "collapse"}` (log₁₀ E = 3.5332 − 0.013455 × collapse % at ρ̄ 200, 3.6301 at 250; `eurelaps_lung/config.py`); `scripts/predict_cohort.py` runs it for every patient. The CT density is in every config (`physics.ct_density`, report only: `E_Pa_ct`, `p0_Pa_ct`). Left: karl10, karl18, karl23 (+1.8 to +5.6 mm).
- [x] Leave-one-out of the transferable model (now phase 3 of `main_calibrate.py`, formerly `scripts/leave_one_out.py`): for each patient, E from the E(collapse %) regression of the other patients, q0 from its collapse % (forward `volume_ratio`), one forward run; ASSD vs its own fitE. Done: analysis/results.md 5f.
- [ ] **The reference is the pleural cavity of the pneumothorax CT** (user, 2026-11-17: one CT per patient; METHODOLOGY.md section 7, item 12). The cavity is enlarged (chest wall out, mediastinum shifted, diaphragm down), so the collapse % is overestimated and the reference shape is wrong medially / caudally. Test whether this explains the E(collapse %) trend and the "missing load": cavity volume against the contralateral lung (normal right / left ≈ 1.1–1.2), mediastinal shift if measurable, then correlate with the fitted E. Done 2026-11-17 (analysis/results.md 5d): enlargement against the contralateral lung median 1.06, too noisy (0.60–2.37), and it does not explain the E trend (log E vs collapse % r −0.87, vs enlargement −0.51, partial coefficient of the wrong sign). Direct test left: patient_6.
- [x] CT density (2026-11-17, analysis/results.md 5e, `scripts/ct_density.py`): ρ̄ of the inflated lung = CT mass / cavity volume, median 163 kg/m³ (85–294), mass ±25 % (patient_6). Lower in the large collapses, so the E(collapse %) trend gets steeper, not flatter. Decided (user, 2026-11-17): the model's ρ̄ becomes **200 kg/m³**, the measured pleural gradient 0.2 cmH₂O/cm (Milic-Emili et al. 1966) / g; E default 1360 Pa (= 1700 × 0.8, same load) and the E rule's a = 3.5332 (= 3.6301 + log₁₀ 0.8), no refit needed. The per-patient CT ρ̄ stays report only. Runs before this date record ρ̄ 250 (their E × 0.8 = E at 200).
- [x] Final fit + prediction on all 16 (2026-11-17, analysis/results.md 5g): prediction = fit +0.09 mm median on the 13 with E in bounds; karl10 / karl18 / karl23 +2.6 to +4.7 mm. Viewer: `scripts/slicer_compare.py` (CT, cavity, real collapsed, fit, prediction).
- [ ] **Re-segment** (analysis/results.md 5h, labelmaps `results/<p>/qa_segmentation.nrrd`): karl10 collapsed lung (heart / mediastinum inside, user 2026-11-17); karl17, karl24 (and karl18) cavity base (diaphragm / abdomen inside). Then on the server: `check_inputs geometry hilum fem_setup`, `fit_final`, `pred`; locally `ct_density.py --config ... --write-config`; recalibrate the E rule if a patient enters the bounds. To decide (user): pleural effusion (patient_1, patient_2): leave it in the cavity, or remove it from the wall.
- [ ] Look in Slicer (with the CT): karl10 (dense anterior collapsed lung, 0.94 g/mL), karl17 / karl18 / karl24 (non-air between lung and cavity: effusion?), patient_2 (collapsed lung almost airless).
- [ ] **patient_6: the only real pair** (user, 2026-11-17): `CZ_dataset/vtk_surfaces_CZ/patient_6_1` = CT before the pneumothorax (`lung_left.vtk` + all lobes), `patient_6_3` = CT with the pneumothorax (`left_lung_collapsed.vtk`, right lobes, no cavity). Measures the error of the cavity as reference: true collapse %, right-lung change between the scans, and (once the left pleural cavity of 6_3 is segmented) cavity vs true lung volume and shape, and fits with either reference. Needs a rigid registration 6_1 → 6_3 (right lung / airways): two different scans, unlike the abandoned cavity-collapsed registration. karl21 has no cavity / collapsed lung: not usable.
- [x] Gravity direction: one CT per patient, so the inflated and collapsed surfaces share frame and position; gravity is the table vertical (+P for a supine scan) whatever the patient's anatomical tilt, so no registration to an "ideal supine". Optional check: fitE with gravity tilted ±10° (roll / pitch) on karl06, karl20, karl23.
- [ ] Look for the missing load on karl10, karl18, karl23 (collapse sequences in Slicer: where the fit misses, medial / diaphragmatic / dorsal).
- [x] Decided (user, 2026-11-17): `newton_maxit` 50, wall smoothing 3 mm kept, patient_2 included with the wall. Still open: `fit.best_by` default (`rms` now).

- [ ] karl19: fixed on the Mac (2026-11-17): `clean()` turned 4 degenerate triangles of `lung_right.vtk` into lines that `vtkTriangleFilter` passed through; `data_io.as_triangles` now keeps only the polygons. To do on the server: `geometry hilum fem_setup`, then the fit and the prediction.
- Patients (2026-11-17): 16 configs = every folder with a cavity and a collapsed lung. Not usable as they are: karl21 (no cavity / collapsed lung segmented, the CT exists), patient_6_1 (CT before the pneumothorax: no collapsed lung), patient_6_3 (collapsed lung, no cavity). Segmenting the cavity of 6_3 adds a patient and the only real pair.
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
- [x] Physical parametrisation (`eurelaps_lung/fem/load.py`): q = q0 + γ (h − h̄), E and p0 reported, E fittable; a common E by `scripts/fit_cohort.py`.
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
