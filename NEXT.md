# Next steps

Open tasks only. Methods in [METHODOLOGY.md](METHODOLOGY.md), results and the evidence behind past decisions in [analysis/results.md](analysis/results.md), code in [CLAUDE.md](CLAUDE.md). What was done is in the git history (`git log -- NEXT.md`; the long version before the 2026-11-19 cleanup: `git show 8cc1308:NEXT.md`).

**Goal.** A FEM model of the lung collapse in VATS, calibrated on the patients (inflated cavity + collapsed lung from one CT), used (1) to predict the collapse of a new lung from its inflated CT (3D Slicer extension) and (2) to generate the training data of a GNN surrogate.

## Server runs

- [ ] **Copy the CTs to the server** (2026-11-19): each `Input_CT/<p>.nii[.gz]` into the patient's `data_dir` (the configs' `inputs.ct`), and check that the contralateral lobe files (`inputs.contralateral`) are there; then `--steps ct` once per patient (writes `output_dir/ct/ct_values.json`, the configs no longer carry `ct_density` / `ct_emphysema`). Needs SimpleITK (or itk) in `lungfem`.
- [ ] **Compression barrier and emphysema** (METHODOLOGY.md 3.4.1, analysis/results.md 5j–5k; commands given 2026-11-19):
  - C1: the cohort `fit_final` with `physics.barrier={"phi": "ct"}` (`run_name=fit_barrier`): ASSD within 0.1 mm of `fit_final`; patient_2 is inside the barrier zone (φ 0.114, observed 0.138) and may change.
  - C2: the model's collapsed density ρ̄₀ / J by antero-posterior thirds against the CT (results 5e: posterior / anterior 1.1–2.4).
  - C3: surgery, `physics.collapse={"p0_Pa": 1600}` (12 mmHg) with the barrier, supine and lateral decubitus (operated side up), E 0.7 / 1.2 / 2 kPa, on karl06, karl04, karl08, patient_2: convergence, final volume against φ, shape against E. If the shape hardly depends on E, the pressure-driven mode uses one E (~1.2 kPa).
  - C4: karl08, karl17 with `physics.emphysema=false`: how much the trapped air changes the final volume.
- [ ] Make `FEM_v2_{HD,CZ}/<p>/fem/setup/` the Netgen setups (copy from `FEM_v2_netgen_*` or rerun `fem_setup`), after `scripts/init_output_dir.py --configs configs/*.json --apply` if not done.
- [ ] Recompute the rules file from the server summaries: `python main_calibrate.py --configs configs/*.json --fit E --fit-run fitE_netgen --skip-fit --no-loo --out calibration/E_netgen.json --make-default`; the E rule should stay 3.5380, −0.012883 and the q0 rule match the Mac one within rounding; commit `rules.json`.
- [ ] Cohort fits and leave-one-out with the hilum computed on the inflated lung (`--set hilum.lung=inflated`, separate output folder), as the prediction does: if the rule changes, it becomes the built-in one (locally the hilum moves 0–3.6 mm on 14 / 16, 39 mm on karl17, 12 mm on karl20: results 7e).
- [ ] Checks of the entry points not yet run on the server: `main_calibrate.py ... --skip-fit` against the earlier leave-one-out table; `main_predict.py --collapse` on a config with `inputs.collapsed: null` in a fresh `output_dir`; the regions fit with the new code; `geometry` + `fem_setup` in a fresh `output_dir` against the existing files.

## Data

- [ ] **Re-segment** (results 5h, labelmaps `results/<p>/qa_segmentation.nrrd`; `04_Data/Patients_status.xlsx`, column "Segmentation check"): karl10 collapsed lung (heart / mediastinum inside); karl17, karl24 (and karl18) cavity base (diaphragm / abdomen inside); karl23 contralateral (right) lobes (dark clusters outside the lung). Then on the server `check_inputs geometry hilum fem_setup`, `fit_final`, `pred`; locally `scripts/ct_density.py --config ... --write-config`; recalibrate the E rule if a patient enters the E bounds.
- [ ] Decide (user): the pleural effusion of patient_1 and patient_2: leave it in the cavity, or remove it from the wall.
- [ ] **patient_6, the only real inflated / collapsed pair:** segment the left pleural cavity of `patient_6_3`; rigid registration 6_1 → 6_3 (right lung / airways). It measures the error of using the cavity as the reference (true collapse %, cavity vs true lung volume and shape) and gives one more patient.
- [ ] karl21: segment the left cavity and the collapsed lung (the CT exists).
- [ ] Emphysema: the phase of the CTs (inspiration or not: LAA% depends on it).

## Model

- [ ] The missing load on karl10, karl18, karl23 (E at a bound, prediction +2.6 to +4.7 mm): first after their re-segmentation; then in Slicer where the fit misses (medial / diaphragmatic / dorsal) and which load would explain it (mediastinum, diaphragm). A physics change: to agree first.
- [ ] Barrier, later: φ per element from the inflated CT of a new patient (density and emphysema per tetrahedron); the trapped fraction r per patient, or higher inside bullae.
- [ ] Optional: trapped air by approximate lobe from the airways (centerline with lobe labels from a Slicer extension, `Input_Data/<p>/`): partition the collapsed lung by the nearest labelled centerline point, air fraction per lobe from the HU, compare with the fit error map. Analysis only.
- [ ] Check the rotation in the collapse sequences where the collapse is large (karl23 73°, patient_2 44°, patient_1 34°, karl04 32°; median 11°).
- [ ] Later: deformable wall (stiffness per part of the cavity: ribs high, diaphragm / mediastinum low); anatomical clamp (hilum + pulmonary ligament).
- [ ] Decide: `fit.best_by` default (`rms` now).

## 3D Slicer extension and package

- [ ] Add to the extension (repository `SlicerEurelapsLung`, prediction only) what was added on 2026-11-19: the compression barrier (`physics.barrier`), emphysema yes / no (a user input) with its volume from the CT, the CT density (`scripts/ct_density.py` logic on the inflated CT), the pressure-driven mode (`p0`, 12 mmHg) with a fixed E, the lateral decubitus position.
- [ ] Tag the package version the extension pins (after the barrier changes: a new version), decide whether the repository is public and its license.
- [ ] Open: hilum clamp from the airways only, or the vessels required too (the extension needs them segmented).
- [ ] Future tests: forward in float32 against float64 (needs a `dtype` option in `TorchSolver` and a looser `newton_tol`); a Mac sparse LU against SciPy SuperLU, only if the linear solve is the bottleneck; a surface remesh without pyacvd (Netgen's STL remesh or gmsh, GPL) if pyacvd breaks in Slicer (different surface nodes: the calibration must be redone).

## Towards the GNN

- [ ] Dataset generator: N loads per inflated lung (the physical load with random E / collapse % / position, plus smooth perturbations), saved as nodes / tets / loads / displacements for PyTorch / PyG.
- [ ] More inflated lungs (contralateral lungs mirrored, public CT datasets with airways and vessels) and, above all, more real inflated + collapsed pairs.
- [ ] GNN surrogate (geometry + load → displacement), tested on the real patients; physics loss from `torch_solver.TorchFEM`. Then, possibly, a network proposing the load from the geometry, trained end to end through it (few degrees of freedom with ~16 patients; validation by leave-one-out).
