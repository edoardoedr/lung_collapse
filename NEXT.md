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
- [ ] `scripts/summarize_fits.py`: the per-patient table (collapse %, mean p/E, slopes, errors) regenerated from `results/`.
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
