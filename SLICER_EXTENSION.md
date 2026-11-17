# 3D Slicer extension: design draft

Draft of 2026-11-17, nothing implemented yet. It covers a 3D Slicer extension that runs the prediction of `eurelaps_lung` (`main_predict.py`) on a patient loaded in Slicer and shows the collapse as a sequence. Fitting and calibration stay outside Slicer: the extension ships a calibration file and uses it.

## 1. What the user does

1. **Load the data:** the CT (any format Slicer reads) and a segmentation with:
   - **lung:** one segment, or the lobes (2 on the left, 3 on the right). Lobes are merged into one segment by the extension.
   - **airways:** the bronchial tree, at least down to the lobar bronchi.
   - **pulmonary arteries / veins:** optional, see 6.2 (the current model's clamp uses them).
2. **Choose the side** (left / right): guessed from the segment's centroid, editable.
3. **Set the collapse** (section 2) and, if wanted, the advanced parameters (section 3).
4. **Click Apply.**
   - The first time, the model geometry is built (surface remesh → hilum → clamp → volume mesh → wall field), tens of seconds.
   - Then the forward run.
   - A new collapse % on the same segmentation reuses the geometry and only re-solves.
5. **Get the output:**
   - a sequence of the collapse, 0 → 100 % of the load, in the 3D view over the CT, coloured by displacement;
   - the final collapsed lung as a model node;
   - a results table: volume inflated / collapsed, collapse % reached, E, p0, gravity slope, wall penetration, solves.

## 2. Which collapse input

The main control is the **collapse %**: 100 · (1 − V_collapsed / V_inflated).

- It is the only input the calibrated model is built on. The E rule is log₁₀ E = a + b · collapse % (`calibration.py`), and q0 is then searched so that the volume matches (`forward.find_q0`, 3–6 solves, monotonic). Without the %, E has no value.
- It is what the surgeon can estimate and compare across patients. Pressure is not measured intra-operatively, and the model only identifies p0 / E, not the two separately.
- **The same control can also be shown as a volume.**
  - The inflated volume is known after the first step, so a spin box in mL can be linked to the % spin box: editing either updates the other.
  - The model works with the ratio; the volume is only a different unit.
- The valid range of the rule is shown next to the control: collapse 23–86 % in the calibration cohort. Outside it the extension warns: the prediction is an extrapolation.

**Pressure** (p0 in Pa, or q0 = p0 / E) is an *advanced* alternative mode.
- E must then be given explicitly, because the rule needs the %.
- The collapse % becomes an output.
- It is useful to explore "what if the pleural pressure were X", not for prediction.

## 3. Parameters shown

**Main panel:** these are the config keys that make sense for a user.

| Control | Config key | Default | Notes |
|---|---|---|---|
| Collapse % (or volume mL) | `physics.collapse.volume_ratio` | 50 % | section 2 |
| Position during the collapse | `physics.gravity.collapse` | lateral decubitus on the healthy side | VATS is done in lateral decubitus. The CT is supine (`gravity.reference`). The difference enters as a body force, already in the model. |
| Sequence frames | `collapse.sequence_frames` | 30 | |

**Advanced panel** (collapsed by default):

| Control | Config key | Default |
|---|---|---|
| Stiffness mode | `physics.E_Pa` | calibration rule, or a value in Pa |
| Pressure mode | `physics.collapse.p0_Pa` / `q0` | off |
| Poisson's ratio | `physics.material.nu` | 0.30 |
| Tissue density | `physics.tissue_density` | 200 kg/m³ |
| CT density | `physics.ct_density` | measured from the CT (`scripts/ct_density.py` logic), report only |
| Wall on / off, margin | `fem_setup.wall.tol_mm` | on, 5 mm |
| Mesh size | `fem_setup.mesh_size_mm` | 14 mm |
| Calibration file | `--params` | the one shipped with the extension |

Not exposed: solver options (fixed per platform, section 5), perturbations and regions (fit only), the fit section.

## 4. Architecture

```
SlicerLungCollapse/                         extension (GitHub repo, Slicer Extensions Index)
├── CMakeLists.txt
├── LungCollapse/                           scripted module
│   ├── LungCollapse.py                     Widget (UI) + Logic + Test
│   ├── Resources/UI/LungCollapse.ui        Qt Designer
│   ├── Resources/calibration/E_200.json    rule file of main_calibrate.py
│   └── Resources/Icons/
└── (eurelaps_lung installed with pip into Slicer's Python, pinned version)
```

**Data flow** (Logic):

1. **Segmentation → surfaces.**
   - Merge the lobes: Segment Editor *Logical operators → Union*, then a small closing (1–2 mm) to fill the fissure gaps.
   - Export the closed-surface representation (vtkPolyData in RAS).
   - Write `.vtk` files with the space in the header (the package reads LPS / RAS, `data_io.read_surface`), into a work folder per segmentation.
2. **Config as a dict.** The patient config is built in memory from `configs/templates/new_patient.json` + the UI values. It needs a `load_config` that accepts a dict (section 7).
3. **Computation in a separate process:**
   ```
   PythonSlicer -m eurelaps_lung.predict --config work/config.json --collapse 50 --params E_200.json --frames 30
   ```
   - Started with `slicer.util.launchConsoleProcess` or a `qt.QProcess`.
   - The log lines are parsed for the progress bar ("geometry", "hilum", "fem_setup", "q0 … volume …").
   - Why a separate process:
     - the GUI stays responsive;
     - Cancel can kill it;
     - a crash of a native library (gmsh, torch, VTK) does not take Slicer down;
     - torch / OpenMP threads do not fight with Qt;
     - `volume_mesh.py` already runs gmsh in a subprocess (`sys.executable`; check that it is `PythonSlicer` inside Slicer).
4. **Output into the scene, directly** (not through the loader script `sequence.py` writes):
   - `vtkMRMLSequenceNode` of model nodes from `runs/<run>/sequence/`, plus the browser;
   - the final model node, with displacement as scalars;
   - a table node from `summary.json`.
   - The code of `SLICER_SCRIPT` in `fem/sequence.py` is the starting point.
5. **Cache.** The work folder is keyed by a hash of the segment geometry + the geometry keys of the config (`geometry`, `hilum`, `fem_setup`). If `fem/setup/problem.npz` is there, only the forward runs, which is what `main_predict.py` already does.

**Later phase.** Once the GNN exists, the forward process can be replaced by the network (milliseconds, float32: this is where MPS / CUDA pay off). The UI and the data flow stay the same.

## 5. Technical problems

### 5.1 Device and precision on the Mac: no MPS for the solve

- **MPS has no float64**, and torch has no sparse direct solver on MPS. The FEM solve needs both:
  - Newton to 1e-7 on a nearly incompressible, large-strain problem with a wall penalty;
  - a sparse LU of the tangent.
- `TorchSolver` already refuses `mps` (`torch_solver.py`). float32 is not a fix: the tangent is ill-conditioned (penalty 20 E/mm, ν 0.3, large strains), and Newton would stall or reject steps.
- **This is not a problem in practice: the problem is small.** karl04: 1355 nodes, 6777 tetrahedra, ~4000 unknowns, 480 surface nodes.
  - A sparse LU of that size takes milliseconds on a CPU.
  - On the server a whole prediction (volume search included) took 0.2 min on the GPU, where the GPU brings nothing at this size.
  - **CPU float64 + SciPy SuperLU** (`linear_solver: "scipy"`) is the target on every platform. To be measured on an M-series Mac: expected under a minute.
- **MPS is useful only for the GNN** (float32 inference), and for `TorchFEM` as a residual (it already runs on any device and dtype).

### 5.2 Linear solvers

| Solver | Status in the extension |
|---|---|
| `pypardiso` | Intel MKL: x86 only, not on Apple Silicon |
| `cudss` (nvmath) | CUDA only |
| `scipy` (SuperLU) | everywhere; enough at this size |

The extension should set `solver.linear_solver: "scipy"` and `device: "cpu"` explicitly, rather than rely on `auto`.

### 5.3 Warp

- NVIDIA Warp runs on macOS only on the CPU (no CUDA), and adds a package and a kernel compile at first use.
- The extension uses the **torch** core only. Warp stays a server option.

### 5.4 PyTorch inside Slicer

- Installing torch into Slicer's Python is best done with the **PyTorch extension** (SlicerPyTorch, module "PyTorch Utils"): it picks the right wheel for the platform / CUDA. It is several hundred MB.
- **Alternative worth considering: a numpy core.**
  - torch is used only for assembly: Neo-Hookean stress and tangent through `torch.func`.
  - A vectorised numpy assembly with the closed-form tangent would make the extension numpy / scipy only:
    - the hand-derived `nh_P` / `nh_dP` already exist in `warp_solver.py`;
    - `NodalSolver` already does everything else;
    - `tests/numpy_core.py` is the slow complex-step version of the same idea.
  - At 6777 tetrahedra the closed form takes milliseconds per assembly.
  - Cost: one more core to validate against torch (same check as torch vs Warp, ~1e-12).
  - Gain: no torch download, no version conflicts with other extensions that pin torch (e.g. TotalSegmentator).
  - **Decision needed.**

### 5.5 Python version

- The package uses `X | None` in dataclass annotations (`config.py`, 24 times). These are evaluated at class creation and need **Python ≥ 3.10** (README).
- Slicer 5.8 and earlier bundle Python 3.9; newer builds moved to a newer Python. **Check the target Slicer version.**
- The fix is cheap and keeps 3.9 working: `from __future__ import annotations` at the top of the modules that use the syntax.

### 5.6 pyvista and pyacvd: plain VTK instead

pyvista is only a thin layer over VTK in this code (checked 2026-11-17). Each use has a VTK filter that ships with Slicer:

| Use | Where | Plain VTK |
|---|---|---|
| `pv.read`, `.save` (.vtk, .vtp, .vtu) | `data_io`, `setup`, `results`, `sequence`, `nodal.export_volume` | `vtkPolyDataReader` / `vtkXMLPolyDataReader` / writers (some already used in `data_io`) |
| `pv.PolyData(points, faces)`, `.points`, `.faces`, `pv.wrap` | everywhere | `vtkPoints` + `vtkCellArray` via `vtkmodules.util.numpy_support` |
| `.clean()` | `surface_mesh`, `hilum`, `check_inputs` | `vtkCleanPolyData` |
| `.fill_holes()`, `.n_open_edges` | `surface_mesh` | `vtkFillHolesFilter`, `vtkFeatureEdges` (or `fem.geometry.face_adjacency`, numpy) |
| `.connectivity()` | `check_inputs` | `vtkPolyDataConnectivityFilter` |
| `.volume` | `surface_mesh` | `vtkMassProperties` or `fem.geometry.signed_volume` (numpy) |
| `grid.compute_implicit_distance` | `fem/wall_field.py` | `vtkImplicitPolyDataDistance` on the grid points (already used in `fem/geometry.py`) |
| `pv.UnstructuredGrid` (tets) | `setup`, `nodal` | `vtkUnstructuredGrid` + `vtkXMLUnstructuredGridWriter` |
| `cell_data` / `point_data` | `hilum`, `sequence` | `GetCellData().AddArray(numpy_to_vtk(...))` |

- Proposed change: one module `eurelaps_lung/vtk_mesh.py` (~100 lines) that holds these helpers and passes `(points, tris)` arrays around, with pyvista removed from the package. The outputs must equal the current files; test with `tests/` and on the server.
- The real dependency is **pyacvd** (ACVD remesh, `geometry` step): it is built on pyvista. Slicer's own Surface Toolbox ("Uniform remesh") pip-installs pyacvd, so pyvista does run on Slicer's VTK. pyacvd 0.3.0 broke inside Slicer (a compiled extension built for PyPI's VTK); the Slicer forum suggested `pyacvd<0.3` or `==0.3.1`. Options:
  - (a) keep pyacvd only for the `geometry` step, installed as the Surface Toolbox does, with its version pinned;
  - (b) call the Surface Toolbox remesh (`SurfaceToolbox` logic) from the extension: same pyacvd, but installed and maintained by Slicer;
  - (c) remesh with gmsh, which is already a dependency (surface reparametrisation + `Mesh.MeshSizeMax`). This is a method change: the meshes differ from the cohort's, so it has to be validated (volume, quality, effect on the collapse).
- (a) or (b) are enough at first: pyacvd runs once per patient, in the separate process. The separate process (section 4) does not isolate packages: `PythonSlicer` has the same site-packages.

### 5.7 Other dependencies

| Package | Where | In Slicer |
|---|---|---|
| `itk` | `data_io` (top-level import, used only for image I/O), `raster`, `check_inputs` | not needed by the prediction. Make the import lazy in `data_io`; Slicer has SimpleITK if images are needed. |
| `scikit-learn` | `fem/regions.py`, imported by `fem/setup.py` at top level | only for the Ward regions (fit). Make the import lazy. |
| `gmsh` | `fem/volume_mesh.py` (subprocess) | pip wheels for macOS arm64 / x86, Linux, Windows bundle the native library. GPL: fine as a runtime pip dependency, to keep in mind if the extension bundles it. |
| `pyacvd` | `surface_mesh.py` | installed by Slicer's Surface Toolbox too; pin a version that works with Slicer's VTK (5.6) |
| `scipy`, `numpy` | everywhere | already in Slicer |

### 5.8 Windows

- The prediction path does not use `run_control.guarded`: that is fit only, and it uses `sh`, `kill` and `os.killpg`, which do not exist on Windows.
- Check `max_solve_s` and the gmsh timeout on Windows.

### 5.9 Coordinates

- Slicer works in RAS, the package in LPS (`data_io` converts on read and write, with the space in the file header). Keep writing the header, and convert back to RAS when loading the frames.
- The gravity directions of `physics.gravity` are in the patient's LPS. They stay right as long as the surfaces are in patient coordinates (Slicer's RAS world from the CT's IJK→RAS), which they are.

### 5.10 Segmentation input

- **Lobes → one lung:**
  - the union leaves gaps along the fissures; a closing of 1–2 mm before the surface export closes them;
  - the ACVD remesh (480 nodes) smooths the rest;
  - check one connected component and no holes before meshing.
- **Checks on the inflated lung only:** `check_inputs` compares with the collapsed lung and is not run. The extension needs its own light checks:
  - closed surface, one component;
  - volume in a plausible range (e.g. 1–4 L);
  - airways touching the lung surface (the hilum needs the rings).

## 6. Model questions this raises

1. **Calibration file.** The extension ships the E rule at ρ̄ 200 (a 3.5332, b −0.013455, `config.py`) as a JSON from `main_calibrate.py`. It records the cohort (12 patients), the collapse range and the residual (0.16 in log₁₀ E, i.e. E ×/÷ 1.45 at 1 sd). It is updated when the cohort grows.
2. **Airways only, without vessels.**
   - The hilum is the centroid of the rings where airways, arteries **and** veins enter the lung, and the clamp spheres are built around all three (`fem_setup.clamp.points`).
   - With airways only, the hilum moves towards the main bronchus and the clamp shrinks: a **different model**, with a different E and different shapes.
   - Two ways forward:
     - (a) require arteries / veins. They can come from an automatic segmentation, e.g. TotalSegmentator in Slicer: check which pulmonary vessel classes it gives.
     - (b) refit the cohort with an airways-only clamp (`main_calibrate.py --set fem_setup.clamp.points='["hilum","airways"]'` after `fem_setup` with airways only) and compare the ASSD with the current fits. If it is close, airways only becomes the extension's default, with its own calibration file.
   - **Decision needed.**
3. **The CT is supine, VATS is lateral.** The default "lateral decubitus on the healthy side" uses the body force (`gravity.reference` ≠ `gravity.collapse`).
   - It was validated on the Mac numpy core and run on the server once (results 5c).
   - The E rule was calibrated on supine → supine fits. Whether the same E holds in the lateral position is not tested: the real collapsed CTs are supine.

## 7. Changes to the package before the extension

| Change | Why |
|---|---|
| `pyproject.toml`, version number, `pip install eurelaps_lung` (from git or a wheel) | the extension installs a pinned version |
| `eurelaps_lung/predict.py` with `run(config, collapse=None, params=None, from_fit=None, frames=None, progress=None)` and `python -m eurelaps_lung.predict`; `main_predict.py` becomes a thin wrapper | a Python API, not only a CLI script |
| `load_config` from a dict (base folder given) | configs built in memory |
| lazy imports of `itk` (`data_io`) and `scikit-learn` (`fem/setup.py` → `regions`) | fewer dependencies for the prediction |
| `from __future__ import annotations` in the modules with `X \| None` | Python 3.9 |
| progress callback or structured log lines (`step`, `solve i / n`) | progress bar |
| sequence frames also returned as arrays (`make_sequence` → data + files) | build the Slicer sequence without re-reading files |
| optional: numpy core with the closed-form Neo-Hookean tangent | no torch in Slicer (5.4) |
| optional: inflated-only input checks | 5.10 |

## 8. Phases

1. **v0, a prototype without the package changes.**
   - A scripted module that exports the surfaces, writes a config file and runs `main_predict.py` with `PythonSlicer` in a subprocess.
   - It loads the sequence with the existing `load_collapse_in_slicer.py`.
   - Test: Mac M-series and Linux, torch CPU, one patient.
2. **v1:** the changes of section 7, in-scene output, cache, UI of sections 2–3, input checks, cancel.
3. **v2:** extension packaging (Extensions Index, dependencies installed on first use), tests, Windows.
4. **v3:** the GNN in place of the forward solve (MPS / CUDA), interactive slider on the collapse %.
