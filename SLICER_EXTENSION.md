# 3D Slicer extension: design

Design of a 3D Slicer 5.12 extension that predicts the collapsed lung of a patient loaded in Slicer and shows the collapse as a sequence. It runs the prediction of `eurelaps_lung` (`eurelaps_lung.predict`), installed from this repository with pip. Fitting and calibration stay outside Slicer: the extension uses the calibrated defaults shipped with the package.

The extension lives in its own repository and is built in two parts (section 9):
- **Part 1, the interface:** the module, its UI and a fake logic. It needs only this file.
- **Part 2, the computation:** package installation, surfaces, config, prediction run and outputs into the scene.

Status of the package side (2026-11-18): ready for both parts (section 2). Sections 3–8 are the specification. Section 10 keeps the technical background (precision, solvers, dependencies, licenses).

## 1. What the user does

1. **Give the data, from the scene or from files:** the CT (optional, for display) and a segmentation of it with:
   - **the lungs:** one segment per lung, or the lobes (2 on the left, 3 on the right);
   - **airways, pulmonary arteries, pulmonary veins:** one segment each. They place the hilum, where the lung is held.

   Either a segmentation already in the scene, or segmentation files chosen in the module: they are loaded into the scene, and from there both cases are the same. Slicer converts the segments to closed surfaces, which is what the package works on.
2. **Choose the lung to collapse:** Left / Right. The lung segment is pre-selected from the segment names or centroids, and is editable.
3. **Optional, for validation: the collapsed lung.** A segment of the collapsed lung, in the same coordinates as the inflated one (same scan, or registered beforehand in Slicer). With it, the collapse % can be taken from it, and the results include the error of the prediction against it.
4. **Set the collapse:** the collapse %, and the position during the collapse (supine as the CT, or lateral decubitus on the healthy side).
5. **Set the number of frames** of the sequence, and, if wanted, the advanced parameters.
6. **Click Apply.**
   - The first time, the model geometry is built (surface remesh, hilum, clamp, volume mesh, wall field): tens of seconds.
   - Then the forward run (q0 is searched until the volume matches the collapse %), then the frames.
   - A new collapse % on the same segmentation reuses the geometry and only re-solves.
7. **Choose where the outputs go** (optional): an output folder keeps every file. Without one, the files go to a temporary folder that is deleted afterwards (section 6); the results stay in the scene either way.
8. **Look at the output:**
   - the collapse as a sequence over the CT, coloured by displacement, with a frame slider and play;
   - the final collapsed lung as a model;
   - the hilum and its rings, to check them;
   - a results table: volumes, collapse % reached, E, p0, solves, time, and with the collapsed lung, the errors (ASSD, mean, 95th percentile, max).

## 2. What the package provides (done, 2026-11-18)

| Item | Where | Notes |
|---|---|---|
| pip-installable package `eurelaps-lung` 0.1.0 | `pyproject.toml` | `import eurelaps_lung`. The default install is the prediction path without `vtk` and `torch` (both come from Slicer, section 4) |
| Prediction API and CLI | `eurelaps_lung/predict.py` | `predict.run(config, collapse=..., frames=..., run_name=..., overrides=[...])` returns the summary dict. CLI: `python -m eurelaps_lung.predict --config c.json --collapse 40 --frames 30` (or the `eurelaps-predict` script) |
| Prediction defaults | `eurelaps_lung/data/predict_default.json`, `predict.default_config()` | the calibrated model (section 5) |
| Config from a dict | `config.load_config(dict, overrides, base_dir)` | no need to write the config file, but writing it is still useful (it is the record) |
| Collapse % from the collapsed lung | `collapse="target"` (`--collapse target`) | validation mode |
| Errors against the collapsed lung | `summary.json`: `assd_mm`, `mean_err_mm`, `median_err_mm`, `p95_err_mm`, `max_err_mm`, `target_volume_ratio`; frames: `Error_mm` | when `inputs.collapsed` is set |
| Sequence frames | `runs/<run>/sequence/frame_000.vtp …` (+ `target.vtp`, `hilum_anchor.mrk.json`, `load_collapse_in_slicer.py`) | `Displacement_mm`, `Disp_mag_mm`, `Pressure_Pa`, `Error_mm`, `WallDistance_mm`; frames = load 0 → 100 % |
| Rule range for the UI | `config.E_RULE_INFO` | 13 patients, collapse 23–86 %, residual sd 0.158 log10 E, supine → supine |
| No GPL code | Netgen (LGPL) replaced gmsh | section 10.6 |
| No pyvista outside pyacvd | `eurelaps_lung/vtk_mesh.py` (Slicer's VTK) | outputs identical to the pyvista code on 16 patients |
| Imports | tested in `tests/test_predict.py` | the prediction modules import without torch, Warp, GetFEM, itk, scikit-learn, nvmath, pypardiso, pyvista, pyacvd, netgen (each is imported only where it is used) |
| Netgen's interpreter | `EURELAPS_PYTHON`, else `sys.executable` | right inside `PythonSlicer` (section 6) |

Checked on the Mac (env `lungct`: real VTK, pyacvd, Netgen, numpy test core instead of torch): `predict.run` from `default_config()` on karl04 with its collapsed lung builds the geometry, collapses to the collapse % of the collapsed lung, reports the errors and writes the frames.

**Still to check in Slicer itself (Part 2):** `pip install` into Slicer 5.12 (no second VTK, the pykdtree OpenMP clash, netgen-mesher wheels), torch from the PyTorch extension, the run time on the Mac CPU.

## 3. Inputs: from segmentation to surfaces

The package reads closed surfaces (`.vtk`, `.vtp`), with the coordinate space in the file (`SPACE=RAS` or `SPACE=LPS` in the legacy header, or the `SPACE` field data). Slicer works in RAS.

**Two input sources, one path.** The input panel has a switch *From the scene* / *From files*:

| Source | What the user picks | What the extension does |
|---|---|---|
| From the scene | a `vtkMRMLSegmentationNode` (and the CT volume) | nothing: the segments are used as they are |
| From files | a segmentation file: `.seg.nrrd`, or a labelmap `.nrrd` / `.nii` / `.nii.gz` (labels named by a colour table or mapped in the UI); optionally the CT file; optionally a second file with the collapsed lung | loads them into the scene (`slicer.util.loadSegmentation`; a labelmap with `slicer.util.loadLabelVolume` + `ImportLabelmapToSegmentationNode`), then as from the scene |
| From files, surfaces (advanced) | ready closed surfaces `.vtk` / `.vtp` / `.stl` per structure | copied to the work folder as they are (the space from the file header, else asked: LPS or RAS) |

Files loaded by the module are added to the scene, so that the user sees and can check what was used; a "Remove loaded inputs" option drops them after the run.

Then, for both sources, the extension:

1. **Merges the lobes** of the chosen side into one segment (Segment Editor logic, *Logical operators → Union*), then a closing of 1–2 mm (*Smoothing → Closing*) to fill the fissure gaps.
2. **Converts the segments to closed surfaces with Slicer** (lung, airways, arteries, veins, and the collapsed lung): `segmentationNode.CreateClosedSurfaceRepresentation()`, `GetClosedSurfaceInternalRepresentation(segmentId)` (vtkPolyData, RAS, in the segmentation's parent transform: harden it first or apply it). The conversion parameters are Slicer's defaults (smoothing factor 0.5); the package remeshes the surface afterwards (ACVD, ~480 nodes), so they matter little, but they are written to the config record.
3. **Writes them** into the work folder as legacy `.vtk` with `SPACE=RAS` in the header. With the package: `data_io.write_surface(vtk_mesh.Surface.from_vtk(polydata), path, "RAS")`. The `geometry` step converts to LPS.
4. **Light checks before the run** (the package's `check_inputs` needs the collapsed lung and is not run):
   - one connected component, closed;
   - lung volume in a plausible range (1–4 L);
   - each tree touches the lung surface (the hilum needs the rings).

The collapsed lung must be in the same coordinates as the inflated one. The extension does not register CTs. If it comes from another scan, the user registers the two in Slicer first.

## 4. Installation

**Tag the package** (in this repository; the user commits and tags):

```bash
git tag -a v0.1.0 -m "eurelaps_lung 0.1.0: prediction API, Netgen mesher, pyproject"
git push origin v0.1.0
```

**In Slicer** (the extension does this on first use, with a confirmation dialog):

```python
import slicer
# torch: from the PyTorch extension (Extensions Manager: "PyTorch"), which picks the wheel for the platform
import PyTorchUtils
torch = PyTorchUtils.PyTorchUtilsLogic().installTorch(askConfirmation=True)        # CPU wheel on the Mac
# the package and its prediction dependencies (numpy, scipy: already in Slicer; pyacvd, netgen-mesher)
slicer.util.pip_install("eurelaps-lung @ git+https://github.com/edoardoedr/lung_collapse.git@v0.1.0")
```

- **Pin the tag** (`@v0.1.0`): the extension and the package versions go together. The extension checks `eurelaps_lung.__version__`.
- **Private repository:** pip needs credentials (a token in the URL, or SSH). A public repository, or a wheel attached to a GitHub release, avoids this.
- **Never pip-install `vtk` into Slicer.** The package does not require it, but pyacvd → pyvista declares it. If `pip_install` pulls a second VTK: install with `--no-deps`, then pyacvd, pyvista, pykdtree, netgen-mesher (and pyvista's pure-Python requirements) with `--no-deps` too. Check this first.
- Not installed in Slicer: the extras `fit` (scikit-learn, itk), `gpu`, `warp`. They are for the server (`pip install -e ".[server]"`).

## 5. The configuration

**One base config: the package defaults** (`eurelaps_lung/data/predict_default.json`, `predict.default_config()`). They are the settings of the calibrated model. The extension does not ship its own config and changes only the keys of section 7. The same defaults on the command line are `configs/templates/new_patient.json`; a test keeps the two equal.

| Group | Value | Why |
|---|---|---|
| Surface | ACVD remesh to ~480 nodes | the cohort's meshes |
| Hilum | from the **inflated** lung | the prediction has no collapsed lung. The cohort fits used the collapsed one: section 8.1 |
| Clamp | spheres at the hilum, airways, arteries, veins rings (radius × 2) | the calibrated model |
| Volume mesh | Netgen, maxh 10 mm, 10 optimisation steps | the calibration meshes (2026-11-18) |
| Wall | the inflated surface + 5 mm, smooth field (grid 1.5 mm, σ 3 mm) | the lung cannot leave the chest cavity |
| Material | Neo-Hookean, ν 0.30 | |
| Stiffness | `E_Pa {"rule": "collapse"}`: log₁₀ E = 3.5380 − 0.012883 × collapse % | the prediction rule of the cohort (leave-one-out: +0.09 mm median ASSD against the patient's own fit) |
| Density | 200 kg/m³ | the measured pleural gradient |
| Gravity | supine → supine | the calibration |
| Collapse | `volume_ratio` = 1 − collapse % / 100 | q0 searched |
| Solver | torch core, CPU, float64, SciPy SuperLU, Newton ≤ 50 iterations, 600 s per solve | the same on every platform (10.1, 10.2) |
| Frames | 30 | |

The config actually used is written to `<output>/logs/config_used_predict.json`, with the overrides applied.

## 6. Architecture

```
SlicerLungCollapse/                         extension repository
├── CMakeLists.txt                          Extension Wizard layout
├── LungCollapse/
│   ├── CMakeLists.txt
│   ├── LungCollapse.py                     Widget + Logic + Test
│   ├── Resources/UI/LungCollapse.ui        Qt Designer
│   └── Resources/Icons/LungCollapse.png
└── README.md                               includes the dependencies and their licenses (10.6)
```

**Data flow (Logic):**

1. **Work folder** per segmentation and side:
   - **output folder chosen:** `<output folder>/<patient or segmentation name>_<side>/`. Everything stays there: `inputs/*.vtk`, `config.json`, `out/` (geometry, `fem/setup`, `runs/<run>/` with the frames, `summary.json`, `logs/`). It is the record of the run and can be reopened (`load_collapse_in_slicer.py`) or rerun on the command line.
   - **no output folder:** a temporary folder, `tempfile.mkdtemp(prefix="LungCollapse_", dir=slicer.app.temporaryPath)`. It is kept while the module is used with the same inputs (the geometry is reused for a new collapse %), and **deleted** when the scene is closed, the inputs change, or Slicer exits (observer on `slicer.mrmlScene.EndCloseEvent` and on `slicer.app.aboutToQuit`; leftover `LungCollapse_*` folders older than a day are removed when the module starts, after a crash).
   - Before deleting, everything the scene needs is already in memory (models, sequence, markups, table are loaded, not linked to the files). The scene can still be saved with *Save* as usual: Slicer writes its own copies.
   - A **"Save outputs…"** button copies the current temporary work folder to a folder chosen afterwards.
   - The geometry is rebuilt only when the inputs change: a hash of the exported surfaces (points) and of the geometry keys of the config (`geometry`, `hilum`, `fem_setup`) is stored in the work folder. If it matches and `out/fem/setup/problem.npz` exists, only the forward runs: `predict.run` does this by itself.
2. **Surfaces** → `inputs/*.vtk` (section 3).
3. **Config**: `predict.default_config()` + patient, `data_dir`, `output_dir`, `inputs` + the UI values, written as `config.json`.
4. **Run in a separate process:**
   ```
   PythonSlicer -m eurelaps_lung.predict --config <work>/config.json --collapse 40 --frames 30 --run pred_40
   ```
   - Started with `qt.QProcess` (or `slicer.util.launchConsoleProcess`), output read line by line.
   - Why a process:
     - the GUI stays responsive;
     - Cancel kills it;
     - a crash of a native library (Netgen, torch, VTK) does not take Slicer down;
     - torch threads do not fight with Qt.
   - Inside `PythonSlicer`, `sys.executable` is a Python, so the Netgen subprocess works. The extension can also set `EURELAPS_PYTHON` to the `PythonSlicer` path.
   - Exit code ≠ 0 = failure: show the last log lines.
5. **Progress** from the log lines:

   | Line contains | Stage |
   |---|---|
   | `=== geometry ===` | 1/5 surface |
   | `=== hilum ===` | 2/5 hilum |
   | `=== fem_setup ===` | 3/5 mesh and wall |
   | `=== collapse ===` / `q0 … volume …` | 4/5 forward (one line per solve) |
   | `frames` / `sequence` | 5/5 frames |
   | `WARNING` | shown in the panel (e.g. "a ring centre is > 50 mm from the hilum") |

6. **Outputs into the scene**, from `out/runs/<run>/`:
   - **sequence:** a `vtkMRMLSequenceNode` of the `sequence/frame_*.vtp` models (index = load %) and a `vtkMRMLSequenceBrowserNode`, coloured by `Disp_mag_mm`. The files say `SPACE=LPS`: Slicer's model reader converts them. The code of `load_collapse_in_slicer.py` (`fem/sequence.py`, `SLICER_SCRIPT`) is the starting point;
   - **final model:** `surface.vtp`;
   - **collapsed lung:** `sequence/target.vtp` as a grey wireframe, when given;
   - **hilum:** `sequence/hilum_anchor.mrk.json` and `out/hilum/rings.vtp`;
   - **table:** a `vtkMRMLTableNode` from `summary.json` (section 7, outputs).

**Later:** once the GNN exists, the forward process can be replaced by the network (milliseconds, float32 on MPS or CUDA), with an interactive slider on the collapse %. The UI and the data flow stay the same.

## 7. Interface (Part 1)

One scripted module, **Lung Collapse**, category "Chest" (or "Simulation"). Four collapsible sections; "Advanced" is closed by default.

**Inputs**

| Control | Widget | Notes |
|---|---|---|
| Input source | radio buttons *From the scene* / *From files* (+ *Surfaces* in Advanced) | section 3 |
| CT | scene: `qMRMLNodeComboBox` (`vtkMRMLScalarVolumeNode`); files: `ctkPathLineEdit` | optional, for display and as the segmentation's reference |
| Segmentation | scene: `qMRMLNodeComboBox` (`vtkMRMLSegmentationNode`); files: `ctkPathLineEdit` (`.seg.nrrd`, `.nrrd`, `.nii`, `.nii.gz`) + "Load" | a file is loaded into the scene, then the segment selectors below fill in |
| Lung to collapse | radio buttons Left / Right | pre-set from the segments: the side with a segment named like `left`/`right`, else by centroid x |
| Lung segment(s) | `qMRMLSegmentSelectorWidget` (multi-selection) | the lobes of that side are pre-selected and merged |
| Airways / Arteries / Veins | three segment selectors | required (the hilum). Pre-selected by name |
| Compare with a collapsed lung | checkbox + (scene: segmentation combo / files: path) + segment selector | optional; same coordinates (section 3) |
| Surfaces (Advanced, source *Surfaces*) | one `ctkPathLineEdit` per structure (`.vtk`, `.vtp`, `.stl`) + space LPS / RAS when the file has no tag | skips the conversion |

**Collapse**

| Control | Widget | Config key | Default |
|---|---|---|---|
| Collapse % | slider + spin box 0–95 | `physics.collapse.volume_ratio` | 50 |
| … as volume | spin box mL, linked to the % once the inflated volume is known | (same) | |
| Use the collapse % of the collapsed lung | checkbox (only with the collapsed lung) | `--collapse target` | off |
| Position during the collapse | combo: *Supine (as the CT)*, *Lateral decubitus, healthy side down*, *Prone* | `physics.gravity.collapse`: `supine`, `right_lateral` for a left lung / `left_lateral` for a right lung, `prone` | Supine |
| Frames | spin box 0–60 | `collapse.sequence_frames` | 30 |

- Next to the collapse %: "calibrated on 23–86 %" (`E_RULE_INFO`). Outside that range: a warning "extrapolation".
- Next to the position: "calibrated on supine CTs". For a lateral position, a note that the rule is not validated there (section 8.2).
- The CT position (`physics.gravity.reference`) is supine. It is shown in Advanced, not in the main panel.

**Advanced** (closed by default)

| Control | Config key | Default |
|---|---|---|
| Stiffness: from the collapse % (rule) / fixed E in Pa | `physics.E_Pa` | rule |
| Poisson's ratio | `physics.material.nu` | 0.30 |
| CT position | `physics.gravity.reference` | supine |
| Wall margin (mm) | `fem_setup.wall.tol_mm` | 5 |
| Device: CPU / CUDA (only if `torch.cuda.is_available()`) | `solver.device` (+ `linear_solver` null on CUDA) | CPU |
| (moved to the main panel, *Run and outputs*) | | |
| Rebuild geometry | `--steps geometry hilum fem_setup` | off |

Not exposed: the other keys of the config (surface remesh, mesh size, clamp, solver tolerances, fit). A "Show config" button opens the config used.

**Run and outputs**

- **Output folder:** `ctkPathLineEdit` (directory), empty by default, with the note "empty = temporary folder, deleted when the scene is closed". **Save outputs…** button after a run with a temporary folder.
- **Apply** (enabled when the inputs are complete), **Cancel** (while running), a progress bar with the stage name, a log box (collapsed, last lines).
- **Results table:**

  | Row | From `summary.json` |
  |---|---|
  | Inflated volume, collapsed volume (mL) | `volume_ratio` × the inflated volume |
  | Collapse % asked / reached | `collapse_percent` |
  | E (Pa), p0 (Pa) | `physics.E_Pa`, `physics.p0_Pa` |
  | Solves, time | `solves`, the run time |
  | Wall penetration (mm) | `wall_max_penetration_mm` |
  | With the collapsed lung: ASSD, mean, median, 95th percentile, max error (mm); its collapse % | `assd_mm`, `mean_err_mm`, `median_err_mm`, `p95_err_mm`, `max_err_mm`, `target_volume_ratio` |

- **Sequence:** the frame slider of the sequence browser (the load %), play / pause; the colour (displacement, or error against the collapsed lung when given).
- **Visibility checkboxes:** CT, inflated lung, collapsed prediction, real collapsed lung, hilum and rings.

**Part 1 acceptance:**
- the module loads in Slicer 5.12;
- every control above exists and is validated: Apply is disabled until the inputs are complete; the range warnings work;
- both input sources work up to the segment selectors (loading a `.seg.nrrd` or a labelmap from file fills them like a scene segmentation);
- the output folder: with one, the fake outputs are written there; without, to a temporary folder deleted on scene close (and "Save outputs…" copies it);
- a fake Logic returns the frames of a sphere shrinking and a fixed summary, so that the sequence, the slider and the table can be tried without the package;
- the Logic interface is the one Part 2 fills: `LungCollapseLogic.run(params: dict, progress_callback, done_callback)`, with `params` = the UI values, named as the config keys above, plus `surfaces` (structure → vtkPolyData in RAS, or a file path), `output_dir` (None = temporary).

## 8. Model questions

1. **Hilum on the inflated lung.** The prediction computes the hilum on the inflated lung; the cohort fits, the E rule and the leave-one-out computed it on the collapsed one (the configs' `hilum.lung: "collapsed"`).
   - On the 16 local patients (2026-11-18, analysis/results.md 7e) the hilum moves by 0–3.6 mm on 14.
   - On karl17 (39 mm: an airways ring found 116 mm away) and karl20 (12 mm: the veins ring 42 mm away) it moves a lot. karl17 triggers the warning "a ring centre is > 50 mm from the hilum".
   - To do on the server: the cohort fits and leave-one-out with `hilum.lung: "inflated"` (NEXT.md). The extension shows the rings and the warning so that the user can check them.
2. **The CT is supine, VATS is lateral.** The lateral position enters as a body force (`gravity.reference` ≠ `gravity.collapse`).
   - It was validated on the Mac numpy core and run on the server once (results 5c).
   - The rule was calibrated on supine → supine: whether the same E holds in the lateral position is not tested (the real collapsed CTs are supine).
3. **Airways only, without vessels.** With airways only the hilum moves towards the main bronchus and the clamp shrinks: a different model. Either the vessels are required (automatic segmentation, e.g. TotalSegmentator), or the cohort is refitted with `fem_setup.clamp.points: ["hilum", "airways"]` and gets its own rule. **Decision needed**; for now the three trees are required.
4. **Calibration updates.** The rule is in the package (`config.E_RULE_A`, `E_RULE_B`, `E_RULE_INFO`). A new calibration = a new package version, and the extension pins it.

## 9. The two parts

**Part 1: interface (Claude Code, extension repository).** From this file only, no package needed.
- Skeleton with the Slicer Extension Wizard: extension `SlicerLungCollapse`, scripted module `LungCollapse`.
- The `.ui` and the Widget of section 7, with the parameter node (`slicer.parameterNodeWrapper`) holding the UI state.
- The input side for real (it is Slicer only, no package): loading segmentation files, segment selection, lobe merge, conversion to closed surfaces (section 3), handed to the Logic as vtkPolyData.
- The work folder (section 6): output folder or temporary one, its deletion, "Save outputs…".
- The fake Logic and the acceptance list of section 7.
- Nothing about the package or the solver.

**Part 2: computation (Claude Code, extension repository).**
- Dependency installation on first use (section 4): PyTorch extension → torch; `pip_install` of the package at the pinned tag; check the VTK and the version.
- The real Logic:
  - surfaces written to the work folder (section 3);
  - config from `predict.default_config()` + the UI (sections 5, 7);
  - the process and the progress (section 6);
  - outputs into the scene, the results table (sections 6, 7).
- Work folder cache (section 6).
- Test: the module's self-test on a sample patient (a public chest CT with segmented lungs, airways and vessels, or one of the cohort's surfaces if they can be shared), on the Mac (CPU) and on Linux.
- README: dependencies and licenses (10.6), what the model is and its limits (section 8).

## 10. Technical background

### 10.1 Device and precision on the Mac: CPU float64

- **MPS has no float64**, and torch has no sparse direct solver on MPS (nor on the CPU: the LU is outside torch, 10.2). `TorchSolver` refuses `mps`.
- **float32 would probably not lose accuracy where it matters, but it has not been measured:**
  - The displacement error of a converged solve is about eps (1.2e-7) × the condition number of the tangent. With a condition number of 1e4–1e5 that is 0.01–0.5 mm, below the segmentation error (~1 mm) and the model error (ASSD ~6–9 mm).
  - The risk is **convergence**: Newton stops at `newton_tol` 1e-7, which float32 cannot reach. Line search, adaptive steps and the volume search would work on a noisier residual.
  - **No speed gain either:** the LU stays on the CPU, so every iteration copies the tangent from MPS. At ~7000 tetrahedra the GPU assembly gains little.
  - Test to settle it (NEXT.md, future): the same forward in float32 vs float64.
- **The problem is small**: ~1200 nodes, ~5500 tetrahedra, ~3600 unknowns. A sparse LU of that size takes milliseconds; on the server a whole prediction took 0.2 min on the GPU, where the GPU brings nothing at this size. **CPU float64 + SciPy SuperLU** is the target on every platform; the time on an M-series Mac is to be measured (Part 2).
- MPS is for the GNN (float32 inference) and for `TorchFEM` as a residual.

### 10.2 Linear solvers

| Solver | In the extension |
|---|---|
| `scipy` (SuperLU) | yes: everywhere, enough at this size |
| `pypardiso` | Intel MKL: x86 only, not on Apple Silicon |
| `cudss` (nvmath) | CUDA only |

- The linear solver is separate from the core: torch assembles the tangent, `LinearSolver` (`fem/solvers/nodal.py`) factorises it on the host. `auto` = cuDSS on CUDA if installed, else pypardiso if installed, else SuperLU.
- So far every server run used torch on CUDA + cuDSS; the Mac tests use the numpy core + SuperLU and reproduce the server fit (karl04: ASSD 9.00 mm, p0 1283 vs 1282 Pa, 11 solves). The solvers are all direct: the backend changes the time, not the result.
- The tangent is not symmetric (follower pressure), so an LU is needed (no Cholesky).
- SuperLU on karl18's pattern (2298 dofs): 27 ms per factorisation + solve on this Mac.
- Other sparse LUs on a Mac, if it is ever the bottleneck (NEXT.md: future test):

  | Option | Notes |
  |---|---|
  | Apple Accelerate `SparseFactorizationLU` | macOS 15.5+, C API only (ctypes / C extension) |
  | `macldlt` (pip, MIT) | Accelerate LDLᵀ only: does not apply; a template for an LU wrapper |
  | UMFPACK, MUMPS, KLU | compiled packages to install in Slicer |
  | dense LU (numpy / torch) | ~0.1 s at 2300 dofs, grows as n³ |

### 10.3 PyTorch inside Slicer

- **Decided (user, 2026-11-18): torch, from the PyTorch extension** (SlicerPyTorch, `PyTorchUtils`). It picks the right wheel for the platform; several hundred MB.
- A numpy core (closed-form Neo-Hookean tangent: `nh_P` / `nh_dP` exist in `warp_solver.py`) would remove torch. It is not needed now; it stays an option if torch versions clash with other extensions (e.g. TotalSegmentator).
- Warp is not used in Slicer (CPU only on macOS, a kernel compile at first use).

### 10.4 Python version

- The package needs Python ≥ 3.10 (`X | None` in dataclasses). Slicer 5.10+ has Python 3.12: fine.

### 10.5 pyvista and pyacvd

- pyvista was only a thin layer over VTK: **replaced by `eurelaps_lung/vtk_mesh.py`** (plain VTK filters set as pyvista set them). On 16 patients the four preprocessing steps give the same files as the pyvista code (analysis/results.md 7d).
- **pyacvd** (ACVD remesh, `geometry` step) is still needed and is built on pyvista. It is confined to `surface_mesh.acvd_remesh`. Slicer's Surface Toolbox pip-installs pyacvd too ("Uniform remesh"). pyacvd 0.3.0 broke inside Slicer; the package requires ≥ 0.3.1.
- Seen on the Mac (env `lungct`): the pip wheel of **pykdtree** (pyacvd dependency) bundles its own `libomp`; with another OpenMP already loaded the process aborts (`OMP: Error #15`). Fixed there with the conda-forge pykdtree. Slicer and torch bundle OpenMP too: check this first in Part 2 (fallback: `KMP_DUPLICATE_LIB_OK=TRUE` in the process environment, or a pykdtree built without OpenMP).
- Alternatives if pyacvd breaks in Slicer: Netgen's STL remesh, or gmsh's (GPL). Either is a method change (different surface nodes): the calibration would have to be redone (NEXT.md: future test).

### 10.6 What gets installed, and the licenses

Prediction path (`geometry` → `hilum` → `fem_setup` → `collapse`):

| Package | Used by | In Slicer 5.12 | License |
|---|---|---|---|
| `numpy`, `scipy` | everywhere; SuperLU | already there | BSD |
| `vtk` | `vtk_mesh.py` | already there (Slicer's): never pip-install it | BSD |
| `eurelaps-lung` | the package | `pip_install` from git, pinned tag | (to choose; the repository has no license file yet) |
| `netgen-mesher` (+ `netgen-occt`) | `fem/volume_mesh.py` (subprocess) | pip wheels for macOS / Linux / Windows | LGPL-2.1 |
| `pyacvd` + `pykdtree` | `surface_mesh.acvd_remesh` | pip | MIT; LGPL-3.0+ |
| `pyvista` | inside pyacvd only | with pyacvd (check the VTK, section 4) | MIT |
| `torch` (CPU) | the FEM core | PyTorch extension | BSD |

Not installed in Slicer: `itk` (check_inputs, images), `scikit-learn` (Ward regions), `warp-lang`, `getfem`, `nvmath-python`, `pypardiso`.

- Slicer is BSD-style and avoids copyleft dependencies. The LGPL ones are fine as separate pip-installed packages (not bundled, not modified). **No GPL code is left on the prediction path**: gmsh (GPL-2.0+) was replaced by Netgen on 2026-11-18 (analysis/results.md 7d: same fits, rule and leave-one-out on 16 patients; better tetrahedra).
- Volume meshers considered: gmsh (GPL), **Netgen (LGPL, chosen)**, TetGen (AGPL / commercial: what SegmentMesher offers), Cleaver2 (MIT, but it meshes the labelmap and does not keep the model's surface).
- **Why not SegmentMesher:** it meshes the segmentation, not the model's ACVD surface; it is compiled Slicer-only tools, while the same meshing must run on the server (cohort fits, GNN data); its surface-keeping engine is TetGen (AGPL).

### 10.7 Windows

- The prediction path does not use `run_control.guarded` (fit only; it needs `sh`, `kill`, `os.killpg`).
- To check: `max_solve_s` and the Netgen timeout (`fem_setup.mesh_timeout_s`, a subprocess timeout) on Windows.

### 10.8 Coordinates

- Slicer works in RAS, the package in LPS. `data_io` converts on read and write, and writes the space in every file (`SPACE=` header / field data).
- The gravity directions are in the patient's LPS. They are right as long as the surfaces are in patient coordinates (Slicer's RAS world from the CT's IJK→RAS), i.e. with the transforms hardened before the export.
