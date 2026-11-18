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

- **MPS has no float64**, and torch has no sparse direct solver on MPS (nor on CPU: the LU is always outside torch, 5.2). `TorchSolver` refuses `mps` (`torch_solver.py`).
- **Would float32 lose accuracy on the result?** Probably not where it matters, but it has not been measured:
  - float32 has ~7 digits (eps 1.2e-7). The displacement error of a converged solve is about eps × the condition number of the tangent. Large strains, the wall penalty (20 E/mm) and the stiff clamp make the tangent poorly conditioned; with a condition number of 1e4–1e5 the error is 1e-3–1e-2 relative, i.e. ~0.01–0.5 mm on displacements of tens of mm. That is below the segmentation error (~1 mm) and the model error (ASSD ~9 mm).
  - The risk is **convergence**, not the final number: Newton stops at `newton_tol` 1e-7 (relative residual), which float32 cannot reach; the tolerance must be ~1e-4–1e-5. Line search, the adaptive load steps and the volume-ratio secant (`find_q0`) then work with a noisier residual, and solves that converge today may fail.
  - Near the clamp the strains are small: Neo-Hookean stress has the cancellation F − F⁻ᵀ and ln J ≈ J − 1 there, so the relative error of the stress grows like eps / strain. Locally only.
  - The fit is worse off than the prediction: its Jacobian (sensitivities, Gauss-Newton) needs more digits than the forward solve.
  - **And it brings no speed:** the LU is still on the CPU (no sparse solver on MPS; SciPy SuperLU runs in float32 too), so every Newton iteration copies the tangent MPS → CPU; at ~7000 tetrahedra the GPU assembly gains little.
  - Test to settle it (server, no Mac torch needed): the same forward with `TorchFEM` in float32 vs float64 (CUDA or CPU, LU in float32), compare surface displacement, solves and failures.
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

- The linear solver is separate from the core: torch (or Warp, numpy) assembles the tangent, and `LinearSolver` (`fem/solvers/nodal.py`) factorises it on the host. `auto` picks cuDSS if the device is CUDA and nvmath is installed, else pypardiso if installed, else SciPy SuperLU.
- What was used so far: **every server run = torch on CUDA + cuDSS** (`auto`; 95 of 95 logs in `results/` say "device cuda, linear solver cudss"). The Mac tests = numpy core + SciPy SuperLU, and they reproduce the server fit (karl04: ASSD 9.00 mm, p0 1283 vs 1282 Pa, 11 solves). The solvers are all direct: the backend changes the time, not the result (up to round-off).
- The tangent is not symmetric (follower pressure on the surface), so an LU is needed: Cholesky (CHOLMOD) does not apply.
- Timing on this Mac (2026-11-17): SuperLU on the sparsity pattern of karl18 (766 nodes, 2298 dofs) takes 27 ms per factorisation + solve; a dense LU 98 ms. karl04 (~4000 dofs) should be well under 0.1 s. A Newton solve has tens of factorisations, a prediction tens of solves: the linear algebra costs seconds; the torch assembly on CPU is still to be measured.
- SuperLU is not the only sparse LU on a Mac, only the one already in the code and in Slicer (SciPy ships with it). Other CPU options, if it is ever the bottleneck (NEXT.md: future test):

  | Option | Notes |
  |---|---|
  | Apple Accelerate, `SparseFactor(SparseFactorizationLU, ...)` | in the macOS SDK since **macOS 15.5** (`Sparse/Solve.h`, checked 2026-11-18); also Cholesky / LDLᵀ (symmetric only) and QR (any matrix, slower) since 10.13. C API only: no official Python binding, needs a small ctypes / C extension; Mac only |
  | `macldlt` (github.com/bodono/macldlt, pip, MIT) | pybind11 wrapper of Accelerate's **LDLᵀ only** (symmetric indefinite; reads one triangle, symmetry not checked): does not apply to the non-symmetric tangent. Wheels macOS 13+ arm64, Python ≥ 3.10, small project (6 stars, no releases; 2026-11-18). Useful as a template for an Accelerate **LU** wrapper (same API: `factor`, `refactor(values)` on a fixed pattern, `solve`) |
  | UMFPACK (`scikit-umfpack`) | SuiteSparse; compiled package to install in Slicer |
  | MUMPS (`python-mumps`, `petsc4py`) | the solver of the GetFEM runs; conda-forge builds, compiled |
  | KLU (SuiteSparse) | fast for small circuit-like matrices; few maintained Python bindings |
  | CHOLMOD (`scikit-sparse`) | Cholesky: does not apply (the tangent is not symmetric) |
  | dense LU (`numpy` / `torch.linalg` on CPU) | no install; ~0.1 s at 2300 dofs, grows as n³ |
  | torch / MPS | no sparse direct solver (torch's sparse solve is CUDA only) |

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
- Slicer 5.8 and earlier bundle Python 3.9; Slicer 5.10 moved to Python 3.12.
- **Target: Slicer 5.12** (user, 2026-11-18), so Python 3.12: no change needed. `from __future__ import annotations` would only matter for Slicer ≤ 5.8.

### 5.6 pyvista and pyacvd: plain VTK instead

pyvista is only a thin layer over VTK in this code (checked 2026-11-17). Each use has a VTK filter that ships with Slicer:

| Use | Where | Plain VTK |
|---|---|---|
| `pv.read`, `.save` (.vtk, .vtp, .vtu) | `data_io`, `setup`, `results`, `sequence`, `nodal.export_volume` | `vtkPolyDataReader` / `vtkXMLPolyDataReader` / writers (some already used in `data_io`) |
| `pv.PolyData(points, faces)`, `.points`, `.faces`, `pv.wrap` | everywhere | `vtkPoints` + `vtkCellArray` via `vtkmodules.util.numpy_support` |
| `.clean()` | `surface_mesh`, `hilum`, `check_inputs` | `vtkCleanPolyData` |
| `.fill_holes()`, `.n_open_edges`, `.extract_feature_edges` | `surface_mesh` | `vtkFillHolesFilter`, `vtkFeatureEdges` (or `fem.geometry.face_adjacency`, numpy) |
| `.smooth_taubin` | `surface_mesh` | `vtkWindowedSincPolyDataFilter` (what pyvista calls) |
| `.connectivity()`, `.extract_largest()` | `check_inputs`, `surface_mesh` | `vtkPolyDataConnectivityFilter` |
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
- pyacvd imports pyvista at module level (and pykdtree): pyvista cannot be avoided while pyacvd is used, but it is now confined to `surface_mesh.acvd_remesh` (2026-11-18).
- Seen on the Mac (env `lungct`, 2026-11-18): the pip wheel of **pykdtree** (a pyacvd dependency) bundles its own `libomp.dylib`; with the conda one already loaded the process aborts (`OMP: Error #15`) inside pyacvd, old and new code alike. Fixed there with the conda-forge pykdtree. Slicer bundles its own OpenMP too: check this first when pyacvd is installed in Slicer 5.12.

### 5.7 What gets installed in Slicer 5.12

Prediction path (`geometry` → `hilum` → `fem_setup` → `collapse`), imports checked 2026-11-18:

| Package | Used by | In Slicer 5.12 |
|---|---|---|
| `numpy`, `scipy` | everywhere; SciPy SuperLU = the linear solver (5.2) | already there |
| `vtk` (`vtkmodules`) | `vtk_mesh.py`: every surface operation and file | already there (Slicer's own VTK): **never pip-install `vtk`** |
| `eurelaps_lung` | the package | `pip install` (git or wheel, pinned version, A1) |
| `gmsh` | `fem/volume_mesh.py` (subprocess) | `pip install gmsh`: the wheel bundles the native library, no dependencies. GPL: fine as a runtime pip dependency |
| `pyacvd` + `pykdtree` | `surface_mesh.acvd_remesh` | `pip install pyacvd` (pinned), as Slicer's Surface Toolbox does; check the OpenMP clash of the pykdtree wheel (5.6) |
| `pyvista` | only inside pyacvd | comes with pyacvd. Its pip requirements include `vtk`, plus matplotlib, pillow, pooch, scooby, cyclopts, pyvista-validation: verify that installing it in Slicer 5.12 does **not** pull a second VTK; if it does, `pip install pyvista --no-deps` + its pure-Python requirements |
| `torch` (CPU) | `fem/solvers/torch_solver.py` (the core) | the PyTorch extension of Slicer (SlicerPyTorch) or `pip install torch` (CPU wheel, ~200-300 MB); or nothing if the numpy core is written (C2) |

Not installed in Slicer (not on the prediction path):

| Package | Used by |
|---|---|
| `itk` | `check_inputs`, `raster`, image I/O of `data_io` (fit only; import now lazy in `data_io`) |
| `scikit-learn` | `fem/regions.py`, only with `fem_setup.regions` (import now lazy) |
| `warp-lang`, `getfem`, `nvmath-python` (cuDSS), `pypardiso` | server cores and solvers |

### 5.7b Licenses

Slicer itself is BSD-style and takes no copyleft code; extensions with a more restrictive license than MIT / BSD / Apache must say why, and Slicer maintainers avoid relying on GPL code (a GPL dependency is accepted as a separate executable, documented as such). Licenses from the installed packages' metadata (2026-11-18):

| Package | License | For the extension |
|---|---|---|
| numpy, scipy, VTK, torch | BSD | fine |
| pyacvd | MIT | fine; it is what Slicer's own Surface Toolbox uses for "Uniform remesh" |
| pyvista | MIT | fine |
| pykdtree (pyacvd dependency) | LGPL-3.0+ | fine as a separate pip-installed package (not bundled, not modified) |
| **gmsh** | **GPL-2.0+** | the only copyleft one. We already run it as a separate process (`volume_mesh.py`), installed by pip at the user's request, not bundled: the "separate executable" case. To state in the extension's README; or replace it |

What the Slicer community uses for volume meshes: the **SegmentMesher** extension, with **Cleaver2** (MIT; meshes the segmentation labelmap itself, so it does not keep a given surface triangulation) and **TetGen** (AGPL / commercial license; constrained Delaunay that keeps the surface). Options if gmsh has to go, keeping the surface exactly as the model needs:

| Option | License | Notes |
|---|---|---|
| keep gmsh, separate process | GPL-2.0+ | no change; documented dependency, installed by the user |
| **Netgen** (`netgen-mesher` 6.2, pip wheels for macOS / Linux / Windows; brings `netgen-occt`) | LGPL-2.1 | volume mesh of a given closed surface mesh (`GenerateVolumeMesh`, `maxh` 10, `optsteps3d` 10), surface kept. **Tested on the Mac 2026-11-18 (analysis/results.md 7d):** surface kept on 14 / 14 patients, < 1.5 s, better tetrahedra than gmsh, collapse as close or closer to a fine-mesh reference than gmsh's (3 / 4 patients closer, karl04 equal) |
| TetGen | AGPL / commercial | what SegmentMesher offers; more restrictive than gmsh |
| Cleaver2 | MIT | does not keep our ACVD surface: the surface nodes would change (a method change, as 5.11) |

Why not SegmentMesher itself: (1) it meshes the segmentation, not our ACVD surface, so the surface nodes of the model (wall, clamp, target matching, calibration, the GNN graph) would change; (2) it is a Slicer extension of compiled command-line tools, while the same meshing must run on the server without Slicer (cohort fits, GNN training data): the server and Slicer would build different meshes; (3) its surface-keeping engine is TetGen (AGPL).

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

### 5.11 Surface remesh: pyacvd or gmsh

The mesh is built in two stages, and only the first one is in question:

1. **Surface** (`geometry` step, `surface_mesh.py`): Slicer surface (dense, staircase) → weld, largest component, Taubin smoothing, fill holes → **ACVD** (pyacvd) to ~480 nodes (`target_nodes`, `max_faces` 1000) → light smoothing → watertightness check (retries with 2 % fewer clusters).
2. **Volume** (`fem_setup` step, `fem/volume_mesh.py`): **gmsh** fills that surface with tetrahedra (Delaunay, `MeshSizeMax` 14 mm inside) and must keep the surface triangulation exactly: the code stops if gmsh moves a surface node or changes a triangle. gmsh stays in any case.

The question is who does stage 1:

| | pyacvd (now) | gmsh surface remesh (alternative) |
|---|---|---|
| How | clusters the dense surface into n Voronoi regions of equal area; one node per region | `classifySurfaces` + `createGeometry` re-parametrise the STL as patches, then a frontal / Delaunay 2D mesh at a size in mm |
| Control | number of nodes (`geometry.target_nodes` 480): the same graph size for every patient | element size in mm. The node count can be targeted too: on the cohort's ACVD meshes the mean edge equals h = √(4A / (√3 F)), F = 2N − 4 (A = area; 15–22 mm for N = 480, area 900–1950 cm²), so h from the smoothed surface's area gives ~N nodes, and one correction of h hits it within a few %. Size can also be graded (finer at the hilum) |
| Quality | near-equilateral triangles; sometimes a few non-manifold edges (the retry loop handles it) | good triangles, but the re-parametrisation of a closed, curved surface depends on an angle threshold and can fail or make bad patches on a noisy input |
| Input | robust on the dense segmentation surface | needs the surface smoothed first (the VTK Taubin step stays) |
| Dependencies | pyacvd + pyvista; pyacvd 0.3.0 broke in Slicer (5.6) | none new (gmsh is there for stage 2); one tool for surface and volume |
| Validation | all the cohort's fits, the E rule and the GNN data use these meshes | **new**: different meshes → check volume, ASSD of the fits, the E rule (mesh dependence) before switching |

- Recommendation: keep pyacvd for the extension (pinned version, or the Surface Toolbox's install), and remove pyvista from everything else (5.6). Test the gmsh remesh later on the cohort, as a fallback if pyacvd breaks again in Slicer; switching needs the calibration redone on the new meshes.
- A VTK-only remesh (`vtkQuadricDecimation` + smoothing) is not an option: it gives non-uniform triangles and no control of the node count.

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

Goal: the extension (a separate repository) does `pip install eurelaps_lung==X.Y` into Slicer's Python and calls the package; nothing of the package is copied into the extension. Every change keeps the server runs identical (same files, same numbers): check with `tests/` and one fit + one prediction on the server.

**A. Packaging and API**

| # | Change | Why |
|---|---|---|
| A1 | `pyproject.toml`, version number, install from git or a wheel; dependencies split into extras: core (numpy, scipy, vtk, pyacvd, gmsh), `[torch]`, `[fit]` (scikit-learn), `[images]` (itk), `[server]` (warp-lang, getfem, nvmath, pypardiso) | the extension installs a pinned version and only what the prediction needs |
| A2 | `eurelaps_lung/predict.py`: `run(config, collapse=None, params=None, from_fit=None, frames=None, progress=None)` + `python -m eurelaps_lung.predict`; `main_predict.py` becomes a thin wrapper | a Python API, not only a CLI script |
| A3 | `load_config` from a dict (base folder given) | configs built in memory from the UI |
| A4 | calibration file shipped in the package (`eurelaps_lung/data/E_rule.json`, read by `calibration.load_rule`) | the extension uses the same rule as `physics.E_Pa {"rule": "collapse"}` |
| A5 | progress callback or structured log lines (`step`, `solve i / n`) | progress bar |
| A6 | sequence frames also returned as arrays (`make_sequence` → data + files) | build the Slicer sequence without re-reading files |

**B. Dependencies that Slicer does not have or that clash**

| # | Change | Why |
|---|---|---|
| B1 | pyvista removed: `eurelaps_lung/vtk_mesh.py` with the plain-VTK helpers, surfaces passed as `(points, tris)` or `vtkPolyData` (5.6) | one VTK, Slicer's |
| B2 | pyacvd kept for the surface remesh, version pinned against Slicer's VTK; gmsh remesh tested later as fallback (5.11) | pyacvd is the only real pyvista user |
| B3 | lazy imports of `itk` (`data_io`, `raster`, `check_inputs`) and `scikit-learn` (`fem/setup.py` → `regions`). **Done for `data_io` and `fem/setup.py` (2026-11-18)**; `raster` / `check_inputs` still import itk at the top, but the prediction does not import them | not needed by the prediction |
| B4 | gmsh subprocess: an interpreter that is Python in Slicer (`PythonSlicer`, not the Slicer executable `sys.executable` may point to), or gmsh in-process with a timeout | `volume_mesh.py` runs `sys.executable -c ...` |
| B5 | ~~`from __future__ import annotations`~~ not needed: Slicer 5.12 has Python 3.12 (5.5) | Python 3.9 (Slicer ≤ 5.8) only |

**C. Compute (to be decided with the torch questions)**

| # | Change | Why |
|---|---|---|
| C1 | prediction on CPU float64 + scipy SuperLU as the documented default; no pypardiso / cuDSS / Warp on that path | Mac and Windows (5.1–5.3) |
| C2 | optional: numpy core with the closed-form Neo-Hookean tangent | no torch in Slicer (5.4) |

**D. Inputs**

| # | Change | Why |
|---|---|---|
| D1 | inflated-only input checks (closed, one component, volume range, airways touch the lung) | `check_inputs` needs the collapsed lung (5.10) |
| D2 | airways-only hilum / clamp, if decided (6.2) | vessels may not be segmented |

Order: B1, B3 and A1 first (the package imports in Slicer), then A2–A6 and B4 (the extension can call it), then C and D.

## 8. Phases

1. **v0, a prototype without the package changes.**
   - A scripted module that exports the surfaces, writes a config file and runs `main_predict.py` with `PythonSlicer` in a subprocess.
   - It loads the sequence with the existing `load_collapse_in_slicer.py`.
   - Test: Mac M-series and Linux, torch CPU, one patient.
2. **v1:** the changes of section 7, in-scene output, cache, UI of sections 2–3, input checks, cancel.
3. **v2:** extension packaging (Extensions Index, dependencies installed on first use), tests, Windows.
4. **v3:** the GNN in place of the forward solve (MPS / CUDA), interactive slider on the collapse %.
