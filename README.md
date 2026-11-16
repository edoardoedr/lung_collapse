# Lung collapse pipeline

A finite-element model of the collapse of a lung in VATS surgery, built from CT segmentations. Given the inflated lung, it predicts the collapsed one from a few physical parameters (stiffness, density, patient position, how much the lung collapses); given also the collapsed lung, it fits those parameters to it. The simulations are the training data for a GNN surrogate.

**The model** (details and equations in [METHODOLOGY.md](METHODOLOGY.md)): the inflated lung is an elastic body, fixed at the hilum. In the pneumothorax it loses the pleural pressure that kept it open, and with it the pressure gradient that carried its weight, so it shrinks and settles on the chest wall:

```
pressure on the surface / E  =  q0  +  γ (h − h_mean),      γ = density · g / E
```

| Parameter | Config key | Meaning |
|---|---|---|
| E | `physics.E_Pa` | effective stiffness of the lung tissue [Pa] |
| ν | `physics.material.nu` | compressibility (the lung loses air) |
| density | `physics.tissue_density` | mean density of the inflated lung [kg/m³] |
| position | `physics.gravity` | patient position in the inflated CT and during the collapse (sets "up": h) |
| q0 | `physics.collapse` | uniform pressure / E: it sets how much the lung collapses. Given as `q0`, as `p0_Pa`, or as the `volume_ratio` to reach |
| wall | `fem_setup.wall`, `physics.wall` | the chest cavity the lung may not leave |

Each physical parameter is either **given** (a number) or **fitted** to the collapsed lung (`{"fit": true, "init": ..., "bounds": [...]}`).

![Physical model](collapse_model.svg)

## Steps

![Pipeline overview](collapsed_lung_project_workflow.svg)

| Step | Code | What it does | Main output |
|---|---|---|---|
| segmentation | manual, 3D Slicer | surfaces of the lungs, airways and vessels | `<data_dir>/*.vtk` |
| `check_inputs` | [check_inputs.py](pipeline/check_inputs.py) | finds segmentation problems before they break the meshing (needs the collapsed lung) | `checks/` |
| `geometry` | [surface_mesh.py](pipeline/surface_mesh.py) | clean, uniform surface meshes (~480 nodes) of the inflated lung and, if given, of the collapsed one | `lung_inflated_mesh.vtp`, `lung_collapsed_mesh.vtp` |
| `hilum` | [hilum.py](pipeline/hilum.py) | where airways, arteries and veins enter the lung | `hilum_anchor.mrk.json` |
| `fem_setup` | [collapse/setup.py](pipeline/collapse/setup.py) | clamp at the hilum, chest wall, tetrahedral mesh | `fem/setup/problem.npz` |
| `collapse` | [collapse/forward.py](pipeline/collapse/forward.py) | **forward:** all parameters given → collapsed lung | `runs/<run_name>/` |
| `fit` | [collapse/fit.py](pipeline/collapse/fit.py) | **inverse:** fits the parameters marked `fit` to the collapsed lung | `runs/<run_name>/` |

## Setup

```bash
pip install -r requirements.txt       # Python >= 3.10
```

The `collapse` and `fit` steps also need the library of the FEM core chosen in `solver.core`:

- **PyTorch** (default): `pip install torch` (≥ 2.0), optionally `"nvmath-python[cu12]"` (cuDSS on GPU) and `pypardiso`.
- **Warp:** `pip install warp-lang`, same optional linear solvers.
- **GetFEM** (validation only): `conda install -c conda-forge getfem`.

## Running

```bash
python main.py --config configs/karl04.json                       # the steps listed in the config
python main.py --config configs/karl04.json --steps fit           # only some steps
```

One-off changes without editing the config: `--set key=value ...` (value parsed as JSON; a misspelt key is an error). Give each variant its own `run_name`, so it goes to its own folder:

```bash
# fit the collapse pressure with another stiffness
python main.py --config configs/karl04.json --steps fit --set physics.E_Pa=1000 run_name=E1000

# fit the stiffness too
python main.py --config configs/karl04.json --steps fit \
    --set 'physics.E_Pa={"fit": true, "init": 1700, "bounds": [300, 10000]}' run_name=fitE

# predict: collapse to half the volume, patient on the left side, with a sequence for Slicer
python main.py --config configs/karl04.json --steps collapse \
    --set 'physics.collapse={"volume_ratio": 0.5}' 'physics.gravity={"reference": "supine", "collapse": "left_lateral"}' \
          collapse.sequence_frames=30 run_name=half_lateral
```

Several patients: `JOBS=4 bash scripts/run_all.sh [case ...] [-- main.py arguments]` (logs in `logs_run_all/`). One parameter shared by all patients, scanned over values: `python scripts/fit_cohort.py --configs configs/karl*.json --param physics.E_Pa --values 1000 1700 3000`. A table of all runs: `python scripts/summarize_runs.py <results folder>`.

Steps always run in pipeline order. If a step fails, the error is in `logs/pipeline.log` and the run stops.

## Input data

One folder per patient (`data_dir`), surface models exported from 3D Slicer (`.vtk`), all from the same scan: the inflated lung (`lung_left.vtk`), the airway, artery and vein trees, and optionally the collapsed lung (`lung_left_collapsed.vtk`). Without the collapsed lung set `inputs.collapsed` to `null`, `hilum.lung` to `"inflated"` and use the `collapse` step only.

## Configuration

One JSON per patient in [`configs/`](configs/); **every key is explained in [`configs/README.md`](configs/README.md).** In short:

| Section | Content |
|---|---|
| `inputs` | the surface files |
| `check_inputs`, `geometry`, `hilum`, `fem_setup` | how the model is built (mesh sizes, clamp, wall) |
| `physics` | the model's parameters, given or fitted |
| `solver` | FEM core (`torch`, `warp`, `getfem`) and its numerical options |
| `collapse`, `fit` | options of the forward and of the inverse run |
| `run_name` | folder of this run under `runs/` |

## Results

```
<output_dir>/
├── lung_inflated_mesh.vtp, lung_collapsed_mesh.vtp    surface meshes (geometry)
├── hilum_anchor.mrk.json                              hilum + airway / artery / vein entries
├── checks/  hilum/
├── fem/setup/           problem.npz, reference.vtp, target.vtp, volume.vtu, setup_summary.json
├── runs/<run_name>/     surface.vtp     collapsed lung: Displacement_mm, Pressure_Pa, Error_mm, WallPenetration_mm
│                        summary.json    physical parameters (E, p0, q0, ν, gravity), collapse %, errors, solves
│                        state.npz       the solution, to re-solve it (sequence)
│                        sequence/       collapse sequence for Slicer / ParaView
│                        history.csv, best_state.npz, best_params.json    (fit)
└── logs/                pipeline.log, config used
```

All surfaces are in LPS, with the space stored in the file.

## Viewing

**Animated:** `python scripts/collapse_sequence.py --config configs/<p>.json [--set run_name=...]` (or `collapse.sequence_frames` in a forward run) writes `runs/<run_name>/sequence/`. Copy the folder to the machine with Slicer and open it:

```bash
/Applications/Slicer.app/Contents/MacOS/Slicer --python-script <folder>/load_collapse_in_slicer.py
```

or, with Slicer open, in its Python console:

```python
p = '<folder>/load_collapse_in_slicer.py'
exec(open(p).read(), {'__file__': p})
```

**Several runs side by side,** in Slicer's Python console:

```python
exec(open('<repo>/scripts/slicer_compare_sequences.py').read())
load('<output_dir>/runs/fit/sequence', '<output_dir>/runs/E1000/sequence')
```

**Static:** drag `runs/<run_name>/surface.vtp` and `fem/setup/target.vtp` into Slicer and colour the surface by `Error_mm`.

## More

| File | Content |
|---|---|
| [configs/README.md](configs/README.md) | every config key |
| [METHODOLOGY.md](METHODOLOGY.md) | the model, the equations, the solver, the fit, what is not modelled (with the [forward / inverse figure](inverse_fem_pressure_fit_detail.svg)) |
| [analysis/results.md](analysis/results.md) | results and the evidence behind the choices |
| [NEXT.md](NEXT.md) | plan and to-do |
| [CLAUDE.md](CLAUDE.md) | development notes: code architecture, validation, tests |
