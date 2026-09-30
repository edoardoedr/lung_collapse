"""Pipeline configuration: one JSON file per patient, one section per step.

Path rules:
  - data_dir / output_dir and settings files (e.g. elastix parameter files) are relative
    to the JSON file itself;
  - step inputs are relative to data_dir, step outputs relative to output_dir;
    an input left null is taken from the previous step's output;
  - absolute paths are used as given.
Keys starting with "_" are ignored, so they can be used as comments.
"""

import json
from dataclasses import dataclass, field, fields
from pathlib import Path

SPACES = ("LPS", "RAS")


@dataclass
class SurfaceMeshConfig:
    """Step 2: Slicer segmentation surface -> uniform, watertight triangle mesh."""
    input: Path
    output: Path
    input_space: str = "auto"          # auto (read file tag, else LPS) | LPS | RAS
    output_space: str = "LPS"
    merge_tolerance: float = 1e-5      # [mm] welding of duplicate points
    pre_smooth_iters: int = 15         # Taubin, removes the segmentation staircase
    post_smooth_iters: int = 5         # Taubin, relaxes ACVD cluster boundaries
    smooth_pass_band: float = 0.1
    target_nodes: int = 480            # ACVD clusters ~= output nodes (closed surface: faces = 2*nodes - 4)
    max_faces: int = 1000              # hard ceiling; target_nodes is shrunk until it holds
    shrink_factor: float = 0.9
    max_remesh_attempts: int = 6
    min_points_per_cluster: int = 12   # source is subdivided until it is this dense
    max_subdivisions: int = 4
    hole_size: float = 1e4             # [mm] fill_holes size

    def __post_init__(self):
        if self.input_space not in ("auto", *SPACES):
            raise ValueError("surface_mesh.input_space must be auto, LPS or RAS")
        if self.output_space not in SPACES:
            raise ValueError("surface_mesh.output_space must be LPS or RAS")


@dataclass
class RegistrationConfig:
    """Step 3: elastix mask registration, collapsed surface -> inflated space (same topology).

    inflated / collapsed: labelmap (.nrrd, .nii, .nii.gz, .mha, .mhd) or closed surface (rasterised)."""
    inflated: Path                     # inflated lung (data_dir)
    output: Path                       # warped surface (output_dir)
    surface: Path | None = None        # collapsed surface to warp; None = output of surface_mesh
    surface_space: str = "auto"
    collapsed: Path | None = None      # collapsed lung; None = rasterise `surface`
    inflated_label: int | None = None  # label value in a labelmap; None = any voxel > 0
    collapsed_label: int | None = None
    direction: str = "inverse"         # inverse: fixed = inflated (Slicer) | forward: fixed = collapsed
    raster_spacing_mm: float = 1.0     # grid for rasterised surfaces
    crop_margin_mm: float = 20.0
    parameter_maps: list = field(default_factory=lambda: ["rigid", "bspline"])  # elastix defaults
    parameter_files: list = field(default_factory=list)   # elastix .txt files, override parameter_maps
    parameter_overrides: dict = field(default_factory=dict)  # applied to every map
    random_seed: int = 42
    inversion_tol_mm: float = 1e-3
    inversion_max_iter: int = 100
    inversion_samples: int = 200000    # lookup grid for the inversion start points
    output_space: str = "LPS"
    workdir: Path | None = None        # elastix files; set to output_dir/registration

    def __post_init__(self):
        if self.surface_space not in ("auto", *SPACES):
            raise ValueError("registration.surface_space must be auto, LPS or RAS")
        if self.direction not in ("forward", "inverse"):
            raise ValueError("registration.direction must be forward or inverse")
        if self.output_space not in SPACES:
            raise ValueError("registration.output_space must be LPS or RAS")


@dataclass
class PipelineConfig:
    patient: str
    data_dir: Path
    output_dir: Path
    steps: list
    source: Path                       # the JSON file this was read from
    surface_mesh: SurfaceMeshConfig | None = None
    registration: RegistrationConfig | None = None


def _section(cls, raw, name):
    """Build a step dataclass from its JSON dict, rejecting unknown keys (typos)."""
    raw = {k: v for k, v in raw.items() if not k.startswith("_")}
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError("unknown key(s) in '%s': %s" % (name, ", ".join(unknown)))
    try:
        return cls(**raw)
    except TypeError as e:
        raise ValueError("section '%s': %s" % (name, e)) from None


def load_config(path):
    path = Path(path).resolve()
    raw = json.loads(path.read_text())
    base = path.parent
    data_dir = (base / raw["data_dir"]).resolve()
    output_dir = (base / raw["output_dir"]).resolve()

    surface_mesh = None
    if "surface_mesh" in raw:
        surface_mesh = _section(SurfaceMeshConfig, raw["surface_mesh"], "surface_mesh")
        surface_mesh.input = data_dir / surface_mesh.input
        surface_mesh.output = output_dir / surface_mesh.output

    registration = None
    if "registration" in raw:
        registration = _section(RegistrationConfig, raw["registration"], "registration")
        r = registration
        r.inflated = data_dir / r.inflated
        r.output = output_dir / r.output
        r.workdir = output_dir / "registration"
        if r.collapsed is not None:
            r.collapsed = data_dir / r.collapsed
        if r.surface is not None:
            r.surface = data_dir / r.surface
        elif surface_mesh is not None:
            r.surface = surface_mesh.output
        else:
            raise ValueError("registration.surface is required when there is no surface_mesh section")
        r.parameter_files = [base / f for f in r.parameter_files]   # relative to the JSON file

    return PipelineConfig(patient=raw["patient"], data_dir=data_dir, output_dir=output_dir,
                          steps=list(raw.get("steps", [])), source=path,
                          surface_mesh=surface_mesh, registration=registration)
