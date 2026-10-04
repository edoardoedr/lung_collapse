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
class CheckInputsConfig:
    """Step 1b: sanity check of the input surfaces (output_dir/checks)."""
    collapsed: Path | None = None      # collapsed lung (data_dir); None = surface_mesh.input
    inflated: Path | None = None       # inflated lung (data_dir); None = registration.inflated
    structures: dict | None = None     # {name: surface} (data_dir); None = hilum.structures
    outside_tol_mm: float = 1.0        # collapsed points farther outside the inflated lung are flagged
    max_outside_fraction: float = 0.01 # warn above this fraction of flagged points
    notch_radius_mm: float = 5.0       # closing ball: notches narrower than ~2x this are found
    notch_depth_mm: float = 3.0        # points deeper than this in a notch are flagged
    max_notch_fraction: float = 0.005  # warn above this fraction of flagged points
    max_volume_ratio: float = 5.0      # warn if inflated / collapsed volume is larger
    raster_spacing_mm: float = 1.0     # voxel size of the notch check
    fail_on_warning: bool = False      # stop the pipeline on any finding
    workdir: Path | None = None        # set to output_dir/checks


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
    """Step 4: elastix mask registration, collapsed surface -> inflated space (same topology).

    inflated / collapsed: labelmap (.nrrd, .nii, .nii.gz, .mha, .mhd) or closed surface (rasterised)."""
    inflated: Path                     # inflated lung (data_dir)
    output: Path                       # warped surface (output_dir)
    surface: Path | None = None        # collapsed surface to warp; None = output of surface_mesh
    surface_space: str = "auto"
    collapsed: Path | None = None      # collapsed lung; None = rasterise `surface`
    inflated_label: int | None = None  # label value in a labelmap; None = any voxel > 0
    collapsed_label: int | None = None
    direction: str = "inverse"         # inverse: fixed = inflated (Slicer) | forward: fixed = collapsed
    landmarks: Path | None = None      # .mrk.json; None = output of the hilum step (if any)
    landmark_points: list = field(default_factory=lambda: ["hilum"])   # labels used as landmarks
    landmark_weight: float = 0.0       # weight of the landmark metric; 0 = landmarks only for QA
    hilum_region_mm: float = 30.0      # QA: displacement of the nodes this close to the hilum
    raster_spacing_mm: float = 1.0     # grid for rasterised surfaces
    crop_margin_mm: float = 20.0
    parameter_maps: list = field(default_factory=lambda: ["rigid", "bspline"])  # elastix defaults
    parameter_files: list = field(default_factory=list)   # elastix .txt files, override parameter_maps
    parameter_overrides: dict = field(default_factory=dict)  # applied to every map
    random_seed: int = 42
    inversion_tol_mm: float = 1e-3
    inversion_max_iter: int = 100
    inversion_samples: int = 200000    # lookup grid for the inversion start points
    max_interpolated_nodes: int = 10   # non-converged nodes interpolated from neighbours; more -> fail
    max_unflip_nodes: int = 40         # nodes re-interpolated to remove flipped triangles; more -> fail
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
class HilumConfig:
    """Step 3: hilum = centroid of the rings where airways, arteries and veins enter the lung."""
    lung: Path                         # lung surface the trees enter (data_dir), same scan as the trees
    structures: dict                   # {name: surface file} (data_dir), e.g. airways/arteries/veins
    output: Path                       # Slicer .mrk.json (output_dir)
    max_ring_distance_mm: float = 50.0 # warn if a hilar ring centre is farther than this from the hilum


@dataclass
class FemSetupConfig:
    """Step 5a: solver-independent preparation of the inverse FEM (output_dir/fem/setup)."""
    reference: Path | None = None      # inflated surface (data_dir); None = registration output
    target: Path | None = None         # collapsed surface, same nodes (data_dir); None = surface_mesh output
    anchor: Path | None = None         # .mrk.json (data_dir); None = hilum output
    anchor_point: str = "hilum"        # point whose sphere centres the clamped region
    anchor_radius_factor: float = 2.0  # clamp radius = sphere radius of anchor_point x this
    anchor_min_points: int = 20        # clamp radius grows until it holds this many points (rank 3)
    anchor_growth: float = 1.1
    anchor_max_growth_steps: int = 10
    wall: Path | str | None = None     # cavity the lung may not leave: "reference" = the registered inflated
                                       # surface itself, or a closed surface file (data_dir); None = no wall
    wall_tol_mm: float = 2.0           # allowed motion beyond the wall (registration noise margin)
    align: str = "hilum_rigid"        # rigid alignment of the target: hilum_rigid | rigid | none
    cluster_field: str = "rigid_residual"   # field clustered into regions: rigid_residual | hilum_rigid | raw
    feature: str = "normal"            # Ward feature: normal (u.n) | vector (u)
    pos_weight: float = 0.3            # spatial compactness in Ward
    levels: list = field(default_factory=lambda: [1, 4, 8, 14, 25, 40])   # regions, coarse -> fine
    mesh_size_mm: float = 8.0          # Gmsh MeshSizeMax (interior; the surface is kept)
    gmsh_timeout_s: int = 300
    workdir: Path | None = None        # set to output_dir/fem/setup

    def __post_init__(self):
        if self.align not in ("hilum_rigid", "rigid", "none"):
            raise ValueError("fem_setup.align must be hilum_rigid, rigid or none")
        if self.cluster_field not in ("rigid_residual", "hilum_rigid", "raw"):
            raise ValueError("fem_setup.cluster_field must be rigid_residual, hilum_rigid or raw")
        if self.feature not in ("normal", "vector"):
            raise ValueError("fem_setup.feature must be normal or vector")


@dataclass
class FemFitConfig:
    """Step 5b: regional pressures fitted with an exchangeable FEM core (reads fem_setup output)."""
    output: Path                       # fitted surface (output_dir)
    solver: str = "getfem"             # core in pipeline/collapse/solvers, or "module:Class"
    solver_options: dict = field(default_factory=dict)   # passed to the core (see its docstring)
    E_Pa: float = 3000.0               # only scales the reported pressures: shapes fix p/E only
    nu: float = 0.30                   # Poisson's ratio (start value if free_nu)
    free_nu: bool = True               # optimise nu too
    nu_bounds: list = field(default_factory=lambda: [0.05, 0.40])
    levels: list | None = None         # subset of the fem_setup levels (K values); None = all
    optimizer: str = "lsq"             # lsq (bounded least squares, TRF) | nm (Nelder-Mead)
    jacobian: str = "analytic"         # lsq: analytic (from the core; finite differences if it has none) | 2-point
    lsq_diff_step: float = 5e-3        # lsq: relative finite-difference step for the Jacobian
    lsq_ftol: float = 1e-6             # lsq: stop on relative change of the cost
    lsq_xtol: float = 1e-6             # lsq: stop on relative change of the parameters
    lsq_gtol: float = 1e-8             # lsq: stop on the gradient norm
    q0: float = 0.5                    # initial p/E
    q_bounds: list = field(default_factory=lambda: [-1.0, 3.0])
    reg: float = 1.0                   # smoothness between adjacent regions [mm per unit p/E]
    sign_check_q: float = 0.2          # p/E of the initial check that q > 0 collapses
    target_error_mm: float = 2.5       # stop when the mean error reaches this
    time_budget_min: float = 240.0
    hard_grace_min: float = 10.0       # watchdog kills the run at budget + grace
    # residual per surface point, d = fitted - target: "point" = d (point-to-point); "plane" = its
    # component along the target normal n, plus loss_tangent_weight x the tangential rest, i.e.
    # (n n^T + w (I - n n^T)) d. "plane" ignores sliding along the surface, so a correspondence
    # that slid tangentially in the registration (it looks like a rotation) is not forced
    loss: str = "point"
    loss_tangent_weight: float = 0.0
    # per level: move on to the next K when the level's best error improved by less than
    # level_min_improve (relative) over the last level_patience optimiser iterations
    # (lsq: one Jacobian each; Nelder-Mead: parameters + 1 forward solves each)
    level_patience: int = 3
    level_min_improve: float = 0.005
    # whole fit: stop when fit_patience levels in a row improve the best error by less than fit_min_improve
    fit_patience: int = 2
    fit_min_improve: float = 0.02
    run_name: str = "fit"              # sub-folder of output_dir/fem for this fit: several fits
                                       # (e.g. other cores) can share one fem_setup
    setup: Path | None = None          # set to output_dir/fem/setup
    workdir: Path | None = None        # set to output_dir/fem/<run_name>

    def __post_init__(self):
        if self.optimizer not in ("lsq", "nm"):
            raise ValueError("fem_fit.optimizer must be lsq or nm")
        if self.jacobian not in ("analytic", "2-point"):
            raise ValueError("fem_fit.jacobian must be analytic or 2-point")
        if self.loss not in ("point", "plane"):
            raise ValueError("fem_fit.loss must be point or plane")
        if not 0.0 <= self.loss_tangent_weight <= 1.0:
            raise ValueError("fem_fit.loss_tangent_weight must be in [0, 1]")


@dataclass
class PipelineConfig:
    patient: str
    data_dir: Path
    output_dir: Path
    steps: list
    source: Path                       # the JSON file this was read from
    check_inputs: CheckInputsConfig | None = None
    surface_mesh: SurfaceMeshConfig | None = None
    registration: RegistrationConfig | None = None
    hilum: HilumConfig | None = None
    fem_setup: FemSetupConfig | None = None
    fem_fit: FemFitConfig | None = None


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


def apply_overrides(raw, overrides):
    """overrides: ["section.key=value", ...] (value parsed as JSON, else taken as a string),
    e.g. fem_fit.loss=plane, fem_fit.solver_options.wall_update="outer", fem_fit.levels=[1,4]."""
    for item in overrides or ():
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError("--set %s: expected key=value" % item)
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass
        *parents, last = key.split(".")
        node = raw
        for k in parents:
            if not isinstance(node.get(k), dict):
                raise ValueError("--set %s: '%s' is not a section of the config" % (item, k))
            node = node[k]
        node[last] = value                     # unknown keys are rejected later, as in the file
    return raw


def load_config(path, overrides=None):
    path = Path(path).resolve()
    raw = apply_overrides(json.loads(path.read_text()), overrides)
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

    hilum = None
    if "hilum" in raw:
        hilum = _section(HilumConfig, raw["hilum"], "hilum")
        hilum.lung = data_dir / hilum.lung
        hilum.structures = {k: data_dir / v for k, v in hilum.structures.items()}
        hilum.output = output_dir / hilum.output

    if registration is not None:
        if registration.landmarks is not None:
            registration.landmarks = data_dir / registration.landmarks
        elif hilum is not None:
            registration.landmarks = hilum.output

    fem_setup = None
    if "fem_setup" in raw:
        fem_setup = s = _section(FemSetupConfig, raw["fem_setup"], "fem_setup")
        s.workdir = output_dir / "fem" / "setup"
        if s.wall is not None and s.wall != "reference":
            s.wall = data_dir / s.wall
        for key, previous in (("reference", registration), ("target", surface_mesh), ("anchor", hilum)):
            value = getattr(s, key)
            if value is not None:
                setattr(s, key, data_dir / value)
            elif previous is not None:
                setattr(s, key, previous.output)
            else:
                raise ValueError("fem_setup.%s is required when there is no step providing it" % key)

    check_inputs = None
    if "check_inputs" in raw:
        check_inputs = c = _section(CheckInputsConfig, raw["check_inputs"], "check_inputs")
        c.workdir = output_dir / "checks"
        c.collapsed = data_dir / c.collapsed if c.collapsed is not None else (surface_mesh and surface_mesh.input)
        c.inflated = data_dir / c.inflated if c.inflated is not None else (registration and registration.inflated)
        if c.structures is not None:
            c.structures = {k: data_dir / v for k, v in c.structures.items()}
        else:
            c.structures = dict(hilum.structures) if hilum is not None else {}
        if c.collapsed is None or c.inflated is None:
            raise ValueError("check_inputs: collapsed and inflated are required without surface_mesh / registration")

    fem_fit = None
    if "fem_fit" in raw:
        fem_fit = _section(FemFitConfig, raw["fem_fit"], "fem_fit")
        fem_fit.output = output_dir / fem_fit.output
        fem_fit.setup = output_dir / "fem" / "setup"
        if fem_fit.run_name in ("", "setup") or "/" in fem_fit.run_name or "\\" in fem_fit.run_name:
            raise ValueError("fem_fit.run_name must be a plain folder name other than 'setup'")
        fem_fit.workdir = output_dir / "fem" / fem_fit.run_name

    return PipelineConfig(patient=raw["patient"], data_dir=data_dir, output_dir=output_dir,
                          steps=list(raw.get("steps", [])), source=path, check_inputs=check_inputs,
                          surface_mesh=surface_mesh, registration=registration, hilum=hilum,
                          fem_setup=fem_setup, fem_fit=fem_fit)
