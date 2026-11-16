"""Pipeline configuration: one JSON file per patient.

    inputs      the segmented surfaces (inflated lung, optional collapsed lung, airways / vessels)
    geometry, hilum, fem_setup    how the model is built from them
    physics     the collapse model: material, E, density, gravity, uniform pressure, wall
    solver      the FEM core and its numerical options
    collapse    forward run (all physics given) -> collapsed lung
    fit         inverse run (the physics parameters marked "fit" are fitted to the collapsed lung)

A physical parameter (physics.E_Pa, physics.material.nu, physics.collapse.q0) is either a number
(fixed) or {"fit": true, "init": ..., "bounds": [lo, hi]} (fitted by the fit step).

Path rules: data_dir / output_dir are relative to the JSON file; inputs are relative to data_dir,
outputs to output_dir; absolute paths are used as given. Keys starting with "_" are comments.
Unknown keys are errors.
"""

import json
from dataclasses import dataclass, field, fields
from pathlib import Path

SPACES = ("LPS", "RAS")

# direction of gravity in LPS (x: to the patient's left, y: posterior, z: superior)
POSITIONS = {"supine": (0.0, 1.0, 0.0), "prone": (0.0, -1.0, 0.0),
             "left_lateral": (1.0, 0.0, 0.0), "right_lateral": (-1.0, 0.0, 0.0)}
AXES = ("ap", "si", "ml")


# ------------------------------------------------------------------------------------------------
# steps before the FEM
# ------------------------------------------------------------------------------------------------
@dataclass
class CheckInputsConfig:
    """Sanity check of the input surfaces (output_dir/checks). Needs the collapsed lung."""
    outside_tol_mm: float = 1.0        # collapsed points farther outside the inflated lung are flagged
    max_outside_fraction: float = 0.01 # warn above this fraction of flagged points
    notch_radius_mm: float = 5.0       # closing ball: notches narrower than ~2x this are found
    notch_depth_mm: float = 3.0        # points deeper than this in a notch are flagged
    max_notch_fraction: float = 0.005  # warn above this fraction of flagged points
    max_volume_ratio: float = 5.0      # warn if inflated / collapsed volume is larger
    raster_spacing_mm: float = 1.0     # voxel size of the notch check
    fail_on_warning: bool = False      # stop the pipeline on any finding
    collapsed: Path | None = None      # set from inputs
    inflated: Path | None = None
    structures: dict | None = None
    workdir: Path | None = None        # set to output_dir/checks


@dataclass
class SurfaceMeshConfig:
    """Slicer segmentation surface -> uniform, watertight triangle mesh (geometry step)."""
    input_space: str = "auto"          # auto (read file tag, else LPS) | LPS | RAS
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
    inflated_output: str = "lung_inflated_mesh.vtp"     # the model's surface (output_dir)
    collapsed_output: str = "lung_collapsed_mesh.vtp"   # the target surface (output_dir)
    output_space: str = "LPS"          # the collapse steps work in LPS
    inflated: Path | None = None       # set from inputs
    collapsed: Path | None = None
    outputs: dict | None = None        # set: {"inflated": path, "collapsed": path}

    def __post_init__(self):
        if self.input_space not in ("auto", *SPACES):
            raise ValueError("geometry.input_space must be auto, LPS or RAS")
        if self.output_space != "LPS":
            raise ValueError("geometry.output_space must be LPS (the collapse steps work in LPS)")


@dataclass
class HilumConfig:
    """Hilum = centroid of the rings where airways, arteries and veins enter the lung."""
    lung: str | Path | None = None     # "collapsed" | "inflated": the lung surface the trees enter;
                                       # None = collapsed if there is one, else inflated
    output: str | Path = "hilum_anchor.mrk.json"   # Slicer .mrk.json (output_dir)
    max_ring_distance_mm: float = 50.0 # warn if a hilar ring centre is farther than this from the hilum
    structures: dict | None = None     # set from inputs


@dataclass
class FemSetupConfig:
    """Model geometry for the FEM, independent of the core (output_dir/fem/setup/problem.npz)."""
    # clamp: surface triangles within (sphere radius x radius_factor) of each anchor point
    clamp_points: list = field(default_factory=lambda: ["hilum", "airways", "arteries", "veins"])
    clamp_radius_factor: float = 2.0
    clamp_min_points: int = 20         # the first ball grows until the clamp holds this many points (rank 3)
    clamp_growth: float = 1.1
    clamp_max_growth_steps: int = 10
    mesh_size_mm: float = 14.0         # Gmsh MeshSizeMax (interior; the surface is kept)
    gmsh_timeout_s: int = 300
    # wall: cavity the lung may not leave. "reference" = the inflated surface itself, a file name
    # (data_dir) = that closed surface, None = no wall
    wall: Path | str | None = "reference"
    wall_tol_mm: float = 5.0           # allowed motion beyond the wall (segmentation margin)
    wall_grid_mm: float | None = 1.5   # the wall as a smooth field (solver wall_update "potential"):
                                       # grid spacing of the signed distance; None = not built
    wall_grid_margin_mm: float = 30.0  # grid = wall bounding box + this (more than any outward motion)
    wall_smoothing_mm: float = 3.0     # Gaussian sigma: rounds the wall's corners by a few mm
    # Ward pressure regions of the target displacement (only for physics.perturbation regions;
    # needs the collapsed lung): numbers of regions, coarse -> fine. None = not computed
    regions: list | None = None
    regions_pos_weight: float = 0.3    # spatial compactness in Ward
    reference: Path | None = None      # set: geometry outputs, hilum output
    target: Path | None = None
    anchor: Path | None = None
    workdir: Path | None = None        # set to output_dir/fem/setup

    def __post_init__(self):
        if self.wall_grid_mm is not None and (self.wall_grid_mm <= 0 or self.wall_grid_margin_mm <= 0
                                              or self.wall_smoothing_mm < 0):
            raise ValueError("fem_setup.wall: grid_mm and grid_margin_mm must be > 0, smoothing_mm >= 0")


# ------------------------------------------------------------------------------------------------
# physics
# ------------------------------------------------------------------------------------------------
@dataclass
class Param:
    """A physical parameter: fixed at `value`, or fitted from `value` (the initial one) within bounds."""
    name: str
    value: float
    fit: bool = False
    bounds: tuple = (-float("inf"), float("inf"))

    @classmethod
    def parse(cls, name, raw, bounds):
        """raw: number, or {"fit": bool, "init" | "value": number, "bounds": [lo, hi]}."""
        if isinstance(raw, bool) or not isinstance(raw, (int, float, dict)):
            raise ValueError("%s: a number, or {\"fit\": true, \"init\": ..., \"bounds\": [lo, hi]}" % name)
        if not isinstance(raw, dict):
            return cls(name, float(raw), False, tuple(bounds))
        raw = {k: v for k, v in raw.items() if not k.startswith("_")}
        unknown = sorted(set(raw) - {"fit", "init", "value", "bounds"})
        if unknown:
            raise ValueError("%s: unknown key(s) %s (fit, init, bounds)" % (name, ", ".join(unknown)))
        if "init" in raw and "value" in raw:
            raise ValueError("%s: give init or value, not both" % name)
        if "init" not in raw and "value" not in raw:
            raise ValueError("%s: needs init (the start value of the fit)" % name)
        b = tuple(float(v) for v in raw.get("bounds", bounds))
        v = float(raw.get("init", raw.get("value")))
        fit = bool(raw.get("fit", True))
        if len(b) != 2 or not b[0] < b[1]:
            raise ValueError("%s.bounds: [lo, hi] with lo < hi" % name)
        if fit and not b[0] <= v <= b[1]:
            raise ValueError("%s: init %g outside its bounds %s" % (name, v, list(b)))
        return cls(name, v, fit, b)


@dataclass
class PhysicsConfig:
    """The collapse model (METHODOLOGY.md): compressible hyperelastic lung, clamped at the hilum,
    loaded by the pleural pressure it loses in the pneumothorax,

        p / E = q0 + gamma (h - h_mean) [+ perturbation],     gamma = tissue_density g / E,

    h = height against gravity in the reference position, plus the wall as contact. With different
    reference and collapse positions the weight's change is a body force gamma (e_collapse - e_reference)."""
    material: str = "neo_hookean"      # solvers/materials.py (torch core); the other cores: neo_hookean
    material_params: dict = field(default_factory=dict)
    nu: Param = None                   # Poisson's ratio
    E_Pa: Param = None                 # effective Young's modulus of the parenchyma [Pa]
    tissue_density: float = 250.0      # mean density of the inflated lung [kg/m3] (assumed: no CT HU)
    gravity: dict | None = None        # None = no gravity term, else {"reference": e_A, "collapse": e_B,
                                       # "names": (..)}: unit vectors of gravity in LPS
    q0: Param | None = None            # uniform p/E (the released recoil pressure / E)
    p0_Pa: float | None = None         # ... or given in Pa (q0 = p0 / E), forward only
    volume_ratio: float | None = None  # ... or found so that V_collapsed / V_inflated is this, forward only
    wall_stiffness: float = 20.0       # wall penalty [E per mm of penetration]
    wall_eps: float = 0.5              # width [mm] of the smooth start of the penalty
    # diagnostic perturbation on top of the physical load, fit only: free linear slopes along
    # anatomical axes (ap + = anterior, si + = superior, ml + = lateral) or one p/E per Ward region
    gradient_axes: list = field(default_factory=list)
    regions: list | None = None        # region counts (levels, coarse -> fine) among fem_setup.regions
    regions_reg: float = 1.0           # smoothness between adjacent regions [mm per unit p/E]

    @property
    def fitted(self):
        """Names of what the fit step fits."""
        out = [p.name for p in (self.q0, self.E_Pa, self.nu) if p is not None and p.fit]
        return out + ["slope_" + a for a in self.gradient_axes] + (["regions"] if self.regions else [])


def _position(name, v):
    if isinstance(v, str):
        if v not in POSITIONS:
            raise ValueError("%s: one of %s, or a vector [x, y, z] in LPS" % (name, ", ".join(POSITIONS)))
        return POSITIONS[v], v
    try:
        vec = tuple(float(c) for c in v)
    except TypeError:
        vec = ()
    n = sum(c * c for c in vec) ** 0.5
    if len(vec) != 3 or n == 0:
        raise ValueError("%s: one of %s, or a non-zero vector [x, y, z] in LPS" % (name, ", ".join(POSITIONS)))
    return tuple(c / n for c in vec), "custom"


def _clean(raw, name, known):
    """raw dict without comment keys, unknown keys rejected."""
    if not isinstance(raw, dict):
        raise ValueError("'%s' must be a section {...}" % name)
    raw = {k: v for k, v in raw.items() if not k.startswith("_")}
    unknown = sorted(set(raw) - set(known))
    if unknown:
        raise ValueError("unknown key(s) in '%s': %s" % (name, ", ".join(unknown)))
    return raw


def _physics(raw):
    raw = _clean(raw, "physics", ("material", "E_Pa", "tissue_density", "gravity", "collapse", "wall", "perturbation"))
    p = PhysicsConfig()
    mat = _clean(raw.get("material", {}), "physics.material", ("name", "nu", "params"))
    p.material, p.material_params = mat.get("name", "neo_hookean"), dict(mat.get("params", {}))
    p.nu = Param.parse("physics.material.nu", mat.get("nu", 0.40), (0.05, 0.45))
    p.E_Pa = Param.parse("physics.E_Pa", raw.get("E_Pa", 1700.0), (300.0, 10000.0))
    p.tissue_density = float(raw.get("tissue_density", 250.0))
    if p.E_Pa.value <= 0 or p.tissue_density <= 0 or not 0.0 < p.nu.value < 0.5:
        raise ValueError("physics: E_Pa and tissue_density must be > 0, nu in (0, 0.5)")
    g = raw.get("gravity", "supine")
    if g is not None:
        if not isinstance(g, dict):
            g = dict(reference=g, collapse=g)
        g = _clean(g, "physics.gravity", ("reference", "collapse"))
        if "reference" not in g:
            raise ValueError("physics.gravity: needs reference (the position of the inflated CT)")
        eA, nA = _position("physics.gravity.reference", g["reference"])
        eB, nB = _position("physics.gravity.collapse", g.get("collapse", g["reference"]))
        p.gravity = dict(reference=eA, collapse=eB, names=(nA, nB))
    c = _clean(raw.get("collapse", {"q0": {"fit": True, "init": 0.5}}), "physics.collapse",
               ("q0", "p0_Pa", "volume_ratio"))
    if len(c) != 1:
        raise ValueError("physics.collapse: exactly one of q0, p0_Pa, volume_ratio")
    if "q0" in c:
        p.q0 = Param.parse("physics.collapse.q0", c["q0"], (-1.0, 3.0))
    elif "p0_Pa" in c:
        p.p0_Pa = float(c["p0_Pa"])
    else:
        p.volume_ratio = float(c["volume_ratio"])
        if not 0.0 < p.volume_ratio <= 1.0:
            raise ValueError("physics.collapse.volume_ratio: V_collapsed / V_inflated in (0, 1]")
    w = _clean(raw.get("wall", {}), "physics.wall", ("stiffness", "eps"))
    p.wall_stiffness, p.wall_eps = float(w.get("stiffness", 20.0)), float(w.get("eps", 0.5))
    pert = raw.get("perturbation")
    if pert is not None:
        pert = _clean(pert, "physics.perturbation", ("gradient", "regions", "reg"))
        if ("gradient" in pert) == ("regions" in pert):
            raise ValueError("physics.perturbation: either gradient (axes) or regions (levels)")
        if "gradient" in pert:
            axes = list(pert["gradient"])
            if not axes or any(a not in AXES for a in axes) or len(set(axes)) != len(axes):
                raise ValueError("physics.perturbation.gradient: distinct axes among ap, si, ml")
            p.gradient_axes = axes
        else:
            p.regions = [int(k) for k in pert["regions"]]
            if not p.regions:
                raise ValueError("physics.perturbation.regions: the region counts, e.g. [1, 4, 8]")
            p.regions_reg = float(pert.get("reg", 1.0))
    if p.E_Pa.fit and p.gravity is None:
        raise ValueError("physics.E_Pa cannot be fitted without gravity: shapes fix only p / E, and E enters "
                         "through the gravity gradient")
    if p.gravity is not None and "ap" in p.gradient_axes and abs(p.gravity["reference"][1]) > 0.99:
        raise ValueError("physics.perturbation.gradient 'ap' is the gravity axis of this position: fit "
                         "physics.E_Pa instead, or set gravity to null for a free ap slope")
    return p


# ------------------------------------------------------------------------------------------------
# solver, runs
# ------------------------------------------------------------------------------------------------
@dataclass
class SolverConfig:
    core: str = "torch"                # core in pipeline/collapse/solvers, or "module:Class"
    options: dict = field(default_factory=dict)   # passed to the core (see solvers/nodal.py)


@dataclass
class CollapseConfig:
    """Forward run: every physics parameter given."""
    sequence_frames: int = 0           # > 0: also write the collapse sequence (load ramped 0 -> 100 %)
    volume_tol: float = 0.005          # physics.collapse.volume_ratio: accepted error of the ratio
    max_search_solves: int = 15


@dataclass
class FitConfig:
    """Inverse run: the physics parameters marked "fit" (and the perturbation) are fitted to the
    collapsed lung by bounded least squares on the surface distance, both directions."""
    symmetric_weight: float = 1.0      # weight of the target -> fitted term (0 = fitted -> target only)
    best_by: str = "rms"               # the result is the state of lowest "rms" (what the optimiser
                                       # minimises) or lowest "mean" distance
    slope_bound: float = 4.0           # bound on |slope| of physics.perturbation.gradient [p/E across the lung]
    sign_check_q: float = 0.2          # p/E of the initial check that q > 0 collapses
    target_error_mm: float = 2.5       # stop when the error reaches this
    time_budget_min: float = 240.0
    hard_grace_min: float = 10.0       # watchdog kills the run at budget + grace
    lsq_ftol: float = 1e-6             # stop on relative change of the cost
    lsq_xtol: float = 1e-6             # stop on relative change of the parameters
    lsq_gtol: float = 1e-8             # stop on the gradient norm
    # per level: stop when its best error improved by less than level_min_improve (relative) over
    # the last level_patience optimiser iterations
    level_patience: int = 3
    level_min_improve: float = 0.005
    # whole fit (regions): stop when fit_patience levels in a row improve by less than fit_min_improve
    fit_patience: int = 2
    fit_min_improve: float = 0.02

    def __post_init__(self):
        if self.best_by not in ("mean", "rms"):
            raise ValueError("fit.best_by must be mean or rms")
        if self.symmetric_weight < 0.0:
            raise ValueError("fit.symmetric_weight must be >= 0")


@dataclass
class PipelineConfig:
    patient: str
    data_dir: Path
    output_dir: Path
    steps: list
    source: Path                       # the JSON file this was read from
    run_name: str
    inflated: Path                     # inputs
    collapsed: Path | None
    structures: dict
    setup_dir: Path                    # output_dir/fem/setup
    run_dir: Path                      # output_dir/runs/<run_name>
    check_inputs: CheckInputsConfig
    geometry: SurfaceMeshConfig
    hilum: HilumConfig
    fem_setup: FemSetupConfig
    physics: PhysicsConfig
    solver: SolverConfig
    collapse: CollapseConfig
    fit: FitConfig

    def solver_options(self, core=None):
        """Options for the core: solver.options plus what physics decides (wall stiffness, material).
        core: another core than solver.core (validation scripts): the options it does not accept
        are dropped (solvers.options_for)."""
        from .collapse.solvers import get_solver, options_for
        o = dict(self.solver.options, wall_stiffness=self.physics.wall_stiffness, wall_eps=self.physics.wall_eps)
        if core is not None and core != self.solver.core:
            o = options_for(core, o, self.solver.core)
        keys = get_solver(core or self.solver.core).accepted_options()
        if keys is None or "material" in keys:
            o.update(material=self.physics.material, material_params=self.physics.material_params)
        elif self.physics.material != "neo_hookean" or self.physics.material_params:
            raise ValueError("solver '%s' has only the neo_hookean material (physics.material: %s)"
                             % (core or self.solver.core, self.physics.material))
        return o


def _section(cls, raw, name, rename=None):
    """Build a dataclass from its JSON dict, rejecting unknown keys (typos)."""
    known = {f.name for f in fields(cls)}
    raw = _clean(raw, name, known | set(rename or ()))
    for old, new in (rename or {}).items():
        if old in raw:
            raw[new] = raw.pop(old)
    try:
        return cls(**raw)
    except TypeError as e:
        raise ValueError("section '%s': %s" % (name, e)) from None


def apply_overrides(raw, overrides):
    """overrides: ["section.key=value", ...] (value parsed as JSON, else taken as a string), e.g.
    physics.E_Pa=1000, physics.collapse={"volume_ratio":0.5}, solver.wall_update=outer, run_name=x.
    Missing intermediate sections are created."""
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
            if k not in node or node[k] is None:
                node[k] = {}
            if not isinstance(node[k], dict):
                raise ValueError("--set %s: '%s' is not a section of the config" % (item, k))
            node = node[k]
        node[last] = value                     # unknown keys are rejected later, as in the file
    return raw


TOP = ("patient", "data_dir", "output_dir", "steps", "run_name", "inputs", "check_inputs", "geometry", "hilum",
       "fem_setup", "physics", "solver", "collapse", "fit")
STEP_NAMES = ("check_inputs", "geometry", "hilum", "fem_setup", "collapse", "fit")


def load_config(path, overrides=None):
    path = Path(path).resolve()
    raw = _clean(apply_overrides(json.loads(path.read_text()), overrides), "the config", TOP)
    for k in ("patient", "data_dir", "output_dir", "inputs"):
        if k not in raw:
            raise ValueError("the config needs '%s'" % k)
    base = path.parent
    data_dir = (base / raw["data_dir"]).resolve()
    output_dir = (base / raw["output_dir"]).resolve()
    steps = list(raw.get("steps", []))
    bad = sorted(set(steps) - set(STEP_NAMES))
    if bad:
        raise ValueError("unknown step(s) %s (available: %s)" % (", ".join(bad), ", ".join(STEP_NAMES)))
    run_name = str(raw.get("run_name", "run"))
    if not run_name or "/" in run_name or "\\" in run_name:
        raise ValueError("run_name must be a plain folder name")

    inp = _clean(raw["inputs"], "inputs", ("inflated", "collapsed", "structures"))
    if "inflated" not in inp:
        raise ValueError("inputs.inflated (the inflated lung surface) is required")
    inflated = data_dir / inp["inflated"]
    collapsed = data_dir / inp["collapsed"] if inp.get("collapsed") else None
    structures = {k: data_dir / v for k, v in (inp.get("structures") or {}).items()}

    check = _section(CheckInputsConfig, raw.get("check_inputs", {}), "check_inputs")
    check.collapsed, check.inflated, check.structures = collapsed, inflated, dict(structures)
    check.workdir = output_dir / "checks"

    geometry = _section(SurfaceMeshConfig, raw.get("geometry", {}), "geometry")
    geometry.inflated, geometry.collapsed = inflated, collapsed
    geometry.outputs = dict(inflated=output_dir / geometry.inflated_output,
                            collapsed=output_dir / geometry.collapsed_output)

    hilum = _section(HilumConfig, raw.get("hilum", {}), "hilum")
    if hilum.lung is None:
        hilum.lung = "collapsed" if collapsed is not None else "inflated"
    if hilum.lung not in ("collapsed", "inflated") or (hilum.lung == "collapsed" and collapsed is None):
        raise ValueError("hilum.lung must be inflated, or collapsed when inputs.collapsed is given")
    hilum.lung = collapsed if hilum.lung == "collapsed" else inflated
    hilum.structures = dict(structures)
    hilum.output = output_dir / hilum.output

    fs = dict(_clean(raw.get("fem_setup", {}), "fem_setup", ("clamp", "wall", "mesh_size_mm", "gmsh_timeout_s", "regions")))
    clamp = _clean(fs.pop("clamp", {}), "fem_setup.clamp", ("points", "radius_factor", "min_points", "growth", "max_growth_steps"))
    wall = _clean(fs.pop("wall", {}), "fem_setup.wall", ("surface", "tol_mm", "grid_mm", "grid_margin_mm", "smoothing_mm"))
    regions = _clean(fs.pop("regions", None) or {}, "fem_setup.regions", ("levels", "pos_weight"))
    fs.update({"clamp_" + k: v for k, v in clamp.items()})
    fs.update({("wall" if k == "surface" else "wall_" + k): v for k, v in wall.items()})
    if "levels" in regions:
        fs["regions"] = regions["levels"]
    if "pos_weight" in regions:
        fs["regions_pos_weight"] = regions["pos_weight"]
    fem_setup = _section(FemSetupConfig, fs, "fem_setup")
    if fem_setup.wall is not None and fem_setup.wall != "reference":
        fem_setup.wall = data_dir / fem_setup.wall
    fem_setup.reference = output_dir / geometry.inflated_output
    fem_setup.target = output_dir / geometry.collapsed_output if collapsed is not None else None
    fem_setup.anchor = hilum.output
    fem_setup.workdir = output_dir / "fem" / "setup"

    physics = _physics(raw.get("physics", {}))
    if physics.regions and fem_setup.regions is None:
        fem_setup.regions = list(physics.regions)           # the setup computes what the physics asks for
    if fem_setup.regions is not None and collapsed is None:
        raise ValueError("pressure regions need the collapsed lung (inputs.collapsed)")

    s = raw.get("solver", {})
    if not isinstance(s, dict):
        raise ValueError("'solver' must be a section {...}")
    s = {k: v for k, v in s.items() if not k.startswith("_")}       # the core rejects unknown options
    solver = SolverConfig(core=s.pop("core", "torch"), options=s)
    for k in ("wall_stiffness", "wall_eps", "material", "material_params"):
        if k in solver.options:
            raise ValueError("solver.%s: set it in physics (wall.stiffness, wall.eps, material)" % k)

    return PipelineConfig(patient=raw["patient"], data_dir=data_dir, output_dir=output_dir, steps=steps,
                          source=path, run_name=run_name, inflated=inflated, collapsed=collapsed,
                          structures=structures, setup_dir=fem_setup.workdir,
                          run_dir=output_dir / "runs" / run_name, check_inputs=check, geometry=geometry,
                          hilum=hilum, fem_setup=fem_setup, physics=physics, solver=solver,
                          collapse=_section(CollapseConfig, raw.get("collapse", {}), "collapse"),
                          fit=_section(FitConfig, raw.get("fit", {}), "fit"))
