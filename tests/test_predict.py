"""The prediction entry point (eurelaps_lung/predict.py) as the 3D Slicer extension uses it: imports
without the optional libraries, the packaged defaults, a run from a config dict.

    python tests/test_predict.py
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import stubs
import fakes  # noqa: F401
import test_wall_potential as W
from eurelaps_lung import predict
from eurelaps_lung.config import E_RULE_A, E_RULE_B, load_config

# Not on the prediction path: importing it must not need these (vtk is Slicer's, netgen runs in a
# subprocess, pyacvd is imported inside the remesh, torch only when the core is built).
OPTIONAL = ("torch", "warp", "getfem", "itk", "sklearn", "nvmath", "pypardiso", "gmsh", "pyvista", "pyacvd", "netgen")
PREDICTION_MODULES = ("eurelaps_lung.predict", "eurelaps_lung.surface_mesh", "eurelaps_lung.hilum",
                      "eurelaps_lung.fem.setup", "eurelaps_lung.fem.forward", "eurelaps_lung.fem.sequence",
                      "eurelaps_lung.fem.results", "eurelaps_lung.fem.solvers.nodal", "eurelaps_lung.calibration")

IMPORT_CHECK = r"""
import importlib.abc, sys
from unittest.mock import MagicMock
BLOCK, MOCK = set(sys.argv[1].split(",")), ("vtk", "vtkmodules")
class F(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        top = name.split(".")[0]
        if top in BLOCK:
            raise ImportError("blocked: " + name)
        if top in MOCK:
            import importlib.machinery as m
            class L(importlib.abc.Loader):
                def create_module(self, spec):
                    x = MagicMock(name=spec.name); x.__path__, x.__spec__, x.__name__ = [], spec, spec.name; return x
                def exec_module(self, module): pass
            return m.ModuleSpec(name, L(), is_package=True)
sys.meta_path.insert(0, F())
sys.path.insert(0, sys.argv[2])
import importlib
for mod in sys.argv[3].split(","):
    importlib.import_module(mod)
print("ok")
"""


def test_imports_without_optional_libraries():
    r = subprocess.run([sys.executable, "-c", IMPORT_CHECK, ",".join(OPTIONAL), str(stubs.ROOT),
                        ",".join(PREDICTION_MODULES)], capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.strip() == "ok", r.stderr[-2000:]
    return "%d modules with %s blocked" % (len(PREDICTION_MODULES), ", ".join(OPTIONAL))


def filled(out, **extra):
    cfg = predict.default_config()
    cfg.update(patient="karl04", data_dir=".", output_dir=str(out),
               inputs={"inflated": "lung.vtk", "collapsed": None, "structures": {}}, **extra)
    return cfg


def test_default_config():
    """The packaged defaults load from a dict and give the calibrated model."""
    cfg = load_config(filled("/tmp/x"), predict.collapse_overrides(40.0))
    ph = cfg.physics
    E = 10 ** (E_RULE_A + E_RULE_B * 40.0)
    assert abs(ph.E_Pa.value - E) < 1e-6 * E and abs(ph.volume_ratio - 0.6) < 1e-12
    assert ph.nu.value == 0.30 and ph.tissue_density == 200.0 and ph.gravity["names"] == ("supine", "supine")
    assert cfg.solver.core == "torch" and cfg.solver.options["device"] == "cpu"
    assert cfg.solver.options["linear_solver"] == "scipy" and cfg.collapse.sequence_frames == 30
    assert cfg.hilum.lung == cfg.inflated and cfg.source is None and cfg.raw["patient"] == "karl04"
    return "E %.0f Pa at 40 %%, hilum on the inflated lung, torch cpu + SuperLU" % E


def test_defaults_match_template():
    """configs/templates/new_patient.json (command line) = the packaged defaults, but the solver platform."""
    t = json.loads((stubs.ROOT / "configs/templates/new_patient.json").read_text())
    d = predict.default_config()
    strip = lambda x: json.loads(json.dumps(x), object_hook=lambda o: {k: v for k, v in o.items() if not k.startswith("_")})
    for k in ("geometry", "fem_setup", "physics"):
        assert strip(t[k]) == strip(d[k]), k
    assert strip(t["hilum"]) == strip(d["hilum"])
    same = {k: v for k, v in strip(d["solver"]).items() if k not in ("device", "linear_solver", "max_solve_s")}
    assert all(strip(t["solver"])[k] == v for k, v in same.items())
    return "geometry, hilum, fem_setup, physics equal"


def test_run_from_dict():
    """predict.run on a config dict: karl04's problem.npz already there, so only the forward runs."""
    p = W.problem()
    if p is None:
        return "skipped"
    out = Path(tempfile.mkdtemp(prefix="lung_predict_")) / "out"
    (out / "fem/setup").mkdir(parents=True)
    p.save(out / "fem/setup/problem.npz")
    cfg = filled(out)
    cfg["solver"] = {"core": "numpy_core:NumpySolver", "wall_update": "potential", "adaptive_steps": True,
                     "warm_substeps": True, "newton_maxit": 50, "load_steps": 4, "max_solve_s": 60}
    s = predict.run(cfg, collapse=20.0, frames=0, run_name="p20")
    assert abs(s["collapse_percent"] - 20.0) <= 100 * 0.006 and abs(s["physics"]["E_rule"]["collapse_percent"] - 20.0) < 1e-9
    assert (out / "logs/config_used_predict.json").is_file() and (out / "runs/p20/state.npz").is_file()
    return "collapse %.2f %% (asked 20), E %.0f Pa from the rule, %d solves" % (
        s["collapse_percent"], s["physics"]["E_Pa"], s["solves"])


def test_drive_by_pressure():
    """p0 instead of the collapse %: the E rule cannot apply (clear error), E given works; collapse and p0
    together are refused; estimate() matches the rules."""
    try:
        load_config(filled("/tmp/x"), predict.pressure_overrides(330.0, None))
        raise AssertionError("the E rule with p0 must fail")
    except ValueError as e:
        assert "give E as a number" in str(e), e
    ph = load_config(filled("/tmp/x"), predict.pressure_overrides(330.0, 1300.0) + predict.E_overrides(1300.0)).physics
    assert ph.p0_Pa == 330.0 and ph.E_Pa.value == 1300.0 and ph.volume_ratio is None and ph.E_rule is None
    for kw in (dict(collapse=40.0, p0=330.0),):
        try:
            predict.run(filled("/tmp/x"), **kw)
            raise AssertionError("collapse and p0 together must fail")
        except ValueError as e:
            assert "not both" in str(e)
    e = predict.estimate(collapse=40.0)
    assert abs(e["E_Pa"] - 10 ** (E_RULE_A + E_RULE_B * 40.0)) < 1e-6
    assert abs(predict.estimate(p0=e["p0_Pa"], E=e["E_Pa"])["collapse_percent"] - 40.0) < 1e-9
    return "p0 330 Pa, E 1300 Pa; at 40 %% the rules expect p0 %.0f Pa" % e["p0_Pa"]


def test_run_driven_by_pressure():
    """predict.run(p0=, E=) on karl04's problem.npz: the collapse % is the result; summary.prediction."""
    p = W.problem()
    if p is None:
        return "skipped"
    out = Path(tempfile.mkdtemp(prefix="lung_predict_")) / "out"
    (out / "fem/setup").mkdir(parents=True)
    p.save(out / "fem/setup/problem.npz")
    cfg = filled(out)
    cfg["solver"] = {"core": "numpy_core:NumpySolver", "wall_update": "potential", "adaptive_steps": True,
                     "warm_substeps": True, "newton_maxit": 50, "load_steps": 4, "max_solve_s": 60}
    s = predict.run(cfg, p0=300.0, E=1300.0, frames=0, run_name="p0")
    pr = s["prediction"]
    assert pr["driven_by"] == "p0" and pr["p0_asked"] == 300.0 and s["physics"]["E_Pa"] == 1300.0
    assert abs(s["physics"]["p0_Pa"] - 300.0) < 1e-6 and pr["estimate"]["from"] == "p0"
    return "collapse %.1f %% (estimate %.1f %%)" % (s["collapse_percent"], pr["estimate"]["collapse_percent"])


if __name__ == "__main__":
    stubs.run_tests(globals())
