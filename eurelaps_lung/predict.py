"""Prediction: the inflated lung of one patient (with airways / vessels for the hilum) collapsed by a
forward run with all the physics given. No collapsed lung is needed; with one in the config, the
errors against it are reported. The entry point of main_predict.py and of the 3D Slicer extension.

    python -m eurelaps_lung.predict --config work/config.json --collapse 40 [--frames 30]
    python -m eurelaps_lung.predict --config work/config.json --collapse target      # validation

Python:
    from eurelaps_lung import predict
    cfg = predict.default_config()            # the prediction defaults (eurelaps_lung/data/predict_default.json)
    cfg.update(patient="p1", data_dir="/work/p1", output_dir="/work/p1/out",
               inputs={"inflated": "lung.vtk", "collapsed": None,
                       "structures": {"airways": "airways.vtk", "arteries": "arteries.vtk", "veins": "veins.vtk"}})
    summary = predict.run(cfg, collapse=40, frames=30)

Parameters (later ones win):
  params FILE      rule file of main_calibrate.py: E (and tissue_density, nu) from the collapse %
                   (and the CT density physics.ct_density for the rules E_rho / sg_rho)
  from_fit RUN     E, nu, tissue_density of this patient's fit run output_dir/runs/RUN (and its q0
                   when collapse is not given: the fit reproduced)
  collapse P       collapse %: q0 is searched so that V_collapsed / V_inflated = 1 - P / 100;
                   "target" = the collapse % of the collapsed lung given in inputs.collapsed;
                   None = the config's physics.collapse (q0, p0_Pa or volume_ratio)
  overrides        ["section.key=value", ...], any config key
Without params / from_fit, E is the config's physics.E_Pa: in the defaults the built-in rule
{"rule": "collapse"} (config.E_RULE_A, E_RULE_B).

The model geometry (geometry -> hilum -> fem_setup) is built first when output_dir/fem/setup/problem.npz
is missing, or when `steps` asks for it. Output: output_dir/runs/<run>/ (surface.vtp, state.npz,
summary.json) and, with frames > 0, the collapse sequence (runs/<run>/sequence/).
"""

import argparse
import copy
import json
import logging
import sys
from importlib import resources

from . import calibration
from .config import load_config
from .runner import run_steps, start

log = logging.getLogger("eurelaps_lung")
GEOMETRY_STEPS = ("geometry", "hilum", "fem_setup")
DEFAULT_CONFIG = "predict_default.json"


def default_config():
    """The prediction defaults as a dict (a fresh copy): the model and solver settings of the calibrated
    model. The caller sets patient, data_dir, output_dir and inputs."""
    text = resources.files("eurelaps_lung").joinpath("data", DEFAULT_CONFIG).read_text()
    return copy.deepcopy(json.loads(text))


def collapse_overrides(collapse_percent):
    if collapse_percent is None:
        return []
    if not 0.0 <= collapse_percent < 100.0:
        raise ValueError("collapse: a percentage in [0, 100)")
    return ['physics.collapse={"volume_ratio": %.6f}' % (1.0 - collapse_percent / 100.0)]


def fit_run_overrides(summary, with_q0):
    """Overrides with the physics of a fit run's summary.json."""
    ph = summary["physics"]
    out = ["physics.E_Pa=%.6g" % ph["E_Pa"], "physics.material.nu=%.6g" % ph["nu"],
           "physics.tissue_density=%.6g" % ph.get("tissue_density_kg_m3", 250.0)]
    if with_q0:
        out.append('physics.collapse={"q0": %.8g}' % ph["q0"])
    return out


def lung_extent(cfg):
    """The lung's extent along gravity [mm] (rule sg_rho), from problem.npz."""
    from .fem.fit import hilum_position
    from .fem.load import LoadModel
    from .fem.problem import CollapseProblem
    model = LoadModel(cfg.physics, CollapseProblem.load(cfg.setup_dir / "problem.npz"), hilum_position(cfg))
    if model.extent is None:
        raise ValueError("the rule sg_rho needs gravity (physics.gravity)")
    return model.extent


def target_collapse_percent(cfg):
    """Collapse % of the collapsed lung of the config (its fem_setup target) against the inflated one."""
    from .fem.geometry import signed_volume
    from .fem.problem import CollapseProblem
    problem = CollapseProblem.load(cfg.setup_dir / "problem.npz")
    if not problem.has_target:
        raise ValueError("collapse 'target' needs the collapsed lung: inputs.collapsed, and fem_setup run with it "
                         "(steps geometry hilum fem_setup)")
    ratio = signed_volume(*problem.target_surface) / signed_volume(problem.reference, problem.tris)
    return 100.0 * (1.0 - ratio)


def run(config, collapse=None, params=None, from_fit=None, frames=None, run_name="predict", steps=None,
        overrides=(), base_dir=None, log_name="predict"):
    """Prediction of one patient; returns the run's summary (dict).

    config: path of a JSON config, or a dict (e.g. default_config() filled in; relative data_dir /
    output_dir are then taken from base_dir, default the current folder)."""
    if params is not None and from_fit is not None:
        raise ValueError("give params or from_fit, not both")
    base = ["run_name=%s" % run_name] + (["collapse.sequence_frames=%d" % frames] if frames is not None else [])
    target = isinstance(collapse, str)
    if target and collapse != "target":
        raise ValueError("collapse: a percentage, 'target' or None")
    coll = [] if target else collapse_overrides(collapse)
    overrides = list(overrides)
    cfg = load_config(config, base + coll + overrides, base_dir=base_dir)
    start(cfg, log_name, overrides)

    if target and cfg.collapsed is None:
        raise ValueError("collapse 'target' needs inputs.collapsed (the collapsed lung)")
    have = (cfg.setup_dir / "problem.npz").is_file()
    steps = list(steps) if steps else ([] if have else list(GEOMETRY_STEPS))
    if steps:
        log.info("building the model geometry: %s", " -> ".join(steps))
        run_steps(cfg, steps)
    if target:
        collapse = target_collapse_percent(cfg)
        log.info("collapse %% of the collapsed lung (inputs.collapsed): %.2f %%", collapse)
        coll = collapse_overrides(collapse)

    phys = []
    if params is not None:
        if collapse is None:
            raise ValueError("params gives E from the collapse %: collapse is needed")
        rule = calibration.load_rule(params)
        cfg = load_config(config, base + coll + overrides, base_dir=base_dir)
        L = lung_extent(cfg) if rule["rule"] == "sg_rho" else None
        phys = calibration.rule_overrides(rule, collapse, cfg.physics.ct_density, L)
        log.info("rule %s of %s (%d patients): %s", rule["rule"], params, rule["n"], " ".join(phys))
    elif from_fit is not None:
        f = cfg.output_dir / "runs" / from_fit / "summary.json"
        if not f.is_file():
            raise ValueError("no fit run %s (%s)" % (from_fit, f))
        phys = fit_run_overrides(json.loads(f.read_text()), with_q0=collapse is None)
        log.info("physics of the fit run %s: %s", from_fit, " ".join(phys))
    if phys or target:
        cfg = load_config(config, base + phys + coll + overrides, base_dir=base_dir)
    ph = cfg.physics
    fitted = [p.name for p in (ph.E_Pa, ph.nu, ph.q0) if p is not None and p.fit]
    if fitted:
        log.warning("marked {fit: true} in the config, used at their init value: %s", ", ".join(fitted))
    run_steps(cfg, ["collapse"])
    return json.loads((cfg.run_dir / "summary.json").read_text())


def main(argv=None):
    ap = argparse.ArgumentParser(description="Predict the collapsed lung: forward run with the physics given")
    ap.add_argument("--config", required=True, help="patient JSON config (inputs.collapsed may be null)")
    ap.add_argument("--collapse", metavar="PERCENT|target",
                    help="collapse %% (100 * (1 - V_collapsed / V_inflated)), or 'target' = that of inputs.collapsed")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--params", metavar="FILE", help="rule file of main_calibrate.py")
    src.add_argument("--from-fit", metavar="RUN", help="physics of this patient's fit run (output_dir/runs/RUN)")
    ap.add_argument("--run", default="predict", help="run name (output_dir/runs/<run>)")
    ap.add_argument("--frames", type=int, help="collapse sequence frames (0 = none; default: the config's)")
    ap.add_argument("--steps", nargs="+", choices=GEOMETRY_STEPS,
                    help="rebuild these geometry steps first (default: only when problem.npz is missing)")
    ap.add_argument("--set", nargs="+", metavar="KEY=VALUE", default=[], help="override config keys")
    args = ap.parse_args(argv)
    collapse = args.collapse
    if collapse is not None and collapse != "target":
        try:
            collapse = float(collapse)
        except ValueError:
            sys.exit("--collapse: a percentage or 'target'")
    try:
        run(args.config, collapse=collapse, params=args.params, from_fit=args.from_fit, frames=args.frames,
            run_name=args.run, steps=args.steps, overrides=args.set)
    except ValueError as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()
