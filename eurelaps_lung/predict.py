"""Prediction: the inflated lung of one patient (with airways / vessels for the hilum) collapsed by a
forward run with all the physics given. No collapsed lung is needed; with one in the config, the
errors against it are reported. The entry point of main_predict.py and of the 3D Slicer extension.

    python -m eurelaps_lung.predict --config work/config.json --collapse 40 [--frames 30]
    python -m eurelaps_lung.predict --config work/config.json --collapse target      # validation
    python -m eurelaps_lung.predict --config work/config.json --p0 330 --E 1300      # pressure given
    python -m eurelaps_lung.predict --config work/config.json --p0 1600 --E 1200 --barrier \
        --emphysema yes --ct-density 180 --ct-emphysema 20                          # surgery, 12 mmHg
    python -m eurelaps_lung.predict --estimate --collapse 40       # what the rules expect (no run, no config)
    python -m eurelaps_lung.predict --estimate --p0 330 --E 1300

Python:
    from eurelaps_lung import predict
    cfg = predict.default_config()            # the prediction defaults (eurelaps_lung/data/predict_default.json)
    cfg.update(patient="p1", data_dir="/work/p1", output_dir="/work/p1/out",
               inputs={"inflated": "lung.vtk", "collapsed": None,
                       "structures": {"airways": "airways.vtk", "arteries": "arteries.vtk", "veins": "veins.vtk"}})
    summary = predict.run(cfg, collapse=40, frames=30)        # collapse % given: E from the rule, q0 searched
    summary = predict.run(cfg, p0=330, E=1300)                # pressure given: the collapse % is the result
    summary = predict.run(cfg, p0=1600, E=1200, barrier=True, emphysema=True, ct_density=180, ct_emphysema=20)
    predict.estimate(collapse=40)    # {collapse_percent, E_Pa, p0_Pa, q0, ...}: the rules' expectation
    predict.estimate(p0=330, E=1300)
    predict.rules()                  # the rules in use (eurelaps_lung/data/rules.json or $EURELAPS_RULES)

Parameters (later ones win):
  params FILE      rule file of main_calibrate.py: E (and tissue_density, nu) from the collapse %
                   (and the CT density physics.ct_density for the rules E_rho / sg_rho)
  from_fit RUN     E, nu, tissue_density of this patient's fit run output_dir/runs/RUN (and its q0
                   when collapse is not given: the fit reproduced)
  collapse P       collapse %: q0 is searched so that V_collapsed / V_inflated = 1 - P / 100;
                   "target" = the collapse % of the collapsed lung given in inputs.collapsed;
                   None = the config's physics.collapse (q0, p0_Pa or volume_ratio)
  p0 P [Pa]        the released pressure instead of the collapse % (q0 = p0 / E): the collapse % is the
                   result. The E rule needs the collapse %, so with p0 give E too (or a number in the config)
  E [Pa]           E given (instead of the config's physics.E_Pa)
  emphysema        True / False: does the CT show emphysema (the user's reading; physics.emphysema)
  ct_density       [kg/m3] density of the inflated lung on the CT (physics.ct_density; eurelaps_lung.ct.inflated_lung)
  ct_emphysema     [%] its emphysema volume, LAA%-950 (physics.ct_emphysema; the same function)
  barrier          True: the compression limit J >= phi, phi = ct_density / 1050 [+ 0.54 ct_emphysema / 100
                   if emphysema] (physics.barrier {"phi": "ct"}; needs ct_density and emphysema). Needed for
                   surgical pressures (p0 1600 Pa = 12 mmHg); METHODOLOGY.md 3.4.1
  overrides        ["section.key=value", ...], any config key
Without params / from_fit, E is the config's physics.E_Pa: in the defaults the rule {"rule": "collapse"}
of the rule file in use (eurelaps_lung/data/rules.json, or $EURELAPS_RULES).

summary.json gets "prediction": how the run was driven (collapse % or p0), what was asked, and the rules'
estimate for it (estimate()), to compare with what the run reached.

The model geometry (geometry -> hilum -> fem_setup) is built first when output_dir/fem/setup/problem.npz
is missing, or when `steps` asks for it; the `ct` step (density and emphysema volume, eurelaps_lung/ct.py)
runs first when inputs.ct is given and output_dir/ct/ct_values.json is missing. Output: output_dir/runs/<run>/ (surface.vtp, state.npz,
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
GEOMETRY_STEPS = ("ct", "geometry", "hilum", "fem_setup")
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


def rules(path=None):
    """The rules in use (a rule file of main_calibrate.py): the packaged eurelaps_lung/data/rules.json, or
    $EURELAPS_RULES, or `path`. Keys: rule, a, b, n, residual_sd, collapse_range, q0 {a, b, ...}, source."""
    return calibration.load_rules(path)


def estimate(collapse=None, p0=None, E=None, params=None):
    """The rules' expectation before a run: collapse % [, E] -> E, q0, p0; or p0 and E -> q0, collapse %.
    params: a rule file instead of the one in use. See calibration.estimate."""
    return calibration.estimate(rules(params), collapse_percent=collapse, p0=p0, E=E)


def pressure_overrides(p0, E):
    """--set items of a run driven by the pressure: physics.collapse p0_Pa (and E when given)."""
    if p0 is None:
        return []
    if not p0 > 0:
        raise ValueError("p0: a pressure > 0 Pa")
    out = ['physics.collapse={"p0_Pa": %.6g}' % float(p0)]
    return out


def E_overrides(E):
    if E is None:
        return []
    if not E > 0:
        raise ValueError("E: a Young's modulus > 0 Pa")
    return ["physics.E_Pa=%.6g" % float(E)]


def ct_overrides(emphysema=None, ct_density=None, ct_emphysema=None, barrier=None):
    """--set items for the CT values, the emphysema yes / no and the compression barrier."""
    out = []
    if emphysema is not None:
        out.append("physics.emphysema=%s" % ("true" if emphysema else "false"))
    if ct_density is not None:
        if not ct_density > 0:
            raise ValueError("ct_density: kg/m3 > 0")
        out.append("physics.ct_density=%.6g" % float(ct_density))
    if ct_emphysema is not None:
        if not 0.0 <= ct_emphysema <= 100.0:
            raise ValueError("ct_emphysema: LAA%-950 in [0, 100] %")
        out.append("physics.ct_emphysema=%.6g" % float(ct_emphysema))
    if barrier is not None:
        out.append('physics.barrier=%s' % ('{"phi": "ct"}' if barrier else "null"))
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
        overrides=(), base_dir=None, log_name="predict", p0=None, E=None, emphysema=None, ct_density=None,
        ct_emphysema=None, barrier=None):
    """Prediction of one patient; returns the run's summary (dict).

    config: path of a JSON config, or a dict (e.g. default_config() filled in; relative data_dir /
    output_dir are then taken from base_dir, default the current folder). collapse (a % or "target") or
    p0 [Pa]: how the collapse is driven; E [Pa]: E given instead of the config's (needed with p0 when the
    config has the E rule). emphysema (True / False, the user's reading of the CT), ct_density, ct_emphysema,
    barrier (True = compression limit from them): see the module docstring."""
    if params is not None and from_fit is not None:
        raise ValueError("give params or from_fit, not both")
    if p0 is not None and collapse is not None:
        raise ValueError("give the collapse % or the pressure p0, not both: with p0 the collapse % is the result")
    if p0 is not None and params is not None and E is None:
        raise ValueError("the rule file gives E from the collapse %, unknown when p0 is given: give E too")
    base = ["run_name=%s" % run_name] + (["collapse.sequence_frames=%d" % frames] if frames is not None else [])
    target = isinstance(collapse, str)
    if target and collapse != "target":
        raise ValueError("collapse: a percentage, 'target' or None")
    coll = [] if target else collapse_overrides(collapse)
    overrides = (list(overrides) + pressure_overrides(p0, E) + E_overrides(E)
                 + ct_overrides(emphysema, ct_density, ct_emphysema, barrier))
    cfg = load_config(config, base + coll + overrides, base_dir=base_dir)
    start(cfg, log_name, overrides)

    if target and cfg.collapsed is None:
        raise ValueError("collapse 'target' needs inputs.collapsed (the collapsed lung)")
    have = (cfg.setup_dir / "problem.npz").is_file()
    steps = list(steps) if steps else ([] if have else [s for s in GEOMETRY_STEPS if s != "ct"])
    if cfg.ct is not None and "ct" not in steps and not (cfg.output_dir / "ct" / "ct_values.json").is_file():
        steps = ["ct"] + steps                                  # density and emphysema volume from the CT
    if steps:
        log.info("building the model geometry: %s", " -> ".join(steps))
        run_steps(cfg, steps)
    if target:
        collapse = target_collapse_percent(cfg)
        log.info("collapse %% of the collapsed lung (inputs.collapsed): %.2f %%", collapse)
        coll = collapse_overrides(collapse)

    phys = []
    if params is not None and p0 is not None:
        # pressure given: E is given too (checked above); the rule file still gives nu and the density
        rule = calibration.load_rule(params)
        phys = ["physics.tissue_density=%.6g" % rule["tissue_density"]] + (
            ["physics.material.nu=%.6g" % rule["nu"]] if rule.get("nu") is not None else [])
        log.info("rule %s of %s with p0 given: E given, %s", rule["rule"], params, " ".join(phys))
    elif params is not None:
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
    path = cfg.run_dir / "summary.json"
    summary = json.loads(path.read_text())
    if p0 is None and collapse is None and cfg.physics.p0_Pa is not None:
        p0 = cfg.physics.p0_Pa                         # driven by the config's physics.collapse p0_Pa
    summary["prediction"] = prediction_record(summary, collapse, p0, E, params)
    path.write_text(json.dumps(summary, indent=2))
    return summary


def prediction_record(summary, collapse, p0, E, params):
    """How the run was driven, and the rules' estimate for it (to compare with what it reached)."""
    ph = summary.get("physics", {})
    rec = {"driven_by": "p0" if p0 is not None else "collapse", "collapse_asked": collapse, "p0_asked": p0,
           "E_given": E, "emphysema": ph.get("emphysema"), "barrier": ph.get("barrier")}
    try:
        r = rules(params)
        rec["rules"] = r.get("source")
        if p0 is not None:
            rec["estimate"] = calibration.estimate(r, p0=p0, E=ph.get("E_Pa"))
        else:
            c = summary.get("collapse_percent") if collapse in (None, "target") else collapse
            rec["estimate"] = calibration.estimate(r, collapse_percent=c, E=ph.get("E_Pa"))
    except (ValueError, KeyError, OSError) as e:
        rec["estimate"] = None
        rec["estimate_error"] = str(e)
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description="Predict the collapsed lung: forward run with the physics given")
    ap.add_argument("--config", help="patient JSON config (inputs.collapsed may be null)")
    drive = ap.add_mutually_exclusive_group()
    drive.add_argument("--collapse", metavar="PERCENT|target",
                       help="collapse %% (100 * (1 - V_collapsed / V_inflated)), or 'target' = that of inputs.collapsed")
    drive.add_argument("--p0", type=float, metavar="PA", help="released pressure p0 [Pa]: the collapse %% is the result")
    ap.add_argument("--E", type=float, metavar="PA", help="E [Pa] instead of the config's (needed with --p0 and the E rule)")
    ap.add_argument("--emphysema", choices=("yes", "no"), help="does the CT show emphysema (your reading)")
    ap.add_argument("--ct-density", type=float, metavar="KG_M3", help="density of the inflated lung on the CT")
    ap.add_argument("--ct-emphysema", type=float, metavar="PERCENT", help="emphysema volume on the CT, LAA%%-950")
    ap.add_argument("--barrier", action="store_true",
                    help="compression limit from the CT values (needed for surgical pressures, e.g. --p0 1600)")
    ap.add_argument("--estimate", action="store_true",
                    help="print the rules' estimate (E, p0, collapse %%) for --collapse or --p0 --E, and exit")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--params", metavar="FILE", help="rule file of main_calibrate.py")
    src.add_argument("--from-fit", metavar="RUN", help="physics of this patient's fit run (output_dir/runs/RUN)")
    ap.add_argument("--run", default="predict", help="run name (output_dir/runs/<run>)")
    ap.add_argument("--frames", type=int, help="collapse sequence frames (0 = none; default: the config's)")
    ap.add_argument("--steps", nargs="+", choices=GEOMETRY_STEPS,
                    help="rebuild these geometry steps first (default: only when problem.npz is missing)")
    ap.add_argument("--set", nargs="+", metavar="KEY=VALUE", default=[], help="override config keys")
    args = ap.parse_args(argv)
    if args.estimate:
        try:
            c = float(args.collapse) if args.collapse is not None else None
            e = estimate(collapse=c, p0=args.p0, E=args.E, params=args.params)
        except ValueError as err:
            sys.exit(str(err))
        print(json.dumps(e, indent=2))
        return
    if not args.config:
        sys.exit("--config is required (except with --estimate)")
    collapse = args.collapse
    if collapse is not None and collapse != "target":
        try:
            collapse = float(collapse)
        except ValueError:
            sys.exit("--collapse: a percentage or 'target'")
    try:
        run(args.config, collapse=collapse, params=args.params, from_fit=args.from_fit, frames=args.frames,
            run_name=args.run, steps=args.steps, overrides=args.set, p0=args.p0, E=args.E,
            emphysema=None if args.emphysema is None else args.emphysema == "yes", ct_density=args.ct_density,
            ct_emphysema=args.ct_emphysema, barrier=True if args.barrier else None)
    except ValueError as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()
