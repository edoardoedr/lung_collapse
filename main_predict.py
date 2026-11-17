"""Predict the collapsed lung of one patient: the inflated lung (with airways / vessels for the hilum)
is collapsed by a forward run with all the physics given. No collapsed lung is needed; with one in
the config, the errors against it are reported.

    python main_predict.py --config configs/<p>.json --collapse 40 --params calibration/E.json [--frames 30]
    python main_predict.py --config configs/<p>.json --collapse 40 --from-fit fitE
    python main_predict.py --config configs/<p>.json --collapse 40          # the config's physics

Parameters (later ones win):
  --params FILE    rule file of main_calibrate.py: E (and tissue_density, nu) from the collapse %
                   (and the CT density physics.ct_density for the rules E_rho / sg_rho)
  --from-fit RUN   E, nu, tissue_density of this patient's fit run output_dir/runs/RUN (and its q0
                   when --collapse is not given: the fit reproduced)
  --collapse P     collapse %: q0 is searched so that V_collapsed / V_inflated = 1 - P / 100;
                   without it, the config's physics.collapse (q0 or p0_Pa)
  --set KEY=VALUE  any config key, e.g. physics.gravity='{"reference":"supine","collapse":"left_lateral"}'

The model geometry (geometry -> hilum -> fem_setup) is built first when output_dir/fem/setup/problem.npz
is missing, or when --steps asks for it. Output: output_dir/runs/<run>/ (surface.vtp, state.npz,
summary.json) and, with --frames > 0, the collapse sequence for 3D Slicer.
"""

import argparse
import json
import logging
import sys

from eurelaps_lung import calibration
from eurelaps_lung.config import load_config
from eurelaps_lung.runner import run_steps, start

log = logging.getLogger("eurelaps_lung")
GEOMETRY_STEPS = ("geometry", "hilum", "fem_setup")


def collapse_overrides(collapse_percent):
    if collapse_percent is None:
        return []
    if not 0.0 <= collapse_percent < 100.0:
        raise ValueError("--collapse: a percentage in [0, 100)")
    return ['physics.collapse={"volume_ratio": %.6f}' % (1.0 - collapse_percent / 100.0)]


def fit_run_overrides(summary, with_q0):
    """--set items with the physics of a fit run's summary.json."""
    ph = summary["physics"]
    out = ["physics.E_Pa=%.6g" % ph["E_Pa"], "physics.material.nu=%.6g" % ph["nu"],
           "physics.tissue_density=%.6g" % ph.get("tissue_density_kg_m3", 250.0)]
    if with_q0:
        out.append('physics.collapse={"q0": %.8g}' % ph["q0"])
    return out


def lung_extent(cfg):
    """The lung's extent along gravity [mm] (rule sg_rho), from problem.npz."""
    from eurelaps_lung.fem.fit import hilum_position
    from eurelaps_lung.fem.load import LoadModel
    from eurelaps_lung.fem.problem import CollapseProblem
    model = LoadModel(cfg.physics, CollapseProblem.load(cfg.setup_dir / "problem.npz"), hilum_position(cfg))
    if model.extent is None:
        raise ValueError("the rule sg_rho needs gravity (physics.gravity)")
    return model.extent


def main():
    ap = argparse.ArgumentParser(description="Predict the collapsed lung: forward run with the physics given")
    ap.add_argument("--config", required=True, help="patient JSON config (inputs.collapsed may be null)")
    ap.add_argument("--collapse", type=float, metavar="PERCENT", help="collapse %% (100 * (1 - V_collapsed / V_inflated))")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--params", metavar="FILE", help="rule file of main_calibrate.py")
    src.add_argument("--from-fit", metavar="RUN", help="physics of this patient's fit run (output_dir/runs/RUN)")
    ap.add_argument("--run", default="predict", help="run name (output_dir/runs/<run>)")
    ap.add_argument("--frames", type=int, help="collapse sequence frames (0 = none; default: the config's)")
    ap.add_argument("--steps", nargs="+", choices=GEOMETRY_STEPS,
                    help="rebuild these geometry steps first (default: only when problem.npz is missing)")
    ap.add_argument("--set", nargs="+", metavar="KEY=VALUE", default=[], help="override config keys")
    args = ap.parse_args()

    base = ["run_name=%s" % args.run] + (["collapse.sequence_frames=%d" % args.frames] if args.frames is not None else [])
    try:
        coll = collapse_overrides(args.collapse)
    except ValueError as e:
        sys.exit(str(e))
    cfg = load_config(args.config, base + coll + args.set)
    start(cfg, "predict", sys.argv[1:])

    steps = args.steps or ([] if (cfg.setup_dir / "problem.npz").is_file() else list(GEOMETRY_STEPS))
    if steps:
        log.info("building the model geometry: %s", " -> ".join(steps))
        run_steps(cfg, steps)

    params = []
    if args.params:
        if args.collapse is None:
            sys.exit("--params gives E from the collapse %%: --collapse is needed")
        rule = calibration.load_rule(args.params)
        L = lung_extent(cfg) if rule["rule"] == "sg_rho" else None
        params = calibration.rule_overrides(rule, args.collapse, cfg.physics.ct_density, L)
        log.info("rule %s of %s (%d patients): %s", rule["rule"], args.params, rule["n"], " ".join(params))
    elif args.from_fit:
        f = cfg.output_dir / "runs" / args.from_fit / "summary.json"
        if not f.is_file():
            sys.exit("no fit run %s (%s)" % (args.from_fit, f))
        params = fit_run_overrides(json.loads(f.read_text()), with_q0=args.collapse is None)
        log.info("physics of the fit run %s: %s", args.from_fit, " ".join(params))
    if params:
        cfg = load_config(args.config, base + params + coll + args.set)
    ph = cfg.physics
    fitted = [p.name for p in (ph.E_Pa, ph.nu, ph.q0) if p is not None and p.fit]
    if fitted:
        log.warning("marked {fit: true} in the config, used at their init value: %s", ", ".join(fitted))
    run_steps(cfg, ["collapse"])


if __name__ == "__main__":
    main()
