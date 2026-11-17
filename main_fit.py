"""Fit one patient: the physics parameters marked {"fit": true} in the config are fitted so that the
inflated lung collapses onto the patient's collapsed lung.

    python main_fit.py --config configs/karl04.json
    python main_fit.py --config configs/karl04.json --steps fit --set physics.E_Pa='{"fit": true, "init": 1360}' run_name=fitE

Runs the steps listed in the config (or in --steps), in order:
check_inputs -> geometry -> hilum -> fem_setup -> fit. Output in output_dir (logs/fit.log,
runs/<run_name>/). Forward runs (parameters given -> collapsed lung) are main_predict.py; the
population parameters from many fits are main_calibrate.py.
"""

import argparse
import sys

from eurelaps_lung.config import load_config
from eurelaps_lung.runner import PREPROCESSING, run_steps, start

STEPS = (*PREPROCESSING, "fit")


def main():
    ap = argparse.ArgumentParser(description="Fit the model to one patient's collapsed lung")
    ap.add_argument("--config", required=True, help="patient JSON config")
    ap.add_argument("--steps", nargs="+", choices=STEPS, help="override the config's step list")
    ap.add_argument("--set", nargs="+", metavar="KEY=VALUE", default=[],
                    help="override config keys for this run, e.g. physics.E_Pa=1000 run_name=gE1000")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    steps = args.steps or cfg.steps
    if "collapse" in steps:
        sys.exit("the collapse step (forward run) is main_predict.py: remove it from the config's steps")
    if cfg.collapsed is None and "fit" in steps:
        sys.exit("the fit needs the collapsed lung (inputs.collapsed); without it use main_predict.py")
    start(cfg, "fit", args.set)
    run_steps(cfg, steps)


if __name__ == "__main__":
    main()
