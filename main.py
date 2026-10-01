"""Collapsed-lung pipeline entry point.

    python main.py --config configs/patient1.json
    python main.py --config configs/patient1.json --steps surface_mesh

Runs the steps listed in the config (or in --steps), always in pipeline order.
Step 1 (segmentation) is done by hand in 3D Slicer and is not here.
"""

import argparse
import logging
import shutil
import sys
import time

from pipeline import check_inputs, hilum, registration, surface_mesh
from pipeline.collapse import fit, setup
from pipeline.config import load_config

# step name -> run(step_config), in pipeline order
STEPS = {
    "check_inputs": check_inputs.run,      # step 1b, sanity check of the input surfaces
    "surface_mesh": surface_mesh.run,      # step 2
    "hilum": hilum.run,                    # step 3, before registration: provides its landmarks
    "registration": registration.run,      # step 4
    "fem_setup": setup.run,                # step 5a, independent of the FEM core
    "fem_fit": fit.run,                    # step 5b, FEM core chosen in the config
}

log = logging.getLogger("pipeline")


def setup_logging(log_dir):
    log_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_dir / "pipeline.log")):
        h.setFormatter(fmt)
        root.addHandler(h)


def main():
    ap = argparse.ArgumentParser(description="Collapsed-lung pipeline")
    ap.add_argument("--config", required=True, help="patient JSON config")
    ap.add_argument("--steps", nargs="+", choices=list(STEPS), help="override the config's step list")
    args = ap.parse_args()

    cfg = load_config(args.config)
    requested = args.steps or cfg.steps
    unknown = sorted(set(requested) - set(STEPS))
    if unknown:
        sys.exit("unknown step(s): %s (available: %s)" % (", ".join(unknown), ", ".join(STEPS)))

    # main results go in output_dir, everything else in sub-folders
    log_dir = cfg.output_dir / "logs"
    setup_logging(log_dir)
    shutil.copy(cfg.source, log_dir / "config_used.json")
    log.info("patient %s, config %s", cfg.patient, cfg.source)

    for name, run in STEPS.items():
        if name not in requested:
            continue
        step_cfg = getattr(cfg, name)
        if step_cfg is None:
            sys.exit("step '%s' requested but the config has no '%s' section" % (name, name))
        log.info("=== %s ===", name)
        t0 = time.time()
        try:
            run(step_cfg)
        except Exception:
            log.exception("=== %s FAILED ===", name)
            sys.exit(1)
        log.info("=== %s done in %.1f s ===", name, time.time() - t0)


if __name__ == "__main__":
    main()
