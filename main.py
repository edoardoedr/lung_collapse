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

from pipeline import registration, surface_mesh
from pipeline.config import load_config

# step name -> run(step_config), in pipeline order
STEPS = {
    "surface_mesh": surface_mesh.run,      # step 2
    "registration": registration.run,      # step 3
}

log = logging.getLogger("pipeline")


def setup_logging(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(output_dir / "pipeline.log")):
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

    setup_logging(cfg.output_dir)
    shutil.copy(cfg.source, cfg.output_dir / "config_used.json")
    log.info("patient %s, config %s", cfg.patient, cfg.source)

    for name, run in STEPS.items():
        if name not in requested:
            continue
        step_cfg = getattr(cfg, name)
        if step_cfg is None:
            sys.exit("step '%s' requested but the config has no '%s' section" % (name, name))
        log.info("=== %s ===", name)
        t0 = time.time()
        run(step_cfg)
        log.info("=== %s done in %.1f s ===", name, time.time() - t0)


if __name__ == "__main__":
    main()
