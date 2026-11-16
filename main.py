"""Lung-collapse pipeline entry point.

    python main.py --config configs/karl04.json
    python main.py --config configs/karl04.json --steps fem_setup fit
    python main.py --config configs/karl04.json --steps collapse --set physics.collapse='{"volume_ratio": 0.5}' run_name=half

Runs the steps listed in the config (or in --steps), always in pipeline order. The segmentation
is done by hand in 3D Slicer and is not here.
"""

import argparse
import logging
import shutil
import sys
import time

from pipeline import check_inputs, hilum, surface_mesh
from pipeline.collapse import fit, forward, setup
from pipeline.config import load_config

# step name -> (run, what it gets from the config), in pipeline order
STEPS = {
    "check_inputs": (check_inputs.run, lambda c: c.check_inputs),   # sanity check of the input surfaces
    "geometry": (surface_mesh.run, lambda c: c.geometry),           # surface meshes: inflated (+ collapsed)
    "hilum": (hilum.run, lambda c: c.hilum),                        # hilum anchor from airways / vessels
    "fem_setup": (setup.run, lambda c: c.fem_setup),                # clamp, wall, volume mesh -> problem.npz
    "collapse": (forward.run, lambda c: c),                         # forward: physics given -> collapsed lung
    "fit": (fit.run, lambda c: c),                                  # inverse: fit the marked parameters
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
    ap.add_argument("--set", nargs="+", metavar="KEY=VALUE", default=[],
                    help="override config keys for this run, e.g. physics.E_Pa=1000 run_name=gE1000")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    requested = args.steps or cfg.steps
    unknown = sorted(set(requested) - set(STEPS))
    if unknown:
        sys.exit("unknown step(s): %s (available: %s)" % (", ".join(unknown), ", ".join(STEPS)))

    # main results go in output_dir, everything else in sub-folders
    log_dir = cfg.output_dir / "logs"
    setup_logging(log_dir)
    shutil.copy(cfg.source, log_dir / ("config_used_%s.json" % cfg.source.stem))
    log.info("patient %s, config %s%s", cfg.patient, cfg.source, ", overrides: " + " ".join(args.set) if args.set else "")

    for name, (run, section) in STEPS.items():
        if name not in requested:
            continue
        log.info("=== %s ===", name)
        t0 = time.time()
        try:
            run(section(cfg))
        except Exception:
            log.exception("=== %s FAILED ===", name)
            sys.exit(1)
        log.info("=== %s done in %.1f s ===", name, time.time() - t0)


if __name__ == "__main__":
    main()
