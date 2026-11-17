"""The steps of the model and how the entry points (main_fit.py, main_predict.py) run them.

    check_inputs -> geometry -> hilum -> fem_setup -> collapse (forward) | fit (inverse)

Steps always run in this order, whatever order they are asked in. The segmentation is done by hand
in 3D Slicer and is not here.
"""

import importlib
import logging
import shutil
import sys
import time

# step name -> (module with run(), what it gets from the config), in order. The modules are imported
# only when their step runs (each needs its own libraries: itk, pyacvd, gmsh, the FEM core).
STEPS = {
    "check_inputs": ("check_inputs", lambda c: c.check_inputs),   # sanity check of the input surfaces
    "geometry": ("surface_mesh", lambda c: c.geometry),           # surface meshes: inflated (+ collapsed)
    "hilum": ("hilum", lambda c: c.hilum),                        # hilum anchor from airways / vessels
    "fem_setup": ("fem.setup", lambda c: c.fem_setup),            # clamp, wall, volume mesh -> problem.npz
    "collapse": ("fem.forward", lambda c: c),                     # forward: physics given -> collapsed lung
    "fit": ("fem.fit", lambda c: c),                              # inverse: fit the marked parameters
}
PREPROCESSING = ("check_inputs", "geometry", "hilum", "fem_setup")

log = logging.getLogger("eurelaps_lung")


def setup_logging(log_dir, name):
    """Log to stdout and to log_dir/<name>.log."""
    log_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_dir / ("%s.log" % name))):
        h.setFormatter(fmt)
        root.addHandler(h)


def start(cfg, name, overrides=()):
    """Logging in output_dir/logs/<name>.log and a copy of the config there."""
    log_dir = cfg.output_dir / "logs"
    setup_logging(log_dir, name)
    shutil.copy(cfg.source, log_dir / ("config_used_%s.json" % cfg.source.stem))
    log.info("patient %s, config %s%s", cfg.patient, cfg.source,
             ", overrides: " + " ".join(overrides) if overrides else "")


def run_steps(cfg, steps):
    """Run the requested steps in order; exit 1 on the first failure."""
    unknown = sorted(set(steps) - set(STEPS))
    if unknown:
        sys.exit("unknown step(s): %s (available: %s)" % (", ".join(unknown), ", ".join(STEPS)))
    for name, (module, section) in STEPS.items():
        if name not in steps:
            continue
        log.info("=== %s ===", name)
        t0 = time.time()
        try:
            importlib.import_module("." + module, __package__).run(section(cfg))
        except Exception:
            log.exception("=== %s FAILED ===", name)
            sys.exit(1)
        log.info("=== %s done in %.1f s ===", name, time.time() - t0)
