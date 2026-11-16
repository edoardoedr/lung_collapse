"""Convert a config of the old schema (configs/<p>_wall_torch.json on main: surface_mesh /
registration / fem_setup / fem_fit sections) to the current one (inputs / geometry / fem_setup /
physics / solver / fit).

    python scripts/migrate_config.py OLD.json [OLD.json ...] --out configs/
    python scripts/migrate_config.py OLD.json --out configs/ --E 1700 --wall-update potential

The output keeps output_dir and the file names of the old config, so the geometry, hilum and
fem_setup outputs already there are found again (run only the collapse / fit step). The physics
is the physical model: gravity supine, E from --E, q0 fitted from the old start value, the wall of
the old config. Numerical settings (solver options, fit stops) are copied; options that no longer
exist are dropped. Old configs: git show main:configs/<name>.
"""

import argparse
import json
import sys
from pathlib import Path

DROPPED_SOLVER = ("wall_stiffness", "wall_eps", "material", "material_params", "slow_ramp", "wall_quadrature",
                  "support_k", "wall_update")
FIT_KEYS = ("target_error_mm", "time_budget_min", "hard_grace_min", "lsq_ftol", "lsq_xtol", "lsq_gtol",
            "level_patience", "level_min_improve", "fit_patience", "fit_min_improve", "sign_check_q")


def clean(d):
    return {k: v for k, v in d.items() if not k.startswith("_")}


def migrate(old, E=1700.0, wall_update="potential", run_name=None):
    sm, reg, hil = clean(old["surface_mesh"]), clean(old["registration"]), clean(old["hilum"])
    fs, ff = clean(old["fem_setup"]), clean(old["fem_fit"])
    if reg.get("method") != "none":
        raise ValueError("only unregistered configs (registration.method 'none') can be converted")
    so = dict(ff.get("solver_options", {}))
    anchor = fs.get("anchor_point", "hilum")
    wall = fs.get("wall")
    new = {
        "patient": old["patient"], "data_dir": old["data_dir"], "output_dir": old["output_dir"],
        "steps": ["check_inputs", "geometry", "hilum", "fem_setup", "fit"],
        "run_name": run_name or "fit",
        "inputs": {
            "_comment": "surfaces from 3D Slicer (data_dir). collapsed: null = no target (collapse step only)",
            "inflated": reg["inflated"], "collapsed": sm["input"], "structures": hil["structures"]},
        "check_inputs": {k: v for k, v in clean(old.get("check_inputs", {})).items()
                         if k not in ("collapsed", "inflated", "structures")},
        "geometry": {**{k: v for k, v in sm.items() if k not in ("input", "output", "output_space")},
                     "inflated_output": reg["output"], "collapsed_output": sm["output"]},
        "hilum": {"lung": "collapsed" if hil["lung"] == sm["input"] else "inflated", "output": hil["output"],
                  "max_ring_distance_mm": hil.get("max_ring_distance_mm", 50.0)},
        "fem_setup": {
            "_comment": "clamp: triangles within sphere radius x radius_factor of the anchor points. wall.surface: "
                        "'reference' (the inflated surface), a file, or null; grid_mm / smoothing_mm: the wall as a "
                        "smooth field (solver.wall_update potential). regions: {levels: [...]} only for "
                        "physics.perturbation regions",
            "clamp": {"points": [anchor] if isinstance(anchor, str) else anchor,
                      "radius_factor": fs.get("anchor_radius_factor", 2.0), "min_points": fs.get("anchor_min_points", 20),
                      "growth": fs.get("anchor_growth", 1.1), "max_growth_steps": fs.get("anchor_max_growth_steps", 10)},
            "mesh_size_mm": fs.get("mesh_size_mm", 14.0), "gmsh_timeout_s": fs.get("gmsh_timeout_s", 300),
            "wall": {"surface": wall, "tol_mm": fs.get("wall_tol_mm", 5.0), "grid_mm": 1.5 if wall else None,
                     "grid_margin_mm": 30.0, "smoothing_mm": 3.0}},
        "physics": {
            "_comment": "a parameter is a number (fixed) or {fit: true, init, bounds} (fitted by the fit step). "
                        "gravity: position of the inflated CT and of the collapse (supine | prone | left_lateral | "
                        "right_lateral | [x, y, z] LPS), null = no gravity. collapse: q0 (p/E) | p0_Pa | volume_ratio. "
                        "perturbation (fit only): null | {gradient: [ap, si, ml]} | {regions: [K, ...]}",
            "material": {"name": so.get("material", "neo_hookean"), "nu": ff.get("nu", 0.40),
                         "params": so.get("material_params", {})},
            "E_Pa": E, "tissue_density": ff.get("tissue_density", 250.0),
            "gravity": {"reference": "supine", "collapse": "supine"},
            "collapse": {"q0": {"fit": True, "init": ff.get("q0", 0.5), "bounds": ff.get("q_bounds", [-1.0, 3.0])}},
            "wall": {"stiffness": so.get("wall_stiffness", 20.0), "eps": so.get("wall_eps", 0.5)},
            "perturbation": None},
        "solver": {"_comment": "core: torch | warp | getfem (validation); the rest are the core's options "
                               "(pipeline/collapse/solvers/nodal.py)",
                   "core": ff.get("solver", "torch"), "wall_update": wall_update,
                   **{k: v for k, v in so.items() if k not in DROPPED_SOLVER}, "newton_maxit": 50},
        "collapse": {"sequence_frames": 0},
        "fit": {"symmetric_weight": ff.get("loss_symmetric_weight", 1.0), "best_by": "rms",
                **{k: ff[k] for k in FIT_KEYS if k in ff}},
    }
    if ff.get("free_nu"):
        new["physics"]["material"]["nu"] = {"fit": True, "init": ff.get("nu", 0.40),
                                            "bounds": ff.get("nu_bounds", [0.05, 0.45])}
    return new


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("configs", nargs="+")
    ap.add_argument("--out", required=True, help="folder for the new configs (<patient>.json)")
    ap.add_argument("--E", type=float, default=1700.0, help="physics.E_Pa")
    ap.add_argument("--wall-update", default="potential", choices=["potential", "outer"])
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in args.configs:
        old = json.loads(Path(f).read_text())
        new = migrate(old, args.E, args.wall_update)
        dst = out / ("%s.json" % old["patient"])
        dst.write_text(json.dumps(new, indent=2) + "\n")
        print("%s -> %s" % (f, dst))


if __name__ == "__main__":
    sys.exit(main())
