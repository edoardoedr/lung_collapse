"""Add the smooth wall field to an existing fem_setup output, without redoing the setup.

    python scripts/add_wall_field.py --config configs/karl04.json [--grid 1.5] [--smoothing 3] [--margin 30]

fem_setup builds the field (fem_setup.wall grid_mm / smoothing_mm); this script does only that
step on the problem.npz already there (same volume mesh, so the existing runs stay valid) and
rewrites it with the field: needed by solver.wall_update "potential". Use it for a fem_setup
output made before the field existed, or to change the grid or the smoothing.
"""

import argparse
import dataclasses
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eurelaps_lung.fem.problem import CollapseProblem                # noqa: E402
from eurelaps_lung.fem.setup import wall_field_arrays                # noqa: E402
from eurelaps_lung.config import load_config                              # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--grid", type=float, default=None, help="grid spacing [mm] (default: fem_setup.wall_grid_mm, else 1.5)")
    ap.add_argument("--smoothing", type=float, default=None, help="Gaussian sigma [mm] (default: fem_setup.wall_smoothing_mm)")
    ap.add_argument("--margin", type=float, default=None, help="grid margin [mm] (default: fem_setup.wall_grid_margin_mm)")
    ap.add_argument("--set", nargs="+", metavar="KEY=VALUE", default=[], help="config overrides as in main_fit.py")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")

    cfg = load_config(args.config, args.set).fem_setup
    cfg = dataclasses.replace(cfg, wall_grid_mm=args.grid or cfg.wall_grid_mm or 1.5,
                              wall_smoothing_mm=cfg.wall_smoothing_mm if args.smoothing is None else args.smoothing,
                              wall_grid_margin_mm=args.margin or cfg.wall_grid_margin_mm)
    path = cfg.workdir / "problem.npz"
    problem = CollapseProblem.load(path)
    if not problem.has_wall:
        sys.exit("%s has no wall (fem_setup.wall.surface is null)" % path)
    arrays, _ = wall_field_arrays(cfg, problem.wall_points, problem.wall_tris, problem.reference,
                                  problem.target_points if problem.has_target else None)
    dataclasses.replace(problem, **arrays).save(path)
    print("wrote %s with the wall field (grid %g mm, smoothing %g mm)" % (path, cfg.wall_grid_mm, cfg.wall_smoothing_mm))


if __name__ == "__main__":
    main()
