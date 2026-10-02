"""Check the analytic Jacobian of the FEM core against finite differences, and compare fits.

    python scripts/check_jacobian.py --config configs/karl04.json
    python scripts/check_jacobian.py --config configs/karl04_wall.json
    python scripts/check_jacobian.py --config configs/karl04.json --compare-fit

Needs the fem_setup output of the config (run the pipeline up to fem_setup first).

1. Jacobian check: converges solve(q, nu) at a non-trivial q on one level, then compares every
   column of solver.jacobian(q, nu, with_nu=True) with central finite differences of solve(),
   each warm-started from the same converged state. Newton is run with --newton-tol (tighter
   than the fit's default) so that the finite differences are not limited by the solve tolerance.
   Expected relative error < 1e-5 per column without a wall; with a wall the Jacobian keeps the
   contact linearisation fixed, so larger errors are possible where points touch the wall.
2. --compare-fit: runs fem_fit twice (jacobian "2-point" and "analytic") into
   output_dir/fem/fit_2-point and fit_analytic, and prints errors, stop reasons, solves, time
   and pressures per level side by side.
"""

import argparse
import dataclasses
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.collapse import fit                                    # noqa: E402
from pipeline.collapse.problem import CollapseProblem                # noqa: E402
from pipeline.collapse.solvers import get_solver                     # noqa: E402
from pipeline.config import load_config                              # noqa: E402


def linear_solvers():
    """Which sparse linear solvers this GetFEM build has (md.solve picks MUMPS if present)."""
    import getfem as gf
    A = gf.Spmat("identity", 3)
    out = {}
    for name in ("mumps", "superlu"):
        try:
            gf.linsolve(name, A, np.ones(3))
            out[name] = "available"
        except Exception as e:
            out[name] = "not available (%s)" % str(e).splitlines()[0][:60]
    return out


def check(cfg, args):
    fc = cfg.fem_fit
    problem = CollapseProblem.load(fc.setup / "problem.npz")
    opts = dict(fc.solver_options, newton_tol=args.newton_tol)
    solver = get_solver(fc.solver)(problem, opts)
    if fc.solver == "getfem":
        print("GetFEM linear solvers:", linear_solvers(), "| linear_solver option:", opts.get("linear_solver"))
    sign = fit.sign_check(solver, problem, fc)

    L = min(args.level, len(problem.levels) - 1)
    K, labels = int(problem.levels[L]), problem.regions[L]
    solver.set_regions(labels, K)
    q = sign * args.q * (1.0 + 0.3 * np.linspace(-1.0, 1.0, K))       # non-uniform pressures
    nu = fc.nu
    print("level K=%d, q = %s, nu = %.3f, wall %s, newton_tol %g, h %g"
          % (K, np.round(q, 3), nu, "on" if problem.has_wall else "off", args.newton_tol, args.h))

    t0 = time.time()
    if solver.solve(q, nu) is None:
        raise SystemExit("solve(q, nu) did not converge with newton_tol %g%s" % (
            args.newton_tol, "; with a wall try --newton-tol 1e-8 --h 1e-4 (the contact penalty is "
            "only C1, Newton may not reach very tight tolerances) or a smaller --q" if problem.has_wall else ""))
    t_solve = time.time() - t0
    U0 = solver.get_state()
    t0 = time.time()
    J = solver.jacobian(q, nu, with_nu=True)
    t_jac = time.time() - t0
    if J is None:
        raise SystemExit("the solver returned no Jacobian")

    def at(qq, nn):
        solver.set_state(U0)                                          # same warm start each time
        Us = solver.solve(qq, nn)
        if Us is None:
            raise SystemExit("finite-difference solve did not converge")
        return Us.ravel()

    h, t_fd, worst = args.h, 0.0, 0.0
    print("%-8s %12s %12s %12s" % ("column", "|J|", "|FD|", "rel. error"))
    for c in range(K + 1):
        t0 = time.time()
        if c < K:
            e = np.zeros(K)
            e[c] = h
            fd = (at(q + e, nu) - at(q - e, nu)) / (2 * h)
        else:
            fd = (at(q, nu + h) - at(q, nu - h)) / (2 * h)
        t_fd += time.time() - t0
        rel = np.linalg.norm(J[:, c] - fd) / max(np.linalg.norm(fd), 1e-300)
        worst = max(worst, rel)
        print("%-8s %12.4e %12.4e %12.2e" % ("q%d" % c if c < K else "nu", np.linalg.norm(J[:, c]),
                                              np.linalg.norm(fd), rel))
    print("max relative error %.2e (%s; threshold 1e-5 without wall)"
          % (worst, "OK" if worst < 1e-5 else "CHECK"))
    print("timing: one solve %.2f s, analytic Jacobian (%d columns) %.2f s, central FD %.2f s"
          % (t_solve, K + 1, t_jac, t_fd))


def compare_fit(cfg):
    fc, rows = cfg.fem_fit, {}
    for mode in ("2-point", "analytic"):
        wd = fc.workdir.parent / ("fit_" + mode)
        c = dataclasses.replace(fc, jacobian=mode, workdir=wd, output=wd / "lung_fem_fit.vtp")
        t0 = time.time()
        fit.run(c)
        res = json.loads((wd / "result_summary.json").read_text())
        res["wall_time_min"] = (time.time() - t0) / 60
        import pyvista as pv
        e = pv.read(wd / "lung_fem_fit.vtp")["Error_mm"]
        res["rms_err_mm"] = float(np.sqrt((e ** 2).mean()))
        rows[mode] = res
    a, b = rows["2-point"], rows["analytic"]
    print("\n%-26s %18s %18s" % ("", "2-point", "analytic"))
    for k in ("mean_err_mm", "rms_err_mm", "assd_mm", "max_err_mm", "nu", "evals", "wall_time_min", "stop_reason"):
        fmt = "%18s" if isinstance(a[k], str) else "%18.3f"
        print("%-26s " % k + fmt % a[k] + " " + fmt % b[k])
    print("\nper level: K, stop, solves, minutes, best mean error")
    for r in (a, b):
        t = [lv["elapsed_min"] for lv in r["levels"]]
        for lv, dt in zip(r["levels"], np.diff([0.0] + t)):
            lv["elapsed_min"] = dt
    for la, lb in zip(a["levels"], b["levels"]):
        print("  K=%-3d  %-22s %5d %7.1f %8.3f   |   %-22s %5d %7.1f %8.3f"
              % (la["K"], la["stop"], la["evals"], la["elapsed_min"], la["best_mean_err_mm"],
                 lb["stop"], lb["evals"], lb["elapsed_min"], lb["best_mean_err_mm"]))
    print("\nfinal pressures [Pa] (K=%d vs K=%d):" % (a["K"], b["K"]))
    for i in range(max(len(a["pressures_Pa"]), len(b["pressures_Pa"]))):
        pa = a["pressures_Pa"][i] if i < len(a["pressures_Pa"]) else float("nan")
        pb = b["pressures_Pa"][i] if i < len(b["pressures_Pa"]) else float("nan")
        print("  region %2d  %10.1f  %10.1f" % (i, pa, pb))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--level", type=int, default=1, help="index of the fem_setup level (default 1 = second)")
    ap.add_argument("--q", type=float, default=0.5, help="mean p/E of the check")
    ap.add_argument("--h", type=float, default=1e-5, help="finite-difference step on q and nu")
    ap.add_argument("--newton-tol", type=float, default=1e-11)
    ap.add_argument("--compare-fit", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)          # print progress also through | tee
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    cfg = load_config(args.config)
    check(cfg, args)
    if args.compare_fit:
        compare_fit(cfg)


if __name__ == "__main__":
    main()
