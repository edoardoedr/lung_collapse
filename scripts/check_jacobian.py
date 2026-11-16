"""Check the analytic Jacobian of the FEM core against finite differences.

    python scripts/check_jacobian.py --config configs/karl04.json
    python scripts/check_jacobian.py --config configs/karl04.json --set solver.wall_update=outer
    python scripts/check_jacobian.py --config configs/karl04.json --solver getfem --set fem_setup.wall.surface=null
    python scripts/check_jacobian.py --config configs/karl04.json --options '{"device": "cpu"}'

Needs the fem_setup output of the config. --solver checks another core than solver.core (the
options it does not accept are dropped), --options is merged into the core's options.

1. Core: the free surface is split into --bands regions along the body axis, solve(q, nu[, body])
   is converged at a non-uniform q, and every column of solver.jacobian (q per region, nu, and the
   body force when the config has different reference / collapse positions) is compared with
   central finite differences of solve(), each warm-started from the same converged state.
2. Load model: the same for the parameters of the config's last fit level (q0, gravity slope,
   perturbation, nu), i.e. the chain rule of load.py on top of the core's Jacobian.

Newton runs with --newton-tol (tighter than the default) so that the finite differences are not
limited by the solve tolerance. Expected relative error < 1e-5: without a wall, and with the wall
as a smooth field (wall_update "potential"). With wall_update "outer" the Jacobian keeps the
contact linearisation fixed, so larger errors are possible where points touch the wall.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.collapse import fit                                    # noqa: E402
from pipeline.collapse.load import LoadModel                         # noqa: E402
from pipeline.collapse.problem import CollapseProblem                # noqa: E402
from pipeline.collapse.solvers import get_solver                     # noqa: E402
from pipeline.config import load_config                              # noqa: E402


def compare(names, J, fd_column):
    """Print |J|, |FD| and the relative error per column; returns the largest error."""
    worst, t_fd = 0.0, 0.0
    print("%-16s %12s %12s %12s" % ("column", "|J|", "|FD|", "rel. error"))
    for c, name in enumerate(names):
        t0 = time.time()
        fd = fd_column(c)
        t_fd += time.time() - t0
        rel = np.linalg.norm(J[:, c] - fd) / max(np.linalg.norm(fd), 1e-300)
        worst = max(worst, rel)
        print("%-16s %12.4e %12.4e %12.2e" % (name, np.linalg.norm(J[:, c]), np.linalg.norm(fd), rel))
    print("max relative error %.2e (%s; threshold 1e-5), central FD %.2f s" % (worst, "OK" if worst < 1e-5 else "CHECK", t_fd))
    return worst


def check(cfg, core, opts, args):
    problem = CollapseProblem.load(cfg.setup_dir / "problem.npz")
    ph = cfg.physics
    model = LoadModel(ph, problem, fit.hilum_position(cfg))
    solver = get_solver(core)(problem, dict(opts, newton_tol=args.newton_tol))
    sign = fit.sign_check(solver, problem, cfg.fit.sign_check_q, ph.nu.value)
    body_on = model.has_body
    if body_on:
        solver.set_body(model.body_direction)
    nu = ph.nu.value
    h = args.h
    hint = ("; with a wall try --newton-tol 1e-8 --h 1e-4 or a smaller --q" if problem.has_wall else "")

    def solve(q, nn, body):
        return solver.solve(q, nn, body) if body_on else solver.solve(q, nn)

    def restart(U0, load, body):
        if body_on:
            solver.set_state(U0, load, body)
        else:
            solver.set_state(U0, load)

    # 1. core: regions = bands along the superior-inferior axis
    free = ~problem.clamped_tri
    z = problem.reference[problem.tris].mean(1)[:, 2]
    K = args.bands
    edges = np.quantile(z[free], np.linspace(0, 1, K + 1)[1:-1])
    labels = np.where(free, np.searchsorted(edges, z), -1)
    solver.set_regions(labels, K)
    q = sign * args.q * (1.0 + 0.3 * np.linspace(-1.0, 1.0, K))       # non-uniform pressures
    body = model.slope_of_E(ph.E_Pa.value) if body_on else 0.0
    print("\n[1] core '%s': %d bands, q = %s, nu = %.3f%s, wall %s, newton_tol %g, h %g"
          % (core, K, np.round(q, 3), nu, ", body %.3f" % body if body_on else "",
             "%s" % opts.get("wall_update", "outer") if problem.has_wall else "off", args.newton_tol, h))
    t0 = time.time()
    if solve(q, nu, body) is None:
        raise SystemExit("solve did not converge with newton_tol %g%s" % (args.newton_tol, hint))
    t_solve = time.time() - t0
    U0 = solver.get_state()
    load = np.where(labels >= 0, q[np.maximum(labels, 0)], np.nan)
    t0 = time.time()
    J = solver.jacobian(q, nu, True, True) if body_on else solver.jacobian(q, nu, True)
    t_jac = time.time() - t0
    if J is None:
        raise SystemExit("the solver returned no Jacobian")

    def at(qq, nn, bb):
        restart(U0, load, body)                                       # same warm start each time
        Us = solve(qq, nn, bb)
        if Us is None:
            raise SystemExit("finite-difference solve did not converge" + hint)
        return Us.ravel()

    def fd_core(c):
        if c < K:
            e = np.zeros(K)
            e[c] = h
            return (at(q + e, nu, body) - at(q - e, nu, body)) / (2 * h)
        if c == K:
            return (at(q, nu + h, body) - at(q, nu - h, body)) / (2 * h)
        return (at(q, nu, body + h) - at(q, nu, body - h)) / (2 * h)

    names = ["q%d" % k for k in range(K)] + ["nu"] + (["body"] if body_on else [])
    worst = compare(names, J, fd_core)
    print("timing: one solve %.2f s, analytic Jacobian (%d columns) %.2f s" % (t_solve, len(names), t_jac))

    # 2. the config's load model
    try:
        level = model.fit_levels(cfg.fit)[-1]
    except ValueError as e:
        print("\n[2] load model: skipped (%s)" % e)
        return worst
    area = model.area
    x = level.x0_from(model.start_q_tri() * args.q / 0.5, nu, area)
    solver.reset()
    solver.set_regions(level.labels, level.n_solver)
    with_nu, with_body = level.nu_index is not None, bool(np.any(level.body_coef != 0.0))
    print("\n[2] load model, %d parameters: %s" % (level.P, ", ".join("%s %.4g" % nv for nv in list(zip(level.names, x))[:8])))

    def run(xx):
        return solve(sign * level.q_solver(xx), level.nu(xx), level.body(xx))

    if run(x) is None:
        raise SystemExit("solve did not converge" + hint)
    U0, load, body = solver.get_state(), sign * level.q_tri(x), level.body(x)
    if with_body:
        Jc = solver.jacobian(sign * level.q_solver(x), level.nu(x), with_nu, True)
    else:
        Jc = solver.jacobian(sign * level.q_solver(x), level.nu(x), with_nu)
    n = level.n_solver
    J = level.chain(sign * Jc[:, :n], Jc[:, n] if with_nu else None, Jc[:, n + int(with_nu)] if with_body else None)

    def fd_level(c):
        out = []
        for sgn in (1.0, -1.0):
            xx = x.copy()
            xx[c] += sgn * h
            restart(U0, load, body)
            Us = run(xx)
            if Us is None:
                raise SystemExit("finite-difference solve did not converge" + hint)
            out.append(Us.ravel())
        return (out[0] - out[1]) / (2 * h)

    cols = list(range(min(level.P, args.max_columns)))
    if level.P > len(cols):
        print("(first %d of %d parameters)" % (len(cols), level.P))
    return max(worst, compare([level.names[c] for c in cols], J[:, cols], lambda i: fd_level(cols[i])))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="+", metavar="KEY=VALUE", default=[], help="config overrides as in main.py")
    ap.add_argument("--bands", type=int, default=4, help="pressure regions of the core check")
    ap.add_argument("--q", type=float, default=0.5, help="mean p/E of the check")
    ap.add_argument("--h", type=float, default=1e-5, help="finite-difference step")
    ap.add_argument("--newton-tol", type=float, default=1e-11)
    ap.add_argument("--max-columns", type=int, default=12, help="parameters checked in the load-model part")
    ap.add_argument("--solver", help="core to check instead of solver.core (getfem, warp, torch)")
    ap.add_argument("--options", default="{}", help="JSON merged into the core's options")
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)          # print progress also through | tee
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    cfg = load_config(args.config, args.set)
    core = args.solver or cfg.solver.core
    opts = dict(cfg.solver_options(core), **json.loads(args.options))
    check(cfg, core, opts, args)


if __name__ == "__main__":
    main()
