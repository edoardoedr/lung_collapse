"""Why does a solve with the wall fail? Newton trace of a Warp / torch core along a load ramp.

    python scripts/wall_diagnose.py --config configs/karl04_wall_torch.json
    python scripts/wall_diagnose.py --config configs/karl04_wall_torch.json --q 0.5 --options '{"wall_update": "newton"}'

Needs the fem_setup output of the config, and a core built on nodal.py (warp, torch): no GetFEM.
With --from-fit RUN it ramps instead to the best pressures of that fit (fem/RUN/best_state.npz), the
load a fit with the wall actually needs (regions up to the p/E bound). With --perturb r ... it then
re-solves from the final state with the pressures perturbed by up to r (relative, per region), as
the fit's trials do, and prints the gap error of every wall round.
Reaches mean p/E --q in --steps increments from the reference (as compare_warp_getfem.py [b]),
each warm-started from the previous one, and prints per increment: converged or not, Newton runs
and iterations, wall rounds, failure reasons, time, contact points, penetration and the smallest
element volume ratio J. For the first failed increment it prints the Newton iterations of its
first attempt (residual, line-search step, points in contact, deepest gap). The full trace is
written to output_dir/fem/<run_name>/wall_diagnose.csv.
"""

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.collapse import fit                                    # noqa: E402
from pipeline.collapse.geometry import WallDistance                  # noqa: E402
from pipeline.collapse.problem import CollapseProblem                # noqa: E402
from pipeline.collapse.solvers import get_solver                     # noqa: E402
from pipeline.collapse.solvers.nodal import NodalSolver              # noqa: E402
from pipeline.config import load_config                              # noqa: E402

FAILS = ("fail_inverted", "fail_maxit", "fail_linear", "fail_nonfinite", "fail_time", "fail_unsettled")


def min_J(s, U):
    """Smallest det(F) over the tetrahedra (1 = undeformed, <= 0 = inverted)."""
    u = U.reshape(-1, 3)[s.tets]
    D = s.Dm + np.stack([u[:, 1] - u[:, 0], u[:, 2] - u[:, 0], u[:, 3] - u[:, 0]], axis=2)
    return float((np.linalg.det(D) / np.linalg.det(s.Dm)).min())


def first_attempt(trace):
    """Records of the first path tried in a solve (up to the next 'path' event)."""
    out, seen = [], 0
    for r in trace:
        if r["event"] == "path":
            seen += 1
            if seen > 1:
                break
        out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--level", type=int, default=1, help="index of the fem_setup level (default 1 = second)")
    ap.add_argument("--q", type=float, default=0.2, help="mean p/E to reach")
    ap.add_argument("--nu", type=float, default=None, help="default: fem_fit.nu")
    ap.add_argument("--steps", type=int, default=10, help="load increments")
    ap.add_argument("--options", default="{}", help="JSON merged into solver_options")
    ap.add_argument("--from-fit", metavar="RUN", default=None,
                    help="ramp to the best pressures of fit RUN (fem/RUN/best_state.npz: its regions, p/E and nu) "
                         "instead of --level / --q")
    ap.add_argument("--perturb", type=float, nargs="*", default=[],
                    help="after the ramp, solves warm-started from the final state with p/E x (1 + r u), "
                         "u uniform in [-1, 1] per region, for each relative size r given (as the fit's trials)")
    ap.add_argument("--trials", type=int, default=3, help="perturbed solves per size (default 3)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    cfg = load_config(args.config)
    fc = cfg.fem_fit
    problem = CollapseProblem.load(fc.setup / "problem.npz")
    opts = dict(fc.solver_options, **json.loads(args.options))
    s = get_solver(fc.solver)(problem, opts)
    if not isinstance(s, NodalSolver):
        raise SystemExit("solver '%s' has no Newton trace: use a warp or torch config" % fc.solver)
    if not problem.has_wall:
        print("note: this fem_setup has no wall")
    wall = WallDistance(problem.wall_points, problem.wall_tris) if problem.has_wall else None

    sign = fit.sign_check(s, problem, fc)
    nu = fc.nu if args.nu is None else args.nu
    if args.from_fit:
        b = np.load(fc.workdir.parent / args.from_fit / "best_state.npz")
        labels, q = np.asarray(b["tri_labels"]), sign * np.asarray(b["q"], float)   # q: as the core sees it
        K = len(q)
        nu = float(b["nu"]) if args.nu is None else args.nu
        src = "best of %s (p/E %.2f .. %.2f)" % (args.from_fit, np.abs(q).min(), np.abs(q).max())
    else:
        L = min(args.level, len(problem.levels) - 1)
        K, labels = int(problem.levels[L]), problem.regions[L]
        q = sign * args.q * (1.0 + 0.3 * np.linspace(-1.0, 1.0, K))
        src = "mean p/E %g" % args.q
    s.set_regions(labels, K)
    o = s.options
    print("case %s, core %s, K=%d, %s, nu %.2f, %d increments | wall_update %s, stiffness %g, eps %g, "
          "settle %g mm, max rounds %d, load_steps %d, warm_substeps %s, slow_ramp %s, max_solve_s %s, support_k %g"
          % (cfg.patient, fc.solver, K, src, nu, args.steps, o["wall_update"], o["wall_stiffness"],
             o["wall_eps"], o["wall_settle_mm"], o["wall_max_updates"], o["load_steps"], o["warm_substeps"],
             o["slow_ramp"], o["max_solve_s"], o.get("support_k", 0.0)))
    print("%5s %4s %6s %6s %6s %6s %-34s %8s %9s %8s %7s %7s"
          % ("load", "ok", "newton", "iters", "rounds", "paths", "failures (inv/maxit/lin/nan/time/unsettled)",
             "time s", "contact", "pen mm", "max|U|", "min J"))
    rows, failed_trace = [], None
    s.reset()
    for k in range(1, args.steps + 1):
        lam = k / args.steps
        s.trace = []
        c0, t0 = s.stats(), time.time()
        Us = s.solve(q * lam, nu)
        dt = time.time() - t0
        c = {key: s.stats()[key] - c0.get(key, 0) for key in s.stats()}
        for r in s.trace:
            rows.append(dict(load=lam, **r))
        fails = "/".join(str(c[f]) for f in FAILS)
        if Us is None:
            print("%4d%% %4s %6d %6d %6d %6d %-34s %8.1f" % (100 * lam, "NO", c["newton_calls"], c["newton_iters"],
                                                            c["wall_updates"], sum(1 for r in s.trace if r["event"] == "path"),
                                                            fails, dt))
            failed_trace = s.trace
            break
        U = s.get_state()
        pen, ncon = np.nan, 0
        if wall is not None:
            d = wall(problem.reference + Us)[0] - problem.wall_allow
            pen, ncon = max(d.max(), 0.0), int((d > -0.5).sum())
        print("%4d%% %4s %6d %6d %6d %6d %-34s %8.1f %9d %8.3f %7.1f %7.3f"
              % (100 * lam, "ok", c["newton_calls"], c["newton_iters"], c["wall_updates"],
                 sum(1 for r in s.trace if r["event"] == "path"), fails, dt, ncon, pen,
                 np.linalg.norm(Us, axis=1).max(), min_J(s, U)))
    s.trace = None

    if args.perturb and failed_trace is None:
        # the fit's situation: a trial near a converged state, warm-started from it with its load
        U_base = s.get_state()
        load = np.where(labels >= 0, q[np.maximum(labels, 0)], np.nan)
        rng = np.random.default_rng(args.seed)
        print("\nperturbed solves from the final state (wall rounds: gap error of each re-linearisation, mm, "
              "first attempt; settled below %g):" % o["wall_settle_mm"])
        print("%7s %5s %4s %8s %6s %-34s  %s" % ("size", "trial", "ok", "time s", "paths",
                                                "failures (inv/maxit/lin/nan/time/unsettled)", "wall rounds"))
        for r in args.perturb:
            for t in range(args.trials):
                qp = q * (1.0 + r * rng.uniform(-1.0, 1.0, K))
                s.set_state(U_base, load)
                s.trace = []
                c0, t0 = s.stats(), time.time()
                Us = s.solve(qp, nu)
                dt = time.time() - t0
                c = {key: s.stats()[key] - c0.get(key, 0) for key in s.stats()}
                rounds = [rr["change"] for rr in first_attempt(s.trace) if rr["event"] == "wall"]
                print("%7.3f %5d %4s %8.1f %6d %-34s  %s%s"
                      % (r, t, "ok" if Us is not None else "NO", dt,
                         sum(1 for rr in s.trace if rr["event"] == "path"), "/".join(str(c[f]) for f in FAILS),
                         " ".join("%.2f" % v for v in rounds[:20]), " ..." if len(rounds) > 20 else ""))
                s.trace = None

    if failed_trace is not None:
        print("\nfirst attempt of the failed increment:")
        print("  %-6s %4s %12s %12s %8s %4s %8s %10s %8s" % ("event", "it", "|R|_1", "criterion", "alpha", "ls",
                                                             "contact", "max gap", "max|U|"))
        for r in first_attempt(failed_trace):
            e = r["event"]
            if e == "iter":
                print("  %-6s %4d %12.4e %12.4e %8s %4d %8s %10s %8.1f"
                      % (e, r["it"], r["res"], r["crit"], "-" if r["alpha"] is None else "%.3g" % r["alpha"],
                         r["n_ls"], r.get("n_contact", "-"),
                         "%.3f" % r["max_gap"] if "max_gap" in r else "-", r["max_u"]))
            elif e == "wall":
                print("  %-6s      wall re-linearised, gap of the previous linearisation off by %.3f mm near the wall" % (e, r["change"]))
            elif e == "fail":
                print("  %-6s      %s" % (e, r["reason"]))
            elif e == "step" and "at" in r:
                print("  %-6s      to %.4f of the path (step %.4f)" % (e, r["at"], r["size"]))
            elif e == "step":
                print("  %-6s      sub-step %d of %d" % (e, r["step"], r["of"]))
            elif e == "path":
                print("  %-6s      %s, %d step(s)" % (e, "from the last converged state" if r["warm"] else
                                                     "ramp from the reference", r["steps"]))
        paths = [r for r in failed_trace if r["event"] in ("path", "fail")]
        print("\nall attempts of the failed increment:")
        for r in paths:
            print("  " + ("%s, %d step(s)" % ("warm" if r["warm"] else "from zero", r["steps"]) if r["event"] == "path"
                          else "   -> failed: %s" % r["reason"]))

    out = fc.workdir / "wall_diagnose.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print("\ntrace: %s" % out)


if __name__ == "__main__":
    main()
