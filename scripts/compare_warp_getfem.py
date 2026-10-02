"""Validate the Warp core against the GetFEM core on a real case (run where both are installed).

    python scripts/compare_warp_getfem.py --config configs/karl04_wall.json            # a, b, c
    python scripts/compare_warp_getfem.py --config configs/karl04_wall.json --timing   # + e
    python scripts/compare_warp_getfem.py --config configs/karl04_wall.json --fit      # + d (long)

Needs the fem_setup output of the config. With a wall in the setup, (a) runs on a copy of the
problem without it and (b) on the problem as is.

  0. face quadrature: GetFEM's IM_TETRAHEDRON(3) face rule vs the one hard-coded in warp_solver
  a. no wall: same mesh, several (q, nu) up to large collapse, solved from the reference by both
     -> max / mean nodal |U_warp - U_getfem| (relative to max |U|), Newton iterations of both
  b. wall: same, plus the residual penetration of both
  c. WarpSolver.jacobian vs central finite differences of its own solve() (newton_tol 1e-11;
     with a wall 1e-8 and h >= 1e-4, the contact penalty being only C1),
     and vs GetFEMSolver.jacobian
  d. full fit with each core (analytic Jacobian): errors, stop reasons, solves, time, pressures
  e. Warp timing per forward solve and per Jacobian, for linear_solver cudss / pardiso / scipy,
     split into assembly and linear solve
"""

import argparse
import dataclasses
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.collapse import fit                                    # noqa: E402
from pipeline.collapse.geometry import WallDistance                  # noqa: E402
from pipeline.collapse.problem import CollapseProblem                # noqa: E402
from pipeline.collapse.solvers import get_solver                     # noqa: E402
from pipeline.collapse.solvers.warp_solver import FACE_QUAD_B, FACE_QUAD_W  # noqa: E402
from pipeline.config import load_config                              # noqa: E402

CASES = [(0.2, 0.30), (0.5, 0.30), (1.0, 0.30), (1.0, 0.40), (2.0, 0.40)]   # (mean p/E, nu)


def without_wall(problem):
    return dataclasses.replace(problem, wall_points=np.zeros((0, 3)), wall_tris=np.zeros((0, 3), dtype=np.int64),
                               wall_allow=np.zeros(0))


def make(name, problem, fc, **opts):
    o = dict(fc.solver_options)
    if name == "warp":
        o.pop("linear_solver", None)                    # GetFEM names (mumps, ...) mean nothing to Warp
    o.update(opts)
    return get_solver(name)(problem, o)


def getfem_nodal(g, problem, U):
    """GetFEM dof vector -> (n_nodes, 3) in problem.nodes order."""
    _, idx = cKDTree(g.mfu.basic_dof_nodes().T).query(problem.nodes, k=3)
    return np.asarray(U)[np.sort(idx, axis=1)]


def pressures(K, scale):
    return scale * (1.0 + 0.3 * np.linspace(-1.0, 1.0, K))


def quadrature():
    import getfem as gf
    im = gf.Integ("IM_TETRAHEDRON(3)")
    print("\n[0] face quadrature")
    try:
        pts, w = np.asarray(im.face_pts(0)), np.asarray(im.face_coeffs(0))
        print("  GetFEM IM_TETRAHEDRON(3) face 0: %d points, weights / sum = %s"
              % (len(w), np.round(np.sort(w / w.sum()), 6).tolist()))
        print("  GetFEM points (reference tet coords):\n%s" % np.round(pts.T if pts.shape[0] == 3 else pts, 6))
    except Exception as e:
        print("  could not read GetFEM face points: %s" % e)
    print("  warp_solver IM_TRIANGLE(3): %d points, weights = %s"
          % (len(FACE_QUAD_W), np.round(np.sort(FACE_QUAD_W), 6).tolist()))


def compare_forward(tag, problem, fc, K, labels, sign):
    print("\n[%s] forward solves from the reference, wall %s" % (tag, "on" if problem.has_wall else "off"))
    g, w = make("getfem", problem, fc), make("warp", problem, fc)
    for s in (g, w):
        s.set_regions(labels, K)
    wall = WallDistance(problem.wall_points, problem.wall_tris) if problem.has_wall else None
    print("  %-6s %-5s %10s %10s %10s %8s %8s %8s %8s %10s %10s"
          % ("p/E", "nu", "max|U|", "max diff", "mean diff", "rel", "it GF", "it Warp", "t GF", "pen GF", "pen Warp"))
    for scale, nu in CASES:
        q = sign * pressures(K, scale)
        res = {}
        for name, s in (("g", g), ("w", w)):
            s.reset()
            c0, t0 = s.stats(), time.time()
            Us = s.solve(q, nu)
            dt = time.time() - t0
            it = s.stats()["newton_iters"] - c0["newton_iters"]
            if Us is None:
                res[name] = None
                continue
            U = getfem_nodal(s, problem, s.get_state()) if name == "g" else s.get_state().reshape(-1, 3)
            pen = (max((wall(problem.reference + Us)[0] - problem.wall_allow).max(), 0.0) if wall else np.nan)
            res[name] = (U, it, dt, pen)
        if res["g"] is None or res["w"] is None:
            print("  %-6.2f %-5.2f  not converged: GetFEM %s, Warp %s"
                  % (scale, nu, res["g"] is not None, res["w"] is not None))
            continue
        d = np.linalg.norm(res["w"][0] - res["g"][0], axis=1)
        umax = np.linalg.norm(res["g"][0], axis=1).max()
        print("  %-6.2f %-5.2f %10.3f %10.2e %10.2e %8.1e %8d %8d %7.1fs %10.3f %10.3f"
              % (scale, nu, umax, d.max(), d.mean(), d.max() / max(umax, 1e-30), res["g"][1], res["w"][1],
                 res["g"][2], res["g"][3], res["w"][3]))
    print("  target: rel < 1e-6 without wall")


def compare_jacobian(problem, fc, K, labels, sign, h):
    # the contact penalty is only C1: with a wall Newton cannot reach 1e-11, so tol and h are relaxed
    tol, h = (1e-8, max(h, 1e-4)) if problem.has_wall else (1e-11, h)
    print("\n[c] Jacobian, wall %s, newton_tol %g, h %g" % ("on" if problem.has_wall else "off", tol, h))
    q, nu = sign * pressures(K, 0.5), fc.nu
    J = {}
    for name in ("warp", "getfem"):
        s = make(name, problem, fc, newton_tol=tol)
        s.set_regions(labels, K)
        if s.solve(q, nu) is None:
            raise SystemExit("%s: solve(q, nu) did not converge" % name)
        U0 = s.get_state()
        t0 = time.time()
        J[name] = s.jacobian(q, nu, with_nu=True)
        print("  %s jacobian: %.2f s" % (name, time.time() - t0))
        if name == "warp":
            def at(qq, nn):
                s.set_state(U0)
                r = s.solve(qq, nn)
                if r is None:
                    raise SystemExit("warp finite-difference solve did not converge")
                return r.ravel()
            fd = []
            for c in range(K + 1):
                if c < K:
                    e = np.zeros(K)
                    e[c] = h
                    fd.append((at(q + e, nu) - at(q - e, nu)) / (2 * h))
                else:
                    fd.append((at(q, nu + h) - at(q, nu - h)) / (2 * h))
    print("  %-6s %14s %16s" % ("column", "Warp vs FD", "Warp vs GetFEM"))
    for c in range(K + 1):
        jw, jg = J["warp"][:, c], J["getfem"][:, c]
        print("  %-6s %14.2e %16.2e" % ("q%d" % c if c < K else "nu",
                                        np.linalg.norm(jw - fd[c]) / np.linalg.norm(fd[c]),
                                        np.linalg.norm(jw - jg) / np.linalg.norm(jg)))
    print("  target: Warp vs FD < 1e-5 without wall")


def timing(problem, fc, K, labels, sign):
    print("\n[e] Warp timing (one forward solve from the reference, one Jacobian)")
    q, nu = sign * pressures(K, 0.5), fc.nu
    for backend in ("cudss", "pardiso", "scipy"):
        try:
            s = make("warp", problem, fc, linear_solver=backend)
        except Exception as e:
            print("  %-8s unavailable (%s)" % (backend, str(e).splitlines()[0][:80]))
            continue
        s.set_regions(labels, K)
        for what in ("forward", "jacobian"):
            s.times = dict(assembly_s=0.0, linear_s=0.0, assemblies=0, linear_solves=0)
            t0 = time.time()
            try:
                ok = s.solve(q, nu) is not None if what == "forward" else s.jacobian(q, nu, True) is not None
            except Exception as e:
                print("  %-8s %-8s failed: %s" % (backend, what, str(e).splitlines()[0][:80]))
                break
            t = s.times
            print("  %-8s %-8s %7.2f s total | assembly %6.2f s (%d) | linear solve %6.2f s (%d) | %s"
                  % (backend, what, time.time() - t0, t["assembly_s"], t["assemblies"], t["linear_s"],
                     t["linear_solves"], "ok" if ok else "FAILED"))


def compare_fit(cfg):
    fc, rows = cfg.fem_fit, {}
    for name in ("getfem", "warp"):
        wd = fc.workdir.parent / ("fit_" + name)
        opts = dict(fc.solver_options)
        if name == "warp":
            opts.pop("linear_solver", None)
        c = dataclasses.replace(fc, solver=name, solver_options=opts, jacobian="analytic", workdir=wd,
                                output=wd / "lung_fem_fit.vtp")
        t0 = time.time()
        fit.run(c)
        res = json.loads((wd / "result_summary.json").read_text())
        res["wall_time_min"] = (time.time() - t0) / 60
        import pyvista as pv
        e = pv.read(wd / "lung_fem_fit.vtp")["Error_mm"]
        res["rms_err_mm"] = float(np.sqrt((e ** 2).mean()))
        rows[name] = res
    a, b = rows["getfem"], rows["warp"]
    print("\n[d] full fit\n%-22s %18s %18s" % ("", "getfem", "warp"))
    for k in ("mean_err_mm", "rms_err_mm", "assd_mm", "max_err_mm", "nu", "evals", "wall_time_min", "stop_reason"):
        fmt = "%18s" if isinstance(a[k], str) else "%18.3f"
        print("%-22s " % k + fmt % a[k] + " " + fmt % b[k])
    for r in (a, b):
        t = [lv["elapsed_min"] for lv in r["levels"]]
        for lv, dt in zip(r["levels"], np.diff([0.0] + t)):
            lv["elapsed_min"] = dt
    print("per level: K, stop, solves, minutes, best mean error")
    for la, lb in zip(a["levels"], b["levels"]):
        print("  K=%-3d  %-22s %5d %7.1f %8.3f   |   %-22s %5d %7.1f %8.3f"
              % (la["K"], la["stop"], la["evals"], la["elapsed_min"], la["best_mean_err_mm"],
                 lb["stop"], lb["evals"], lb["elapsed_min"], lb["best_mean_err_mm"]))
    print("final pressures [Pa] (K=%d vs K=%d):" % (a["K"], b["K"]))
    for i in range(max(len(a["pressures_Pa"]), len(b["pressures_Pa"]))):
        pa = a["pressures_Pa"][i] if i < len(a["pressures_Pa"]) else float("nan")
        pb = b["pressures_Pa"][i] if i < len(b["pressures_Pa"]) else float("nan")
        print("  region %2d  %10.1f  %10.1f" % (i, pa, pb))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--level", type=int, default=1, help="index of the fem_setup level (default 1 = second)")
    ap.add_argument("--h", type=float, default=1e-5, help="finite-difference step for (c)")
    ap.add_argument("--timing", action="store_true", help="also run (e)")
    ap.add_argument("--fit", action="store_true", help="also run (d), two full fits")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    cfg = load_config(args.config)
    fc = cfg.fem_fit
    problem = CollapseProblem.load(fc.setup / "problem.npz")
    L = min(args.level, len(problem.levels) - 1)
    K, labels = int(problem.levels[L]), problem.regions[L]
    print("case %s: %d nodes, %d tets, level K=%d, wall in setup: %s"
          % (cfg.patient, len(problem.nodes), len(problem.tets), K, problem.has_wall))

    quadrature()
    signs = {}
    for name in ("getfem", "warp"):
        s = make(name, without_wall(problem), fc)
        signs[name] = fit.sign_check(s, without_wall(problem), fc)
    print("\nsign check: getfem %+g, warp %+g (must be equal)" % (signs["getfem"], signs["warp"]))
    sign = signs["getfem"]

    compare_forward("a", without_wall(problem), fc, K, labels, sign)
    if problem.has_wall:
        compare_forward("b", problem, fc, K, labels, sign)
    compare_jacobian(without_wall(problem), fc, K, labels, sign, args.h)
    if problem.has_wall:
        compare_jacobian(problem, fc, K, labels, sign, args.h)
    if args.timing:
        timing(without_wall(problem), fc, K, labels, sign)
    if args.fit:
        logging.getLogger().setLevel(logging.INFO)
        compare_fit(cfg)


if __name__ == "__main__":
    main()
