"""Step 5b - fem_fit: regional pleural pressures that collapse the inflated lung onto the target.

  1. load output_dir/fem/setup/problem.npz and the FEM core named in the config
  2. sign check: q > 0 must move the surface inward (else q is flipped for this core)
  3. coarse-to-fine levels (regions from fem_setup), each warm-started from the best so far;
     per level, bounded least squares (lsq) on the point residual (loss "point": fitted - target;
     "plane": its component along the target normal, tangential part weighted by
     loss_tangent_weight) + smoothness between adjacent regions, or Nelder-Mead (nm); Poisson's ratio optionally free. The lsq Jacobian
     comes from the core (one linear solve per column) when it provides one, else from finite
     differences (one nonlinear solve per column)
     with a cavity wall in fem_setup, the core also keeps the surface inside it (contact)
  4. stops on: target error | time budget | plateau | levels exhausted | Ctrl+C / SIGTERM
  5. output: fitted surface (main result) + output_dir/fem/fit/ (history, checkpoints, summary)

Only the solver knows the FEM library: see solvers/base.py for the contract.
"""

import json
import logging
import time

import numpy as np
from scipy.optimize import least_squares, minimize
from scipy.optimize._numdiff import approx_derivative

from ..data_io import write_surface
from .geometry import (WallDistance, assd, face_adjacency, kabsch, polydata, rotation_deg, tri_geometry,
                       vertex_normals)
from .problem import CollapseProblem
from .regions import region_pairs
from .run_control import Stop, Tracker, guarded
from .solvers import get_solver

log = logging.getLogger(__name__)


def sign_check(solver, problem, cfg):
    """+1 if q > 0 collapses with this core, -1 if the core has the opposite convention."""
    solver.set_regions(np.where(problem.clamped_tri, -1, 0), 1)
    Us = solver.solve(np.array([cfg.sign_check_q]), cfg.nu)
    if Us is None:
        raise RuntimeError("first forward solve failed (clamp / volume mesh quality?)")
    free = np.setdiff1d(np.arange(len(Us)), problem.clamped_points)
    un = np.einsum("ij,ij->i", Us, vertex_normals(problem.reference, problem.tris))[free].mean()
    solver.reset()
    if un > 0:
        log.warning("sign check: q > 0 moves the surface outward with '%s' -> q is flipped "
                    "(fix the core's sign convention)", cfg.solver)
        return -1.0
    log.info("sign check OK: q = %.2f moves the surface %.2f mm inward on average", cfg.sign_check_q, -un)
    return 1.0


class Loss:
    """Residual of the free surface points, d = fitted - target: P d with P = I ("point") or
    n n^T + w (I - n n^T) ("plane", n = target vertex normal, w = loss_tangent_weight). The same P
    applies to the Jacobian rows, so the analytic Jacobian of any core still works.

    Clamped points are left out: they do not move, so their error is a constant that does not
    change the fit, but it would weigh on the reported errors, the target-error stop and the
    plateau tests."""

    def __init__(self, cfg, X_tgt, tris, free):
        self.plane = cfg.loss == "plane"
        self.w = cfg.loss_tangent_weight
        self.n = vertex_normals(X_tgt, tris)                 # all points (export)
        self.free = np.asarray(free, bool)
        self.nf = self.n[self.free]

    def project(self, d):
        """d (N, 3) for all points -> residual (n_free, 3)."""
        d = d[self.free]
        if not self.plane:
            return d
        dn = np.einsum("ij,ij->i", d, self.nf)[:, None] * self.nf
        return dn + self.w * (d - dn)

    def jacobian(self, J):
        """J (3N, P) of the displacement of all points -> of the residual (3 n_free, P)."""
        J3 = J.reshape(len(self.n), 3, -1)[self.free]
        if self.plane:
            Jn = self.nf[:, :, None] * np.einsum("ik,ikp->ip", self.nf, J3)[:, None, :]
            J3 = Jn + self.w * (J3 - Jn)
        return J3.reshape(-1, J.shape[1])


def rigid_part(X_from, X_to):
    """Rotation [deg] and centroid shift [mm] of the best rigid fit X_from -> X_to."""
    R, t = kabsch(X_from, X_to)
    c = X_from.mean(0)
    return rotation_deg(R), float(np.linalg.norm(R @ c + t - c))


def run_level(L, K, tri_labels, x0, lb, ub, ctx):
    """One level of K regions; returns (stop reason, best mean error of the level)."""
    solver, trk, cfg, sign = ctx["solver"], ctx["trk"], ctx["cfg"], ctx["sign"]
    X_ref, X_tgt, loss = ctx["X_ref"], ctx["X_tgt"], ctx["loss"]
    pairs = region_pairs(tri_labels, ctx["adj"])
    nu_fixed = cfg.nu
    solver.set_regions(tri_labels, K)
    trk.new_level()
    stats0 = solver.stats()
    log.info("--- level %d: K=%d regions, %d parameters, %d adjacent pairs, %s ---",
             L, K, len(x0), len(pairs), cfg.optimizer)

    def unpack(x):
        return (x[:K], x[K]) if cfg.free_nu else (x, nu_fixed)

    last = dict(x=None, r=None, ok=False, J=None)          # last residual evaluation / Jacobian
    jstats = dict(n=0, s=0.0, fd=0)

    def evaluate(x):
        trk.check()
        q, nu = unpack(x)
        t0 = time.time()
        Us = solver.solve(sign * q, nu)
        dt = time.time() - t0
        last.update(x=np.array(x, dtype=float), ok=Us is not None)
        if cfg.optimizer == "nm" and (trk.level_evals + 1) % (len(x0) + 1) == 0:
            trk.iteration()
        rreg = cfg.reg * (q[pairs[:, 0]] - q[pairs[:, 1]])
        if Us is None:
            trk.n_fail += 1
            trk.log(L, K, np.nan, np.nan, nu, 0, dt)
            base = trk.level_best_r if trk.level_best_r is not None else ctx["r_zero"]
            last["r"] = np.r_[2.0 * base, rreg]
            return last["r"], np.inf
        r = loss.project(X_ref + Us - X_tgt).ravel()
        d = np.linalg.norm(r.reshape(-1, 3), axis=1)            # per point, in the loss's metric
        err, rms = float(d.mean()), float(np.sqrt((d ** 2).mean()))
        trk.log(L, K, err, rms, nu, 1, dt)
        q_tri = np.where(tri_labels >= 0, q[np.maximum(tri_labels, 0)], np.nan)
        state = dict(Us=Us.copy(), solver_state=solver.get_state(), q=q.copy(), nu=float(nu),
                     tri_labels=tri_labels.copy(), q_tri=q_tri, err=err, rms=rms, K=K, level=L,
                     E_Pa=cfg.E_Pa, pressures_Pa=[float(v) for v in q * cfg.E_Pa],
                     elapsed_min=trk.elapsed() / 60)
        if trk.level_evals % 5 == 1 or err < trk.level_best_err:
            log.info("[L%d K=%d] eval %4d  t=%6.1f min  mean %.3f mm  rms %.3f  nu %.3f  p=[%s%s] Pa  best %.3f",
                     L, K, trk.level_evals, trk.elapsed() / 60, err, rms, nu,
                     ", ".join("%.0f" % v for v in q[:6] * cfg.E_Pa), ", ..." if K > 6 else "",
                     min(err, trk.best_err))
        last["r"] = np.r_[r, rreg]
        trk.improve(err, r, state)
        return last["r"], err + np.dot(rreg, rreg) / max(1, len(rreg))

    # regularisation rows of the Jacobian: d(reg (q_a - q_b))/dx, constant
    R = np.zeros((len(pairs), len(x0)))
    R[np.arange(len(pairs)), pairs[:, 0]] = cfg.reg
    R[np.arange(len(pairs)), pairs[:, 1]] = -cfg.reg

    def jacobian(x):
        """Jacobian of the lsq residual [r, rreg] at x (one call per lsq iteration): from the core
        if available (not counted as a forward evaluation), else finite differences exactly as
        least_squares(jac="2-point") would compute them."""
        if last["x"] is None or not np.array_equal(x, last["x"]):
            evaluate(x)                                     # scipy normally calls fun(x) first
        trk.iteration()
        if not last["ok"] and last["J"] is not None:
            return last["J"]                                # forward failed at x: last good one
        J = None
        if last["ok"] and cfg.jacobian == "analytic":
            q, nu = unpack(x)
            t0 = time.time()
            J = solver.jacobian(sign * q, nu, cfg.free_nu)
            if J is not None:
                J = np.array(J, dtype=float)
                J[:, :K] *= sign                            # the core sees sign * q
                J = np.vstack([loss.jacobian(J), R])
                jstats["n"] += 1
                jstats["s"] += time.time() - t0
        if J is None:                                       # 2-point, or core without Jacobian
            jstats["fd"] += 1
            J = approx_derivative(lambda z: evaluate(z)[0], x, method="2-point",
                                  rel_step=cfg.lsq_diff_step, f0=last["r"], bounds=(lb, ub))
        last["J"] = J
        return J

    reason = "converged"
    try:
        if cfg.optimizer == "lsq":
            least_squares(lambda x: evaluate(x)[0], x0, bounds=(lb, ub), method="trf",
                          jac=jacobian,
                          x_scale=np.maximum(np.abs(x0), 0.1), diff_step=cfg.lsq_diff_step,
                          ftol=cfg.lsq_ftol, xtol=cfg.lsq_xtol, gtol=cfg.lsq_gtol, max_nfev=100000)
        else:
            minimize(lambda x: evaluate(np.clip(x, lb, ub))[1], x0, method="Nelder-Mead",
                     bounds=list(zip(lb, ub)), options=dict(maxfev=100000, xatol=1e-4, fatol=1e-3,
                                                            adaptive=True))
    except Stop as s:
        reason = s.reason
    stats = {k: v - stats0.get(k, 0) for k, v in solver.stats().items()}
    log.info("level %d timing: %d solves, %.2f s per solve; %d analytic Jacobians (%.2f s each), "
             "%d by finite differences%s", L, trk.level_evals, trk.level_solve_s / max(trk.level_evals, 1),
             jstats["n"], jstats["s"] / max(jstats["n"], 1), jstats["fd"],
             "; " + ", ".join("%s %d" % kv for kv in stats.items()) if stats else "")
    return reason, trk.level_best_err


def run(cfg):
    problem = CollapseProblem.load(cfg.setup / "problem.npz")
    out = cfg.workdir
    out.mkdir(parents=True, exist_ok=True)
    X_ref, X_tgt, tris = problem.reference, problem.target, problem.tris
    _, _, area = tri_geometry(X_ref, tris)
    adj, _ = face_adjacency(tris)
    free = np.ones(len(X_ref), bool)
    free[problem.clamped_points] = False
    loss = Loss(cfg, X_tgt, tris, free)
    d0 = np.linalg.norm(loss.project(X_ref - X_tgt), axis=1)
    mism = np.linalg.norm(X_ref - X_tgt, axis=1)[~free]
    log.info("problem: %d surface points (%d clamped, left out of the error: %.2f mm mean / %.2f max from "
             "their target), %d volume nodes, %d tets, levels K = %s; loss %s%s, no-deformation error "
             "%.2f mm (point-to-point %.2f)", len(X_ref), (~free).sum(), mism.mean(), mism.max(),
             len(problem.nodes), len(problem.tets), ", ".join(map(str, problem.levels)), cfg.loss,
             " (tangent weight %g)" % cfg.loss_tangent_weight if loss.plane else "", d0.mean(),
             np.linalg.norm(X_ref - X_tgt, axis=1)[free].mean())

    cls = get_solver(cfg.solver)
    if problem.has_wall and not cls.supports_wall:
        raise RuntimeError("fem_setup has a wall but solver '%s' does not support it" % cfg.solver)
    solver = cls(problem, cfg.solver_options)
    sign = sign_check(solver, problem, cfg)

    levels = list(zip(problem.levels, problem.regions))
    if cfg.levels is not None:
        missing = sorted(set(cfg.levels) - set(problem.levels.tolist()))
        if missing:
            raise ValueError("fem_fit.levels %s not in the fem_setup levels %s" % (missing, problem.levels.tolist()))
        levels = [(K, lab) for K, lab in levels if K in cfg.levels]

    trk = Tracker(cfg, out)
    ctx = dict(solver=solver, trk=trk, cfg=cfg, sign=sign, X_ref=X_ref, X_tgt=X_tgt, adj=adj, loss=loss,
               r_zero=loss.project(X_ref - X_tgt).ravel())
    q_tri, nu_best = np.where(problem.clamped_tri, np.nan, cfg.q0), cfg.nu
    summary, stall, prev = [], 0, d0.mean()
    stop_reason = "levels exhausted"
    with guarded(trk, int(60 * (cfg.time_budget_min + cfg.hard_grace_min))):
        for L, (K, tri_labels) in enumerate(levels, start=1):
            K = int(K)
            if trk.best is not None:                       # warm start from the best so far
                q_tri, nu_best = trk.best["q_tri"], trk.best["nu"]
                solver.set_state(trk.best["solver_state"])
            x0 = np.array([np.nansum(q_tri[tri_labels == k] * area[tri_labels == k]) /
                           area[tri_labels == k].sum() for k in range(K)])
            lb, ub = np.full(K, cfg.q_bounds[0]), np.full(K, cfg.q_bounds[1])
            if cfg.free_nu:
                x0, lb, ub = np.r_[x0, nu_best], np.r_[lb, cfg.nu_bounds[0]], np.r_[ub, cfg.nu_bounds[1]]
            x0 = np.clip(x0, lb + 1e-6, ub - 1e-6)
            reason, lvl_err = run_level(L, K, tri_labels, x0, lb, ub, ctx)
            imp = (prev - trk.best_err) / max(prev, 1e-9)
            summary.append(dict(level=L, K=K, best_mean_err_mm=float(lvl_err), stop=reason,
                                evals=trk.level_evals, elapsed_min=trk.elapsed() / 60))
            log.info("level %d done (%s): level best %.3f mm, overall best %.3f mm, improvement %.1f%%",
                     L, reason, lvl_err, trk.best_err, 100 * imp)
            trk.checkpoint()
            if reason in ("target error reached", "time budget") or reason.startswith("signal"):
                stop_reason = reason
                break
            stall = stall + 1 if imp < cfg.fit_min_improve else 0
            prev = trk.best_err
            if stall >= cfg.fit_patience:
                stop_reason = "fit plateau (%d levels < %.0f%% gain)" % (stall, 100 * cfg.fit_min_improve)
                break
    trk.close()

    b = trk.best
    if b is None:
        raise RuntimeError("no successful forward solve, nothing to export")
    Xs = X_ref + b["Us"]
    err_all = np.linalg.norm(Xs - X_tgt, axis=1)                # point-to-point, whatever the loss
    err_n_all = np.einsum("ij,ij->i", Xs - X_tgt, loss.n)      # signed, along the target normal
    err, err_n = err_all[free], err_n_all[free]                  # the metrics: free points only
    rot = dict(target=rigid_part(X_ref, X_tgt), fitted=rigid_part(X_ref, Xs))
    s = polydata(Xs, tris)
    s.point_data["Displacement_mm"] = b["Us"]
    s.point_data["Error_mm"] = err_all
    s.point_data["NormalError_mm"] = err_n_all                  # > 0 = outside the target surface
    s.point_data["Clamped"] = (~free).astype(np.uint8)
    s.cell_data["PressureRegion"] = b["tri_labels"]
    s.cell_data["Pressure_Pa"] = np.nan_to_num(b["q_tri"] * cfg.E_Pa, nan=0.0)
    s.cell_data["Clamped"] = problem.clamped_tri.astype(np.uint8)
    pen = None
    if problem.has_wall:
        pen = WallDistance(problem.wall_points, problem.wall_tris)(Xs)[0] - problem.wall_allow
        s.point_data["WallPenetration_mm"] = pen               # > 0 = beyond the allowed position
    write_surface(s, cfg.output, "LPS")
    if b["solver_state"] is not None:
        try:
            solver.export_volume(out / "volume_best.vtk", b["solver_state"])
        except NotImplementedError:
            pass

    res = dict(solver=cfg.solver, stop_reason=stop_reason, loss=cfg.loss,
               loss_tangent_weight=cfg.loss_tangent_weight, mean_err_mm=float(err.mean()),
               mean_normal_err_mm=float(np.abs(err_n).mean()), best_loss_err_mm=float(b["err"]),
               metric_points=int(free.sum()), clamped_points=int((~free).sum()),
               clamped_mismatch_mm=dict(mean=float(mism.mean()), max=float(mism.max())),
               rigid_rotation_deg=dict(target=rot["target"][0], fitted=rot["fitted"][0]),
               rigid_shift_mm=dict(target=rot["target"][1], fitted=rot["fitted"][1]),
               median_err_mm=float(np.median(err)), p95_err_mm=float(np.percentile(err, 95)),
               max_err_mm=float(err.max()), assd_mm=assd(Xs, X_tgt, tris),
               baseline_mean_mm=float(np.linalg.norm(X_ref - X_tgt, axis=1)[free].mean()),
               baseline_loss_mm=float(d0.mean()), K=b["K"], nu=b["nu"], E_Pa=cfg.E_Pa,
               pressures_Pa=b["pressures_Pa"], mean_pressure_Pa=float(np.nanmean(b["q_tri"]) * cfg.E_Pa),
               pressure_sign=sign, evals=trk.n_eval, failed_solves=trk.n_fail,
               solve_s_mean=trk.solve_s / max(trk.n_eval, 1), solver_stats=solver.stats(),
               elapsed_min=trk.elapsed() / 60, levels=summary, wall=bool(problem.has_wall),
               wall_max_penetration_mm=None if pen is None else float(max(pen.max(), 0.0)),
               wall_points_beyond=None if pen is None else int((pen > 0).sum()))
    (out / "result_summary.json").write_text(json.dumps(res, indent=2))
    log.info("result (%s): mean %.3f mm point-to-point, %.3f along the normal (no deformation %.3f point-to-point), "
             "median %.3f, p95 %.3f, max %.3f, ASSD %.3f mm; K=%d, nu=%.3f, mean p=%.0f Pa (E=%.0f Pa); "
             "%d solves (%d failed), %.1f min",
             stop_reason, res["mean_err_mm"], res["mean_normal_err_mm"], res["baseline_mean_mm"], res["median_err_mm"],
             res["p95_err_mm"], res["max_err_mm"], res["assd_mm"], b["K"], b["nu"],
             res["mean_pressure_Pa"], cfg.E_Pa, trk.n_eval, trk.n_fail, res["elapsed_min"])
    log.info("rigid part vs the inflated lung: target %.1f deg / %.1f mm, fitted %.1f deg / %.1f mm",
             *rot["target"], *rot["fitted"])
    if pen is not None:
        log.info("wall: %d points beyond the allowed position, max %.2f mm",
                 res["wall_points_beyond"], res["wall_max_penetration_mm"])
    for lv in summary:
        log.info("  L%d K=%-3d best %.3f mm (%s, %d solves)", lv["level"], lv["K"],
                 lv["best_mean_err_mm"], lv["stop"], lv["evals"])
    return cfg.output
