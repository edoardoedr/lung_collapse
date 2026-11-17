"""fit step: the physics parameters marked "fit" in the config (and the optional perturbation) are
fitted so that the inflated lung collapses onto the collapsed one.

  1. load output_dir/fem/setup/problem.npz and the FEM core
  2. sign check: q > 0 must move the surface inward (else q is flipped for this core)
  3. the levels of the load model (load.py), each warm-started from the best state so far; per
     level a bounded least squares (scipy TRF) on the surface distance in both directions (Loss)
     + the smoothness between adjacent regions, with the analytic Jacobian of the core
  4. stops on: target error | time budget | plateau | levels exhausted | Ctrl+C / SIGTERM
  5. output_dir/runs/<run_name>/: surface.vtp, state.npz, summary.json (results.py), history.csv,
     best_state.npz / best_params.json (checkpoints)

Only the solver knows the FEM library: see solvers/base.py for the contract.
"""

import json
import logging
import time

import numpy as np
from scipy.optimize import least_squares

from .geometry import WallDistance, closest_on_surface, tri_geometry, vertex_normals
from .load import LoadModel
from .problem import CollapseProblem
from .results import write_run
from .run_control import Stop, Tracker, guarded
from .solvers import get_solver

log = logging.getLogger(__name__)


def make_solver(cfg, problem, model=None):
    """The FEM core of the config on this problem; with a load model, also its body force."""
    cls = get_solver(cfg.solver.core)
    if problem.has_wall and not cls.supports_wall:
        raise RuntimeError("fem_setup has a wall but solver '%s' does not support it" % cfg.solver.core)
    solver = cls(problem, cfg.solver_options())
    if model is not None and model.has_body:
        if not solver.supports_body:
            raise RuntimeError("solver '%s' has no body force: it cannot run different reference / collapse "
                               "positions" % cfg.solver.core)
        solver.set_body(model.body_direction)
    return solver


def sign_check(solver, problem, q, nu):
    """+1 if q > 0 collapses with this core, -1 if the core has the opposite convention."""
    solver.set_regions(np.where(problem.clamped_tri, -1, 0), 1)
    Us = solver.solve(np.array([q]), nu)
    if Us is None:
        raise RuntimeError("first forward solve failed (clamp / volume mesh quality?)")
    free = np.setdiff1d(np.arange(len(Us)), problem.clamped_points)
    un = np.einsum("ij,ij->i", Us, vertex_normals(problem.reference, problem.tris))[free].mean()
    solver.reset()
    if un > 0:
        log.warning("sign check: q > 0 moves the surface outward with this core -> q is flipped "
                    "(fix the core's sign convention)")
        return -1.0
    log.info("sign check OK: q = %.2f moves the surface %.2f mm inward on average", q, -un)
    return 1.0


def hilum_position(cfg):
    """(3,) LPS position of the hilum from the setup summary, None if not there."""
    ss = cfg.setup_dir / "setup_summary.json"
    return np.array(json.loads(ss.read_text())["anchor_center_lps"][0]) if ss.is_file() else None


def describe_load(model, ph):
    g = ph.gravity
    if g is None:
        return "no gravity"
    return ("gravity %s -> %s, extent %.0f mm, slope %.3f p/E at E %.0f Pa%s"
            % (*g["names"], model.extent, model.slope_of_E(ph.E_Pa.value), ph.E_Pa.value,
               " (+ body force)" if model.has_body else ""))


class Loss:
    """Distance between the fitted surface and the collapsed one, in both directions.

    Fitted -> target: for each free fitted point x its closest point c on the target surface and
    the normal n there (interpolated vertex normals), residual n n^T (x - c): the distance along
    the normal, so sliding along the surface costs nothing. c and n are recomputed at every
    evaluation; the Jacobian holds them fixed (Gauss-Newton of point-to-plane ICP).

    That term alone is one-sided: a lung that shrinks onto part of the target scores well while
    the rest of the target stays uncovered. Target -> fitted (weight ws): for every target point y
    its closest point z on the fitted surface, z = sum_k b_k x_k over the vertices of the fitted
    triangle (barycentric b), residual z - y, zero when z falls on a clamped triangle. Both terms
    are averaged over their own number of points: cost = sum fwd^2 + ws (n_free / n_target) sum bwd^2.
    The Jacobian holds the triangle and b fixed (exact gradient of the squared distance).

    Clamped points are left out: they do not move, so their error is a constant."""

    def __init__(self, problem, symmetric_weight):
        free = np.ones(len(problem.reference), bool)
        free[problem.clamped_points] = False
        self.free, self.n_free = free, int(free.sum())
        self.tris = problem.tris
        self.surface = WallDistance(*problem.target_surface)
        self.sym = float(symmetric_weight)
        self.Y = self.surface.P
        self.clamped_tri = np.asarray(problem.clamped_tri, bool)
        self.scale = np.sqrt(self.sym * self.n_free / len(self.Y)) if self.sym else 0.0
        self.back = None                                     # (fitted triangle, weights, kept) per target point
        self.n = self.nf = None

    def residual(self, Xs):
        """Fitted surface points Xs (N, 3) -> flat residual: 3 n_free rows (fitted -> target), then,
        with the symmetric term, 3 n_target rows (target -> fitted, scaled)."""
        target, _, self.n = self.surface.closest(Xs)
        self.nf = self.n[self.free]
        d = (Xs - target)[self.free]
        r = (np.einsum("ij,ij->i", d, self.nf)[:, None] * self.nf).ravel()
        return np.r_[r, self.scale * self.backward(Xs).ravel()] if self.sym else r

    def backward(self, Xs):
        """(n_target, 3) closest point on the fitted surface Xs - target point, 0 where that closest
        point is on a clamped triangle; keeps the triangles / weights for the Jacobian."""
        cell, b, z = closest_on_surface(Xs, self.tris, self.Y)
        keep = ~self.clamped_tri[cell]
        self.back = (self.tris[cell], b, keep)
        return (z - self.Y) * keep[:, None]

    def error(self, r):
        """Flat residual -> (mean, rms) distance per point; with the symmetric term (fwd + ws bwd) /
        (1 + ws) of the two means, bwd over the kept target points."""
        r = r.reshape(-1, 3)
        df = np.linalg.norm(r[:self.n_free], axis=1)
        if not self.sym:
            return float(df.mean()), float(np.sqrt((df ** 2).mean()))
        db = np.linalg.norm(r[self.n_free:], axis=1)[self.back[2]] / self.scale
        if not len(db):
            db = np.zeros(1)
        ws = self.sym
        return (float((df.mean() + ws * db.mean()) / (1 + ws)),
                float(np.sqrt(((df ** 2).mean() + ws * (db ** 2).mean()) / (1 + ws))))

    def jacobian(self, J):
        """J (3N, P) of the displacement of all points -> of the residual (of the last residual())."""
        Jall = J.reshape(len(self.n), 3, -1)
        J3 = Jall[self.free]
        J3 = (self.nf[:, :, None] * np.einsum("ik,ikp->ip", self.nf, J3)[:, None, :]).reshape(-1, J.shape[1])
        if not self.sym:
            return J3
        tri, b, keep = self.back
        Jb = np.einsum("mk,mkjp->mjp", b * keep[:, None], Jall[tri])
        return np.vstack([J3, self.scale * Jb.reshape(-1, J.shape[1])])


def run_level(L, level, x0, ctx):
    """One level of the load model; returns (stop reason, best error of the level)."""
    solver, trk, cfg, sign = ctx["solver"], ctx["trk"], ctx["cfg"], ctx["sign"]
    X_ref, loss, model = ctx["X_ref"], ctx["loss"], ctx["model"]
    P, lb, ub = level.P, level.lb, level.ub
    solver.set_regions(level.labels, level.n_solver)
    trk.new_level()
    stats0 = solver.stats()
    log.info("--- level %d: %d parameters (%s%s), %d core regions, %d adjacent pairs ---", L, P,
             ", ".join(level.names[:6]), ", ..." if P > 6 else "", level.n_solver, len(level.pairs))
    with_nu, with_body = level.nu_index is not None, bool(np.any(level.body_coef != 0.0))
    body_on = model.has_body

    last = dict(x=None, r=None, ok=False, J=None)          # last residual evaluation / Jacobian
    jstats = dict(n=0, s=0.0)
    # warm start of every solve: the state of the lowest lsq cost so far in this level (the
    # optimiser's current iterate), not the last converged solve, which is often a rejected trial
    # far from the next one
    anchor = dict(cost=np.inf, state=None, load=None, body=0.0, current=True)

    def evaluate(x):
        trk.check()
        if not anchor["current"]:
            if body_on:
                solver.set_state(anchor["state"], anchor["load"], anchor["body"])
            else:
                solver.set_state(anchor["state"], anchor["load"])
            anchor["current"] = True
        qs, nu, body = level.q_solver(x), level.nu(x), level.body(x)
        t0 = time.time()
        Us = solver.solve(sign * qs, nu, body) if body_on else solver.solve(sign * qs, nu)
        dt = time.time() - t0
        last.update(x=np.array(x, dtype=float), ok=Us is not None)
        rreg = level.reg_residual(x)
        if Us is None:
            trk.n_fail += 1
            trk.log(L, P, np.nan, np.nan, nu, 0, dt)
            log.info("[L%d] eval %4d failed after %.0f s", L, trk.level_evals, dt)
            base = trk.level_best_r if trk.level_best_r is not None else ctx["r_zero"]
            last["r"] = np.r_[2.0 * base, rreg]             # a rejected trial
            return last["r"]
        r = loss.residual(X_ref + Us)
        err, rms = loss.error(r)
        trk.log(L, P, err, rms, nu, 1, dt)
        q_tri = level.q_tri(x)
        # q / tri_labels as the core sees them, so that the state can be re-solved (sequence)
        state = dict(Us=Us.copy(), solver_state=solver.get_state(), q=np.array(qs, float), nu=float(nu),
                     body=float(body), body_direction=model.body_direction, sign=sign,
                     tri_labels=level.labels.copy(), q_tri=q_tri, err=err, rms=rms, level=L,
                     params=np.array(x, float), param_names=level.names, physics=model.describe(level, x),
                     elapsed_min=trk.elapsed() / 60)
        score = rms if cfg.fit.best_by == "rms" else err
        if trk.level_evals % 5 == 1 or score < trk.level_best_err:
            log.info("[L%d] eval %4d  t=%6.1f min  mean %.3f mm  rms %.3f  %s%s  best %.3f", L, trk.level_evals,
                     trk.elapsed() / 60, err, rms,
                     ", ".join("%s %.4g" % nv for nv in list(zip(level.names, x))[:6]), ", ..." if P > 6 else "",
                     min(score, trk.best_err))
        last["r"] = np.r_[r, rreg]
        cost = float(np.dot(last["r"], last["r"]))
        if cost < anchor["cost"]:
            anchor.update(cost=cost, state=state["solver_state"], load=sign * q_tri, body=body, current=True)
        else:
            anchor["current"] = anchor["state"] is None     # the solver now holds this trial
        trk.improve(score, r, state)
        return last["r"]

    R = level.reg_jacobian()

    def jacobian(x):
        """Jacobian of [r, rreg] at x (one call per lsq iteration), from the core."""
        if last["x"] is None or not np.array_equal(x, last["x"]):
            evaluate(x)                                     # scipy normally calls fun(x) first
        trk.iteration()
        if not last["ok"] and last["J"] is not None:
            return last["J"]                                # forward failed at x: last good one
        if not last["ok"]:
            raise Stop("first solve failed")
        t0 = time.time()
        if with_body:
            J = solver.jacobian(sign * level.q_solver(x), level.nu(x), with_nu, True)
        else:
            J = solver.jacobian(sign * level.q_solver(x), level.nu(x), with_nu)
        if J is None:
            raise RuntimeError("solver '%s' has no analytic Jacobian for this model" % cfg.solver.core)
        J = np.array(J, dtype=float)
        n = level.n_solver
        Jnu = J[:, n] if with_nu else None
        Jbody = J[:, n + int(with_nu)] if with_body else None
        J = np.vstack([loss.jacobian(level.chain(sign * J[:, :n], Jnu, Jbody)), R])
        jstats["n"] += 1
        jstats["s"] += time.time() - t0
        last["J"] = J
        return J

    reason = "converged"
    try:
        least_squares(evaluate, x0, bounds=(lb, ub), method="trf", jac=jacobian,
                      x_scale=np.maximum(np.abs(x0), 0.1), ftol=cfg.fit.lsq_ftol, xtol=cfg.fit.lsq_xtol,
                      gtol=cfg.fit.lsq_gtol, max_nfev=100000)
    except Stop as s:
        reason = s.reason
    stats = {k: v - stats0.get(k, 0) for k, v in solver.stats().items()}
    log.info("level %d timing: %d solves, %.2f s per solve; %d Jacobians (%.2f s each)%s", L, trk.level_evals,
             trk.level_solve_s / max(trk.level_evals, 1), jstats["n"], jstats["s"] / max(jstats["n"], 1),
             "; " + ", ".join("%s %d" % kv for kv in stats.items() if kv[1]) if stats else "")
    return reason, trk.level_best_err


def run(cfg):
    problem = CollapseProblem.load(cfg.setup_dir / "problem.npz")
    if not problem.has_target:
        raise RuntimeError("the fit needs the collapsed lung: set inputs.collapsed and rerun geometry and fem_setup")
    out = cfg.run_dir
    out.mkdir(parents=True, exist_ok=True)
    X_ref, tris = problem.reference, problem.tris
    _, _, area = tri_geometry(X_ref, tris)
    ph, fc = cfg.physics, cfg.fit
    loss = Loss(problem, fc.symmetric_weight)
    r_zero = loss.residual(X_ref)
    e0, rms0 = loss.error(r_zero)
    log.info("problem: %d surface points (%d clamped, left out of the error), %d volume nodes, %d tets; loss: "
             "distance along the target normal%s, no-deformation error %.2f mm (rms %.2f)", len(X_ref),
             len(X_ref) - loss.n_free, len(problem.nodes), len(problem.tets),
             " + target -> fitted (weight %g)" % loss.sym if loss.sym else "", e0, rms0)

    model = LoadModel(ph, problem, hilum_position(cfg))
    levels = model.fit_levels(fc)
    solver = make_solver(cfg, problem, model)
    sign = sign_check(solver, problem, fc.sign_check_q, ph.nu.value)
    log.info("load model: %s; fitted per level: %s", describe_load(model, ph),
             " | ".join("%d: %s%s" % (i + 1, ", ".join(lv.names[:4]), ", ..." if lv.P > 4 else "")
                        for i, lv in enumerate(levels)))

    trk = Tracker(fc, out)
    ctx = dict(solver=solver, trk=trk, cfg=cfg, sign=sign, X_ref=X_ref, loss=loss, model=model,
               r_zero=r_zero.ravel())
    q_tri, nu_best = model.start_q_tri(), ph.nu.value
    base0 = rms0 if fc.best_by == "rms" else e0             # in the measure that picks the best
    summary, stall, prev = [], 0, base0
    stop_reason = "levels exhausted"
    with guarded(trk, int(60 * (fc.time_budget_min + fc.hard_grace_min))):
        for L, level in enumerate(levels, 1):
            if trk.best is not None:                        # warm start from the best so far
                b = trk.best
                q_tri, nu_best = b["q_tri"], b["nu"]
                if model.has_body:
                    solver.set_state(b["solver_state"], sign * b["q_tri"], b["body"])
                else:
                    solver.set_state(b["solver_state"], sign * b["q_tri"])
            x0 = level.x0_from(q_tri, nu_best, area)
            reason, lvl_err = run_level(L, level, x0, ctx)
            imp = (prev - trk.best_err) / max(prev, 1e-9)
            summary.append(dict(level=L, parameters=level.P, best_err_mm=float(lvl_err), stop=reason,
                                evals=trk.level_evals, elapsed_min=trk.elapsed() / 60))
            log.info("level %d done (%s): level best %.3f mm, overall best %.3f mm, improvement %.1f%%",
                     L, reason, lvl_err, trk.best_err, 100 * imp)
            trk.checkpoint()
            if reason in ("target error reached", "time budget") or reason.startswith("signal"):
                stop_reason = reason
                break
            stall = stall + 1 if imp < fc.fit_min_improve else 0
            prev = trk.best_err
            if stall >= fc.fit_patience:
                stop_reason = "fit plateau (%d levels < %.0f%% gain)" % (stall, 100 * fc.fit_min_improve)
                break
    trk.close()

    b = trk.best
    if b is None:
        raise RuntimeError("no successful forward solve, nothing to export (%s)" % stop_reason)
    res = write_run(cfg, problem, solver, b, "fit", dict(
        stop_reason=stop_reason, best_by=fc.best_by, symmetric_weight=loss.sym, best_loss_err_mm=float(b["err"]),
        best_loss_rms_mm=float(b["rms"]), baseline_loss_mm=float(e0),
        parameters=dict(zip(b["param_names"], (float(v) for v in b["params"]))),
        evals=trk.n_eval, failed_solves=trk.n_fail, solve_s_mean=trk.solve_s / max(trk.n_eval, 1),
        elapsed_min=trk.elapsed() / 60, levels=summary))
    log.info("fit (%s): %d solves (%d failed), %.1f min", stop_reason, trk.n_eval, trk.n_fail, res["elapsed_min"])
    for lv in summary:
        log.info("  L%d  %d parameters  best %.3f mm (%s, %d solves)", lv["level"], lv["parameters"],
                 lv["best_err_mm"], lv["stop"], lv["evals"])
    return out
