"""collapse step: the forward run. Every physics parameter is given in the config and the
inflated lung is collapsed with them; no collapsed lung is needed (with one, the errors against
it are reported).

The uniform pressure is given as physics.collapse q0 (p/E), p0_Pa, or volume_ratio: with the
latter, q0 is searched so that V_collapsed / V_inflated has that value (the volume decreases
monotonically with q0: secant steps inside a bracket, each solve warm-started from the last one).

Output: output_dir/runs/<run_name>/ (results.py) and, with collapse.sequence_frames > 0, the
collapse sequence (sequence.py).
"""

import logging
import time

import numpy as np

from .fit import describe_load, hilum_position, make_solver, sign_check
from .geometry import signed_volume
from .load import LoadModel
from .problem import CollapseProblem
from .results import write_run
from .sequence import make_sequence

log = logging.getLogger(__name__)


def run(cfg):
    problem = CollapseProblem.load(cfg.setup_dir / "problem.npz")
    ph, fc = cfg.physics, cfg.collapse
    model = LoadModel(ph, problem, hilum_position(cfg))
    level = model.forward_level()
    solver = make_solver(cfg, problem, model)
    sign = sign_check(solver, problem, cfg.fit.sign_check_q, ph.nu.value)
    solver.set_regions(level.labels, level.n_solver)
    X_ref, tris = problem.reference, problem.tris
    V0 = signed_volume(X_ref, tris)
    nu = ph.nu.value
    log.info("load model: %s; E %.0f Pa, nu %.3f, wall %s", describe_load(model, ph), ph.E_Pa.value, nu,
             "on" if problem.has_wall else "off")
    t0 = time.time()
    n_solves = [0, 0]                                           # done, failed

    def solve(q0):
        """(surface displacement or None, volume ratio) at uniform p/E q0, warm-started."""
        x = np.array([q0])
        body = level.body(x)
        n_solves[0] += 1
        Us = solver.solve(sign * level.q_solver(x), nu, body) if model.has_body else \
            solver.solve(sign * level.q_solver(x), nu)
        if Us is None:
            n_solves[1] += 1
            log.info("  q0 %.4f: solve failed", q0)
            return None, None
        r = signed_volume(X_ref + Us, tris) / V0
        log.info("  q0 %.4f: volume %.1f %% of the inflated one", q0, 100 * r)
        return Us, r

    search = None
    if ph.volume_ratio is None:
        q0 = ph.q0.value if ph.q0 is not None else ph.p0_Pa / ph.E_Pa.value
        Us, _ = solve(q0)
        if Us is None:
            raise RuntimeError("the forward solve failed (q0 %.3f): see the solver statistics in the log" % q0)
    else:
        q0, Us, search = find_q0(solve, ph.volume_ratio, nu, fc)
    x = np.array([q0])
    q_tri = level.q_tri(x)
    physics = model.describe(level, x)
    physics["fitted"] = []                                      # nothing is fitted in a forward run
    state = dict(Us=Us, solver_state=solver.get_state(), q=level.q_solver(x), nu=nu, body=level.body(x),
                 body_direction=model.body_direction, sign=sign, tri_labels=level.labels.copy(), q_tri=q_tri,
                 params=x, physics=physics)
    write_run(cfg, problem, solver, state, "collapse", dict(
        solves=n_solves[0], failed_solves=n_solves[1], elapsed_min=(time.time() - t0) / 60, volume_search=search))
    if fc.sequence_frames > 0:
        make_sequence(cfg, fc.sequence_frames, solver=solver)
    return cfg.run_dir


def find_q0(solve, target, nu, fc):
    """q0 with volume ratio `target` (tolerance fc.volume_tol): (q0, displacement, search record).
    First guess from the hydrostatic response, 1 - V/V0 = q / (K/E); then secant steps kept inside
    the bracket; a failed solve halves the step from the last good q0."""
    K = 1.0 / (3.0 * (1.0 - 2.0 * nu))
    lo = (0.0, 1.0)                                             # (q0, ratio) with ratio > target
    hi = None                                                   # ... with ratio < target
    good = None                                                 # last converged (q0, Us, ratio)
    q = max((1.0 - target) * K, 1e-3)
    history = []
    for _ in range(fc.max_search_solves):
        Us, r = solve(q)
        if Us is None:
            base = good[0] if good is not None else 0.0
            q = base + 0.5 * (q - base)
            if abs(q - base) < 1e-4:
                break
            continue
        history.append((float(q), float(r)))
        good = (q, Us, r)
        if abs(r - target) <= fc.volume_tol:
            log.info("volume ratio %.3f reached with q0 %.4f (%d solves)", r, q, len(history))
            return q, Us, dict(target=target, reached=float(r), q0=float(q), history=history)
        if r > target:
            lo = (q, r)
        else:
            hi = (q, r)
        if hi is None:                                          # not bracketed yet: extrapolate, at most x2
            slope = (lo[1] - 1.0) / lo[0] if lo[0] > 0 else -1.0 / K
            q = min(lo[0] + (target - lo[1]) / slope, 2.0 * lo[0] + 0.05) if slope < 0 else 2.0 * lo[0] + 0.05
        else:
            qs = lo[0] + (target - lo[1]) * (hi[0] - lo[0]) / (hi[1] - lo[1])
            w = hi[0] - lo[0]
            q = min(max(qs, lo[0] + 0.1 * w), hi[0] - 0.1 * w)   # keep away from the ends
    if good is None:
        raise RuntimeError("volume search: no forward solve converged")
    best = min(history, key=lambda h: abs(h[1] - target))
    raise RuntimeError("volume ratio %.3f not reached in %d solves (closest: %.3f at q0 %.4f); raise "
                       "collapse.max_search_solves or check the failed solves" % (target, fc.max_search_solves,
                                                                                 best[1], best[0]))
