"""What a run (forward or fit) writes in output_dir/runs/<run_name>/: the collapsed surface, the
state (to re-solve it, e.g. for a sequence) and summary.json with the physical parameters, the
volume, the wall penetration and, when there is a collapsed lung, the errors against it."""

import json
import logging

import numpy as np

from ..data_io import write_surface
from .geometry import WallDistance, assd, closest_on_surface, kabsch, polydata, rotation_deg, signed_volume

log = logging.getLogger(__name__)


def rigid_part(X_from, X_to):
    """Rotation [deg] and centroid shift [mm] of the best rigid fit X_from -> X_to."""
    R, t = kabsch(X_from, X_to)
    c = X_from.mean(0)
    return rotation_deg(R), float(np.linalg.norm(R @ c + t - c))


def state_arrays(state):
    """The arrays of state.npz / best_state.npz: enough to re-solve the state with any core."""
    a = dict(surface_displacement=state["Us"], q=state["q"], tri_labels=state["tri_labels"], nu=state["nu"],
             body=state["body"], body_direction=state["body_direction"], params=state["params"],
             pressure_sign=state["sign"])
    if isinstance(state.get("solver_state"), np.ndarray):
        a["solver_state"] = state["solver_state"]
    return a


def target_errors(problem, Xs, free):
    """Errors of the surface Xs against the collapsed lung (free points only)."""
    tris = problem.tris
    T_end, _, n = WallDistance(*problem.target_surface).closest(Xs)
    d = Xs - T_end
    err_all = np.linalg.norm(d, axis=1)
    err_n_all = np.einsum("ij,ij->i", d, n)                     # signed, along the target normal
    err, err_n = err_all[free], err_n_all[free]
    Y = problem.target_points
    cell, _, z = closest_on_surface(Xs, tris, Y)
    keep = ~problem.clamped_tri[cell]
    t2f = float(np.linalg.norm(z - Y, axis=1)[keep].mean()) if keep.any() else 0.0
    X_ref = problem.reference
    base = np.linalg.norm(X_ref - WallDistance(*problem.target_surface).closest(X_ref)[0], axis=1)
    out = dict(assd_mm=assd(Xs, *problem.target_surface, tris_a=tris), mean_err_mm=float(err.mean()),
               mean_normal_err_mm=float(np.abs(err_n).mean()), target_to_fitted_mm=t2f,
               median_err_mm=float(np.median(err)), p95_err_mm=float(np.percentile(err, 95)),
               max_err_mm=float(err.max()), baseline_mean_mm=float(base[free].mean()),
               baseline_assd_mm=assd(X_ref, *problem.target_surface, tris_a=tris),
               clamped_mismatch_mm=dict(mean=float(base[~free].mean()), max=float(base[~free].max())),
               target_volume_ratio=signed_volume(*problem.target_surface) / signed_volume(X_ref, tris))
    return out, err_all, err_n_all


def write_run(cfg, problem, solver, state, kind, extra=None):
    """Write surface.vtp, state.npz, summary.json (and volume.vtk) of a run; returns the summary.
    state: dict with Us, q, tri_labels, nu, body, body_direction, params, sign, q_tri, physics,
    solver_state. kind: "collapse" or "fit"."""
    out = cfg.run_dir
    out.mkdir(parents=True, exist_ok=True)
    X_ref, tris = problem.reference, problem.tris
    Xs = X_ref + state["Us"]
    free = np.ones(len(X_ref), bool)
    free[problem.clamped_points] = False
    ph = state["physics"]
    E = ph["E_Pa"]

    s = polydata(Xs, tris)
    s.point_data["Displacement_mm"] = state["Us"]
    s.point_data["Clamped"] = (~free).astype(np.uint8)
    s.cell_data["Pressure_Pa"] = np.nan_to_num(state["q_tri"] * E, nan=0.0)
    s.cell_data["Clamped"] = problem.clamped_tri.astype(np.uint8)
    res = dict(kind=kind, patient=cfg.patient, run_name=cfg.run_name, solver=cfg.solver.core, physics=ph,
               volume_ratio=signed_volume(Xs, tris) / signed_volume(X_ref, tris),
               collapse_percent=100.0 * (1.0 - signed_volume(Xs, tris) / signed_volume(X_ref, tris)),
               mean_pressure_Pa=float(np.nanmean(state["q_tri"]) * E), pressure_sign=state["sign"],
               rigid_rotation_deg=None, rigid_shift_mm=None,
               metric_points=int(free.sum()), clamped_points=int((~free).sum()))
    res["rigid_rotation_deg"], res["rigid_shift_mm"] = rigid_part(X_ref, Xs)
    if problem.has_target:
        errs, err_all, err_n_all = target_errors(problem, Xs, free)
        res.update(errs)
        s.point_data["Error_mm"] = err_all
        s.point_data["NormalError_mm"] = err_n_all              # > 0 = outside the target surface
    res["wall"] = bool(problem.has_wall)
    if problem.has_wall:
        pen = WallDistance(problem.wall_points, problem.wall_tris)(Xs)[0] - problem.wall_allow
        s.point_data["WallPenetration_mm"] = pen                # > 0 = beyond the allowed position
        res.update(wall_max_penetration_mm=float(max(pen.max(), 0.0)), wall_points_beyond=int((pen > 0).sum()))
    if problem.has_wall_field:                                  # the same with the smooth wall field
        fld = problem.wall_field()
        ins = fld.inside(Xs)
        pen_f = np.where(ins, 0.0, np.nan)
        pen_f[ins] = fld(Xs[ins], derivatives=0) - problem.wall_field_allow[ins]
        s.point_data["WallFieldPenetration_mm"] = pen_f
        res.update(wall_field_max_penetration_mm=float(max(np.nanmax(pen_f), 0.0)),
                   wall_field_points_beyond=int((pen_f > 0).sum()))
    res["solver_stats"] = solver.stats()
    res.update(extra or {})

    write_surface(s, out / "surface.vtp", "LPS")
    np.savez(out / "state.npz", **state_arrays(state))
    if state.get("solver_state") is not None:
        try:
            solver.export_volume(out / "volume.vtk", state["solver_state"])
        except NotImplementedError:
            pass
    (out / "summary.json").write_text(json.dumps(res, indent=2))

    g = ph.get("gravity")
    log.info("physics: E %.0f Pa, p0 %.0f Pa (q0 %.3f), nu %.3f%s; fitted: %s", E, ph["p0_Pa"], ph["q0"], ph["nu"],
             ", gravity %s -> %s: slope %.3f p/E over %.0f mm" % (*g["positions"], g["slope_pE"], g["extent_mm"])
             if g else ", no gravity", ", ".join(ph["fitted"]) or "nothing")
    if ph.get("E_rule"):
        r = ph["E_rule"]
        log.info("E from the collapse %%: log10 E = %.4f %+.6f x %.1f %% -> %.0f Pa", r["a"], r["b"],
                 r["collapse_percent"], E)
    if ph.get("ct_density_kg_m3"):
        log.info("with the CT density %.0f kg/m3 (model %.0f): E %.0f Pa, p0 %.0f Pa", ph["ct_density_kg_m3"],
                 ph["tissue_density_kg_m3"], ph["E_Pa_ct"], ph["p0_Pa_ct"])
    log.info("collapsed to %.1f %% of the inflated volume (collapse %.1f %%); rigid part %.1f deg / %.1f mm",
             100 * res["volume_ratio"], res["collapse_percent"], res["rigid_rotation_deg"], res["rigid_shift_mm"])
    if problem.has_target:
        log.info("vs the collapsed lung (volume %.1f %%): ASSD %.3f mm (no deformation %.3f); fitted -> target mean "
                 "%.3f, along the normal %.3f, median %.3f, p95 %.3f, max %.3f; target -> fitted %.3f mm",
                 100 * res["target_volume_ratio"], res["assd_mm"], res["baseline_assd_mm"], res["mean_err_mm"],
                 res["mean_normal_err_mm"], res["median_err_mm"], res["p95_err_mm"], res["max_err_mm"],
                 res["target_to_fitted_mm"])
    if problem.has_wall:
        log.info("wall: %d points beyond the allowed position, max %.2f mm%s", res["wall_points_beyond"],
                 res["wall_max_penetration_mm"],
                 " (smoothed field: %d, max %.2f mm)" % (res["wall_field_points_beyond"],
                                                         res["wall_field_max_penetration_mm"])
                 if problem.has_wall_field else "")
    log.info("wrote %s/ (surface.vtp, state.npz, summary.json)", out)
    return res
