"""fem_setup step: the geometry of the collapse model, independent of the FEM core.

  1. reference surface = the remeshed inflated lung (outward triangles, watertight); optional
     target = the remeshed collapsed lung, matched by closest point (its own nodes)
  2. clamped region: triangles within (sphere radius x clamp_radius_factor) of the anchor points:
     the hilum and the airway / artery / vein ring centres (union of the balls)
  3. cavity wall (default: the inflated surface itself): closed surface the lung may not leave,
     with a margin per point (wall_tol_mm + how far the point already lies outside), stored as
     the surface and as a smooth field (wall_field.py: smoothed signed distance on a grid)
  4. optional Ward pressure regions of the target displacement (regions; diagnostic fits only)
  5. tetrahedral volume mesh (Gmsh or Netgen, fem_setup.mesher), surface nodes and triangles unchanged
  6. output_dir/fem/setup/: problem.npz (read by the collapse and fit steps) + files to inspect
"""

import json
import logging
import time

import numpy as np

from .. import vtk_mesh as vm
from ..data_io import read_fiducial_radii, read_fiducials, read_surface, write_surface
from .anchor import clamped_triangles
from .geometry import WallDistance, assd, face_adjacency, faces_of, polydata, signed_volume, tri_geometry
from .problem import CollapseProblem
from .volume_mesh import tet_mesh
from .wall_field import WallField, signed_distance_grid

log = logging.getLogger(__name__)


def closed_surface(path):
    """(points, outward triangles) of a watertight surface file (LPS)."""
    s = read_surface(path, space="LPS")
    P, tris = np.asarray(s.points, float), faces_of(s)
    bad = face_adjacency(tris)[1]
    if bad:
        raise RuntimeError("%s is not watertight (%d bad edges)" % (path.name, bad))
    if signed_volume(P, tris) < 0:
        tris = tris[:, ::-1].copy()
    return P, tris


def wall_field_arrays(cfg, wall_pts, wall_tris, X_ref, target_pts):
    """The wall as a smooth field (wall_field.py): (the CollapseProblem arrays wall_grid, wall_origin,
    wall_spacing, wall_smoothing_mm, wall_field_allow; smoothed distance of the reference points).
    target_pts: the collapsed lung's points (checked against the largest margin), or None."""
    t0 = time.time()
    grid, origin = signed_distance_grid(wall_pts, wall_tris, cfg.wall_grid_mm, cfg.wall_grid_margin_mm,
                                        cfg.wall_smoothing_mm)
    fld = WallField(grid, origin, cfg.wall_grid_mm)
    phis, grad = fld(X_ref, derivatives=1)
    gn = np.linalg.norm(grad, axis=1)
    log.info("wall field: grid %s at %.2f mm (margin %.0f mm), smoothing %.1f mm, %.0f s; smoothed "
             "distance of the reference points median %.2f, min %.2f, max %.2f mm (their margin = "
             "tolerance + the positive part); |gradient| there %.3f-%.3f", "x".join(map(str, grid.shape)),
             cfg.wall_grid_mm, cfg.wall_grid_margin_mm, cfg.wall_smoothing_mm, time.time() - t0, np.median(phis),
             phis.min(), phis.max(), gn.min(), gn.max())
    if target_pts is not None and len(target_pts):
        ex = fld(target_pts, derivatives=0) - cfg.wall_tol_mm - max(phis.max(), 0.0)
        log.info("wall field: %d target points beyond the allowed position (max %.2f mm; the fit cannot reach these)",
                 (ex > 0).sum(), max(ex.max(), 0.0))
    return dict(wall_grid=grid, wall_origin=origin, wall_spacing=np.array(float(cfg.wall_grid_mm)),
                wall_smoothing_mm=np.array(float(cfg.wall_smoothing_mm)),
                wall_field_allow=cfg.wall_tol_mm + np.maximum(phis, 0.0)), phis


def run(cfg):
    out = cfg.workdir
    out.mkdir(parents=True, exist_ok=True)

    # 1. surfaces
    X_ref, tris = closed_surface(cfg.reference)
    cen, nrm, area = tri_geometry(X_ref, tris)
    adj, _ = face_adjacency(tris)
    has_target = cfg.target is not None
    T_pts, T_tris = np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)
    if has_target:
        T_pts, T_tris = closed_surface(cfg.target)
        X_cp = WallDistance(T_pts, T_tris).closest(X_ref)[0]    # closest point of each reference point
        log.info("reference (inflated) %.1f mL, %d points, %d triangles; target (collapsed) %.1f mL, %d points, "
                 "%d triangles, matched by closest point", signed_volume(X_ref, tris) / 1000, len(X_ref), len(tris),
                 signed_volume(T_pts, T_tris) / 1000, len(T_pts), len(T_tris))
    else:
        X_cp = X_ref.copy()
        log.info("reference (inflated) %.1f mL, %d points, %d triangles; no target (prediction only)",
                 signed_volume(X_ref, tris) / 1000, len(X_ref), len(tris))

    # 2. clamped hilum
    names = list(cfg.clamp_points)
    fid, rad = read_fiducials(cfg.anchor), read_fiducial_radii(cfg.anchor)
    missing = [n for n in names if fid.get(n) is None or rad.get(n) is None]
    if missing:
        raise RuntimeError("%s has no sphere point(s) %s (run the hilum step; available: %s)"
                           % (cfg.anchor.name, missing, ", ".join(sorted(rad))))
    centers = np.array([fid[n] for n in names])
    spheres = np.array([rad[n] for n in names])
    clamped, grown, sv = clamped_triangles(X_ref, tris, centers, spheres * cfg.clamp_radius_factor,
                                           cfg.clamp_min_points, cfg.clamp_growth, cfg.clamp_max_growth_steps)
    cpts = np.unique(tris[clamped])
    radii = spheres * cfg.clamp_radius_factor
    radii[0] *= grown                                         # only the first ball grows
    log.info("clamp: %s, spheres %s mm x %.2f%s -> radii %s mm; %d triangles / %d points, up to %.0f mm "
             "from '%s' (singular values %.1f %.1f %.1f)", "+".join(names),
             "/".join("%.1f" % s for s in spheres), cfg.clamp_radius_factor,
             " (%s grown x%.2f)" % (names[0], grown) if grown > 1 else "", "/".join("%.1f" % r for r in radii),
             clamped.sum(), len(cpts), np.linalg.norm(X_ref[cpts] - centers[0], axis=1).max(), names[0], *sv)
    if has_target:
        d0 = np.linalg.norm(X_cp - X_ref, axis=1)
        log.info("distance to the target: clamped points %.2f mm, all points %.2f mm mean", d0[cpts].mean(), d0.mean())

    # 3. cavity wall
    wall_pts, wall_tris, wall_allow, phi0 = np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64), np.zeros(0), None
    wfield, phis = {}, None
    if cfg.wall is not None:
        if cfg.wall == "reference":                   # the inflated surface itself: the points lie on it
            wall_name, wall_pts, wall_tris = "reference surface", X_ref.copy(), tris.copy()
            wall = WallDistance(wall_pts, wall_tris)
            phi0 = np.zeros(len(X_ref))
        else:
            wall_name = cfg.wall.name
            wall_pts, wall_tris = closed_surface(cfg.wall)
            wall = WallDistance(wall_pts, wall_tris)
            phi0 = wall(X_ref)[0]
        wall_allow = cfg.wall_tol_mm + np.maximum(phi0, 0.0)   # points already outside may not go further
        msg = ""
        if has_target:
            excess = wall(T_pts)[0] - cfg.wall_tol_mm - max(phi0.max(), 0.0)
            msg = "; target %d points beyond the allowed position (max %.2f mm, the fit cannot reach these)" \
                  % ((excess > 0).sum(), max(excess.max(), 0.0))
        log.info("wall %s (%.1f mL), tolerance %.1f mm: reference %d/%d points outside (median %.2f, max %.2f mm)%s",
                 wall_name, signed_volume(wall_pts, wall_tris) / 1000, cfg.wall_tol_mm, (phi0 > 0).sum(), len(phi0),
                 np.median(phi0), max(phi0.max(), 0.0), msg)
        if cfg.wall_grid_mm is not None:              # the same wall as a smooth field (wall_update "potential")
            wfield, phis = wall_field_arrays(cfg, wall_pts, wall_tris, X_ref, T_pts if has_target else None)

    # 4. pressure regions (diagnostic fits): Ward clusters of the normal displacement to the target
    u = X_cp - X_ref
    levels, regions = np.zeros(0, dtype=int), np.zeros((0, len(tris)), dtype=int)
    if cfg.regions:
        if not has_target:
            raise RuntimeError("fem_setup.regions needs the collapsed lung (inputs.collapsed)")
        un = np.einsum("ij,ij->i", u[tris].mean(axis=1), nrm)
        from .regions import all_levels                       # scikit-learn: only for the regions
        levels, regions = all_levels(un[:, None], adj, ~clamped, cfg.regions, cen, cfg.regions_pos_weight)
        log.info("pressure regions on the closest-point displacement (|u| mean %.1f mm, u.n mean %.1f mm, "
                 "negative = inward): K = %s", np.linalg.norm(u, axis=1).mean(), un.mean(), ", ".join(map(str, levels)))

    # 5. volume mesh
    size = cfg.mesh_size_mm if cfg.mesher == "gmsh" else cfg.netgen_maxh_mm
    nodes, tets, surface_nodes = tet_mesh(X_ref, tris, out / cfg.mesher, size, cfg.gmsh_timeout_s, cfg.mesher,
                                          cfg.netgen_optsteps3d)

    # 6. outputs
    problem = CollapseProblem(nodes=nodes, tets=tets, surface_nodes=surface_nodes, tris=tris,
                              clamped_tri=clamped, target=X_cp, levels=levels, regions=regions,
                              clean_field=u, wall_points=wall_pts, wall_tris=wall_tris,
                              wall_allow=wall_allow, target_points=T_pts, target_tris=T_tris, **wfield)
    problem.save(out / "problem.npz")

    s = polydata(X_ref, tris)
    s.cell_data["Clamped"] = clamped.astype(np.uint8)
    if has_target:
        s.point_data["TargetDisplacement_mm"] = u               # to the closest point of the collapsed lung
    if phi0 is not None:
        s.point_data["WallDistance_mm"] = phi0
    if phis is not None:
        s.point_data["WallFieldDistance_mm"] = phis
    for K, lab in zip(levels, regions):
        s.cell_data["Regions_K%d" % K] = lab
    write_surface(s, out / "reference.vtp", "LPS")
    if has_target:
        write_surface(polydata(T_pts, T_tris), out / "target.vtp", "LPS")
    clamped_nodes = np.isin(np.arange(len(nodes)), surface_nodes[cpts]).astype(np.uint8)
    vm.write_grid(vm.tet_grid(nodes, tets, {"Clamped": clamped_nodes}, {"SPACE": np.array(["LPS"])}),
                  out / "volume.vtu")

    (out / "setup_summary.json").write_text(json.dumps(dict(
        reference=str(cfg.reference), target=str(cfg.target) if has_target else None, anchor=str(cfg.anchor),
        clamp_points=names, anchor_center_lps=centers.tolist(), anchor_sphere_radius_mm=spheres.tolist(),
        clamp_radii_mm=radii.tolist(), clamp_radius_factor=cfg.clamp_radius_factor,
        clamped_triangles=int(clamped.sum()), clamped_points=int(len(cpts)),
        inflated_volume_mL=signed_volume(X_ref, tris) / 1000,
        collapsed_volume_mL=signed_volume(T_pts, T_tris) / 1000 if has_target else None,
        baseline_assd_mm=assd(X_ref, T_pts, T_tris, tris_a=tris) if has_target else None,
        regions=levels.tolist(), wall=str(cfg.wall) if cfg.wall is not None else None, wall_tol_mm=cfg.wall_tol_mm,
        wall_grid_mm=cfg.wall_grid_mm if wfield else None, wall_smoothing_mm=cfg.wall_smoothing_mm if wfield else None,
        volume_nodes=int(len(nodes)), tets=int(len(tets)), mesher=cfg.mesher, mesh_size_mm=size,
        netgen_optsteps3d=cfg.netgen_optsteps3d if cfg.mesher == "netgen" else None), indent=2))
    log.info("wrote %s/ (problem.npz, reference.vtp, volume.vtu%s)", out, ", target.vtp" if has_target else "")
    return out
