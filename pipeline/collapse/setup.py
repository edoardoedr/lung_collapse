"""Step 5a - fem_setup: everything the inverse FEM needs, independent of the FEM core.

  1. reference (inflated, registration output) and target (collapsed) surfaces with the same
     nodes, outward triangles, watertight
  2. clamped region: triangles within (sphere radius x anchor_radius_factor) of the anchor point(s):
     the hilum, optionally also the airway / artery / vein ring centres (union of the balls)
  3. rigid alignment of the target (default: on the clamped points, consistent with u = 0 there);
     optionally also clamp the triangles that barely move between the inflated and the aligned
     collapsed lung (clamp_still_mm), chosen after the alignment so that it is not affected
  4. optional cavity wall: closed surface the lung may not leave during the collapse, "reference"
     = the registered inflated surface itself (allowed outward margin per point = wall_tol_mm +
     how far the point is already outside)
  5. clean displacement field (rigid part removed) -> Ward pressure regions for every level
  6. tetrahedral volume mesh (Gmsh), surface nodes and triangles unchanged
  7. output_dir/fem/setup/: problem.npz (read by fem_fit) + files to inspect in Slicer / ParaView
"""

import json
import logging

import numpy as np
import pyvista as pv

from ..data_io import read_fiducial_radii, read_fiducials, read_surface, write_surface
from .anchor import clamped_triangles
from .geometry import (WallDistance, assd, face_adjacency, faces_of, kabsch, polydata, rotation_deg,
                       signed_volume, tri_geometry)
from .problem import CollapseProblem
from .regions import all_levels
from .volume_mesh import tet_mesh

log = logging.getLogger(__name__)


def rigid(X_from, X_to, weights=None):
    """X_from rigidly moved onto X_to; also rotation [deg] and centroid shift [mm]."""
    R, t = kabsch(X_from, X_to, weights)
    c = X_from.mean(0)
    return X_from @ R.T + t, rotation_deg(R), float(np.linalg.norm(R @ c + t - c))


def run(cfg):
    out = cfg.workdir
    out.mkdir(parents=True, exist_ok=True)

    # 1. surfaces
    ref, tgt = read_surface(cfg.reference, space="LPS"), read_surface(cfg.target, space="LPS")
    tris = faces_of(ref)
    if ref.n_points != tgt.n_points or not np.array_equal(tris, faces_of(tgt)):
        raise RuntimeError("%s and %s do not have the same nodes and triangles: the reference must be "
                           "the registration output of the target" % (cfg.reference.name, cfg.target.name))
    X_ref, X_raw = np.asarray(ref.points, float), np.asarray(tgt.points, float)
    if signed_volume(X_ref, tris) < 0:
        tris = tris[:, ::-1].copy()
    adj, bad = face_adjacency(tris)
    if bad:
        raise RuntimeError("%s is not watertight (%d bad edges)" % (cfg.reference.name, bad))
    cen, nrm, area = tri_geometry(X_ref, tris)
    log.info("reference %.1f mL, target %.1f mL, %d points, %d triangles",
             signed_volume(X_ref, tris) / 1000, signed_volume(X_raw, tris) / 1000, len(X_ref), len(tris))

    # 2. clamped hilum
    names = [cfg.anchor_point] if isinstance(cfg.anchor_point, str) else list(cfg.anchor_point)
    fid, rad = read_fiducials(cfg.anchor), read_fiducial_radii(cfg.anchor)
    missing = [n for n in names if fid.get(n) is None or rad.get(n) is None]
    if missing:
        raise RuntimeError("%s has no sphere point(s) %s (run the hilum step; available: %s)"
                           % (cfg.anchor.name, missing, ", ".join(sorted(rad))))
    centers = np.array([fid[n] for n in names])
    spheres = np.array([rad[n] for n in names])
    clamped, grown, sv = clamped_triangles(X_ref, tris, centers, spheres * cfg.anchor_radius_factor,
                                           cfg.anchor_min_points, cfg.anchor_growth, cfg.anchor_max_growth_steps)
    cpts = np.unique(tris[clamped])
    radii = spheres * cfg.anchor_radius_factor
    radii[0] *= grown                                         # only the first ball grows
    center = centers[0]
    log.info("clamp: %s, spheres %s mm x %.2f%s -> radii %s mm; %d triangles / %d points, up to %.0f mm "
             "from '%s' (singular values %.1f %.1f %.1f)", "+".join(names),
             "/".join("%.1f" % s for s in spheres), cfg.anchor_radius_factor,
             " (%s grown x%.2f)" % (names[0], grown) if grown > 1 else "", "/".join("%.1f" % r for r in radii),
             clamped.sum(), len(cpts), np.linalg.norm(X_ref[cpts] - center, axis=1).max(), names[0], *sv)
    radius = float(radii.max())

    # 3. rigid alignment of the target
    w_hil = np.zeros(len(X_ref))
    w_hil[cpts] = 1.0
    _, ang, sh = rigid(X_raw, X_ref)
    _, ang_h, sh_h = rigid(X_raw, X_ref, w_hil)
    log.info("rigid part target -> reference: full %.2f deg / %.2f mm, on the clamp %.2f deg / %.2f mm; "
             "clamped points move %.2f mm unaligned", ang, sh, ang_h, sh_h,
             np.linalg.norm(X_raw[cpts] - X_ref[cpts], axis=1).mean())
    if cfg.align == "none":
        X_tgt = X_raw
    else:
        X_tgt = rigid(X_raw, X_ref, w_hil if cfg.align == "hilum_rigid" else None)[0]
    d0 = np.linalg.norm(X_tgt - X_ref, axis=1)
    log.info("target alignment '%s': clamped points %.2f mm from their target, no-deformation error "
             "%.2f mm mean", cfg.align, d0[cpts].mean(), d0.mean())
    clamp_kind = clamped.astype(np.uint8)                     # 1 = anchor balls, 2 = still
    if cfg.clamp_still_mm is not None:
        still = (d0[tris] < cfg.clamp_still_mm).all(axis=1) & ~clamped
        clamped = clamped | still
        clamp_kind[still] = 2
        new = np.setdiff1d(np.unique(tris[still]), cpts)
        cpts = np.unique(tris[clamped])
        log.info("clamp_still_mm %.1f: +%d triangles / %d points moving less than that (%.2f mm mean); "
                 "clamped now %d triangles / %d points", cfg.clamp_still_mm, still.sum(), len(new),
                 d0[new].mean() if len(new) else 0.0, clamped.sum(), len(cpts))

    # 4. cavity wall (optional)
    wall_pts, wall_tris, wall_allow, phi0 = np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64), np.zeros(0), None
    if cfg.wall is not None:
        if cfg.wall == "reference":                   # the registered inflated surface (outward, closed)
            wall_name, wall_pts, wall_tris = "reference surface", X_ref.copy(), tris.copy()
        else:
            w = read_surface(cfg.wall, space="LPS")
            wall_name, wall_pts, wall_tris = cfg.wall.name, np.asarray(w.points, float), faces_of(w)
            if face_adjacency(wall_tris)[1]:
                raise RuntimeError("wall %s is not watertight" % cfg.wall.name)
            if signed_volume(wall_pts, wall_tris) < 0:
                wall_tris = wall_tris[:, ::-1].copy()
        wall = WallDistance(wall_pts, wall_tris)
        phi0, phit = wall(X_ref)[0], wall(X_tgt)[0]
        if cfg.wall == "reference":
            phi0 = np.zeros(len(X_ref))               # the reference points lie on it
        wall_allow = cfg.wall_tol_mm + np.maximum(phi0, 0.0)   # points already outside may not go further
        excess = phit - wall_allow
        log.info("wall %s (%.1f mL), tolerance %.1f mm: reference %d/%d points outside (median %.2f, "
                 "max %.2f mm); aligned target %d points beyond their allowed position (max %.2f mm "
                 "beyond, the fit cannot reach these)", wall_name,
                 signed_volume(wall_pts, wall_tris) / 1000, cfg.wall_tol_mm, (phi0 > 0).sum(), len(phi0),
                 np.median(phi0), max(phi0.max(), 0.0), (excess > 0).sum(), max(excess.max(), 0.0))

    # 5. clustering field and pressure regions
    if cfg.cluster_field == "raw":
        Xc = X_raw
    else:
        Xc = rigid(X_raw, X_ref, w_hil if cfg.cluster_field == "hilum_rigid" else None)[0]
    u = Xc - X_ref
    u_tri = u[tris].mean(axis=1)
    un = np.einsum("ij,ij->i", u_tri, nrm)
    feat = un[:, None] if cfg.feature == "normal" else u_tri
    levels, regions = all_levels(feat, adj, ~clamped, cfg.levels, cen, cfg.pos_weight)
    log.info("clustering field '%s': |u| mean %.1f mm, u.n mean %.1f mm (negative = inward); "
             "levels K = %s", cfg.cluster_field, np.linalg.norm(u, axis=1).mean(), un.mean(),
             ", ".join(map(str, levels)))

    # 6. volume mesh
    nodes, tets, surface_nodes = tet_mesh(X_ref, tris, out / "gmsh", cfg.mesh_size_mm, cfg.gmsh_timeout_s)

    # 7. outputs
    problem = CollapseProblem(nodes=nodes, tets=tets, surface_nodes=surface_nodes, tris=tris,
                              clamped_tri=clamped, target=X_tgt, levels=levels, regions=regions,
                              clean_field=u, wall_points=wall_pts, wall_tris=wall_tris,
                              wall_allow=wall_allow)
    problem.save(out / "problem.npz")

    s = polydata(X_ref, tris)
    s.point_data["CleanDisplacement_mm"] = u
    s.point_data["TargetDisplacement_mm"] = X_tgt - X_ref
    s.cell_data["Clamped"] = clamp_kind                         # 1 = anchor balls, 2 = still (clamp_still_mm)
    if phi0 is not None:
        s.point_data["WallDistance_mm"] = phi0
    for K, lab in zip(levels, regions):
        s.cell_data["Regions_K%d" % K] = lab
    write_surface(s, out / "reference.vtp", "LPS")
    write_surface(polydata(X_tgt, tris), out / "target_aligned.vtp", "LPS")
    vol = pv.UnstructuredGrid({pv.CellType.TETRA: tets}, nodes)
    vol.point_data["Clamped"] = np.isin(np.arange(len(nodes)), surface_nodes[cpts]).astype(np.uint8)
    vol.field_data["SPACE"] = np.array(["LPS"])
    vol.save(out / "volume.vtu")

    (out / "setup_summary.json").write_text(json.dumps(dict(
        reference=str(cfg.reference), target=str(cfg.target), anchor=str(cfg.anchor),
        anchor_point=cfg.anchor_point, anchor_center_lps=centers.tolist(), anchor_sphere_radius_mm=spheres.tolist(),
        clamp_radii_mm=radii.tolist(),
        anchor_radius_factor=cfg.anchor_radius_factor, clamp_radius_mm=radius,
        clamped_triangles=int(clamped.sum()), clamped_points=int(len(cpts)),
        clamp_still_mm=cfg.clamp_still_mm, still_triangles=int((clamp_kind == 2).sum()),
        align=cfg.align, rigid_full_deg=ang, rigid_clamp_deg=ang_h,
        baseline_mean_error_mm=float(d0.mean()), baseline_assd_mm=assd(X_ref, X_tgt, tris),
        cluster_field=cfg.cluster_field, feature=cfg.feature, levels=levels.tolist(),
        wall=str(cfg.wall) if cfg.wall is not None else None, wall_tol_mm=cfg.wall_tol_mm,
        volume_nodes=int(len(nodes)), tets=int(len(tets)), mesh_size_mm=cfg.mesh_size_mm), indent=2))
    log.info("wrote %s/ (problem.npz, reference.vtp, target_aligned.vtp, volume.vtu)", out)
    return out
