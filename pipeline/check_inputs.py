"""Step 1b - sanity check of the input surfaces, before any processing.

  1. every surface: readable, coordinate space tag, closed (no open / non-manifold edges),
     one connected piece, volume
  2. same scan: every structure's bounding box overlaps the collapsed lung's
  3. containment: the collapsed lung must lie inside the inflated one (same scan, the lung only
     shrinks); points outside by more than outside_tol_mm are flagged
  4. narrow notches (folds, open fissures) in both lungs: the mask is closed with a ball of
     notch_radius_mm and points lying deeper than notch_depth_mm inside the closed volume are
     flagged. They break the registration (flipped triangles) and Gmsh (self-intersections)
  5. volume ratio inflated / collapsed

Findings are logged as warnings; the step fails only if fail_on_warning. Outputs in
output_dir/checks/: collapsed_check.vtp / inflated_check.vtp (point data to colour in Slicer)
and check_summary.json.
"""

import json
import logging

import itk
import numpy as np
import pyvista as pv
from scipy.ndimage import distance_transform_edt, map_coordinates

from .collapse.geometry import WallDistance, face_adjacency, faces_of, signed_volume
from .data_io import detect_space, read_surface, write_surface
from .registration import geometry, grid_around, rasterize

log = logging.getLogger(__name__)


def lps_ras(p):
    """'LPS [..] / RAS [..]' (Slicer shows RAS)."""
    return "LPS [%.1f, %.1f, %.1f] / RAS [%.1f, %.1f, %.1f]" % (*p, -p[0], -p[1], p[2])


def integrity(name, path, warnings, lung=True):
    """Basic checks of one surface; returns it (LPS; for a lung its largest piece) or None.

    Extra pieces are a finding for the lungs (islands or internal holes of the segmentation),
    normal for vessel and airway trees."""
    try:
        surf = read_surface(path, space="LPS")
    except Exception as e:
        warnings.append("%s: cannot be read (%s)" % (name, e))
        return None
    if detect_space(pv.read(path), path) is None:
        warnings.append("%s: no coordinate space tag, LPS assumed" % name)
    tris = faces_of(surf)
    _, bad = face_adjacency(tris)
    n_parts = surf.connectivity().point_data["RegionId"].max() + 1
    vol = abs(signed_volume(np.asarray(surf.points), tris)) / 1000.0
    log.info("%-10s %s: %d points, %d triangles, %d piece(s), %d bad edges, %.1f mL",
             name, path.name, surf.n_points, surf.n_cells, n_parts, bad, vol)
    if bad:
        warnings.append("%s: not closed, %d open or non-manifold edges" % (name, bad))
    if lung and n_parts > 1:
        main = read_surface_piece(surf)
        rest = vol - abs(signed_volume(np.asarray(main.points), faces_of(main))) / 1000.0
        warnings.append("%s: %d extra pieces besides the lung (%d points, %.2f mL): islands or internal "
                        "holes of the segmentation (Segment Editor: Islands > Keep largest, Smoothing > "
                        "Closing); the checks below use the largest piece"
                        % (name, n_parts - 1, surf.n_points - main.n_points, abs(rest)))
        surf = main
    return surf


def read_surface_piece(surf):
    """Largest connected piece, as triangles."""
    main = surf.connectivity(extraction_mode="largest")
    return pv.PolyData(np.asarray(main.points), main.faces).clean()


def notch_depth(surf, radius, spacing):
    """Depth [mm] of each surface point inside the surface closed with a ball of `radius`."""
    img = rasterize(surf, grid_around(surf, np.full(3, spacing), radius + 3 * spacing))
    origin, sp, _, _ = geometry(img)
    m = itk.array_from_image(img) > 0                                  # z, y, x
    sp_zyx = sp[::-1]
    dilated = distance_transform_edt(~m, sampling=sp_zyx) <= radius
    closed = distance_transform_edt(dilated, sampling=sp_zyx) > radius
    depth = distance_transform_edt(closed, sampling=sp_zyx)
    idx = ((np.asarray(surf.points) - origin) / sp)[:, ::-1].T          # continuous z, y, x
    return np.maximum(map_coordinates(depth, idx, order=1) - 0.5 * sp.min(), 0.0)


def flag_notches(name, surf, cfg, warnings):
    d = notch_depth(surf, cfg.notch_radius_mm, cfg.raster_spacing_mm)
    deep = d > cfg.notch_depth_mm
    log.info("%-10s notches (closing %.1f mm): %.2f%% of points deeper than %.1f mm, max %.1f mm",
             name, cfg.notch_radius_mm, 100 * deep.mean(), cfg.notch_depth_mm, d.max())
    if deep.mean() > cfg.max_notch_fraction:
        warnings.append("%s: %.2f%% of the surface in narrow notches deeper than %.1f mm (max %.1f mm "
                        "at %s): fill them (Segment Editor, Smoothing > Closing)"
                        % (name, 100 * deep.mean(), cfg.notch_depth_mm, d.max(),
                           lps_ras(np.asarray(surf.points)[d.argmax()])))
    return d


def run(cfg):
    out = cfg.workdir
    out.mkdir(parents=True, exist_ok=True)
    warnings = []

    # 1. integrity
    collapsed = integrity("collapsed", cfg.collapsed, warnings)
    inflated = integrity("inflated", cfg.inflated, warnings)
    structures = {k: integrity(k, p, warnings, lung=False) for k, p in cfg.structures.items()}

    # 2. same scan
    if collapsed is not None:
        b0 = np.array(collapsed.bounds).reshape(3, 2)
        for name, s in [("inflated", inflated), *structures.items()]:
            if s is None:
                continue
            b = np.array(s.bounds).reshape(3, 2)
            if np.any(b[:, 1] < b0[:, 0]) or np.any(b[:, 0] > b0[:, 1]):
                warnings.append("%s does not overlap the collapsed lung: different scan or space?" % name)

    summary = dict(collapsed=str(cfg.collapsed), inflated=str(cfg.inflated))
    if collapsed is not None and inflated is not None:
        # 3. containment
        P = np.asarray(collapsed.points)
        wt = faces_of(inflated)
        if signed_volume(np.asarray(inflated.points), wt) < 0:
            wt = wt[:, ::-1].copy()
        phi = WallDistance(np.asarray(inflated.points), wt).distance(P)
        outside = phi > cfg.outside_tol_mm
        log.info("containment: %.2f%% of the collapsed points outside the inflated lung, %.2f%% by more "
                 "than %.1f mm, max %.1f mm", 100 * (phi > 0).mean(), 100 * outside.mean(),
                 cfg.outside_tol_mm, phi.max())
        if outside.mean() > cfg.max_outside_fraction:
            warnings.append("collapsed lung outside the inflated one: %.2f%% of the points by more than "
                            "%.1f mm, max %.1f mm at %s (Segment Editor: Logical operators > Add the "
                            "collapsed segment to the inflated one)" % (100 * outside.mean(), cfg.outside_tol_mm,
                                                                         phi.max(), lps_ras(P[phi.argmax()])))

        # 4. notches
        dc = flag_notches("collapsed", collapsed, cfg, warnings)
        di = flag_notches("inflated", inflated, cfg, warnings)

        # 5. volume ratio
        vc = abs(signed_volume(P, faces_of(collapsed))) / 1000.0
        vi = abs(signed_volume(np.asarray(inflated.points), wt)) / 1000.0
        log.info("volume inflated / collapsed: %.1f / %.1f mL = x%.2f", vi, vc, vi / vc)
        if vi / vc > cfg.max_volume_ratio:
            warnings.append("inflated / collapsed volume x%.1f > %.1f: the registration will be hard"
                            % (vi / vc, cfg.max_volume_ratio))

        c = collapsed.copy()
        c.point_data["DistanceToInflated_mm"] = phi
        c.point_data["NotchDepth_mm"] = dc
        write_surface(c, out / "collapsed_check.vtp", "LPS")
        i = inflated.copy()
        i.point_data["NotchDepth_mm"] = di
        write_surface(i, out / "inflated_check.vtp", "LPS")
        summary.update(outside_fraction=float((phi > 0).mean()), outside_tol_fraction=float(outside.mean()),
                       outside_max_mm=float(phi.max()), notch_fraction_collapsed=float((dc > cfg.notch_depth_mm).mean()),
                       notch_max_collapsed_mm=float(dc.max()), notch_fraction_inflated=float((di > cfg.notch_depth_mm).mean()),
                       notch_max_inflated_mm=float(di.max()), volume_collapsed_ml=vc, volume_inflated_ml=vi)

    summary["warnings"] = warnings
    (out / "check_summary.json").write_text(json.dumps(summary, indent=2))
    for w in warnings:
        log.warning(w)
    if not warnings:
        log.info("all input checks passed")
    if warnings and cfg.fail_on_warning:
        raise RuntimeError("%d input check(s) failed, see above" % len(warnings))
    log.info("wrote %s/ (collapsed_check.vtp, inflated_check.vtp, check_summary.json)", out)
    return out
