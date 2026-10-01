"""Step 3 - hilum from the airway and vessel segmentations.

Each tree (airways, arteries, veins) pierces the lung surface in closed rings. The exact
surface-surface intersection gives these rings; per structure the largest one (perimeter)
is where it enters the lung at the hilum, the others are small peripheral branches.

  1. rings: exact intersection tree / lung surface, split into connected closed curves
  2. per structure: largest ring -> centre and mean radius (weighted by segment length)
  3. hilum = centroid of the ring centres, radius = mean of the ring radii
  4. write hilum_anchor.mrk.json (LPS) with one point list per ring + one for the hilum, each
     drawn in Slicer as a sphere with its diameter (radius also in the point description),
     plus hilum/ with the numbers (json) and all the rings (vtp).

The trees and the lung must be in the same scan space.
"""

import json
import logging

import numpy as np
import pyvista as pv
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from vtkmodules.vtkCommonCore import vtkLogger
from vtkmodules.vtkFiltersGeneral import vtkIntersectionPolyDataFilter

from .data_io import fiducial_list, read_surface, write_markups, write_surface

log = logging.getLogger(__name__)

COLORS = {"airways": (0.6, 0.9, 1.0), "arteries": (0.85, 0.2, 0.2), "veins": (0.2, 0.35, 0.9),
          "hilum": (1.0, 0.85, 0.0)}


def intersection_segments(a, b):
    """Exact intersection of two surfaces: (points, line segments as index pairs)."""
    f = vtkIntersectionPolyDataFilter()
    f.SetInputData(0, a)
    f.SetInputData(1, b)
    f.SplitFirstOutputOff()
    f.SplitSecondOutputOff()
    old = vtkLogger.GetCurrentVerbosityCutoff()
    vtkLogger.SetStderrVerbosity(vtkLogger.VERBOSITY_ERROR)   # degenerate-triangle warnings
    try:
        f.Update()
    finally:
        vtkLogger.SetStderrVerbosity(old)
    x = pv.wrap(f.GetOutput(0)).clean()
    if x.n_lines == 0:
        return np.zeros((0, 3)), np.zeros((0, 2), dtype=int)
    return np.asarray(x.points), x.lines.reshape(-1, 3)[:, 1:]


def split_rings(pts, seg):
    """Connected curves -> list of dicts (centre, radius, perimeter, segments)."""
    n = len(pts)
    adj = coo_matrix((np.ones(len(seg)), (seg[:, 0], seg[:, 1])), shape=(n, n))
    _, lab = connected_components(adj, directed=False)
    rings = []
    for k in np.unique(lab[seg[:, 0]]):
        s = seg[lab[seg[:, 0]] == k]
        a, b = pts[s[:, 0]], pts[s[:, 1]]
        length = np.linalg.norm(b - a, axis=1)
        mid = 0.5 * (a + b)
        center = (length[:, None] * mid).sum(0) / length.sum()
        radius = (length * np.linalg.norm(mid - center, axis=1)).sum() / length.sum()
        rings.append(dict(center=center, radius=float(radius), perimeter=float(length.sum()),
                          segments=s))
    return sorted(rings, key=lambda r: -r["perimeter"])


def run(cfg):
    lung = read_surface(cfg.lung, space="LPS")
    chosen, all_rings = {}, {}
    for name, path in cfg.structures.items():
        pts, seg = intersection_segments(read_surface(path, space="LPS"), lung)
        if len(seg) == 0:
            raise RuntimeError("%s does not intersect %s: wrong files or different scan?"
                               % (path.name, cfg.lung.name))
        rings = split_rings(pts, seg)
        all_rings[name] = (pts, rings)
        chosen[name] = rings[0]
        r = rings[0]
        log.info("%-10s %d ring(s); hilar ring: centre [%.1f, %.1f, %.1f], radius %.1f mm, "
                 "perimeter %.1f mm", name, len(rings), *r["center"], r["radius"], r["perimeter"])

    hilum = np.mean([r["center"] for r in chosen.values()], axis=0)
    hilum_radius = float(np.mean([r["radius"] for r in chosen.values()]))
    dist = {k: float(np.linalg.norm(r["center"] - hilum)) for k, r in chosen.items()}
    _, closest = lung.find_closest_cell(hilum[None], return_closest_point=True)
    surf_dist = float(np.linalg.norm(closest[0] - hilum))
    log.info("hilum LPS [%.1f, %.1f, %.1f], radius %.1f mm (mean of the rings), %.1f mm from "
             "the %s surface; ring centres at %s", *hilum, hilum_radius, surf_dist, cfg.lung.name,
             ", ".join("%s %.1f mm" % kv for kv in dist.items()))
    if max(dist.values()) > cfg.max_ring_distance_mm:
        log.warning("a ring centre is > %.0f mm from the hilum: check the rings in hilum/rings.vtp",
                    cfg.max_ring_distance_mm)

    # one list per point: Slicer sizes glyphs per list, so each sphere shows its own radius
    lists = [fiducial_list(k, {k: r["center"]}, {k: "radius_mm=%.2f" % r["radius"]},
                           sphere_mm=2 * r["radius"], color=COLORS[k])
             for k, r in chosen.items()]
    lists.append(fiducial_list("hilum", {"hilum": hilum},
                               {"hilum": "radius_mm=%.2f (mean of the rings); centroid of the "
                                         "ring centres" % hilum_radius},
                               sphere_mm=2 * hilum_radius, color=COLORS["hilum"]))
    write_markups(lists, cfg.output)

    qa = cfg.output.parent / "hilum"
    qa.mkdir(parents=True, exist_ok=True)
    (qa / "hilum_anchor.json").write_text(json.dumps(dict(
        hilum_lps=hilum.tolist(), hilum_radius_mm=hilum_radius, lung=str(cfg.lung),
        distance_to_lung_surface_mm=surf_dist,
        rings={k: dict(center_lps=r["center"].tolist(), radius_mm=r["radius"],
                       perimeter_mm=r["perimeter"], distance_to_hilum_mm=dist[k],
                       n_rings_found=len(all_rings[k][1])) for k, r in chosen.items()}), indent=2))
    write_surface(rings_polydata(all_rings), qa / "rings.vtp", "LPS")
    log.info("wrote %s (+ %s/)", cfg.output, qa.name)
    return cfg.output


def rings_polydata(all_rings):
    """All intersection rings as lines; cell data Structure (index) and Hilar (1 = chosen)."""
    blocks, struct, hilar, offset = [], [], [], 0
    pts_all, lines = [], []
    for i, (name, (pts, rings)) in enumerate(all_rings.items()):
        pts_all.append(pts)
        for j, r in enumerate(rings):
            lines.append(r["segments"] + offset)
            struct.append(np.full(len(r["segments"]), i))
            hilar.append(np.full(len(r["segments"]), int(j == 0)))
        offset += len(pts)
    seg = np.vstack(lines)
    out = pv.PolyData(np.vstack(pts_all), lines=np.hstack([np.full((len(seg), 1), 2), seg]).ravel())
    out.cell_data["Structure"] = np.concatenate(struct)
    out.cell_data["Hilar"] = np.concatenate(hilar)
    out.field_data["StructureNames"] = np.array(list(all_rings))
    return out
