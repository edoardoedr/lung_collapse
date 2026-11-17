"""Clamped (Dirichlet) region: surface triangles inside balls around the hilum (and, optionally,
the hilar airway / vessel ring centres)."""

import logging

import numpy as np

from .geometry import tri_geometry

log = logging.getLogger(__name__)


def clamped_triangles(P, tris, centers, radii, min_points, growth, max_steps):
    """Triangles whose centroid lies within radii[i] of centers[i] for some i (union of balls).

    The first ball (the primary anchor, the hilum) grows by `growth` (up to `max_steps` times)
    until the clamped points are at least `min_points` and not close to a plane or a line (third
    singular value > 1 mm): a clamp without rank 3 leaves rigid motions free and the tangent
    matrix singular. The other balls are added as they are, so they only ever enlarge the clamp.
    Returns (mask, growth factor of the first ball, singular values).
    """
    cen, _, _ = tri_geometry(P, tris)
    centers, radii = np.atleast_2d(np.asarray(centers, float)), np.atleast_1d(np.asarray(radii, float))
    rel = np.linalg.norm(cen[:, None, :] - centers[None], axis=2) / radii[None]   # inside ball i if <= 1
    extra = (rel[:, 1:] <= 1.0).any(axis=1)
    for step in range(max_steps + 1):
        f = growth ** step
        mask = (rel[:, 0] <= f) | extra
        pts = np.unique(tris[mask])
        sv = (np.linalg.svd(P[pts] - P[pts].mean(0), compute_uv=False) if len(pts) >= 3
              else np.zeros(3))
        if len(pts) >= min_points and sv[2] > 1.0:
            break
    else:
        log.warning("clamped region still small after %d growth steps (%d points)", max_steps, len(pts))
    if len(pts) < 3 or sv[2] <= 1e-6:
        raise RuntimeError("clamped region is not rank 3 (%d points inside the anchor balls grown x%.2f, "
                           "nearest triangle at %.1f x its ball radius): wrong anchor or space?"
                           % (len(pts), f, rel.min()))
    if step:
        log.info("clamp radii grown x%.2f", f)
    return mask, float(f), sv
