"""Clamped (Dirichlet) hilum region: surface triangles inside a ball around the hilum."""

import logging

import numpy as np

from .geometry import tri_geometry

log = logging.getLogger(__name__)


def clamped_triangles(P, tris, center, radius, min_points, growth, max_steps):
    """Triangles whose centroid lies within `radius` of `center`.

    The radius grows by `growth` (up to `max_steps` times) until the clamped points are at least
    `min_points` and not close to a plane or a line (third singular value > 1 mm): a clamp
    without rank 3 leaves rigid motions free and the tangent matrix singular.
    Returns (mask, radius used, singular values).
    """
    cen, _, _ = tri_geometry(P, tris)
    dist = np.linalg.norm(cen - center, axis=1)
    for step in range(max_steps + 1):
        r = radius * growth ** step
        mask = dist <= r
        pts = np.unique(tris[mask])
        sv = (np.linalg.svd(P[pts] - P[pts].mean(0), compute_uv=False) if len(pts) >= 3
              else np.zeros(3))
        if len(pts) >= min_points and sv[2] > 1.0:
            break
    else:
        log.warning("clamped region still small after %d growth steps (%d points)", max_steps, len(pts))
    if len(pts) < 3 or sv[2] <= 1e-6:
        raise RuntimeError("clamped region is not rank 3 (%d points within %.1f mm of the hilum, "
                           "nearest triangle at %.1f mm): wrong anchor or space?"
                           % (len(pts), r, dist.min()))
    if step:
        log.info("clamp radius grown x%.2f: %.1f -> %.1f mm", growth ** step, radius, r)
    return mask, float(r), sv
