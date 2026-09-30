"""Step 2 - surface meshing.

Slicer segmentation surface (dense, staircase artefacts) -> smooth, watertight,
near-uniform triangle mesh with ~target_nodes nodes, via ACVD clustering (pyacvd).

  1. clean: weld duplicate points, keep the largest component, Taubin smoothing, fill holes
  2. remesh: ACVD with target_nodes clusters, shrunk until faces <= max_faces
  3. relax: light Taubin smoothing, fill holes, watertightness check

All operations are rigid-motion invariant, so the work is done directly in the
output space (LPS <-> RAS is a 180 deg rotation about z).
"""

import logging
import math

import numpy as np
import pyacvd

from .data_io import as_triangles, read_surface, write_surface

log = logging.getLogger(__name__)


def bad_edge_count(surf):
    """Number of boundary + non-manifold edges (0 = watertight manifold)."""
    e = surf.extract_feature_edges(boundary_edges=True, non_manifold_edges=True,
                                   feature_edges=False, manifold_edges=False)
    return e.n_cells


def close_holes(surf, hole_size):
    if surf.n_open_edges > 0:
        surf = as_triangles(surf.fill_holes(hole_size))
    return surf


def smooth(surf, n_iter, pass_band):
    return surf.smooth_taubin(n_iter=n_iter, pass_band=pass_band) if n_iter > 0 else surf


def clean_surface(surf, cfg):
    surf = as_triangles(surf.clean(tolerance=cfg.merge_tolerance))
    surf = surf.extract_largest()                  # drop speckle islands
    surf = smooth(surf, cfg.pre_smooth_iters, cfg.smooth_pass_band)
    return close_holes(surf, cfg.hole_size)


def acvd_remesh(surf, n_clusters, min_points_per_cluster, max_subdivisions):
    """ACVD isotropic remesh to ~n_clusters nodes."""
    # each subdivision multiplies the point count by ~4
    ratio = min_points_per_cluster * n_clusters / max(surf.n_points, 1)
    n_sub = 0 if ratio <= 1 else min(max_subdivisions, math.ceil(math.log(ratio, 4)))
    clus = pyacvd.Clustering(surf)
    if n_sub > 0:
        clus.subdivide(n_sub)
    clus.cluster(n_clusters)
    return as_triangles(clus.create_mesh())


def remesh_surface(surf, cfg):
    n_clusters = cfg.target_nodes
    for _ in range(cfg.max_remesh_attempts):
        mesh = acvd_remesh(surf, n_clusters, cfg.min_points_per_cluster, cfg.max_subdivisions)
        log.info("  clusters=%d -> %d pts, %d tris", n_clusters, mesh.n_points, mesh.n_cells)
        if mesh.n_cells <= cfg.max_faces:
            return mesh
        n_clusters = int(n_clusters * cfg.shrink_factor)
    raise RuntimeError("still > %d faces after %d attempts: lower target_nodes"
                       % (cfg.max_faces, cfg.max_remesh_attempts))


def min_angles_deg(surf):
    """Smallest interior angle of every triangle [deg]."""
    tri = surf.points[surf.faces.reshape(-1, 4)[:, 1:]]
    angles = []
    for i in range(3):
        u = tri[:, (i + 1) % 3] - tri[:, i]
        v = tri[:, (i + 2) % 3] - tri[:, i]
        cos = np.einsum("ij,ij->i", u, v) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1))
        angles.append(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
    return np.min(angles, axis=0)


def run(cfg):
    """Remesh cfg.input and write it to cfg.output. Returns the output path."""
    surf = read_surface(cfg.input, space=cfg.output_space, input_space=cfg.input_space)

    surf = clean_surface(surf, cfg)
    log.info("cleaned: %d pts, %d tris, bad edges %d",
             surf.n_points, surf.n_cells, bad_edge_count(surf))

    mesh = remesh_surface(surf, cfg)
    mesh = smooth(mesh, cfg.post_smooth_iters, cfg.smooth_pass_band)
    mesh = close_holes(mesh, cfg.hole_size)

    n_bad = bad_edge_count(mesh)
    if n_bad:
        raise RuntimeError("remeshed surface is not watertight (%d bad edges)" % n_bad)
    ang = min_angles_deg(mesh)
    log.info("final: %d pts, %d tris, volume %.1f mL, min angle min/mean %.1f/%.1f deg",
             mesh.n_points, mesh.n_cells, mesh.volume / 1000.0, ang.min(), ang.mean())

    write_surface(mesh, cfg.output, cfg.output_space)
    log.info("wrote %s (%s)", cfg.output, cfg.output_space)
    return cfg.output
