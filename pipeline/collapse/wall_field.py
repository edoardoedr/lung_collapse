"""Cavity wall as a smooth field: signed distance to the wall (> 0 outside) sampled on a regular
grid, smoothed, and interpolated with a tricubic B-spline, so that value, gradient and Hessian are
continuous in the position (C2). The contact of wall_update "potential" is the energy of this
field at the current positions, inside Newton (solvers/nodal.py).

Evaluation (WallField) needs numpy / scipy only; building the grid (signed_distance_grid) needs
pyvista / VTK and is done once in fem_setup.
"""

import numpy as np
from scipy import ndimage


def signed_distance_grid(points, tris, spacing, margin, smoothing):
    """Smoothed signed distance to the closed surface (points, tris; outward triangles) on a
    regular grid: (grid (nx, ny, nz) float32, origin (3,)). spacing, margin (around the surface's
    bounding box) and smoothing (Gaussian sigma) in mm."""
    import pyvista as pv
    P = np.asarray(points, float)
    lo, hi = P.min(0) - margin, P.max(0) + margin
    dims = np.ceil((hi - lo) / spacing).astype(int) + 1
    image = getattr(pv, "ImageData", None) or pv.UniformGrid
    grid = image(dimensions=tuple(int(d) for d in dims), spacing=(spacing,) * 3, origin=tuple(lo))
    surf = pv.PolyData(P, np.column_stack([np.full(len(tris), 3), tris]).ravel())
    d = np.asarray(grid.compute_implicit_distance(surf)["implicit_distance"], float)
    d = d.reshape(tuple(dims), order="F")                      # VTK order: x fastest
    if d[0, 0, 0] < 0:                                         # a corner of the box is outside
        d = -d
    if smoothing > 0:
        d = ndimage.gaussian_filter(d, smoothing / spacing, mode="nearest")
    return d.astype(np.float32), lo


def _basis(f):
    """Cubic B-spline weights of the 4 coefficients around a point at fraction f of its cell, with
    first and second derivatives with respect to f: three arrays (..., 4)."""
    g = 1.0 - f
    w = np.stack([g ** 3 / 6.0, (3.0 * f ** 3 - 6.0 * f ** 2 + 4.0) / 6.0,
                  (-3.0 * f ** 3 + 3.0 * f ** 2 + 3.0 * f + 1.0) / 6.0, f ** 3 / 6.0], axis=-1)
    d = np.stack([-0.5 * g ** 2, 1.5 * f ** 2 - 2.0 * f, -1.5 * f ** 2 + f + 0.5, 0.5 * f ** 2], axis=-1)
    dd = np.stack([g, 3.0 * f - 2.0, -3.0 * f + 1.0, f], axis=-1)
    return w, d, dd


class WallField:
    """Tricubic B-spline interpolation of a grid (nx, ny, nz) with the given origin and spacing
    [mm]: field(X) -> (phi (N,), gradient (N, 3), Hessian (N, 3, 3)), exact at the grid points."""

    def __init__(self, grid, origin, spacing):
        self.c = ndimage.spline_filter(np.asarray(grid, float), order=3, mode="mirror")
        self.origin = np.asarray(origin, float).reshape(3)
        self.h = float(spacing)
        self.n = np.array(self.c.shape)
        if np.any(self.n < 4):
            raise ValueError("wall field: the grid needs at least 4 points per axis")

    def inside(self, X):
        """True for the points the grid covers (one cell from its border)."""
        t = (np.atleast_2d(X) - self.origin) / self.h
        return np.all((t >= 1.0) & (t < self.n - 2.0), axis=1)

    def __call__(self, X, derivatives=2):
        """derivatives: 0 = phi only, 1 = (phi, gradient), 2 = (phi, gradient, Hessian).
        Raises ValueError for points outside the grid (see inside())."""
        X = np.atleast_2d(np.asarray(X, float))
        t = (X - self.origin) / self.h
        if not np.all((t >= 1.0) & (t < self.n - 2.0)):
            raise ValueError("wall field: %d point(s) outside the grid (increase fem_setup.wall_grid_margin_mm)"
                             % int((~self.inside(X)).sum()))
        i = np.floor(t).astype(np.int64)
        f = t - i
        o = np.arange(-1, 3)
        C = self.c[(i[:, 0, None] + o)[:, :, None, None], (i[:, 1, None] + o)[:, None, :, None],
                   (i[:, 2, None] + o)[:, None, None, :]]                    # (N, 4, 4, 4)
        (wx, dx, ddx), (wy, dy, ddy), (wz, dz, ddz) = _basis(f[:, 0]), _basis(f[:, 1]), _basis(f[:, 2])

        def s(a, b, c):
            return np.einsum("nijk,ni,nj,nk->n", C, a, b, c)

        phi = s(wx, wy, wz)
        if derivatives == 0:
            return phi
        h = self.h
        grad = np.stack([s(dx, wy, wz), s(wx, dy, wz), s(wx, wy, dz)], axis=1) / h
        if derivatives == 1:
            return phi, grad
        xy, xz, yz = s(dx, dy, wz), s(dx, wy, dz), s(wx, dy, dz)
        H = np.empty((len(X), 3, 3))
        H[:, 0, 0], H[:, 1, 1], H[:, 2, 2] = s(ddx, wy, wz), s(wx, ddy, wz), s(wx, wy, ddz)
        H[:, 0, 1] = H[:, 1, 0] = xy
        H[:, 0, 2] = H[:, 2, 0] = xz
        H[:, 1, 2] = H[:, 2, 1] = yz
        return phi, grad, H / h ** 2


def penalty(g, k, e):
    """Wall force law per unit area and its derivative: w = (k / 2e) (pos(g)^2 - pos(g - e)^2), zero
    for g <= 0, quadratic up to e, then linear with slope k (the law of the linearised wall)."""
    pg, pe = np.maximum(g, 0.0), np.maximum(g - e, 0.0)
    return k / (2.0 * e) * (pg ** 2 - pe ** 2), k / e * (pg - pe)


def penalty_energy(g, k, e):
    """Energy per unit area, d/dg = penalty(g)[0]."""
    pg, pe = np.maximum(g, 0.0), np.maximum(g - e, 0.0)
    return k / (6.0 * e) * (pg ** 3 - pe ** 3)
