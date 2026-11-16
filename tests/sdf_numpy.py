"""Signed distance to a closed triangle surface in numpy (test fixture: fem_setup uses VTK)."""

import numpy as np


def _point_tri_dist2(P, A, B, C):
    """Squared distance of points P (n, 1, 3) to triangles A, B, C (1, m, 3): closest of the
    plane projection (when inside) and the three edges."""
    def seg(P, A, B):
        ab = B - A
        t = np.clip(((P - A) * ab).sum(-1) / (ab * ab).sum(-1), 0.0, 1.0)
        d = P - (A + t[..., None] * ab)
        return (d * d).sum(-1)
    d2 = np.minimum(np.minimum(seg(P, A, B), seg(P, B, C)), seg(P, C, A))
    n = np.cross(B - A, C - A)
    nn = (n * n).sum(-1)
    dist = ((P - A) * n).sum(-1)
    Q = P - (dist / nn)[..., None] * n
    inside = ((np.cross(B - A, Q - A) * n).sum(-1) >= 0) & ((np.cross(C - B, Q - B) * n).sum(-1) >= 0) \
        & ((np.cross(A - C, Q - C) * n).sum(-1) >= 0)
    return np.where(inside, dist ** 2 / nn, d2)


def signed_distance(points, tris, X, chunk=2000):
    """> 0 outside (sign from the winding number)."""
    A, B, C = (np.asarray(points, float)[tris[:, k]][None] for k in range(3))
    out = np.empty(len(X))
    for s in range(0, len(X), chunk):
        P = X[s:s + chunk, None, :]
        d = np.sqrt(_point_tri_dist2(P, A, B, C).min(1))
        a, b, c = A - P, B - P, C - P
        la, lb, lc = (np.linalg.norm(v, axis=-1) for v in (a, b, c))
        num = (a * np.cross(b, c)).sum(-1)
        den = la * lb * lc + (a * b).sum(-1) * lc + (b * c).sum(-1) * la + (c * a).sum(-1) * lb
        wind = (2.0 * np.arctan2(num, den)).sum(1) / (4.0 * np.pi)
        out[s:s + chunk] = np.where(np.abs(wind) > 0.5, -d, d)
    return out


def grid(points, tris, spacing, margin, smoothing):
    """As wall_field.signed_distance_grid, without VTK."""
    from scipy import ndimage
    P = np.asarray(points, float)
    lo, hi = P.min(0) - margin, P.max(0) + margin
    dims = np.ceil((hi - lo) / spacing).astype(int) + 1
    ax = [lo[k] + spacing * np.arange(dims[k]) for k in range(3)]
    X = np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3)
    d = signed_distance(P, tris, X).reshape(tuple(dims))
    if smoothing > 0:
        d = ndimage.gaussian_filter(d, smoothing / spacing, mode="nearest")
    return d.astype(np.float32), lo
