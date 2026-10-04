"""Surface helpers for the collapse steps. Points (N, 3), triangles (T, 3) index arrays."""

import numpy as np
import pyvista as pv
from vtkmodules.vtkFiltersCore import vtkImplicitPolyDataDistance


def faces_of(surf):
    return np.asarray(surf.faces).reshape(-1, 4)[:, 1:].astype(np.int64)


def polydata(points, tris):
    return pv.PolyData(np.asarray(points, float), np.column_stack([np.full(len(tris), 3), tris]).ravel())


def tri_geometry(P, tris):
    """Centroids, unit normals and areas of the triangles."""
    a, b, c = P[tris[:, 0]], P[tris[:, 1]], P[tris[:, 2]]
    cr = np.cross(b - a, c - a)
    area = 0.5 * np.linalg.norm(cr, axis=1)
    return (a + b + c) / 3.0, cr / np.maximum(2.0 * area[:, None], 1e-12), area


def signed_volume(P, tris):
    a, b, c = P[tris[:, 0]], P[tris[:, 1]], P[tris[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def vertex_normals(P, tris):
    """Area-weighted unit normals at the points."""
    _, n, area = tri_geometry(P, tris)
    acc = np.zeros_like(P, dtype=float)
    for k in range(3):
        np.add.at(acc, tris[:, k], n * area[:, None])
    return acc / np.maximum(np.linalg.norm(acc, axis=1, keepdims=True), 1e-12)


def face_adjacency(tris):
    """(pairs of triangles sharing an edge, number of edges not shared by exactly 2 triangles)."""
    e = np.sort(np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]), axis=1)
    f = np.tile(np.arange(len(tris)), 3)
    order = np.lexsort((e[:, 1], e[:, 0]))
    e, f = e[order], f[order]
    _, start, count = np.unique(e, axis=0, return_index=True, return_counts=True)
    two = count == 2
    return np.column_stack([f[start[two]], f[start[two] + 1]]), int((~two).sum())


def kabsch(A, B, w=None):
    """R, t minimising sum w_i |R A_i + t - B_i|^2."""
    w = np.ones(len(A)) if w is None else np.asarray(w, float)
    w = w / w.sum()
    ca, cb = w @ A, w @ B
    H = ((A - ca) * w[:, None]).T @ (B - cb)
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, cb - R @ ca


def rotation_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))))


class WallDistance:
    """Distance to a closed surface (> 0 outside) and the outward direction, for the wall contact.

    __call__ gives the contact data, continuous in X: closest point on the surface, normal =
    area-weighted vertex normals interpolated at it (barycentric), phi = (X - closest point) . normal.
    The raw gradient of the distance would be the direction to the closest point, which on an
    edge or vertex (and the "reference" wall has the lung's surface nodes on its vertices) turns
    arbitrarily under tiny displacements; Newton cannot converge on that.
    distance() is the exact signed distance (reports, checks).

    The surface must have outward triangles (positive signed_volume)."""

    def __init__(self, points, tris):
        self.P = np.asarray(points, float)
        self.T = np.asarray(tris, np.int64)
        self.surf = polydata(self.P, self.T)
        self.N = vertex_normals(self.P, self.T)
        self.f = vtkImplicitPolyDataDistance()
        self.f.SetInput(self.surf)

    def distance(self, X):
        return np.array([self.f.EvaluateFunction(x) for x in np.asarray(X, float)])

    def __call__(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        cell, cp = self.surf.find_closest_cell(X, return_closest_point=True)
        return self._contact(X, np.atleast_1d(cell), np.atleast_2d(cp))

    def _contact(self, X, cell, cp):
        """(phi, n) from the closest triangle and point of each X."""
        tri = self.T[cell]
        a, b, c = self.P[tri[:, 0]], self.P[tri[:, 1]], self.P[tri[:, 2]]
        v0, v1, v2 = b - a, c - a, cp - a
        d00, d01, d11 = (v0 * v0).sum(1), (v0 * v1).sum(1), (v1 * v1).sum(1)
        d20, d21 = (v2 * v0).sum(1), (v2 * v1).sum(1)
        den = np.maximum(d00 * d11 - d01 * d01, 1e-300)
        wb, wc = (d11 * d20 - d01 * d21) / den, (d00 * d21 - d01 * d20) / den
        w = np.clip(np.column_stack([1.0 - wb - wc, wb, wc]), 0.0, None)
        w /= np.maximum(w.sum(1, keepdims=True), 1e-300)
        n = np.einsum("ik,ikj->ij", w, self.N[tri])
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        return np.einsum("ij,ij->i", X - cp, n), n


def assd(PA, PB, tris):
    """Symmetric mean point-to-surface distance between two surfaces with the same triangles."""
    A, B = polydata(PA, tris), polydata(PB, tris)
    _, cb = B.find_closest_cell(A.points, return_closest_point=True)
    _, ca = A.find_closest_cell(B.points, return_closest_point=True)
    return float(np.r_[np.linalg.norm(A.points - cb, axis=1), np.linalg.norm(B.points - ca, axis=1)].mean())
