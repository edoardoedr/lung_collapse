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
    """Signed distance to a closed surface (> 0 outside) and the outward direction at the points.

    The surface must have outward triangles (positive signed_volume)."""

    def __init__(self, points, tris):
        self.f = vtkImplicitPolyDataDistance()
        self.f.SetInput(polydata(points, tris))

    def distance(self, X):
        return np.array([self.f.EvaluateFunction(x) for x in np.asarray(X, float)])

    def __call__(self, X):
        phi, n, g = np.empty(len(X)), np.empty((len(X), 3)), [0.0, 0.0, 0.0]
        for i, x in enumerate(np.asarray(X, float)):
            phi[i] = self.f.EvaluateFunction(x)
            self.f.EvaluateGradient(x, g)
            n[i] = g
        return phi, n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def assd(PA, PB, tris):
    """Symmetric mean point-to-surface distance between two surfaces with the same triangles."""
    A, B = polydata(PA, tris), polydata(PB, tris)
    _, cb = B.find_closest_cell(A.points, return_closest_point=True)
    _, ca = A.find_closest_cell(B.points, return_closest_point=True)
    return float(np.r_[np.linalg.norm(A.points - cb, axis=1), np.linalg.norm(B.points - ca, axis=1)].mean())
