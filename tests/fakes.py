"""numpy stand-ins for the VTK-based helpers of eurelaps_lung/fem/geometry.py and for the surface
writer, so that the collapse and fit steps run on the Mac. Import after stubs, before the steps."""

import numpy as np

import stubs  # noqa: F401
import sdf_numpy
from eurelaps_lung import data_io
from eurelaps_lung.fem import geometry as G

written = {}                                                    # path name -> FakeSurface


class FakeSurface:
    def __init__(self, points, tris):
        self.points, self.tris = np.asarray(points, float), np.asarray(tris)
        self.point_data, self.cell_data, self.field_data = {}, {}, {}

    @property
    def volume(self):
        return abs(G.signed_volume(self.points, self.tris))


class WallDistance(G.WallDistance):
    def __init__(self, points, tris):
        self.P, self.T = np.asarray(points, float), np.asarray(tris, np.int64)
        self.N = G.vertex_normals(self.P, self.T)

    def distance(self, X):
        return sdf_numpy.signed_distance(self.P, self.T, np.asarray(X, float))

    def closest(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        cell, cp = sdf_numpy.closest_points(self.P, self.T, X)
        return (cp, *self._contact(X, cell, cp))


def closest_on_surface(P, tris, X):
    cell, cp = sdf_numpy.closest_points(P, tris, np.atleast_2d(np.asarray(X, float)))
    w = G.barycentric(P, tris[cell], cp)
    return cell, w, np.einsum("ik,ikj->ij", w, P[tris[cell]])


def assd(PA, PB, tris, tris_a=None):
    ta = tris if tris_a is None else tris_a
    _, cb = sdf_numpy.closest_points(PB, tris, PA)
    _, ca = sdf_numpy.closest_points(PA, ta, PB)
    return float(np.r_[np.linalg.norm(PA - cb, axis=1), np.linalg.norm(PB - ca, axis=1)].mean())


def write_surface(mesh, path, space):
    written[str(path)] = mesh


G.WallDistance, G.closest_on_surface, G.assd, G.polydata = WallDistance, closest_on_surface, assd, FakeSurface
data_io.write_surface = write_surface
