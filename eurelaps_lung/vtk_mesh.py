"""Surfaces as numpy arrays, and the VTK filters the steps use. Plain VTK, no pyvista: 3D Slicer
ships VTK (vtkmodules) but not pyvista, and this package must run inside Slicer.

Surface: points (N, 3) as read (a legacy .vtk file gives float32 points, kept as they are),
triangles `tris` (T, 3) or line segments `lines` (L, 2), and dicts of point / cell / field data
(numpy arrays; strings for field data). Surface.to_vtk() / from_vtk() convert to and from
vtkPolyData.

The filters set VTK exactly as the pyvista methods used before (pyvista 0.49): clean,
fill_holes, smooth_taubin, extract_largest, n_open_edges, volume, find_closest_cell,
compute_implicit_distance, read / save. Same VTK in, same numbers out.

pyacvd (surface_mesh.py) still needs pyvista: that is the only place it is imported.
"""

from pathlib import Path

import numpy as np
from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy
from vtkmodules.vtkCommonCore import reference, vtkDoubleArray, vtkPoints, vtkStringArray
from vtkmodules.vtkCommonDataModel import (VTK_TETRA, vtkCellArray, vtkGenericCell, vtkImageData,
                                           vtkPolyData, vtkStaticCellLocator, vtkUnstructuredGrid)
from vtkmodules.vtkFiltersCore import (vtkCleanPolyData, vtkConnectivityFilter, vtkFeatureEdges,
                                       vtkImplicitPolyDataDistance, vtkMassProperties, vtkTriangleFilter,
                                       vtkWindowedSincPolyDataFilter)
from vtkmodules.vtkFiltersGeometry import vtkDataSetSurfaceFilter
from vtkmodules.vtkFiltersModeling import vtkFillHolesFilter
from vtkmodules.vtkIOGeometry import vtkOBJReader, vtkSTLReader, vtkSTLWriter
from vtkmodules.vtkIOLegacy import vtkDataSetReader, vtkPolyDataWriter, vtkUnstructuredGridWriter
from vtkmodules.vtkIOPLY import vtkPLYReader
from vtkmodules.vtkIOXML import vtkXMLPolyDataReader, vtkXMLPolyDataWriter, vtkXMLUnstructuredGridWriter

ID = np.int64


class Surface:
    """Points with triangles or line segments (not both), and their data arrays."""

    def __init__(self, points, tris=None, lines=None):
        self.points = np.asarray(points)
        self.tris = np.zeros((0, 3), ID) if tris is None else np.asarray(tris, ID).reshape(-1, 3)
        self.lines = np.zeros((0, 2), ID) if lines is None else np.asarray(lines, ID).reshape(-1, 2)
        if len(self.tris) and len(self.lines):
            raise ValueError("a Surface has triangles or lines, not both")
        self.point_data, self.cell_data, self.field_data = {}, {}, {}

    @property
    def n_points(self):
        return len(self.points)

    @property
    def n_cells(self):
        return len(self.tris) + len(self.lines)

    @property
    def bounds(self):
        """(xmin, xmax, ymin, ymax, zmin, zmax) over all points, as VTK."""
        lo, hi = self.points.min(0), self.points.max(0)
        return tuple(float(v) for pair in zip(lo, hi) for v in pair)

    @property
    def volume(self):
        return volume(self)

    def copy(self):
        s = Surface(self.points.copy(), self.tris.copy(), self.lines.copy())
        for mine, theirs in ((s.point_data, self.point_data), (s.cell_data, self.cell_data),
                             (s.field_data, self.field_data)):
            mine.update({k: np.array(v, copy=True) for k, v in theirs.items()})
        return s

    def to_vtk(self):
        pd = vtkPolyData()
        pts = vtkPoints()
        pts.SetData(numpy_to_vtk(np.ascontiguousarray(self.points), deep=1))
        pd.SetPoints(pts)
        if len(self.tris):
            pd.SetPolys(_cell_array(self.tris))
        if len(self.lines):
            pd.SetLines(_cell_array(self.lines))
        for name, a in self.point_data.items():
            pd.GetPointData().AddArray(_vtk_array(name, a))
        for name, a in self.cell_data.items():
            pd.GetCellData().AddArray(_vtk_array(name, a))
        for name, a in self.field_data.items():
            pd.GetFieldData().AddArray(_vtk_array(name, a))
        return pd

    @classmethod
    def from_vtk(cls, pd, kind="tris"):
        """Surface of the triangles (kind "tris") or the 2-point lines ("lines") of a vtkPolyData.
        Triangles with other cells around (verts, lines) lose their point / cell data, as the
        pyvista code did."""
        P = vtk_to_numpy(pd.GetPoints().GetData()).copy() if pd.GetPoints() else np.zeros((0, 3))
        nv, nl, npoly = pd.GetNumberOfVerts(), pd.GetNumberOfLines(), pd.GetNumberOfPolys()
        if kind == "tris":
            s, first, n, others = cls(P, tris=_cells(pd.GetPolys(), 3, "triangle")), nv + nl, npoly, nv + nl
        elif kind == "lines":
            s, first, n, others = cls(P, lines=_cells(pd.GetLines(), 2, "2-point line")), nv, nl, nv + npoly
        else:
            raise ValueError("kind: tris or lines")
        others += pd.GetNumberOfStrips()
        s.field_data.update(field_data(pd))
        if kind == "tris" and others:
            return s
        for src, dst, sl in ((pd.GetPointData(), s.point_data, slice(None)),
                             (pd.GetCellData(), s.cell_data, slice(first, first + n))):
            for i in range(src.GetNumberOfArrays()):
                a = src.GetAbstractArray(i)
                if a is not None and a.GetName():
                    dst[a.GetName()] = _numpy_array(a)[sl]
        return s


def field_data(pd):
    """{name: numpy array} of the field data of a VTK data object."""
    fd, out = pd.GetFieldData(), {}
    for i in range(fd.GetNumberOfArrays()):
        a = fd.GetAbstractArray(i)
        if a is not None and a.GetName():
            out[a.GetName()] = _numpy_array(a)
    return out


def _cell_array(conn):
    k = conn.shape[1]
    ca = vtkCellArray()
    ca.SetData(numpy_to_vtkIdTypeArray(np.arange(0, (len(conn) + 1) * k, k, dtype=ID), deep=1),
               numpy_to_vtkIdTypeArray(np.ascontiguousarray(conn, dtype=ID).ravel(), deep=1))
    return ca


def _cells(ca, k, what):
    if ca is None or ca.GetNumberOfCells() == 0:
        return np.zeros((0, k), ID)
    off = vtk_to_numpy(ca.GetOffsetsArray())
    if np.any(np.diff(off) != k):
        raise ValueError("cells other than %ss" % what)
    return vtk_to_numpy(ca.GetConnectivityArray()).astype(ID).reshape(-1, k)


def _vtk_array(name, a):
    a = np.asarray(a)
    if a.dtype.kind in "US":
        v = vtkStringArray()
        for x in a.ravel():
            v.InsertNextValue(str(x))
    else:
        if a.dtype == bool:
            a = a.astype(np.uint8)
        v = numpy_to_vtk(np.ascontiguousarray(a), deep=1)
    v.SetName(name)
    return v


def _numpy_array(a):
    if isinstance(a, vtkStringArray):
        return np.array([a.GetValue(i) for i in range(a.GetNumberOfValues())])
    return vtk_to_numpy(a).copy()


def _poly(s):
    return s.to_vtk() if isinstance(s, Surface) else s


def _run(alg, s):
    alg.SetInputData(_poly(s))
    alg.Update()
    return alg.GetOutput()


# ------------------------------------------------------------------------------------------------
# filters (input: Surface or vtkPolyData; output: Surface)
# ------------------------------------------------------------------------------------------------
def triangles(s):
    """Triangles only (vtkTriangleFilter: polygons split, verts and lines dropped)."""
    f = vtkTriangleFilter()
    f.PassVertsOff()
    f.PassLinesOff()
    return Surface.from_vtk(_run(f, s), "tris")


def clean(s, tolerance=None, kind="tris"):
    """vtkCleanPolyData (merge points, drop degenerate cells; pyvista's defaults); tolerance [mm]
    absolute. kind "tris": the result as triangles (degenerate triangles become lines: dropped)."""
    f = vtkCleanPolyData()
    f.SetPointMerging(True)
    f.SetConvertLinesToPoints(True)
    f.SetConvertPolysToLines(True)
    f.SetConvertStripsToPolys(True)
    if tolerance is not None:
        f.ToleranceIsAbsoluteOn()
        f.SetAbsoluteTolerance(tolerance)
    out = _run(f, s)
    if out.GetNumberOfPoints() < 1 and _poly(s).GetNumberOfCells() > 0:
        raise ValueError("clean tolerance is too high: empty mesh")
    return triangles(out) if kind == "tris" else Surface.from_vtk(out, kind)


def fill_holes(s, hole_size):
    """Holes up to hole_size (radius of the bounding circle) [mm] closed, as triangles."""
    f = vtkFillHolesFilter()
    f.SetHoleSize(hole_size)
    return triangles(_run(f, s))


def smooth_taubin(s, n_iter=20, pass_band=0.1):
    """Windowed-sinc (Taubin) smoothing with pyvista's defaults (edge 15, feature 45 deg, boundary
    smoothing on, Nuttall window)."""
    f = vtkWindowedSincPolyDataFilter()
    f.SetNumberOfIterations(n_iter)
    f.SetFeatureEdgeSmoothing(False)
    f.SetNonManifoldSmoothing(False)
    f.SetFeatureAngle(45.0)
    f.SetEdgeAngle(15.0)
    f.SetBoundarySmoothing(True)
    f.SetPassBand(pass_band)
    f.SetNormalizeCoordinates(False)
    if hasattr(f, "SetWindowFunctionToNuttall"):                 # VTK >= 9.4
        f.SetWindowFunctionToNuttall()
    return Surface.from_vtk(_run(f, s), "tris")


def _connectivity(s, largest):
    f = vtkConnectivityFilter()
    f.ColorRegionsOn()
    f.SetRegionIdAssignmentMode(f.CELL_COUNT_DESCENDING)
    if largest:
        f.SetExtractionModeToLargestRegion()
    else:
        f.SetExtractionModeToAllRegions()
    out = _run(f, s)
    return f, out


def largest_component(s):
    """The largest connected piece (by cells). As pyvista: the largest-region pass leaves the
    unused points, a second all-regions pass removes them."""
    _, out = _connectivity(s, True)
    _, out = _connectivity(out, False)
    r = Surface.from_vtk(out, "tris")
    r.point_data.pop("RegionId", None)
    r.cell_data.pop("RegionId", None)
    return r


def n_components(s):
    f, _ = _connectivity(s, False)
    return int(f.GetNumberOfExtractedRegions())


def bad_edges(s):
    """Number of boundary + non-manifold edges (0 = watertight manifold)."""
    f = vtkFeatureEdges()
    f.FeatureEdgesOff()
    f.ManifoldEdgesOff()
    f.BoundaryEdgesOn()
    f.NonManifoldEdgesOn()
    f.SetColoring(False)
    return int(_run(f, s).GetNumberOfCells())


def boundary_edges(s):
    """Number of boundary (open) edges only."""
    f = vtkFeatureEdges()
    f.FeatureEdgesOff()
    f.ManifoldEdgesOff()
    f.NonManifoldEdgesOff()
    f.BoundaryEdgesOn()
    f.SetColoring(False)
    return int(_run(f, s).GetNumberOfCells())


def volume(s):
    """Enclosed volume (vtkMassProperties; > 0 whatever the orientation)."""
    if _poly(s).GetNumberOfPoints() == 0 or _poly(s).GetNumberOfCells() == 0:
        return 0.0
    f = vtkMassProperties()
    f.SetInputData(_poly(s))
    f.Update()
    return float(f.GetVolume())


class CellLocator:
    """Closest triangle and closest point on a surface (vtkStaticCellLocator, as pyvista's
    find_closest_cell)."""

    def __init__(self, s):
        self.pd = _poly(s)
        self.loc = vtkStaticCellLocator()
        self.loc.SetDataSet(self.pd)
        self.loc.BuildLocator()
        self.cell = vtkGenericCell()

    def closest(self, X):
        """(cell index (M,), closest point (M, 3)) of each X (M, 3)."""
        X = np.atleast_2d(np.asarray(X, float))
        cells, pts = np.zeros(len(X), ID), np.zeros((len(X), 3))
        cp, cid, sub, d2 = [0.0, 0.0, 0.0], reference(0), reference(0), reference(0.0)
        for i, x in enumerate(X):
            self.loc.FindClosestPoint(x, cp, self.cell, cid, sub, d2)
            cells[i], pts[i] = int(cid), cp
        return cells, pts


def implicit_distance(s, X):
    """Signed distance of the points X (M, 3) to the closed surface s (vtkImplicitPolyDataDistance)."""
    f = vtkImplicitPolyDataDistance()
    f.SetInput(_poly(s))
    d = vtkDoubleArray()
    f.FunctionValue(numpy_to_vtk(np.ascontiguousarray(X, dtype=float), deep=1), d)
    return vtk_to_numpy(d).copy()


def grid_points(dims, spacing, origin):
    """Points of a regular grid in VTK order (x fastest), as pyvista's ImageData.points: VTK's own
    points (VTK >= 9.4), else the same cumulative sums pyvista used."""
    img = vtkImageData()
    img.SetDimensions(*(int(d) for d in dims))
    img.SetSpacing(*(float(h) for h in spacing))
    img.SetOrigin(*(float(o) for o in origin))
    if hasattr(img, "GetPoints"):
        return vtk_to_numpy(img.GetPoints().GetData()).copy()
    ax = [np.insert(np.cumsum(np.full(int(n) - 1, h)), 0, 0.0) + o for n, h, o in zip(dims, spacing, origin)]
    xx, yy, zz = np.meshgrid(*ax, indexing="ij")
    return np.c_[xx.ravel(order="F"), yy.ravel(order="F"), zz.ravel(order="F")]


# ------------------------------------------------------------------------------------------------
# files
# ------------------------------------------------------------------------------------------------
def read_polydata(path):
    """vtkPolyData of a .vtk / .vtp / .stl / .ply / .obj file (readers set as pyvista.read)."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".vtk":
        r = vtkDataSetReader()
        for m in ("ReadAllScalarsOn", "ReadAllColorScalarsOn", "ReadAllNormalsOn", "ReadAllTCoordsOn",
                  "ReadAllVectorsOn", "ReadAllFieldsOn", "ReadAllTensorsOn"):
            getattr(r, m)()
    else:
        readers = {".vtp": vtkXMLPolyDataReader, ".stl": vtkSTLReader, ".ply": vtkPLYReader, ".obj": vtkOBJReader}
        if suffix not in readers:
            raise ValueError("%s: unsupported surface format (%s)" % (path.name, ", ".join([".vtk", *readers])))
        r = readers[suffix]()
    r.SetFileName(str(path))
    r.Update()
    out = r.GetOutput()
    if out is None:
        raise OSError("could not read %s" % path)
    if not isinstance(out, vtkPolyData):                         # e.g. an unstructured grid: its surface
        f = vtkDataSetSurfaceFilter()
        f.SetInputData(out)
        f.Update()
        out = f.GetOutput()
    return out


def write_polydata(s, path, header=None):
    """.vtp (XML), .vtk (legacy binary, `header` as its title line) or .stl."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".vtp":
        w = vtkXMLPolyDataWriter()
    elif suffix == ".vtk":
        w = vtkPolyDataWriter()
        if header:
            w.SetHeader(header)
        w.SetFileTypeToBinary()
    elif suffix == ".stl":
        w = vtkSTLWriter()
        w.SetFileTypeToBinary()
    else:
        raise ValueError("%s: write .vtp, .vtk or .stl" % path.name)
    w.SetFileName(str(path))
    w.SetInputData(_poly(s))
    if not w.Write():
        raise OSError("could not write %s" % path)
    return path


def tet_grid(nodes, tets, point_data=None, field_data=None, cell_data=None):
    """vtkUnstructuredGrid of linear tetrahedra."""
    g = vtkUnstructuredGrid()
    pts = vtkPoints()
    pts.SetData(numpy_to_vtk(np.ascontiguousarray(nodes, dtype=float), deep=1))
    g.SetPoints(pts)
    g.SetCells(VTK_TETRA, _cell_array(np.asarray(tets, ID)))
    for name, a in (point_data or {}).items():
        g.GetPointData().AddArray(_vtk_array(name, a))
    for name, a in (field_data or {}).items():
        g.GetFieldData().AddArray(_vtk_array(name, a))
    for name, a in (cell_data or {}).items():
        g.GetCellData().AddArray(_vtk_array(name, a))
    return g


def write_grid(g, path, binary=True):
    """.vtu (XML) or .vtk (legacy; ASCII if not binary)."""
    path = Path(path)
    if path.suffix.lower() == ".vtu":
        w = vtkXMLUnstructuredGridWriter()
        if binary:
            w.SetDataModeToBinary()
        else:
            w.SetDataModeToAscii()
    elif path.suffix.lower() == ".vtk":
        w = vtkUnstructuredGridWriter()
        if binary:
            w.SetFileTypeToBinary()
        else:
            w.SetFileTypeToASCII()
    else:
        raise ValueError("%s: write .vtu or .vtk" % path.name)
    w.SetFileName(str(path))
    w.SetInputData(g)
    if not w.Write():
        raise OSError("could not write %s" % path)
    return path
