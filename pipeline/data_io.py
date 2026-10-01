"""Surface and image I/O with explicit coordinate spaces.

Every surface inside the pipeline is an all-triangle pv.PolyData in a known space
(LPS or RAS; they differ by the sign of x and y). The space is stored in the
"SPACE" field-data array and, for legacy .vtk files, also in the header line the
way 3D Slicer writes it ("... SPACE=LPS").

Images are itk images; ITK physical coordinates are always LPS.
"""

import json
import logging
from pathlib import Path

import itk
import numpy as np
import pyvista as pv
from vtkmodules.vtkFiltersCore import vtkTriangleFilter
from vtkmodules.vtkIOLegacy import vtkPolyDataWriter
from vtkmodules.vtkIOXML import vtkXMLPolyDataWriter

log = logging.getLogger(__name__)

SPACES = ("LPS", "RAS")
DEFAULT_SPACE = "LPS"   # Slicer >= 4.11 writes models in LPS


def as_triangles(mesh):
    """Surface of `mesh` as PolyData with triangle cells only (no verts / lines)."""
    if not isinstance(mesh, pv.PolyData):
        mesh = mesh.extract_surface()
    f = vtkTriangleFilter()
    f.SetInputData(mesh)
    f.PassVertsOff()
    f.PassLinesOff()
    f.Update()
    tri = pv.wrap(f.GetOutput())
    if not tri.is_all_triangles:
        raise ValueError("triangulation left non-triangle cells")
    return tri


def detect_space(mesh, path):
    """'LPS' / 'RAS' from the SPACE field data or the legacy header, else None."""
    if "SPACE" in mesh.field_data:
        space = str(np.asarray(mesh.field_data["SPACE"]).ravel()[0]).upper()
        if space in SPACES:
            return space
    if Path(path).suffix.lower() == ".vtk":
        with open(path, "rb") as f:
            head = f.read(512).decode("latin-1", errors="ignore").upper()
        for space in SPACES:
            if "SPACE=" + space in head:
                return space
    return None


def convert_space(mesh, src, dst):
    """Copy of `mesh` moved from space `src` to `dst` (LPS <-> RAS = negate x, y)."""
    out = mesh.copy()
    if src != dst:
        pts = np.array(out.points, dtype=float)
        pts[:, :2] *= -1.0
        out.points = pts
    out.field_data["SPACE"] = np.array([dst])
    return out


def read_surface(path, space=DEFAULT_SPACE, input_space="auto"):
    """Read a surface (.vtk/.vtp/.stl/...) as triangles, returned in `space`.

    input_space="auto" trusts the file tag and falls back to LPS when there is none.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    mesh = pv.read(path)
    src = input_space
    if src == "auto":
        src = detect_space(mesh, path)
        if src is None:
            log.warning("%s has no SPACE tag, assuming %s", path.name, DEFAULT_SPACE)
            src = DEFAULT_SPACE
    surf = as_triangles(mesh)
    log.info("read %s: %d pts, %d tris, %s -> %s",
             path.name, surf.n_points, surf.n_cells, src, space)
    return convert_space(surf, src, space)


def write_surface(mesh, path, space):
    """Write `mesh` (already in `space`) tagged with its space. .vtp or .vtk keep the tag."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh = mesh.copy()
    mesh.field_data["SPACE"] = np.array([space])
    suffix = path.suffix.lower()
    if suffix == ".vtp":
        w = vtkXMLPolyDataWriter()
    elif suffix == ".vtk":
        w = vtkPolyDataWriter()
        w.SetHeader("lung_collapse pipeline. SPACE=%s" % space)
        w.SetFileTypeToBinary()
    else:
        log.warning("%s: format cannot store the coordinate space (%s)", path.name, space)
        mesh.save(path)
        return path
    w.SetFileName(str(path))
    w.SetInputData(mesh)
    if not w.Write():
        raise OSError("could not write %s" % path)
    return path


def read_mask(path, label=None):
    """Binary float32 mask (1 inside) from a labelmap: voxels == label, or > 0 if label is None.

    Expects a 3D labelmap (Slicer: segmentation -> Export visible segments to binary labelmap).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    img = itk.imread(str(path))
    arr = itk.array_view_from_image(img)
    if arr.ndim != 3:
        raise ValueError("%s: expected a 3D labelmap, got %dD (export a binary labelmap from Slicer)"
                         % (path.name, arr.ndim))
    mask = (arr == label) if label is not None else (arr > 0)
    if not mask.any():
        raise ValueError("%s: mask is empty (label=%s)" % (path.name, label))
    out = itk.image_from_array(mask.astype(np.float32))
    out.CopyInformation(img)
    log.info("read %s: %s voxels, spacing %s mm, mask %.1f mL", path.name,
             "x".join(map(str, arr.shape[::-1])), np.round(tuple(img.GetSpacing()), 3),
             mask.sum() * np.prod(tuple(img.GetSpacing())) / 1000.0)
    return out


MARKUPS_SCHEMA = ("https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/"
                  "Markups/Resources/Schema/markups-schema-v1.0.3.json#")


def write_fiducials(points, path, descriptions=None):
    """Slicer point list (.mrk.json) from {label: LPS position}, optional {label: description}."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptions = descriptions or {}
    cps = [dict(id=str(i + 1), label=label, description=descriptions.get(label, ""),
                position=[float(v) for v in pos], locked=True, visibility=True,
                positionStatus="defined")
           for i, (label, pos) in enumerate(points.items())]
    doc = {"@schema": MARKUPS_SCHEMA,
           "markups": [dict(type="Fiducial", coordinateSystem="LPS", coordinateUnits="mm",
                            locked=True, controlPoints=cps)]}
    path.write_text(json.dumps(doc, indent=2))
    return path


def read_fiducials(path):
    """{label: LPS position} from a Slicer .mrk.json (first markup)."""
    mk = json.loads(Path(path).read_text())["markups"][0]
    flip = np.array([-1.0, -1.0, 1.0]) if mk.get("coordinateSystem", "LPS").upper() == "RAS" else 1.0
    return {cp["label"]: np.asarray(cp["position"], dtype=float) * flip for cp in mk["controlPoints"]}


def write_image(img, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    itk.imwrite(img, str(path), compression=True)
    return path
