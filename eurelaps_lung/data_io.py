"""Surface and image I/O with explicit coordinate spaces.

Every surface inside the pipeline is an all-triangle vtk_mesh.Surface in a known space
(LPS or RAS; they differ by the sign of x and y). The space is stored in the
"SPACE" field-data array and, for legacy .vtk files, also in the header line the
way 3D Slicer writes it ("... SPACE=LPS").

Images are itk images; ITK physical coordinates are always LPS. itk is imported only by the
image functions (the surface steps and the prediction do not need it).
"""

import json
import logging
from pathlib import Path

import numpy as np

from . import vtk_mesh as vm

log = logging.getLogger(__name__)

SPACES = ("LPS", "RAS")
DEFAULT_SPACE = "LPS"   # Slicer >= 4.11 writes models in LPS


def as_triangles(mesh):
    """Surface of `mesh` (vtkPolyData or Surface) with triangle cells only (no verts / lines)."""
    return vm.triangles(mesh)


def detect_space(mesh, path):
    """'LPS' / 'RAS' from the SPACE field data (Surface or vtkPolyData) or the legacy header, else None."""
    fields = mesh.field_data if isinstance(mesh, vm.Surface) else vm.field_data(mesh)
    if "SPACE" in fields:
        space = str(np.asarray(fields["SPACE"]).ravel()[0]).upper()
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
    raw = vm.read_polydata(path)
    src = input_space
    if src == "auto":
        src = detect_space(raw, path)
        if src is None:
            log.warning("%s has no SPACE tag, assuming %s", path.name, DEFAULT_SPACE)
            src = DEFAULT_SPACE
    surf = as_triangles(raw)
    log.info("read %s: %d pts, %d tris, %s -> %s",
             path.name, surf.n_points, surf.n_cells, src, space)
    return convert_space(surf, src, space)


def write_surface(mesh, path, space):
    """Write `mesh` (a Surface, already in `space`) tagged with its space. .vtp or .vtk keep the tag."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh = mesh.copy()
    mesh.field_data["SPACE"] = np.array([space])
    if path.suffix.lower() not in (".vtp", ".vtk"):
        log.warning("%s: format cannot store the coordinate space (%s)", path.name, space)
    return vm.write_polydata(mesh, path, header="eurelaps_lung. SPACE=%s" % space)


def read_mask(path, label=None):
    """Binary float32 mask (1 inside) from a labelmap: voxels == label, or > 0 if label is None.

    Expects a 3D labelmap (Slicer: segmentation -> Export visible segments to binary labelmap).
    """
    import itk
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


def fiducial_list(name, points, descriptions=None, sphere_mm=None, color=None):
    """One Slicer point list from {label: LPS position}, optional {label: description}.

    Slicer draws every point of a list with the same glyph, so per-point sizes need one list
    per point: sphere_mm draws the points as 3D spheres of that absolute size (diameter) [mm];
    color is RGB in 0..1."""
    descriptions = descriptions or {}
    cps = [dict(id=str(i + 1), label=label, description=descriptions.get(label, ""),
                position=[float(v) for v in pos], locked=True, visibility=True,
                positionStatus="defined")
           for i, (label, pos) in enumerate(points.items())]
    markup = dict(type="Fiducial", name=name, coordinateSystem="LPS", coordinateUnits="mm",
                  locked=True, controlPoints=cps)
    display = {}
    if sphere_mm is not None:
        display.update(glyphType="Sphere3D", useGlyphScale=False, glyphSize=float(sphere_mm), opacity=0.6)
    if color is not None:
        display.update(color=list(color), selectedColor=list(color))
    if display:
        markup["display"] = display
    return markup


def write_markups(markups, path):
    """.mrk.json with several lists; Slicer loads each one as its own node."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"@schema": MARKUPS_SCHEMA, "markups": list(markups)}, indent=2))
    return path


def read_fiducials(path):
    """{label: LPS position} over all point lists of a Slicer .mrk.json."""
    out = {}
    for mk in json.loads(Path(path).read_text())["markups"]:
        flip = np.array([-1.0, -1.0, 1.0]) if mk.get("coordinateSystem", "LPS").upper() == "RAS" else 1.0
        for cp in mk.get("controlPoints", []):
            out[cp["label"]] = np.asarray(cp["position"], dtype=float) * flip
    return out


def read_fiducial_radii(path):
    """{label: radius [mm]} for the points of lists drawn as absolute-size spheres (fiducial_list
    with sphere_mm), i.e. the sphere visible in Slicer; other points are left out."""
    out = {}
    for mk in json.loads(Path(path).read_text())["markups"]:
        disp = mk.get("display", {})
        if disp.get("glyphType") == "Sphere3D" and disp.get("useGlyphScale") is False:
            for cp in mk.get("controlPoints", []):
                out[cp["label"]] = float(disp["glyphSize"]) / 2.0
    return out


def write_image(img, path):
    import itk
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    itk.imwrite(img, str(path), compression=True)
    return path
