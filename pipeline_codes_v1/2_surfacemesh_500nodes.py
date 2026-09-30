import math
import numpy as np
import pyvista as pv

try:
    import pyacvd
except ImportError:
    raise SystemExit("pyacvd not installed. Run:  pip install pyacvd")

try:
    from vtkmodules.vtkFiltersCore import vtkTriangleFilter
except ImportError:
    from vtk import vtkTriangleFilter

# ---- I/O -------------------------------------------------------------------
# \\wsl.localhost\Ubuntu\home\pr502\3DSlicer_output\left_lung_segmentedmodel.vtk
INPUT_VTK  = "/home/pr502/3DSlicer_output/segmented_patient1_models/left_lung.vtk"
OUTPUT_VTP = "/home/pr502/patient1_mesh/model_left_lung_mesh.vtp"

# ---- knobs -----------------------------------------------------------------
TAUBIN_ITERS       = 15     # pre-smooth to remove Slicer staircase before remeshing
MAX_FACES          = 1000    # HARD ceiling on triangle count
TARGET_CLUSTERS    = 480    # ~ output node count. Closed surface -> faces ~= 2*nodes
                            # 290 -> ~576 tris. Auto-reduced if it overshoots 600.
MIN_SRC_RATIO      = 12     # want >= this many source points per cluster for uniformity
SUBDIVIDE_CAP      = 4      # safety cap on subdivision passes
CONVERT_RAS_TO_LPS = False  # Slicer models are RAS. True negates x,y for LPS/GetFEM.
POST_TAUBIN        = 5      # gentle smoothing after remesh to relax cluster boundaries
# ----------------------------------------------------------------------------


def _all_tris(m):
    flag = m.is_all_triangles
    return flag() if callable(flag) else bool(flag)


def as_clean_triangles(mesh):
    """GUARANTEED all-triangle PolyData."""
    if not isinstance(mesh, pv.PolyData):
        mesh = mesh.extract_surface(algorithm="dataset_surface")
    f = vtkTriangleFilter()
    f.SetInputData(mesh)
    f.PassVertsOff()
    f.PassLinesOff()
    f.Update()
    tri = pv.wrap(f.GetOutput())
    if _all_tris(tri):
        return tri
    faces = np.asarray(tri.faces)
    keep, i = [], 0
    while i < faces.size:
        n = faces[i]
        if n == 3:
            keep.append(faces[i + 1:i + 4])
        i += n + 1
    if not keep:
        return tri
    keep = np.asarray(keep, dtype=np.int64)
    nf = np.empty((len(keep), 4), dtype=np.int64)
    nf[:, 0] = 3
    nf[:, 1:] = keep
    return pv.PolyData(tri.points, nf.ravel())


def load_and_clean(path):
    """Read the Slicer model surface -> clean, watertight, all-triangle mesh."""
    mesh = pv.read(path)
    surf = as_clean_triangles(mesh)
    surf = surf.clean(tolerance=1e-5)          # weld duplicate / bowtie points
    surf = as_clean_triangles(surf)
    surf = surf.extract_largest()              # drop stray speckle islands
    surf = as_clean_triangles(surf)
    if TAUBIN_ITERS > 0:
        surf = surf.smooth_taubin(n_iter=TAUBIN_ITERS, pass_band=0.1)
    surf = as_clean_triangles(surf)
    if surf.n_open_edges > 0:
        surf = surf.fill_holes(hole_size=1e4)
        surf = as_clean_triangles(surf)
    return surf


def uniform_remesh(surf, n_clusters):
    """ACVD isotropic remesh to ~n_clusters evenly sized triangles."""
    # Make sure the source is dense enough that clustering is well-posed.
    ratio = MIN_SRC_RATIO * n_clusters / max(surf.n_points, 1)
    nsub = 0 if ratio <= 1 else min(SUBDIVIDE_CAP, math.ceil(math.log(ratio, 4)))
    clus = pyacvd.Clustering(surf)
    if nsub > 0:
        clus.subdivide(nsub)
    clus.cluster(n_clusters)
    remesh = clus.create_mesh()
    return as_clean_triangles(remesh)


def quality_str(poly):
    parts = []
    try:
        q = poly.cell_quality(["aspect_ratio", "min_angle"])
        if "aspect_ratio" in q.cell_data:
            ar = np.asarray(q.cell_data["aspect_ratio"])
            parts.append(f"AR mean/max={np.nanmean(ar):.2f}/{np.nanmax(ar):.1f}")
        if "min_angle" in q.cell_data:
            ma = np.asarray(q.cell_data["min_angle"])
            parts.append(f"min-angle min/mean={np.nanmin(ma):.1f}/{np.nanmean(ma):.1f} deg")
    except Exception:
        pass
    return "   ".join(parts) if parts else "(quality metrics unavailable)"


# ---- run -------------------------------------------------------------------
surf = load_and_clean(INPUT_VTK)
print(f"Loaded surface: {surf.n_cells:,} tris, {surf.n_points:,} nodes, "
      f"open edges = {surf.n_open_edges}")

# Remesh, shrinking cluster count if we overshoot the face ceiling.
n_clusters = TARGET_CLUSTERS
for _ in range(6):
    poly = uniform_remesh(surf, n_clusters)
    print(f"  clusters={n_clusters} -> {poly.n_cells:,} tris, {poly.n_points:,} nodes")
    if poly.n_cells < MAX_FACES:
        break
    n_clusters = int(n_clusters * 0.9)

if poly.n_cells >= MAX_FACES:
    print(f"Still >= {MAX_FACES}; lower TARGET_CLUSTERS.")

# Gentle post-smoothing to relax any faceting at cluster boundaries.
if POST_TAUBIN > 0:
    poly = poly.smooth_taubin(n_iter=POST_TAUBIN, pass_band=0.1)
    poly = as_clean_triangles(poly)

if poly.n_open_edges > 0:
    poly = poly.fill_holes(hole_size=1e4)
    poly = as_clean_triangles(poly)

if CONVERT_RAS_TO_LPS:
    pts = poly.points.copy()
    pts[:, 0] *= -1.0   # R -> L
    pts[:, 1] *= -1.0   # A -> P
    poly.points = pts

poly.save(OUTPUT_VTP)
print(f"\nFinal surface: {poly.n_cells:,} tris, {poly.n_points:,} nodes   "
      f"open edges = {poly.n_open_edges}   {quality_str(poly)}")
print(f"Saved -> {OUTPUT_VTP}")