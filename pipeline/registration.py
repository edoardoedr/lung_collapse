"""Step 3 - registration (replaces the 3D Slicer "General Registration (Elastix)" workflow).

Collapsed surface mesh (step 2) -> inflated-space surface with the SAME nodes and faces.

  1. masks: inflated lung; collapsed lung, or (default) the step-2 surface, so the collapsed
     image boundary is exactly the mesh that gets warped. Labelmaps are cropped to the
     mask + margin, surfaces are rasterised; both cast to float.
  2. elastix (default: Slicer's rigid + bspline preset). elastix returns T: fixed -> moving
     (the resampling direction), so how the nodes are mapped depends on `direction`:
       "inverse": fixed = inflated, moving = collapsed, as in the Slicer workflow. The nodes
                  need T^-1 (what Slicer does when hardening a transform on a model): each node
                  y is mapped to the x with T(x) = y. Ill-conditioned when T compresses strongly
                  (large collapse), and fails where T folds.
       "forward": fixed = collapsed, moving = inflated. T maps collapsed -> inflated and is
                  applied to the nodes directly; no inversion.
  3. QA: displacement, flipped triangles, volume, Dice vs the inflated mask.
"""

import logging

import itk
import numpy as np
from vtkmodules.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkImagingStencil import vtkImageStencilToImage, vtkPolyDataToImageStencil

from .data_io import convert_space, read_mask, read_surface, write_image, write_surface

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------------------------------
# images
# ----------------------------------------------------------------------------------------------
def geometry(img):
    """origin, spacing, direction (3x3) and size (x, y, z) of an itk image."""
    return (np.array(tuple(img.GetOrigin())), np.array(tuple(img.GetSpacing())),
            itk.array_from_matrix(img.GetDirection()),
            np.array(tuple(img.GetLargestPossibleRegion().GetSize())))


def make_image(arr, origin, spacing, direction):
    img = itk.image_from_array(np.ascontiguousarray(arr, dtype=np.float32))
    img.SetOrigin([float(v) for v in origin])
    img.SetSpacing([float(v) for v in spacing])
    img.SetDirection(itk.matrix_from_array(np.asarray(direction, dtype=float)))
    return img


def crop_to_mask(img, margin_mm):
    """Crop to the bounding box of the mask plus margin, keeping physical positions."""
    arr = itk.array_from_image(img)                       # z, y, x
    origin, spacing, direction, _ = geometry(img)
    nz = np.argwhere(arr > 0)
    pad = np.ceil(margin_mm / spacing[::-1]).astype(int)
    lo = np.maximum(nz.min(0) - pad, 0)
    hi = np.minimum(nz.max(0) + pad + 1, arr.shape)
    sub = arr[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    return make_image(sub, origin + direction @ (spacing * lo[::-1]), spacing, direction)


def grid_around(surf, spacing, margin_mm):
    """Empty axis-aligned image covering the surface plus margin."""
    b = np.array(surf.bounds).reshape(3, 2)
    lo, hi = b[:, 0] - margin_mm, b[:, 1] + margin_mm
    size = np.ceil((hi - lo) / spacing).astype(int) + 1
    return make_image(np.zeros(size[::-1]), lo, spacing, np.eye(3))


def rasterize(surf, ref):
    """Float32 mask on ref's grid, 1 inside the closed LPS surface `surf`."""
    origin, spacing, direction, size = geometry(ref)
    idx = surf.copy()
    idx.points = ((np.asarray(surf.points) - origin) @ direction) / spacing   # continuous index
    st = vtkPolyDataToImageStencil()
    st.SetInputData(idx)
    st.SetOutputOrigin(0.0, 0.0, 0.0)
    st.SetOutputSpacing(1.0, 1.0, 1.0)
    st.SetOutputWholeExtent(0, int(size[0]) - 1, 0, int(size[1]) - 1, 0, int(size[2]) - 1)
    conv = vtkImageStencilToImage()
    conv.SetInputConnection(st.GetOutputPort())
    conv.SetInsideValue(1.0)
    conv.SetOutsideValue(0.0)
    conv.SetOutputScalarTypeToFloat()
    conv.Update()
    arr = vtk_to_numpy(conv.GetOutput().GetPointData().GetScalars()).reshape(size[::-1])
    return make_image(arr, origin, spacing, direction)


IMAGE_SUFFIXES = (".nrrd", ".nii", ".nii.gz", ".mha", ".mhd")


def surface_mask(surf, cfg):
    """Rasterise a closed LPS surface on an axis-aligned grid around it."""
    spacing = np.full(3, float(cfg.raster_spacing_mm))
    return rasterize(surf, grid_around(surf, spacing, cfg.crop_margin_mm))


def load_mask(path, label, cfg):
    """Cropped float mask from a labelmap, or from a closed surface (rasterised)."""
    if path.name.lower().endswith(IMAGE_SUFFIXES):
        return crop_to_mask(read_mask(path, label), cfg.crop_margin_mm)
    return surface_mask(read_surface(path, space="LPS"), cfg)


def mask_volume_ml(img):
    return float(itk.array_view_from_image(img).sum() * np.prod(tuple(img.GetSpacing())) / 1000.0)


def dice(a, b):
    a, b = itk.array_view_from_image(a) > 0.5, itk.array_view_from_image(b) > 0.5
    return float(2.0 * (a & b).sum() / (a.sum() + b.sum()))


# ----------------------------------------------------------------------------------------------
# elastix
# ----------------------------------------------------------------------------------------------
def parameter_object(cfg):
    po = itk.ParameterObject.New()
    if cfg.parameter_files:
        for f in cfg.parameter_files:
            po.AddParameterFile(str(f))
    else:
        for name in cfg.parameter_maps:
            po.AddParameterMap(po.GetDefaultParameterMap(name))
    overrides = {"RandomSeed": cfg.random_seed, **cfg.parameter_overrides}
    for i in range(po.GetNumberOfParameterMaps()):
        for key, value in overrides.items():
            po.SetParameter(i, key, [str(v) for v in np.atleast_1d(value)])
    return po


def register(fixed, moving, cfg, workdir):
    """Run elastix; returns (T: fixed -> moving as itk transform, moving resampled on fixed)."""
    po = parameter_object(cfg)
    reg = itk.ElastixRegistrationMethod.New(fixed, moving)
    reg.SetParameterObject(po)
    reg.SetOutputDirectory(str(workdir))
    reg.SetLogToConsole(False)
    reg.SetLogToFile(True)
    reg.UpdateLargestPossibleRegion()      # also writes elastix.log, TransformParameters.N.txt
    for f in workdir.glob("result.*"):     # per-stage result images, duplicates of the output
        f.unlink()
    return reg.GetCombinationTransform(), reg.GetOutput()


def transform_points(T, pts):
    return np.array([tuple(T.TransformPoint([float(c) for c in p])) for p in pts])


def lookup_start(T, y, ref, n_samples):
    """Initial guesses for T^-1(y): the grid point of `ref` (~n_samples over its box) whose
    image under T is closest to each y. Keeps the solver away from wrong local minima."""
    from scipy.spatial import cKDTree
    origin, spacing, direction, size = geometry(ref)
    stride = max(1, int(np.ceil((np.prod(size) / n_samples) ** (1 / 3))))
    idx = np.stack(np.meshgrid(*[np.arange(0, n, stride) for n in size], indexing="ij"), -1).reshape(-1, 3)
    P = origin + (idx * spacing) @ direction.T
    _, j = cKDTree(transform_points(T, P)).query(y)
    return P[j]


def jacobians(T, x, h=0.1):
    J = np.empty((len(x), 3, 3))
    for k in range(3):
        e = np.zeros(3)
        e[k] = h
        J[:, :, k] = (transform_points(T, x + e) - transform_points(T, x - e)) / (2 * h)
    return J


def invert_points(T, y, x0, tol, max_iter):
    """x with T(x) = y for every row of y, from x0. Damped Gauss-Newton (Levenberg-Marquardt):
    only steps that reduce the residual are accepted, so near-singular Jacobians (where the
    transform is close to folding) do not throw the iterate away."""
    x = x0.copy()
    r = transform_points(T, x) - y
    err = np.linalg.norm(r, axis=1)
    lam = np.full(len(y), 1e-3)
    for _ in range(max_iter):
        act = np.where(err > tol)[0]
        if len(act) == 0:
            break
        J = jacobians(T, x[act])
        JtJ = np.einsum("nki,nkj->nij", J, J)
        g = np.einsum("nki,nk->ni", J, r[act])
        scale = np.trace(JtJ, axis1=1, axis2=2)[:, None, None] / 3
        dx = np.linalg.solve(JtJ + lam[act, None, None] * scale * np.eye(3), g[:, :, None])[:, :, 0]
        x_new = x[act] - dx
        r_new = transform_points(T, x_new) - y[act]
        e_new = np.linalg.norm(r_new, axis=1)
        ok = e_new < err[act]
        a = act[ok]
        x[a], r[a], err[a] = x_new[ok], r_new[ok], e_new[ok]
        lam[a] /= 3.0
        lam[act[~ok]] *= 4.0
    return x, err


def tri_normals(surf):
    tri = np.asarray(surf.points)[surf.faces.reshape(-1, 4)[:, 1:]]
    return np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])


# ----------------------------------------------------------------------------------------------
# step
# ----------------------------------------------------------------------------------------------
def run(cfg):
    """Warp the collapsed surface cfg.surface into inflated space and write cfg.output."""
    workdir = cfg.workdir
    workdir.mkdir(parents=True, exist_ok=True)

    surf = read_surface(cfg.surface, space="LPS", input_space=cfg.surface_space)
    inflated = load_mask(cfg.inflated, cfg.inflated_label, cfg)
    if cfg.collapsed is not None:
        collapsed = load_mask(cfg.collapsed, cfg.collapsed_label, cfg)
    else:
        collapsed = surface_mask(surf, cfg)
        log.info("collapsed mask rasterised from %s", cfg.surface.name)
    log.info("inflated %.1f mL, collapsed %.1f mL, surface %.1f mL",
             mask_volume_ml(inflated), mask_volume_ml(collapsed), surf.volume / 1000.0)
    write_image(inflated, workdir / "inflated_mask.nrrd")
    write_image(collapsed, workdir / "collapsed_mask.nrrd")

    log.info("elastix (%s): %s", cfg.direction,
             ", ".join(map(str, cfg.parameter_files or cfg.parameter_maps)))
    X_col = np.asarray(surf.points, dtype=float)
    if cfg.direction == "inverse":
        # T: inflated -> collapsed; the collapsed nodes need T^-1
        T, registered = register(inflated, collapsed, cfg, workdir)
        log.info("image Dice after registration: %.3f", dice(inflated, registered))
        x0 = lookup_start(T, X_col, inflated, cfg.inversion_samples)
        X_inf, res = invert_points(T, X_col, x0, cfg.inversion_tol_mm, cfg.inversion_max_iter)
        log.info("inverse residual max %.2e mm", res.max())
        if res.max() > cfg.inversion_tol_mm:
            raise RuntimeError("transform inversion did not converge: max residual %.3g mm "
                               "(%d nodes) - the transform folds or is near-singular there"
                               % (res.max(), (res > cfg.inversion_tol_mm).sum()))
    else:
        # T: collapsed -> inflated, applied to the nodes directly
        T, registered = register(collapsed, inflated, cfg, workdir)
        log.info("image Dice after registration: %.3f", dice(collapsed, registered))
        X_inf = transform_points(T, X_col)
    write_image(registered, workdir / "registered.nrrd")
    warped = surf.copy()
    warped.points = X_inf

    u = np.linalg.norm(X_inf - X_col, axis=1)
    flipped = int((np.einsum("ij,ij->i", tri_normals(surf), tri_normals(warped)) <= 0).sum())
    log.info("node displacement mean %.2f, max %.2f mm", u.mean(), u.max())
    log.info("warped surface %.1f mL, Dice vs inflated mask %.3f, flipped triangles %d",
             warped.volume / 1000.0, dice(inflated, rasterize(warped, inflated)), flipped)
    if flipped:
        log.warning("%d triangles flipped orientation: the warped mesh self-intersects", flipped)

    # convert both before differencing so the displacement vectors are in the output space
    out_col = convert_space(surf, "LPS", cfg.output_space)
    out = convert_space(warped, "LPS", cfg.output_space)
    out.point_data["RegistrationDisplacement_mm"] = np.asarray(out.points) - np.asarray(out_col.points)
    write_surface(out, cfg.output, cfg.output_space)
    log.info("wrote %s (%s)", cfg.output, cfg.output_space)
    return cfg.output
