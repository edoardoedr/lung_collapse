"""Rasterisation of closed surfaces on a regular grid (itk images, LPS), for the input checks."""

import itk
import numpy as np
from vtkmodules.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkImagingStencil import vtkImageStencilToImage, vtkPolyDataToImageStencil


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


def grid_around(surf, spacing, margin_mm):
    """Empty axis-aligned image covering the surface(s) plus margin."""
    surfs = surf if isinstance(surf, (list, tuple)) else [surf]
    b = np.array([s.bounds for s in surfs]).reshape(-1, 3, 2)
    lo, hi = b[:, :, 0].min(0) - margin_mm, b[:, :, 1].max(0) + margin_mm
    size = np.ceil((hi - lo) / spacing).astype(int) + 1
    return make_image(np.zeros(size[::-1]), lo, spacing, np.eye(3))


def rasterize(surf, ref):
    """Float32 mask on ref's grid, 1 inside the closed LPS surface `surf`."""
    origin, spacing, direction, size = geometry(ref)
    idx = surf.copy()
    idx.points = ((np.asarray(surf.points) - origin) @ direction) / spacing   # continuous index
    st = vtkPolyDataToImageStencil()
    st.SetInputData(idx.to_vtk())
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
