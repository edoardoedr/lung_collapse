# make_hilum_anchor.py -- writes results/hilum_anchor.mrk.json
# Run in 3D Slicer's Python console:
#   exec(open('/home/pr502/IBT_pr502_riggio_HIWI/make_hilum_anchor.py').read())
#
# Centre and radius were derived once from hilum markups (closed curve + interior
# fiducials, centroid + max-distance radius) and verified to capture a rank-3 patch
# of the transformed_left_lung surface. Values are hardcoded here so the anchor is
# reproducible without the original markup nodes.
import numpy as np
import vtk
import slicer

HILUM_LPS = np.array([31.745, 47.380, -984.298])   # verified anchor, LPS
RADIUS    = 41.557                                  # mm, for reference
MESH      = "transformed_left_lung"                 # surface the anchor must touch
OUT       = "/home/pr502/IBT_pr502_riggio_HIWI/results/hilum_anchor.mrk.json"

flip = np.array([-1, -1, 1])          # LPS -> RAS (Slicer works in RAS)
hilum_ras = HILUM_LPS * flip

old = slicer.mrmlScene.GetFirstNodeByName("hilum_anchor")
if old:
    slicer.mrmlScene.RemoveNode(old)
fid = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "hilum_anchor")
fid.AddControlPointWorld(vtk.vtkVector3d(*hilum_ras), "hilum")

# verify the anchor captures a non-degenerate (rank-3) patch of surface
from vtk.util.numpy_support import vtk_to_numpy
pts_lps = vtk_to_numpy(slicer.util.getNode(MESH).GetPolyData().GetPoints().GetData()) * flip
d = np.linalg.norm(pts_lps - HILUM_LPS, axis=1)
cap = pts_lps[d <= RADIUS]
print("captured %d points, nearest surface %.2f mm" % (len(cap), d.min()))
if len(cap) >= 3:
    sv = np.linalg.svd(cap - cap.mean(axis=0), compute_uv=False)
    print("rank %d, singular values %s" % (np.linalg.matrix_rank(cap - cap.mean(axis=0), tol=1e-6),
                                           np.round(sv, 1)))

slicer.util.saveNode(fid, OUT)
print("wrote %s" % OUT)