"""Tetrahedral mesh of the reference surface with Netgen (LGPL), surface triangulation kept as it is.

Netgen replaced Gmsh (GPL) on 2026-11-18: better tetrahedra, collapse closer to a fine mesh, same
fits and leave-one-out (analysis/results.md 7d). It runs in a subprocess with a hard timeout (a bad
surface can make it hang, a crash of the native library does not take the caller down).
"""

import logging
import os
import subprocess
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

log = logging.getLogger(__name__)

NETGEN_SCRIPT = """
import sys
import numpy as np
from netgen.meshing import Element2D, FaceDescriptor, Mesh, MeshingParameters, MeshPoint, Pnt
d = np.load("surface.npz")
m = Mesh(dim=3)
ids = [m.Add(MeshPoint(Pnt(*map(float, p)))) for p in d["points"]]
fd = m.Add(FaceDescriptor(surfnr=1, domin=1, domout=0, bc=1))     # outward triangles, inside = domain 1
for t in d["tris"]:
    m.Add(Element2D(fd, [ids[int(k)] for k in t]))
m.GenerateVolumeMesh(MeshingParameters(maxh=float(sys.argv[1]), optsteps3d=int(sys.argv[2])))
xyz = np.array([[p[0], p[1], p[2]] for p in m.Points()], float)
tets = np.array([[v.nr for v in el.vertices] for el in m.Elements3D()], np.int64).reshape(-1, 4)
np.savez("volume_raw.npz", tags=np.arange(1, len(xyz) + 1), xyz=xyz, tets=tets)
m.Save("volume.vol")
"""


def python_executable():
    """The Python that runs Netgen: $EURELAPS_PYTHON, else this one. Inside the 3D Slicer GUI
    sys.executable may be the Slicer application, not a Python: the extension runs the prediction
    with PythonSlicer, where sys.executable is right, or sets EURELAPS_PYTHON."""
    return os.environ.get("EURELAPS_PYTHON") or sys.executable


def boundary_faces(tets):
    """Faces used by exactly one tetrahedron, as sorted node triplets."""
    f = np.sort(np.concatenate([tets[:, [0, 1, 2]], tets[:, [0, 1, 3]],
                                tets[:, [0, 2, 3]], tets[:, [1, 2, 3]]]), axis=1)
    u, count = np.unique(f, axis=0, return_counts=True)
    return u[count == 1]


def tet_mesh(P, tris, workdir, maxh, timeout_s, optsteps3d=10):
    """(nodes, tets, surface_nodes): surface point i is volume node surface_nodes[i]. maxh [mm]:
    Netgen maximum element size inside; optsteps3d: Netgen 3D optimisation steps."""
    workdir.mkdir(parents=True, exist_ok=True)
    name = "Netgen"
    np.savez(workdir / "surface.npz", points=np.asarray(P, float), tris=np.asarray(tris, np.int64))
    cmd = [python_executable(), "-c", NETGEN_SCRIPT, "%r" % float(maxh), str(int(optsteps3d))]
    t0 = time.time()
    try:
        r = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        raise RuntimeError("%s exceeded %d s (raise fem_setup.mesh_size_mm or mesh_timeout_s)" % (name, timeout_s)) from None
    raw = workdir / "volume_raw.npz"
    if r.returncode != 0 or not raw.is_file():
        msg = (r.stderr or r.stdout)[-2000:]
        hint = "\ncheck the reference surface: self-intersections (a fold in the segmentation?), a hole"
        raise RuntimeError("%s failed:\n%s%s" % (name, msg, hint))
    with np.load(raw) as d:
        tags, xyz, conn = d["tags"], d["xyz"], d["tets"]
    raw.unlink()

    order = np.argsort(tags)                  # node tags (1-based) -> 0-based indices
    nodes = xyz[order]
    tets = np.searchsorted(tags[order], conn)
    # positive orientation for every core
    a, b, c, d = (nodes[tets[:, k]] for k in range(4))
    neg = np.einsum("ij,ij->i", b - a, np.cross(c - a, d - a)) < 0
    tets[neg] = tets[neg][:, [0, 2, 1, 3]]

    dist, surface_nodes = cKDTree(nodes).query(P)
    if dist.max() > 1e-6:
        raise RuntimeError("%s moved the surface points (max %.2e mm)" % (name, dist.max()))
    bf = boundary_faces(tets)
    sf = np.unique(np.sort(surface_nodes[tris], axis=1), axis=0)
    if len(bf) != len(sf) or not np.array_equal(bf, sf):
        raise RuntimeError("%s changed the surface triangulation (%d boundary faces, %d surface "
                           "triangles)" % (name, len(bf), len(sf)))
    vol = np.abs(np.einsum("ij,ij->i", b - a, np.cross(c - a, d - a))) / 6.0
    log.info("%s: %d nodes (%d on the surface), %d tets, %.1f mL, min tet %.3f mm^3, %.1f s", name,
             len(nodes), len(P), len(tets), vol.sum() / 1000.0, vol.min(), time.time() - t0)
    return nodes, tets, surface_nodes
