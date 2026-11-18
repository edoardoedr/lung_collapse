"""Tetrahedral mesh of the reference surface, surface triangulation kept as it is, with Gmsh (GPL)
or Netgen (LGPL; analysis/results.md 7d: better tetrahedra, closer to a fine mesh).

The mesher runs in a subprocess with a hard timeout (a bad surface can make it hang).
"""

import logging
import subprocess
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

log = logging.getLogger(__name__)

GMSH_SCRIPT = """
import gmsh, numpy as np
gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)
gmsh.open("volume.geo")
gmsh.model.mesh.generate(3)
tags, xyz, _ = gmsh.model.mesh.getNodes()
_, conn = gmsh.model.mesh.getElementsByType(4)
np.savez("volume_raw.npz", tags=tags, xyz=xyz.reshape(-1, 3), tets=conn.reshape(-1, 4))
gmsh.write("volume.msh")
gmsh.finalize()
"""


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


def write_stl_ascii(P, tris, path):
    """ASCII STL in full precision (binary STL is float32 and would move the surface)."""
    with open(path, "w") as f:
        f.write("solid lung\n")
        for t in tris:
            a, b, c = P[t]
            n = np.cross(b - a, c - a)
            n = n / max(np.linalg.norm(n), 1e-300)
            f.write(" facet normal %.9e %.9e %.9e\n  outer loop\n" % tuple(n))
            for v in (a, b, c):
                f.write("   vertex %.15e %.15e %.15e\n" % tuple(v))
            f.write("  endloop\n endfacet\n")
        f.write("endsolid lung\n")


def boundary_faces(tets):
    """Faces used by exactly one tetrahedron, as sorted node triplets."""
    f = np.sort(np.concatenate([tets[:, [0, 1, 2]], tets[:, [0, 1, 3]],
                                tets[:, [0, 2, 3]], tets[:, [1, 2, 3]]]), axis=1)
    u, count = np.unique(f, axis=0, return_counts=True)
    return u[count == 1]


def tet_mesh(P, tris, workdir, mesh_size, timeout_s, mesher="gmsh", optsteps3d=10):
    """(nodes, tets, surface_nodes): surface point i is volume node surface_nodes[i]. mesh_size [mm]:
    gmsh MeshSizeMax or Netgen maxh; optsteps3d: Netgen only."""
    workdir.mkdir(parents=True, exist_ok=True)
    name = {"gmsh": "Gmsh", "netgen": "Netgen"}[mesher]
    if mesher == "gmsh":
        write_stl_ascii(P, tris, workdir / "surface.stl")
        (workdir / "volume.geo").write_text(
            'Merge "surface.stl";\nSurface Loop(1) = {1};\nVolume(1) = {1};\nPhysical Volume(1) = {1};\n'
            "Mesh.Algorithm3D = 1;\nMesh.MeshSizeMax = %g;\nMesh.Optimize = 1;\nMesh.OptimizeNetgen = 0;\n"
            "Mesh.MshFileVersion = 2.2;\n" % mesh_size)
        cmd = [sys.executable, "-c", GMSH_SCRIPT]
    else:
        np.savez(workdir / "surface.npz", points=np.asarray(P, float), tris=np.asarray(tris, np.int64))
        cmd = [sys.executable, "-c", NETGEN_SCRIPT, "%r" % float(mesh_size), str(int(optsteps3d))]
    t0 = time.time()
    try:
        r = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        raise RuntimeError("%s exceeded %d s (raise the mesh size or gmsh_timeout_s)" % (name, timeout_s)) from None
    raw = workdir / "volume_raw.npz"
    if r.returncode != 0 or not raw.is_file():
        msg = (r.stderr or r.stdout)[-2000:]
        hint = ("\nthe surface intersects itself (a fold in the segmentation?): "
                "check the reference surface" if "PLC Error" in msg else "")
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
    log.info("%s: %d nodes (%d on the surface), %d tets, %.1f mL, min tet %.3f mm^3, %.1f s", mesher,
             len(nodes), len(P), len(tets), vol.sum() / 1000.0, vol.min(), time.time() - t0)
    return nodes, tets, surface_nodes
