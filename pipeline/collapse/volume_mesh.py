"""Tetrahedral mesh of the reference surface with Gmsh, surface triangulation kept as it is.

Gmsh runs in a subprocess with a hard timeout (a bad surface can make it hang).
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


def tet_mesh(P, tris, workdir, mesh_size, timeout_s):
    """(nodes, tets, surface_nodes): surface point i is volume node surface_nodes[i]."""
    workdir.mkdir(parents=True, exist_ok=True)
    write_stl_ascii(P, tris, workdir / "surface.stl")
    (workdir / "volume.geo").write_text(
        'Merge "surface.stl";\nSurface Loop(1) = {1};\nVolume(1) = {1};\nPhysical Volume(1) = {1};\n'
        "Mesh.Algorithm3D = 1;\nMesh.MeshSizeMax = %g;\nMesh.Optimize = 1;\nMesh.OptimizeNetgen = 0;\n"
        "Mesh.MshFileVersion = 2.2;\n" % mesh_size)
    t0 = time.time()
    try:
        r = subprocess.run([sys.executable, "-c", GMSH_SCRIPT], cwd=workdir, capture_output=True,
                           text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        raise RuntimeError("Gmsh exceeded %d s (raise mesh_size_mm or gmsh_timeout_s)" % timeout_s) from None
    raw = workdir / "volume_raw.npz"
    if r.returncode != 0 or not raw.is_file():
        msg = (r.stderr or r.stdout)[-2000:]
        hint = ("\nthe surface intersects itself (flipped triangles from the registration?): "
                "check the reference surface" if "PLC Error" in msg else "")
        raise RuntimeError("Gmsh failed:\n%s%s" % (msg, hint))
    with np.load(raw) as d:
        tags, xyz, conn = d["tags"], d["xyz"], d["tets"]
    raw.unlink()

    order = np.argsort(tags)                  # gmsh node tags -> 0-based indices
    nodes = xyz[order]
    tets = np.searchsorted(tags[order], conn)
    # positive orientation for every core
    a, b, c, d = (nodes[tets[:, k]] for k in range(4))
    neg = np.einsum("ij,ij->i", b - a, np.cross(c - a, d - a)) < 0
    tets[neg] = tets[neg][:, [0, 2, 1, 3]]

    dist, surface_nodes = cKDTree(nodes).query(P)
    if dist.max() > 1e-6:
        raise RuntimeError("Gmsh moved the surface points (max %.2e mm)" % dist.max())
    bf = boundary_faces(tets)
    sf = np.unique(np.sort(surface_nodes[tris], axis=1), axis=0)
    if len(bf) != len(sf) or not np.array_equal(bf, sf):
        raise RuntimeError("Gmsh changed the surface triangulation (%d boundary faces, %d surface "
                           "triangles)" % (len(bf), len(sf)))
    vol = np.abs(np.einsum("ij,ij->i", b - a, np.cross(c - a, d - a))) / 6.0
    log.info("gmsh: %d nodes (%d on the surface), %d tets, %.1f mL, min tet %.3f mm^3, %.1f s",
             len(nodes), len(P), len(tets), vol.sum() / 1000.0, vol.min(), time.time() - t0)
    return nodes, tets, surface_nodes
