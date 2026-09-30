#!/usr/bin/env python3
"""
lung_inverse_fem_fit.py  --  standalone GetFEM inverse FEM for VATS lung collapse
===============================================================================

Inputs (both SURFACE meshes, .vtp / .vtk / .stl):
  --deformable  transformed_left_lung      (reference / inflated-space surface; HARDEN the transform before export!)
  --target      model_left_collapsed_mesh  (collapsed surface; raw or already aligned)

What it does
  1. Reads both surfaces, converts to LPS, orients normals, checks watertightness.
  2. Detects index-wise correspondence (identical topology -> exact point-to-point loss),
     otherwise falls back to a closest-surface loss.
  3. Rigid handling: the ~9 deg rigid artifact is removed from the CLUSTERING field (full Kabsch);
     the FEM target is aligned hilum-to-anchor by default (--align), consistent with the fixed hilum.
  4. Hilum Dirichlet anchor (ball) with rank-3 check and automatic radius growth.
  5. Volumetric tet mesh from the deformable surface via Gmsh (surface nodes preserved exactly).
  6. Compressible Neo-Hookean ([c1,d1] = [mu/2, K/2], the corrected parameterisation),
     follower pleural pressure (Nanson: J F^-T N), non-dimensionalised by E (only p/E is identifiable).
  7. Coarse-to-fine pressure regions: connectivity-constrained Ward clustering of the clean
     normal displacement field, K = 1 -> 4 -> 8 -> 14 -> 25 -> 40 ..., warm-started between levels.
  8. Optimiser per level: bounded Levenberg-Marquardt style least squares (scipy TRF) on the
     point-to-point residual vector (default), or Nelder-Mead (--optimizer nm).
     Optional free Poisson ratio (changes deformation mode, NOT degenerate with p/E).
  9. Stops on: target error reached | time budget | plateau | levels exhausted | Ctrl+C / SIGTERM.
     A separate watchdog process hard-kills the run after budget + grace, and the best result is
     checkpointed to disk on every improvement, so a forced stop never loses the best solution.

Run (system Python, because GetFEM lives there):
  /usr/bin/python3 lung_inverse_fem_fit.py --deformable ... --target ...
Missing scipy / scikit-learn / pyvista / gmsh?  Use a venv that still sees system GetFEM:
  /usr/bin/python3 -m venv --system-site-packages ~/venvs/fem
  ~/venvs/fem/bin/pip install "numpy<2" scipy scikit-learn pyvista gmsh
  ~/venvs/fem/bin/python lung_inverse_fem_fit.py ...
(numpy<2 because Ubuntu's python3-getfem 5.4 is built against numpy 1.x)
"""

import os
import sys
import json
import time
import argparse
import signal
import subprocess
import shutil
import datetime
import traceback

import numpy as np

ROOT = "/home/pr502/IBT_pr502_riggio_HIWI"
RES = os.path.join(ROOT, "results")


# ----------------------------------------------------------------------------------------------
# 0. arguments & dependencies
# ----------------------------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Standalone GetFEM inverse FEM for lung collapse")
    # inputs
    ap.add_argument("--deformable", default=os.path.join(RES, "transformed_left_lung.vtp"))
    ap.add_argument("--target", default=os.path.join(RES, "model_left_collapsed_mesh.vtp"))
    ap.add_argument("--input-space", default="auto", choices=["auto", "LPS", "RAS"],
                    help="coordinate system of the input files (auto = read SPACE tag, else LPS)")
    ap.add_argument("--out", default=None, help="output folder (default results/fem_fit_<timestamp>)")
    # alignment
    ap.add_argument("--align", default="hilum_rigid", choices=["none", "rigid", "hilum_rigid"],
                    help="rigid pre-alignment of the FEM TARGET (needs correspondence). hilum_rigid = "
                         "make the target hilum coincide with the fixed anchor (physically consistent "
                         "with the Dirichlet BC); rigid = full Kabsch (= model_left_collapsed_aligned); "
                         "none = use target as given")
    ap.add_argument("--cluster-field", default="rigid_residual",
                    choices=["raw", "rigid_residual", "hilum_rigid"],
                    help="displacement field used for Ward clustering (rigid artifact removed by default)")
    # hilum
    ap.add_argument("--hilum-json", default=os.path.join(RES, "hilum_anchor.mrk.json"))
    ap.add_argument("--hilum-center", type=float, nargs=3, default=None, help="LPS, overrides json")
    ap.add_argument("--hilum-radius", type=float, default=41.557)
    ap.add_argument("--hilum-min-pts", type=int, default=20)
    # material
    ap.add_argument("--E", type=float, default=3000.0, help="Young's modulus [Pa] (reporting scale)")
    ap.add_argument("--nu", type=float, default=0.30)
    ap.add_argument("--fix-nu", action="store_true", help="do NOT optimise Poisson's ratio")
    ap.add_argument("--nu-bounds", type=float, nargs=2, default=None)
    # discretisation
    ap.add_argument("--order", type=int, default=1, choices=[1, 2], help="FEM order (P2 = slower, no locking)")
    ap.add_argument("--mesh-size", type=float, default=8.0, help="Gmsh MeshSizeMax [mm] (interior)")
    ap.add_argument("--gmsh-timeout", type=int, default=300)
    ap.add_argument("--load-steps", type=int, default=4)
    ap.add_argument("--newton-tol", type=float, default=1e-7)
    ap.add_argument("--newton-maxit", type=int, default=30)
    # clustering
    ap.add_argument("--levels", default="1,4,8,14,25,40", help="region counts, coarse -> fine")
    ap.add_argument("--feature", default="normal", choices=["normal", "vector"],
                    help="Ward feature: normal displacement u.n (default) or full 3D clean vector")
    ap.add_argument("--pos-weight", type=float, default=0.3, help="spatial compactness weight in Ward")
    # optimisation
    ap.add_argument("--optimizer", default="lsq", choices=["lsq", "nm"])
    ap.add_argument("--q0", type=float, default=0.5, help="initial p/E (0.5*3000 Pa ~ 1500 Pa)")
    ap.add_argument("--q-bounds", type=float, nargs=2, default=[-1.0, 3.0], help="bounds on p/E")
    ap.add_argument("--reg", type=float, default=1.0,
                    help="smoothness penalty between adjacent regions [mm per unit p/E]")
    ap.add_argument("--target-error", type=float, default=2.5, help="stop when mean error [mm] <= this")
    ap.add_argument("--time-budget-min", type=float, default=240.0)
    ap.add_argument("--hard-grace-min", type=float, default=10.0)
    ap.add_argument("--level-max-evals", type=int, default=600, help="forward solves per level (incl. FD)")
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--min-improve", type=float, default=0.02)
    return ap.parse_args()


def check_deps():
    missing = []
    for mod in ["numpy", "scipy", "sklearn", "pyvista", "vtk", "getfem"]:
        try:
            __import__(mod)
        except Exception as e:
            missing.append("%s (%s)" % (mod, str(e).splitlines()[0][:80]))
    gmsh_ok = shutil.which("gmsh") is not None
    try:
        import gmsh  # noqa: F401
        gmsh_ok = True
    except Exception:
        pass
    if not gmsh_ok:
        missing.append("gmsh (python module or CLI binary)")
    if missing:
        print("\nMissing dependencies:\n  - " + "\n  - ".join(missing))
        print("\nFix (keeps system GetFEM visible):\n"
              "  /usr/bin/python3 -m venv --system-site-packages ~/venvs/fem\n"
              "  ~/venvs/fem/bin/pip install 'numpy<2' scipy scikit-learn pyvista gmsh\n"
              "  ~/venvs/fem/bin/python %s ...\n" % os.path.basename(__file__))
        sys.exit(1)


# ----------------------------------------------------------------------------------------------
# 1. I/O helpers
# ----------------------------------------------------------------------------------------------
def sniff_space_from_header(path):
    """Slicer legacy .vtk files carry 'SPACE=LPS/RAS' in the header line."""
    try:
        with open(path, "rb") as f:
            head = f.read(512).decode("latin-1", errors="ignore").upper()
        if "SPACE=RAS" in head:
            return "RAS"
        if "SPACE=LPS" in head:
            return "LPS"
    except Exception:
        pass
    return None


def read_surface(path):
    import pyvista as pv
    if not os.path.isfile(path):
        sys.exit("ERROR: file not found: %s" % path)
    m = pv.read(path)
    if not isinstance(m, pv.PolyData):
        m = m.extract_surface()
    m = m.triangulate()
    space = None
    if "SPACE" in m.field_data:
        try:
            space = str(np.asarray(m.field_data["SPACE"]).ravel()[0]).upper()
        except Exception:
            space = None
    if space is None:
        space = sniff_space_from_header(path)
    return m, space


def write_vtp_lps(poly, path):
    """vtkXMLPolyDataWriter directly (pyvista .save() can drop the SPACE tag)."""
    import vtk
    poly = poly.copy()
    poly.field_data["SPACE"] = np.array(["LPS"])
    w = vtk.vtkXMLPolyDataWriter()
    w.SetFileName(path)
    w.SetInputData(poly)
    w.Write()


def write_stl_ascii(points, tris, path):
    """ASCII STL with full double precision (binary STL is float32)."""
    with open(path, "w") as f:
        f.write("solid lung\n")
        for t in tris:
            a, b, c = points[t[0]], points[t[1]], points[t[2]]
            n = np.cross(b - a, c - a)
            nn = np.linalg.norm(n)
            n = n / nn if nn > 0 else n
            f.write(" facet normal %.9e %.9e %.9e\n  outer loop\n" % tuple(n))
            for v in (a, b, c):
                f.write("   vertex %.12e %.12e %.12e\n" % tuple(v))
            f.write("  endloop\n endfacet\n")
        f.write("endsolid lung\n")


def read_markup_point(path):
    """First control point of a Slicer .mrk.json, returned in LPS."""
    with open(path) as f:
        d = json.load(f)
    mk = d["markups"][0]
    cs = str(mk.get("coordinateSystem", "LPS")).upper()
    p = np.array(mk["controlPoints"][0]["position"], dtype=float)
    if cs == "RAS":
        p[:2] *= -1.0
    return p


# ----------------------------------------------------------------------------------------------
# 2. geometry helpers
# ----------------------------------------------------------------------------------------------
def faces_of(poly):
    return poly.faces.reshape(-1, 4)[:, 1:].astype(np.int64)


def orient(poly):
    """Consistent outward normals; point order is NOT changed (split_vertices=False)."""
    out = poly.compute_normals(cell_normals=True, point_normals=True, split_vertices=False,
                               consistent_normals=True, auto_orient_normals=True)
    assert out.n_points == poly.n_points and np.allclose(out.points, poly.points), \
        "normal computation changed point ordering"
    return out


def watertight_report(poly, name):
    e = poly.extract_feature_edges(boundary_edges=True, non_manifold_edges=True,
                                   feature_edges=False, manifold_edges=False)
    print("  %-11s pts=%d tris=%d  open/non-manifold edges=%d  volume=%.1f mL"
          % (name, poly.n_points, poly.n_cells, e.n_cells, abs(poly.volume) / 1000.0))
    return e.n_cells == 0


def kabsch(A, B, w=None):
    """R, t minimising sum w_i |R A_i + t - B_i|^2."""
    w = np.ones(len(A)) if w is None else w
    w = w / w.sum()
    ca, cb = (w[:, None] * A).sum(0), (w[:, None] * B).sum(0)
    H = ((A - ca) * w[:, None]).T @ (B - cb)
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, cb - R @ ca


def face_adjacency(tris):
    edges = {}
    for fi, t in enumerate(tris):
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            edges.setdefault((min(a, b), max(a, b)), []).append(fi)
    pairs = [(v[0], v[1]) for v in edges.values() if len(v) == 2]
    return np.array(pairs, dtype=np.int64)


def tri_geometry(P, tris):
    a, b, c = P[tris[:, 0]], P[tris[:, 1]], P[tris[:, 2]]
    cr = np.cross(b - a, c - a)
    area = 0.5 * np.linalg.norm(cr, axis=1)
    n = cr / np.maximum(2 * area[:, None], 1e-12)
    return (a + b + c) / 3.0, n, area


def assd(A, B):
    """Symmetric mean vertex-to-surface distance between two pyvista PolyData."""
    _, cpB = B.find_closest_cell(A.points, return_closest_point=True)
    _, cpA = A.find_closest_cell(B.points, return_closest_point=True)
    dA = np.linalg.norm(A.points - cpB, axis=1)
    dB = np.linalg.norm(B.points - cpA, axis=1)
    return float((dA.sum() + dB.sum()) / (len(dA) + len(dB)))


# ----------------------------------------------------------------------------------------------
# 3. volumetric meshing (Gmsh in a subprocess with hard timeout, surface kept as-is)
# ----------------------------------------------------------------------------------------------
def gmsh_volume(points, tris, workdir, mesh_size, timeout_s):
    stl = os.path.join(workdir, "deformable_surface.stl")
    geo = os.path.join(workdir, "deformable_volume.geo")
    msh = os.path.join(workdir, "deformable_volume.msh")
    write_stl_ascii(points, tris, stl)
    with open(geo, "w") as f:
        f.write('Merge "%s";\n' % os.path.basename(stl))
        f.write("Surface Loop(1) = {1};\nVolume(1) = {1};\nPhysical Volume(1) = {1};\n")
        f.write("Mesh.Algorithm3D = 1;\nMesh.MeshSizeMax = %g;\nMesh.Optimize = 1;\n" % mesh_size)
        f.write("Mesh.OptimizeNetgen = 0;\nMesh.MshFileVersion = 2.2;\n")
    try:
        import gmsh  # noqa: F401
        code = ("import gmsh; gmsh.initialize(); gmsh.option.setNumber('General.Terminal', 0); "
                "gmsh.open(%r); gmsh.model.mesh.generate(3); gmsh.write(%r); gmsh.finalize()"
                % (os.path.basename(geo), os.path.basename(msh)))
        cmd = [sys.executable, "-c", code]
    except Exception:
        cmd = [shutil.which("gmsh"), os.path.basename(geo), "-3", "-format", "msh22",
               "-o", os.path.basename(msh)]
    t0 = time.time()
    try:
        r = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        sys.exit("ERROR: Gmsh exceeded %d s (increase --mesh-size or --gmsh-timeout)" % timeout_s)
    if r.returncode != 0 or not os.path.isfile(msh):
        print(r.stdout[-2000:], r.stderr[-2000:])
        sys.exit("ERROR: Gmsh volumetrisation failed (watertight surface? see output above)")
    print("  gmsh done in %.1f s" % (time.time() - t0))
    return msh


# ----------------------------------------------------------------------------------------------
# 4. clustering: connectivity-constrained Ward on the clean field
# ----------------------------------------------------------------------------------------------
def ward_labels(feat, adj_pairs, active, K, pos, pos_weight):
    """Returns per-triangle labels (0..K-1), -1 for inactive (hilum) triangles."""
    from sklearn.cluster import AgglomerativeClustering
    from scipy.sparse import coo_matrix
    lab = -np.ones(len(active), dtype=int)
    idx = np.where(active)[0]
    K = int(min(K, len(idx)))
    if K <= 1:
        lab[idx] = 0
        return lab, 1
    remap = -np.ones(len(active), dtype=int)
    remap[idx] = np.arange(len(idx))
    pr = adj_pairs[(remap[adj_pairs[:, 0]] >= 0) & (remap[adj_pairs[:, 1]] >= 0)]
    i, j = remap[pr[:, 0]], remap[pr[:, 1]]
    n = len(idx)
    conn = coo_matrix((np.ones(2 * len(i)), (np.r_[i, j], np.r_[j, i])), shape=(n, n)).tocsr()
    f = feat[idx]
    f = f / (f.std(axis=0, keepdims=True) + 1e-12)
    g = pos[idx] - pos[idx].mean(0)
    g = g / (g.std() + 1e-12) * pos_weight
    X = np.column_stack([f, g])
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sub = AgglomerativeClustering(n_clusters=K, connectivity=conn, linkage="ward").fit_predict(X)
    lab[idx] = sub
    return lab, K


def region_pairs(tri_labels, adj_pairs):
    a, b = tri_labels[adj_pairs[:, 0]], tri_labels[adj_pairs[:, 1]]
    m = (a >= 0) & (b >= 0) & (a != b)
    pr = np.unique(np.sort(np.column_stack([a[m], b[m]]), axis=1), axis=0)
    return pr


# ----------------------------------------------------------------------------------------------
# 5. GetFEM model
# ----------------------------------------------------------------------------------------------
def mat_params(nu):
    # non-dimensional (E = 1). GetFEM 'Compressible_Neo_Hookean' takes [c1, d1] = [mu/2, K/2]
    return [1.0 / (4.0 * (1.0 + nu)), 1.0 / (6.0 * (1.0 - 2.0 * nu))]


class LungFEM:
    RID_DIR = 10

    def __init__(self, msh_path, order, surf_pts, tri_centroids, hilum_tri, args):
        import getfem as gf
        from scipy.spatial import cKDTree
        self.gf = gf
        try:
            gf.util_trace_level(0)
            gf.util_warning_level(0)
        except Exception:
            pass
        self.args = args
        self.mesh = gf.Mesh("import", "gmsh", msh_path)
        self.mfu = gf.MeshFem(self.mesh, 3)
        self.mfu.set_classical_fem(order)
        self.mim = gf.MeshIm(self.mesh, gf.Integ("IM_TETRAHEDRON(%d)" % (3 if order == 1 else 5)))
        self.ndof = self.mfu.nbdof()
        print("  GetFEM mesh: %d nodes, %d tets, %d dofs (P%d)"
              % (self.mesh.nbpts(), self.mesh.nbcvs(), self.ndof, order))

        # outer faces -> surface triangles
        self.of = self.mesh.outer_faces()
        pts = self.mesh.pts()
        cents = np.array([pts[:, self.mesh.pid_in_faces(self.of[:, [i]])].mean(axis=1)
                          for i in range(self.of.shape[1])])
        d, self.tri_of_face = cKDTree(tri_centroids).query(cents)
        print("  outer faces: %d (surface tris %d), face->tri match max %.2e mm"
              % (self.of.shape[1], len(tri_centroids), d.max()))
        if d.max() > 1e-3:
            print("  WARNING: Gmsh changed the surface triangulation; face labels use nearest triangle.")

        # Dirichlet (hilum) region
        dmask = hilum_tri[self.tri_of_face]
        if dmask.sum() == 0:
            sys.exit("ERROR: no GetFEM boundary face inside the hilum ball")
        self.mesh.set_region(self.RID_DIR, self.of[:, dmask])
        self.press_face = ~dmask

        # surface points -> vector dofs (vertex dofs coincide with surface points)
        dn = self.mfu.basic_dof_nodes().T
        dd, idx = cKDTree(dn).query(surf_pts, k=3)
        idx = np.sort(idx, axis=1)
        self.use_interp = not (dd.max() < 1e-4 and np.all(np.diff(idx, axis=1) == 1)
                               and np.all(idx[:, 0] % 3 == 0))
        self.surf_dof = idx
        self.surf_pts = surf_pts
        if self.use_interp:
            print("  WARNING: surface points not on dofs -> using gf.compute_interpolate_on")

        self.md = None
        self.level_id = 0
        self.sign = 1.0
        self.U_cache = None
        self.cur_nu = None
        self.K = 0

    # ---- model per clustering level ----
    def build(self, face_labels, K, nu):
        gf = self.gf
        self.level_id += 1
        base = 1000 * self.level_id
        md = gf.Model("real")
        md.add_fem_variable("u", self.mfu)
        md.add_initialized_data("params", mat_params(nu))
        md.add_finite_strain_elasticity_brick(self.mim, "Compressible_Neo_Hookean", "u", "params")
        md.add_Dirichlet_condition_with_simplification("u", self.RID_DIR)
        F = "(Id(meshdim)+Grad_u)"
        for k in range(K):
            sel = np.where(face_labels == k)[0]
            rid = base + k
            self.mesh.set_region(rid, self.of[:, sel])
            md.add_initialized_data("q%d" % k, [0.0])
            # follower pressure (Nanson): positive q pushes INWARD (verified by sign check)
            expr = "(%g)*q%d*Det(%s)*((Inv(%s))'*Normal).Test_u" % (self.sign, k, F, F)
            if hasattr(md, "add_nonlinear_term"):
                md.add_nonlinear_term(self.mim, expr, rid)
            else:
                md.add_nonlinear_generic_assembly_brick(self.mim, expr, rid)
        self.md, self.K, self.cur_nu = md, K, nu

    def _set_q(self, q, lam):
        for k in range(self.K):
            self.md.set_variable("q%d" % k, [float(q[k]) * lam])

    def _newton(self):
        try:
            r = self.md.solve("max_iter", self.args.newton_maxit, "max_res", self.args.newton_tol,
                              "lsearch", "simplest")
            conv = bool(r[1]) if isinstance(r, (tuple, list)) and len(r) > 1 else True
        except Exception:
            return False
        U = self.md.variable("u")
        return conv and np.all(np.isfinite(U))

    def forward(self, q, nu):
        """Returns full displacement vector or None if Newton failed."""
        if nu != self.cur_nu:
            self.md.set_variable("params", mat_params(nu))
            self.cur_nu = nu
        if self.U_cache is not None:                          # warm start, full load
            self.md.set_variable("u", self.U_cache)
            self._set_q(q, 1.0)
            if self._newton():
                self.U_cache = self.md.variable("u").copy()
                return self.U_cache
        for n in (self.args.load_steps, 3 * self.args.load_steps):   # ramp from zero
            self.md.set_variable("u", np.zeros(self.ndof))
            ok = True
            for s in range(1, n + 1):
                self._set_q(q, s / n)
                if not self._newton():
                    ok = False
                    break
            if ok:
                self.U_cache = self.md.variable("u").copy()
                return self.U_cache
        return None

    def surface_disp(self, U):
        if not self.use_interp:
            return U[self.surf_dof]
        r = np.asarray(self.gf.compute_interpolate_on(self.mfu, U, self.surf_pts.T))
        return r.reshape(3, -1).T


# ----------------------------------------------------------------------------------------------
# 6. run control: time budget, checkpoints, stop reasons
# ----------------------------------------------------------------------------------------------
class Stop(Exception):
    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


class Tracker:
    def __init__(self, args, outdir):
        self.t0 = time.time()
        self.budget = args.time_budget_min * 60.0
        self.target = args.target_error
        self.outdir = outdir
        self.best_err = np.inf
        self.best = None
        self.level_evals = 0
        self.level_cap = args.level_max_evals
        self.level_best_err = np.inf
        self.level_best_r = None
        self.n_eval = 0
        self.n_fail = 0
        self.last_ckpt = 0.0
        self.hist = open(os.path.join(outdir, "history.csv"), "w")
        self.hist.write("eval,level,K,elapsed_min,mean_err_mm,rms_mm,nu,ok\n")
        self.stop_requested = None

    def elapsed(self):
        return time.time() - self.t0

    def check(self):
        if self.stop_requested:
            raise Stop(self.stop_requested)
        if self.elapsed() > self.budget:
            raise Stop("time budget")
        if self.level_evals >= self.level_cap:
            raise Stop("level eval cap")

    def new_level(self):
        self.level_evals = 0
        self.level_best_err = np.inf
        self.level_best_r = None

    def log(self, level, K, err, rms, nu, ok):
        self.n_eval += 1
        self.level_evals += 1
        self.hist.write("%d,%d,%d,%.3f,%.5f,%.5f,%.4f,%d\n"
                        % (self.n_eval, level, K, self.elapsed() / 60, err, rms, nu, ok))
        if self.n_eval % 10 == 0:
            self.hist.flush()

    def improve(self, err, r, state):
        if err < self.level_best_err:
            self.level_best_err, self.level_best_r = err, r.copy()
        if err < self.best_err:
            self.best_err, self.best = err, state
            if time.time() - self.last_ckpt > 15:
                self.checkpoint()
        if err <= self.target:
            self.checkpoint()
            raise Stop("target error reached")

    def checkpoint(self):
        if self.best is None:
            return
        b = self.best
        np.savez(os.path.join(self.outdir, "best_state.npz"), U=b["U"], q=b["q"], nu=b["nu"],
                 tri_labels=b["tri_labels"], err=b["err"])
        with open(os.path.join(self.outdir, "best_params.json"), "w") as f:
            json.dump({k: v for k, v in b.items() if k in
                       ("err", "rms", "K", "level", "nu", "pressures_Pa", "E_Pa", "elapsed_min")},
                      f, indent=2)
        self.last_ckpt = time.time()


def start_watchdog(seconds):
    """Separate process: SIGTERM at budget, SIGKILL 60 s later (survives GIL-blocking C calls)."""
    pid = os.getpid()
    cmd = "sleep %d; kill -TERM %d 2>/dev/null; sleep 60; kill -KILL %d 2>/dev/null" % (seconds, pid, pid)
    return subprocess.Popen(["sh", "-c", cmd], start_new_session=True,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ----------------------------------------------------------------------------------------------
# 7. one optimisation level
# ----------------------------------------------------------------------------------------------
def run_level(L, K, tri_labels, x0, lb, ub, free_nu, nu_fixed, ctx):
    from scipy.optimize import least_squares, minimize
    fem, trk, args = ctx["fem"], ctx["trk"], ctx["args"]
    X_def, X_tgt, corr = ctx["X_def"], ctx["X_tgt"], ctx["corr"]
    pairs = region_pairs(tri_labels, ctx["adj"])
    face_labels = np.where(fem.press_face, tri_labels[fem.tri_of_face], -1)
    nu0 = x0[-1] if free_nu else nu_fixed
    fem.build(face_labels, K, nu0)
    trk.new_level()
    print("\n=== Level %d: K=%d regions, %d params, %d adjacency pairs, optimizer=%s ==="
          % (L, K, len(x0), len(pairs), args.optimizer))

    def unpack(x):
        return (x[:K], x[K]) if free_nu else (x, nu_fixed)

    def geo_residual(Xs):
        if corr:
            r = (Xs - X_tgt)
            d = np.linalg.norm(r, axis=1)
            return r.ravel(), d
        dpoly = ctx["def_poly"].copy()
        dpoly.points = Xs
        _, cp = ctx["tgt_poly"].find_closest_cell(Xs, return_closest_point=True)
        _, cp2 = dpoly.find_closest_cell(X_tgt, return_closest_point=True)
        d1 = np.linalg.norm(Xs - cp, axis=1)
        d2 = np.linalg.norm(X_tgt - cp2, axis=1)
        return np.r_[d1, d2], np.r_[d1, d2]

    def evaluate(x):
        trk.check()
        q, nu = unpack(x)
        U = fem.forward(q, nu)
        rreg = args.reg * (q[pairs[:, 0]] - q[pairs[:, 1]]) if len(pairs) else np.zeros(0)
        if U is None:
            trk.n_fail += 1
            trk.log(L, K, np.nan, np.nan, nu, 0)
            base = trk.level_best_r if trk.level_best_r is not None else ctx["r_zero"]
            return None, np.r_[2.0 * base[:len(ctx["r_zero"])], rreg], np.inf
        Us = fem.surface_disp(U)
        r, d = geo_residual(X_def + Us)
        err = float(d.mean())
        rms = float(np.sqrt((d ** 2).mean()))
        trk.log(L, K, err, rms, nu, 1)
        q_tri = np.where(tri_labels >= 0, q[np.maximum(tri_labels, 0)], np.nan)
        state = dict(U=U.copy(), q=q.copy(), nu=float(nu), tri_labels=tri_labels.copy(),
                     q_tri=q_tri, err=err, rms=rms, K=K, level=L, E_Pa=args.E,
                     pressures_Pa=[float(v) for v in q * args.E],
                     elapsed_min=trk.elapsed() / 60)
        if trk.level_evals % 5 == 1 or err < trk.level_best_err:
            print("  [L%d K=%d] eval %4d  t=%6.1f min  mean=%.3f mm  rms=%.3f  nu=%.3f  "
                  "p=[%s%s] Pa  best=%.3f"
                  % (L, K, trk.level_evals, trk.elapsed() / 60, err, rms, nu,
                     ", ".join("%.0f" % v for v in (q[:6] * args.E)), ", ..." if K > 6 else "",
                     min(err, trk.best_err)))
        full_r = np.r_[r, rreg]
        trk.improve(err, r, state)
        return U, full_r, err + (np.dot(rreg, rreg) / max(1, len(rreg)))

    reason = "converged"
    try:
        if args.optimizer == "lsq":
            least_squares(lambda x: evaluate(x)[1], x0, bounds=(lb, ub), method="trf",
                          x_scale=np.maximum(np.abs(x0), 0.1), diff_step=5e-3,
                          ftol=1e-6, xtol=1e-6, gtol=1e-8, max_nfev=100000)
        else:
            minimize(lambda x: evaluate(np.clip(x, lb, ub))[2], x0, method="Nelder-Mead",
                     bounds=list(zip(lb, ub)),
                     options=dict(maxfev=100000, xatol=1e-4, fatol=1e-3, adaptive=True))
    except Stop as s:
        reason = s.reason
    return reason, trk.level_best_err


# ----------------------------------------------------------------------------------------------
# 8. main
# ----------------------------------------------------------------------------------------------
def main():
    args = parse_args()
    check_deps()
    import pyvista as pv  # noqa: F401
    from scipy.spatial import cKDTree

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = args.out or os.path.join(RES, "fem_fit_" + stamp)
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "run_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    print("Output folder:", outdir)

    # ---------------- surfaces ----------------
    print("\n[1] Reading surfaces")
    dpoly, dspace = read_surface(args.deformable)
    tpoly, tspace = read_surface(args.target)
    for poly, sp, nm in ((dpoly, dspace, "deformable"), (tpoly, tspace, "target")):
        space = sp if args.input_space == "auto" else args.input_space
        space = space or "LPS"
        if space == "RAS":
            poly.points[:, :2] *= -1.0
        print("  %s: %s space -> LPS" % (nm, space))
    dpoly, tpoly = orient(dpoly), orient(tpoly)
    ok1 = watertight_report(dpoly, "deformable")
    watertight_report(tpoly, "target")
    if not ok1:
        sys.exit("ERROR: deformable surface is not watertight -> Gmsh will fail. Repair first.")

    X_def = np.asarray(dpoly.points, dtype=float).copy()
    tris = faces_of(dpoly)
    X_tgt = np.asarray(tpoly.points, dtype=float).copy()
    corr = (dpoly.n_points == tpoly.n_points and dpoly.n_cells == tpoly.n_cells and
            np.array_equal(np.sort(faces_of(dpoly), 1), np.sort(faces_of(tpoly), 1)))
    print("  index-wise correspondence (identical topology): %s" % corr)
    if not corr:
        print("  -> using closest-surface loss (slower, weaker). Point-to-point is strongly preferred.")

    # ---------------- hilum anchor ----------------
    print("\n[2] Hilum Dirichlet anchor")
    if args.hilum_center is not None:
        hc = np.array(args.hilum_center, float)
    elif os.path.isfile(args.hilum_json):
        hc = read_markup_point(args.hilum_json)
        print("  read", args.hilum_json)
    else:
        hc = np.array([31.745, 47.380, -984.298])
        print("  using stored default anchor")
    cen, nrm_t, area = tri_geometry(X_def, tris)
    radius = args.hilum_radius

    def capture(center, rad):
        m = np.linalg.norm(cen - center, axis=1) <= rad
        v = np.unique(tris[m]) if m.any() else np.zeros(0, int)
        return m, v

    m, v = capture(hc, radius)
    if len(v) == 0:
        flip = hc * np.array([-1, -1, 1])
        mf, vf = capture(flip, radius)
        if len(vf) > 0:
            print("  anchor captured nothing; RAS/LPS-flipped anchor captures %d pts -> using flipped" % len(vf))
            hc, m, v = flip, mf, vf
    grow = 0
    while True:
        m, v = capture(hc, radius)
        sv = np.linalg.svd(X_def[v] - X_def[v].mean(0), compute_uv=False) if len(v) >= 3 else np.zeros(3)
        if (len(v) >= args.hilum_min_pts and sv[2] > 1.0) or grow >= 10:
            break
        radius *= 1.1
        grow += 1
    print("  center LPS [%.3f, %.3f, %.3f]  radius %.2f mm%s" % (*hc, radius,
          "  (grown x%.2f)" % (1.1 ** grow) if grow else ""))
    print("  captured %d surface pts / %d tris, singular values [%.1f, %.1f, %.1f]"
          % (len(v), m.sum(), *sv))
    if len(v) < 3 or sv[2] <= 1e-6:
        sys.exit("ERROR: anchor is not rank-3 -> singular tangent matrix. Check center/space.")
    hilum_tri = m
    np.save(os.path.join(outdir, "hilum_tri_mask.npy"), hilum_tri)

    # ---------------- rigid alignment ----------------
    X_tgt_raw = X_tgt.copy()

    def rigid_fit(mode):
        w = None
        if mode == "hilum_rigid":
            w = np.zeros(len(X_def))
            w[v] = 1.0
        R, t = kabsch(X_tgt_raw, X_def, w)
        ang = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
        c = X_tgt_raw.mean(0)
        shift = np.linalg.norm(R @ c + t - c)          # centroid shift (t alone depends on origin)
        return R, t, ang, shift

    print("\n[3] Rigid alignment")
    if corr:
        _, _, ang, sh = rigid_fit("rigid")
        print("  full Kabsch (diagnostic):  rotation %.2f deg, centroid shift %.2f mm" % (ang, sh))
        _, _, angh, shh = rigid_fit("hilum_rigid")
        print("  hilum-only Kabsch:         rotation %.2f deg, centroid shift %.2f mm" % (angh, shh))
        hil_raw = np.linalg.norm(X_tgt_raw[v] - X_def[v], axis=1).mean()
        print("  raw hilum motion (mean over anchor pts): %.2f mm" % hil_raw)
    if args.align != "none":
        if not corr:
            print("  FEM target alignment skipped (needs correspondence)")
        else:
            R_, t_, a_, sh_ = rigid_fit(args.align)
            X_tgt = X_tgt_raw @ R_.T + t_
            print("  FEM target aligned with '%s' (%.2f deg, centroid shift %.2f mm)" % (args.align, a_, sh_))
            np.savez(os.path.join(outdir, "rigid_transform_target.npz"), R=R_, t=t_)
    else:
        print("  FEM target used as given")
    tpoly.points = X_tgt
    write_vtp_lps(tpoly, os.path.join(outdir, "target_used.vtp"))

    # ---------------- clean field & clustering features ----------------
    print("\n[4] Clustering field: %s" % args.cluster_field)
    if corr:
        if args.cluster_field == "raw":
            Xc = X_tgt_raw
        else:
            R_, t_, _, _ = rigid_fit("rigid" if args.cluster_field == "rigid_residual" else "hilum_rigid")
            Xc = X_tgt_raw @ R_.T + t_
        u = Xc - X_def
    else:
        _, cp = tpoly.find_closest_cell(X_def, return_closest_point=True)
        u = cp - X_def
    np.savez(os.path.join(outdir, "clean_field.npz"), pts=X_def, disp=u, tris=tris)
    u_tri = u[tris].mean(axis=1)
    dn_tri = np.einsum("ij,ij->i", u_tri, nrm_t)
    feat = dn_tri[:, None] if args.feature == "normal" else u_tri
    du = np.linalg.norm(u, axis=1)
    print("  |u| mean %.2f mm, max %.2f mm; u.n mean %.2f mm (negative = inward)"
          % (du.mean(), du.max(), dn_tri.mean()))
    adj = face_adjacency(tris)

    # ---------------- volume mesh & FEM ----------------
    print("\n[5] Volumetric mesh")
    msh = gmsh_volume(X_def, tris, outdir, args.mesh_size, args.gmsh_timeout)
    fem = LungFEM(msh, args.order, X_def, cen, hilum_tri, args)

    nu_b = args.nu_bounds or ([0.05, 0.40] if args.order == 1 else [0.05, 0.45])
    free_nu = not args.fix_nu
    print("  material: compressible Neo-Hookean, E=%.0f Pa (scale), nu0=%.3f %s"
          % (args.E, args.nu, "(free in %s)" % nu_b if free_nu else "(fixed)"))

    # ---------------- signal handling & watchdog ----------------
    trk = Tracker(args, outdir)
    hard = int(args.time_budget_min * 60 + args.hard_grace_min * 60)
    wd = start_watchdog(hard)

    def on_signal(signum, frame):
        trk.stop_requested = "signal %d" % signum
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    ctx = dict(fem=fem, trk=trk, args=args, X_def=X_def, X_tgt=X_tgt, corr=corr, adj=adj,
               def_poly=dpoly, tgt_poly=tpoly)
    r0, d0 = ((X_def - X_tgt).ravel(), np.linalg.norm(X_def - X_tgt, axis=1)) if corr else (None, None)
    if not corr:
        _, cpa = tpoly.find_closest_cell(X_def, return_closest_point=True)
        _, cpb = dpoly.find_closest_cell(X_tgt, return_closest_point=True)
        d0 = np.r_[np.linalg.norm(X_def - cpa, axis=1), np.linalg.norm(X_tgt - cpb, axis=1)]
        r0 = d0.copy()
    ctx["r_zero"] = r0
    print("\n  baseline (no deformation): mean error %.3f mm, ASSD %.3f mm" % (d0.mean(), assd(dpoly, tpoly)))
    print("  reference floors from earlier diagnostics: registration ~2.0 mm, discretisation ~1.0 mm")

    # ---------------- sign check ----------------
    lab1 = np.where(hilum_tri, -1, 0)
    fem.build(np.where(fem.press_face, lab1[fem.tri_of_face], -1), 1, args.nu)
    U = fem.forward(np.array([0.2]), args.nu)
    if U is None:
        sys.exit("ERROR: first forward solve failed (anchor / mesh quality?)")
    pn = dpoly.point_normals
    un = np.einsum("ij,ij->i", fem.surface_disp(U), pn).mean()
    if un > 0:
        fem.sign = -1.0
        print("  sign check: q>0 moved surface outward -> flipping pressure sign")
    else:
        print("  sign check OK: q>0 collapses (mean u.n = %.2f mm at p/E=0.2)" % un)
    fem.U_cache = None

    # ---------------- coarse-to-fine levels ----------------
    levels = [int(s) for s in args.levels.split(",") if s.strip()]
    q_tri = np.where(hilum_tri, np.nan, args.q0)
    nu_best = args.nu
    summary, stall, prev = [], 0, d0.mean()
    stop_reason = "levels exhausted"
    for L, Kreq in enumerate(levels, start=1):
        tri_labels, K = ward_labels(feat, adj, ~hilum_tri, Kreq, cen, args.pos_weight)
        np.save(os.path.join(outdir, "tri_labels_K%d.npy" % K), tri_labels)
        # warm start: area-weighted mean of previous best per-triangle pressure
        if trk.best is not None:
            q_tri, nu_best = trk.best["q_tri"], trk.best["nu"]
            fem.U_cache = trk.best["U"].copy()
        x0 = np.array([np.nansum(q_tri[tri_labels == k] * area[tri_labels == k]) /
                       area[tri_labels == k].sum() for k in range(K)])
        lb = np.full(K, args.q_bounds[0])
        ub = np.full(K, args.q_bounds[1])
        if free_nu:
            x0 = np.r_[x0, nu_best]
            lb, ub = np.r_[lb, nu_b[0]], np.r_[ub, nu_b[1]]
        x0 = np.clip(x0, lb + 1e-6, ub - 1e-6)
        reason, lvl_err = run_level(L, K, tri_labels, x0, lb, ub, free_nu, args.nu, ctx)
        imp = (prev - trk.best_err) / max(prev, 1e-9)
        summary.append(dict(level=L, K=K, best_mean_err_mm=float(lvl_err), stop=reason,
                            evals=trk.level_evals, elapsed_min=trk.elapsed() / 60))
        print("  -> level %d done (%s): level best %.3f mm, global best %.3f mm, improvement %.1f%%"
              % (L, reason, lvl_err, trk.best_err, 100 * imp))
        trk.checkpoint()
        if reason in ("target error reached", "time budget") or reason.startswith("signal"):
            stop_reason = reason
            break
        stall = stall + 1 if imp < args.min_improve else 0
        prev = trk.best_err
        if stall >= args.patience:
            stop_reason = "plateau (%d levels < %.0f%% gain)" % (stall, 100 * args.min_improve)
            break

    # ---------------- export ----------------
    wd.kill()
    trk.hist.close()
    print("\n[6] Export (stop reason: %s)" % stop_reason)
    b = trk.best
    if b is None:
        sys.exit("No successful forward solve -> nothing to export.")
    Us = fem.surface_disp(b["U"])
    Xs = X_def + Us
    out = dpoly.copy()
    out.points = Xs
    out.point_data["Displacement"] = Us
    if corr:
        err_pt = np.linalg.norm(Xs - X_tgt, axis=1)
    else:
        _, cp = tpoly.find_closest_cell(Xs, return_closest_point=True)
        err_pt = np.linalg.norm(Xs - cp, axis=1)
    out.point_data["Error_mm"] = err_pt
    out.cell_data["PressureRegion"] = b["tri_labels"]
    out.cell_data["Pressure_Pa"] = np.nan_to_num(b["q_tri"] * args.E, nan=0.0)
    write_vtp_lps(out, os.path.join(outdir, "deformed_best.vtp"))
    try:
        fem.mfu.export_to_vtk(os.path.join(outdir, "volume_best.vtk"), "ascii",
                              fem.mfu, b["U"], "Displacement")
    except Exception as e:
        print("  volume export failed:", e)
    final_assd = assd(out, tpoly)
    res = dict(stop_reason=stop_reason, mean_err_mm=float(err_pt.mean()),
               median_err_mm=float(np.median(err_pt)), p95_err_mm=float(np.percentile(err_pt, 95)),
               max_err_mm=float(err_pt.max()), assd_mm=final_assd, baseline_mean_mm=float(d0.mean()),
               K=b["K"], nu=b["nu"], E_Pa=args.E, pressures_Pa=b["pressures_Pa"],
               mean_pressure_Pa=float(np.nanmean(b["q_tri"]) * args.E),
               evals=trk.n_eval, failed_solves=trk.n_fail, elapsed_min=trk.elapsed() / 60,
               levels=summary, correspondence=bool(corr), align=args.align,
               hilum_center_lps=hc.tolist(), hilum_radius=radius)
    with open(os.path.join(outdir, "result_summary.json"), "w") as f:
        json.dump(res, f, indent=2)
    print("\n================ RESULT ================")
    print("  mean  %.3f mm   (baseline %.3f)" % (res["mean_err_mm"], res["baseline_mean_mm"]))
    print("  median %.3f | p95 %.3f | max %.3f mm | ASSD %.3f mm"
          % (res["median_err_mm"], res["p95_err_mm"], res["max_err_mm"], final_assd))
    print("  K=%d  nu=%.3f  mean p=%.0f Pa (E=%.0f Pa; only p/E is identifiable)"
          % (b["K"], b["nu"], res["mean_pressure_Pa"], args.E))
    print("  %d forward solves (%d failed), %.1f min" % (trk.n_eval, trk.n_fail, res["elapsed_min"]))
    for s in summary:
        print("    L%d K=%-3d best %.3f mm  (%s, %d evals)" % (s["level"], s["K"], s["best_mean_err_mm"],
                                                           s["stop"], s["evals"]))
    print("  files: deformed_best.vtp, target_used.vtp, volume_best.vtk, result_summary.json, history.csv")
    print("  (all LPS; volume_best.vtk is in the reference configuration -> Warp By Vector in ParaView)")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(1)
