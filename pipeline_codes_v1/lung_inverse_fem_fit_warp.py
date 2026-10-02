#!/usr/bin/env python3
"""
lung_inverse_fem_fit_warp.py  --  NVIDIA Warp backend for 6_1_lung_inverse_fem_fit.py
=====================================================================================

Drop-in replacement for the GetFEM part of the original script. Everything else
(I/O, LPS handling, hilum anchor, Gmsh volume mesh, Ward clustering, coarse-to-fine
levels, tracker/watchdog, export) is imported UNCHANGED from the original file.

What changes
  * LungFEM  -> WarpLungFEM : same physics as the GetFEM model
      - P1 tetrahedra, GetFEM 'Compressible_Neo_Hookean' energy
            W = c1 (J^-2/3 I1 - 3) + d1 (J - 1)^2,   [c1, d1] = mat_params(nu)  (E = 1)
      - follower pleural pressure  sign * q_k * J F^-T N . du   per region (Nanson)
      - Dirichlet u = 0 on the hilum faces
    residual + exact tangent assembled on GPU with Warp kernels (float64),
    Newton with backtracking line search, same load ramp / warm start logic.
  * run_level -> analytic Jacobian for scipy least_squares (point-to-point loss):
        du/dq_k = -K_t^-1 dR/dq_k ,  du/dnu = -K_t^-1 dR/dnu
    one factorisation + (K+1) back-substitutions instead of (K+1) nonlinear solves.
    Falls back to finite differences for the closest-surface loss or Nelder-Mead.

Linear solver (--solver)
  cudss   : NVIDIA cuDSS sparse direct solver via nvmath-python (GPU)
  pardiso : MKL Pardiso via pypardiso (CPU, multithreaded)
  scipy   : SuperLU (CPU, always available, slowest)
  auto    : cudss if a CUDA device + nvmath are available, else pardiso, else scipy

Run
  pip install warp-lang "nvmath-python[cu12]" pypardiso   # cu12 extra pulls cuDSS; nvmath/pypardiso optional
  python lung_inverse_fem_fit_warp.py --base 6_1_lung_inverse_fem_fit.py \
         --deformable ... --target ...  [all original options]  [--device cuda --solver cudss]
No GetFEM needed. --order 2 is not implemented (P1 only).
"""

import os
import sys
import time
import argparse
import importlib.util

import numpy as np
import scipy.sparse as sp

import warp as wp

wp.config.quiet = True

f64 = wp.float64


# ----------------------------------------------------------------------------------------------
# Warp kernels (float64)
# ----------------------------------------------------------------------------------------------
@wp.func
def shape_grad(Bi: wp.mat33d, a: int):
    # rows of Dm^-1 are the gradients of N1..N3; N0 = 1 - N1 - N2 - N3
    if a == 0:
        return -(Bi[0] + Bi[1] + Bi[2])
    return Bi[a - 1]


@wp.func
def nh_P(F: wp.mat33d, c1: f64, d1: f64):
    J = wp.determinant(F)
    FiT = wp.transpose(wp.inverse(F))
    I1 = wp.ddot(F, F)
    a = wp.pow(J, -f64(2.0) / f64(3.0))
    two = f64(2.0)
    t23 = f64(2.0) / f64(3.0)
    return c1 * a * (two * F - t23 * I1 * FiT) + two * d1 * (J * J - J) * FiT


@wp.func
def nh_dP(F: wp.mat33d, dF: wp.mat33d, c1: f64, d1: f64):
    J = wp.determinant(F)
    Fi = wp.inverse(F)
    FiT = wp.transpose(Fi)
    I1 = wp.ddot(F, F)
    a = wp.pow(J, -f64(2.0) / f64(3.0))
    two = f64(2.0)
    t23 = f64(2.0) / f64(3.0)
    tr = wp.trace(Fi @ dF)
    da = -t23 * a * tr
    dI1 = two * wp.ddot(F, dF)
    dFiT = -(FiT @ wp.transpose(dF) @ FiT)
    dJ = J * tr
    t1 = c1 * da * (two * F - t23 * I1 * FiT) + c1 * a * (two * dF - t23 * dI1 * FiT - t23 * I1 * dFiT)
    t2 = two * d1 * (two * J - f64(1.0)) * dJ * FiT + two * d1 * (J * J - J) * dFiT
    return t1 + t2


@wp.kernel
def elastic_kernel(U: wp.array(dtype=wp.vec3d), tets: wp.array(dtype=wp.vec4i),
                   Bi_arr: wp.array(dtype=wp.mat33d), vol: wp.array(dtype=f64),
                   c1: f64, d1: f64, compute_K: int,
                   f: wp.array(dtype=wp.vec3d), Kb: wp.array(dtype=wp.mat33d),
                   Jdet: wp.array(dtype=f64)):
    t = wp.tid()
    T = tets[t]
    Bi = Bi_arr[t]
    V = vol[t]
    F = wp.identity(n=3, dtype=f64)
    for a in range(4):
        F = F + wp.outer(U[T[a]], shape_grad(Bi, a))
    J = wp.determinant(F)
    Jdet[t] = J
    if J <= f64(0.0):
        return
    P = nh_P(F, c1, d1)
    for a in range(4):
        wp.atomic_add(f, T[a], V * (P @ shape_grad(Bi, a)))
    if compute_K != 0:
        for a in range(4):
            ga = shape_grad(Bi, a)
            for b in range(4):
                gb = shape_grad(Bi, b)
                blk = wp.mat33d()
                for j in range(3):
                    ej = wp.vec3d()
                    ej[j] = f64(1.0)
                    dP = nh_dP(F, wp.outer(ej, gb), c1, d1)
                    blk = blk + wp.outer(V * (dP @ ga), ej)
                Kb[t * 16 + a * 4 + b] = blk


@wp.func
def skew(v: wp.vec3d):
    return wp.mat33d(f64(0.0), -v[2], v[1],
                     v[2], f64(0.0), -v[0],
                     -v[1], v[0], f64(0.0))


@wp.kernel
def pressure_kernel(X: wp.array(dtype=wp.vec3d), U: wp.array(dtype=wp.vec3d),
                    faces: wp.array(dtype=wp.vec3i), fq: wp.array(dtype=f64), compute_K: int,
                    f: wp.array(dtype=wp.vec3d), Fb: wp.array(dtype=wp.mat33d)):
    i = wp.tid()
    fc = faces[i]
    q = fq[i]
    x0 = X[fc[0]] + U[fc[0]]
    x1 = X[fc[1]] + U[fc[1]]
    x2 = X[fc[2]] + U[fc[2]]
    half = f64(0.5)
    A = half * wp.cross(x1 - x0, x2 - x0)          # current area vector = J F^-T N dA
    w = q / f64(3.0)
    for r in range(3):
        wp.atomic_add(f, fc[r], w * A)
    if compute_K != 0:
        S0 = w * skew(half * (x2 - x1))            # dA/dx0
        S1 = w * skew(half * (x0 - x2))            # dA/dx1
        S2 = w * skew(half * (x1 - x0))            # dA/dx2
        for r in range(3):
            Fb[i * 9 + r * 3 + 0] = S0
            Fb[i * 9 + r * 3 + 1] = S1
            Fb[i * 9 + r * 3 + 2] = S2


# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------
def read_msh22(path):
    """Minimal ASCII Gmsh 2.2 reader: nodes + 4-node tetrahedra."""
    with open(path) as fh:
        lines = fh.read().split("\n")
    i = 0
    ids, xyz, tets = None, None, []
    while i < len(lines):
        s = lines[i].strip()
        if s == "$Nodes":
            n = int(lines[i + 1])
            arr = np.array([lines[i + 2 + k].split() for k in range(n)], dtype=float)
            ids, xyz = arr[:, 0].astype(np.int64), arr[:, 1:4]
            i += n + 2
        elif s == "$Elements":
            n = int(lines[i + 1])
            for k in range(n):
                p = lines[i + 2 + k].split()
                if int(p[1]) == 4:
                    nt = int(p[2])
                    tets.append([int(v) for v in p[3 + nt:3 + nt + 4]])
            i += n + 2
        else:
            i += 1
    if ids is None or not tets:
        raise RuntimeError("no nodes / tetrahedra in %s (binary .msh is not supported)" % path)
    remap = {v: k for k, v in enumerate(ids)}
    T = np.vectorize(remap.get)(np.array(tets, dtype=np.int64))
    used = np.unique(T)
    new = -np.ones(len(xyz), dtype=np.int64)
    new[used] = np.arange(len(used))
    return xyz[used], new[T]


def mat_params_np(nu):
    return 1.0 / (4.0 * (1.0 + nu)), 1.0 / (6.0 * (1.0 - 2.0 * nu))


def dmat_params_dnu(nu):
    return -1.0 / (4.0 * (1.0 + nu) ** 2), 1.0 / (3.0 * (1.0 - 2.0 * nu) ** 2)


class LinearSolver:
    """Sparse direct solve A X = B (B: vector or matrix), A in CSR."""

    def __init__(self, backend, device):
        self.backend = backend
        if backend == "auto":
            self.backend = "scipy"
            try:
                import pypardiso  # noqa: F401
                self.backend = "pardiso"
            except Exception:
                pass
            if str(device).startswith("cuda"):
                try:
                    import nvmath  # noqa: F401
                    self.backend = "cudss"
                except Exception:
                    pass
        print("  linear solver: %s" % self.backend)

    def solve(self, A, B):
        if self.backend == "cudss":
            from nvmath.sparse.advanced import DirectSolver
            b = np.asfortranarray(B)
            with DirectSolver(A.tocsr(), b) as s:
                s.plan()
                s.factorize()
                x = s.solve()
            return np.asarray(x)
        if self.backend == "pardiso":
            import pypardiso
            return pypardiso.spsolve(A.tocsr(), np.ascontiguousarray(B))
        from scipy.sparse.linalg import splu
        return splu(A.tocsc()).solve(B)


class _VolumeExporter:
    """Stands in for fem.mfu so the original export call keeps working."""

    def __init__(self, fem):
        self.fem = fem

    def export_to_vtk(self, path, fmt, mf, U, name):
        import pyvista as pv
        g = pv.UnstructuredGrid({pv.CellType.TETRA: self.fem.tets}, self.fem.X.copy())
        g.point_data[name] = np.asarray(U).reshape(-1, 3)
        g.save(path, binary=False)


# ----------------------------------------------------------------------------------------------
# Warp FEM model (same interface as LungFEM in the original script)
# ----------------------------------------------------------------------------------------------
class WarpLungFEM:
    def __init__(self, msh_path, order, surf_pts, tri_centroids, hilum_tri, args):
        from scipy.spatial import cKDTree
        if order != 1:
            print("  WARNING: Warp backend implements P1 only -> using order 1")
        self.args = args
        self.device = wp.get_device(getattr(args, "device", None) or None)
        self.lin = LinearSolver(getattr(args, "solver", "auto"), self.device)

        X, T = read_msh22(msh_path)
        # positive orientation
        Dm = np.stack([X[T[:, 1]] - X[T[:, 0]], X[T[:, 2]] - X[T[:, 0]], X[T[:, 3]] - X[T[:, 0]]], axis=2)
        det = np.linalg.det(Dm)
        neg = det < 0
        T[neg] = T[neg][:, [0, 2, 1, 3]]
        Dm = np.stack([X[T[:, 1]] - X[T[:, 0]], X[T[:, 2]] - X[T[:, 0]], X[T[:, 3]] - X[T[:, 0]]], axis=2)
        det = np.linalg.det(Dm)
        if np.any(det <= 0):
            raise RuntimeError("degenerate tetrahedra in the Gmsh mesh")
        self.X, self.tets = X, T
        self.nn = len(X)
        self.ndof = 3 * self.nn
        Bi = np.linalg.inv(Dm)                     # rows = grad N1..N3
        vol = det / 6.0
        print("  Warp mesh: %d nodes, %d tets, %d dofs (P1), device %s"
              % (self.nn, len(T), self.ndof, self.device))

        # outer faces with outward orientation
        loc = np.array([[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]])
        opp = np.array([0, 1, 2, 3])
        allf = T[:, loc].reshape(-1, 3)
        allo = T[:, opp].reshape(-1)
        key = np.sort(allf, axis=1)
        _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
        outer = cnt[inv.ravel()] == 1
        faces, oppv = allf[outer], allo[outer]
        nrm = np.cross(X[faces[:, 1]] - X[faces[:, 0]], X[faces[:, 2]] - X[faces[:, 0]])
        flip = np.einsum("ij,ij->i", nrm, X[faces].mean(1) - X[oppv]) < 0
        faces[flip] = faces[flip][:, [0, 2, 1]]
        self.faces = faces
        cents = X[faces].mean(axis=1)
        d, self.tri_of_face = cKDTree(tri_centroids).query(cents)
        print("  outer faces: %d (surface tris %d), face->tri match max %.2e mm"
              % (len(faces), len(tri_centroids), d.max()))
        if d.max() > 1e-3:
            print("  WARNING: Gmsh changed the surface triangulation; face labels use nearest triangle.")

        dmask = hilum_tri[self.tri_of_face]
        if dmask.sum() == 0:
            sys.exit("ERROR: no boundary face inside the hilum ball")
        self.press_face = ~dmask
        fixed_nodes = np.unique(faces[dmask])
        fixed = np.zeros(self.ndof, bool)
        fixed[(3 * fixed_nodes[:, None] + np.arange(3)).ravel()] = True
        self.free = np.where(~fixed)[0]
        self.nfree = len(self.free)
        self.dof2free = -np.ones(self.ndof, np.int64)
        self.dof2free[self.free] = np.arange(self.nfree)

        dd, self.surf_node = cKDTree(X).query(surf_pts)
        if dd.max() > 1e-4:
            print("  WARNING: surface points not on mesh nodes (max %.2e mm) -> nearest node" % dd.max())
        self.surf_pts = surf_pts

        self._build_sparsity()

        dev = self.device
        self.wX = wp.array(X, dtype=wp.vec3d, device=dev)
        self.wT = wp.array(T.astype(np.int32), dtype=wp.vec4i, device=dev)
        self.wBi = wp.array(Bi, dtype=wp.mat33d, device=dev)
        self.wV = wp.array(vol, dtype=f64, device=dev)
        self.wF = wp.array(faces.astype(np.int32), dtype=wp.vec3i, device=dev)
        self.wf = wp.zeros(self.nn, dtype=wp.vec3d, device=dev)
        self.wKb = wp.zeros(len(T) * 16, dtype=wp.mat33d, device=dev)
        self.wFb = wp.zeros(len(faces) * 9, dtype=wp.mat33d, device=dev)
        self.wJ = wp.zeros(len(T), dtype=f64, device=dev)
        self.wfq = wp.zeros(len(faces), dtype=f64, device=dev)

        self.mfu = _VolumeExporter(self)
        self.sign = 1.0
        self.U_cache = None
        self.cur_nu = None
        self.K = 0
        self.face_labels = None
        self.n_newton = 0

    # ---- sparsity (computed once) ----
    def _build_sparsity(self):
        T, Fc = self.tets, self.faces
        ii, jj = np.meshgrid(np.arange(3), np.arange(3), indexing="ij")
        a, b = np.meshgrid(np.arange(4), np.arange(4), indexing="ij")
        # element blocks: [tet, a, b, i, j]
        re = (3 * T[:, a.ravel()][:, :, None, None] + ii[None, None])
        ce = (3 * T[:, b.ravel()][:, :, None, None] + jj[None, None])
        re = np.broadcast_to(re, (len(T), 16, 3, 3)).ravel()
        ce = np.broadcast_to(ce, (len(T), 16, 3, 3)).ravel()
        r3, c3 = np.meshgrid(np.arange(3), np.arange(3), indexing="ij")
        rf = np.broadcast_to(3 * Fc[:, r3.ravel()][:, :, None, None] + ii[None, None], (len(Fc), 9, 3, 3)).ravel()
        cf = np.broadcast_to(3 * Fc[:, c3.ravel()][:, :, None, None] + jj[None, None], (len(Fc), 9, 3, 3)).ravel()
        rows = np.r_[re, rf]
        cols = np.r_[ce, cf]
        fr, fc = self.dof2free[rows], self.dof2free[cols]
        self.keep = (fr >= 0) & (fc >= 0)
        keys = fr[self.keep] * self.nfree + fc[self.keep]
        uniq, self.inv = np.unique(keys, return_inverse=True)
        self.inv = self.inv.ravel()
        r = uniq // self.nfree
        self.indices = (uniq % self.nfree).astype(np.int64)
        self.indptr = np.r_[0, np.cumsum(np.bincount(r, minlength=self.nfree))].astype(np.int64)
        self.nnz = len(uniq)

    # ---- model per clustering level ----
    def build(self, face_labels, K, nu):
        self.face_labels = np.asarray(face_labels)
        self.K = K
        self.cur_nu = nu

    def _face_q(self, q, lam):
        fq = np.zeros(len(self.faces))
        m = self.face_labels >= 0
        fq[m] = self.sign * lam * np.asarray(q, float)[self.face_labels[m]]
        return fq

    def _assemble(self, U, fq, nu, compute_K, c=None):
        c1, d1 = mat_params_np(nu) if c is None else c
        dev = self.device
        wU = wp.array(U.reshape(-1, 3), dtype=wp.vec3d, device=dev)
        self.wf.zero_()
        self.wfq.assign(wp.array(fq, dtype=f64, device=dev))
        k = 1 if compute_K else 0
        wp.launch(elastic_kernel, dim=len(self.tets), device=dev,
                  inputs=[wU, self.wT, self.wBi, self.wV, f64(c1), f64(d1), k, self.wf, self.wKb, self.wJ])
        if fq is not None and np.any(fq != 0.0):
            wp.launch(pressure_kernel, dim=len(self.faces), device=dev,
                      inputs=[self.wX, wU, self.wF, self.wfq, k, self.wf, self.wFb])
        elif compute_K:
            self.wFb.zero_()
        R = self.wf.numpy().ravel().copy()
        if self.wJ.numpy().min() <= 0.0:
            return None, None
        A = None
        if compute_K:
            vals = np.r_[self.wKb.numpy().ravel(), self.wFb.numpy().ravel()][self.keep]
            data = np.bincount(self.inv, weights=vals, minlength=self.nnz)
            A = sp.csr_matrix((data, self.indices, self.indptr), shape=(self.nfree, self.nfree))
        return R, A

    def _newton(self, U, fq, nu):
        tol, maxit = self.args.newton_tol, self.args.newton_maxit
        R, A = self._assemble(U, fq, nu, True)
        if R is None:
            return None
        for _ in range(maxit):
            r = R[self.free]
            rn = np.abs(r).max()
            if rn < tol:
                return U
            try:
                dx = self.lin.solve(A, -r)
            except Exception:
                return None
            self.n_newton += 1
            alpha, n0 = 1.0, np.linalg.norm(r)
            for ls in range(8):
                Ut = U.copy()
                Ut[self.free] += alpha * dx
                Rt, _ = self._assemble(Ut, fq, nu, False)
                if Rt is not None and (np.linalg.norm(Rt[self.free]) < n0 or ls == 7):
                    break
                alpha *= 0.5
            if Rt is None or not np.all(np.isfinite(Ut)):
                return None
            U = Ut
            if np.abs(alpha * dx).max() < 1e-12 * (1.0 + np.abs(U).max()):
                R, A = self._assemble(U, fq, nu, True)
                return U if R is not None and np.abs(R[self.free]).max() < 1e3 * tol else None
            R, A = self._assemble(U, fq, nu, True)
            if R is None:
                return None
        return U if np.abs(R[self.free]).max() < tol else None

    def forward(self, q, nu):
        """Returns full displacement vector (node-major xyz) or None if Newton failed."""
        self.cur_nu = nu
        if self.U_cache is not None:
            U = self._newton(self.U_cache.copy(), self._face_q(q, 1.0), nu)
            if U is not None:
                self.U_cache = U.copy()
                return self.U_cache
        for n in (self.args.load_steps, 3 * self.args.load_steps):
            U = np.zeros(self.ndof)
            ok = True
            for s in range(1, n + 1):
                U = self._newton(U, self._face_q(q, s / n), nu)
                if U is None:
                    ok = False
                    break
            if ok:
                self.U_cache = U.copy()
                return self.U_cache
        return None

    def surface_disp(self, U):
        return np.asarray(U).reshape(-1, 3)[self.surf_node]

    # ---- analytic sensitivities at the converged state U (full load) ----
    def jacobian(self, U, q, nu, with_nu):
        R, A = self._assemble(U, self._face_q(q, 1.0), nu, True)
        if A is None:
            return None
        x = (self.X + U.reshape(-1, 3))
        Fc = self.faces
        Avec = 0.5 * np.cross(x[Fc[:, 1]] - x[Fc[:, 0]], x[Fc[:, 2]] - x[Fc[:, 0]]) / 3.0
        G = np.zeros((self.nn, 3, self.K))
        m = self.face_labels >= 0
        for r in range(3):
            np.add.at(G, (Fc[m, r], slice(None), self.face_labels[m]), self.sign * Avec[m])
        G = G.reshape(self.ndof, self.K)
        cols = [G]
        if with_nu:
            f1, _ = self._assemble(U, None, nu, False, c=(1.0, 0.0))
            f2, _ = self._assemble(U, None, nu, False, c=(0.0, 1.0))
            dc1, dd1 = dmat_params_dnu(nu)
            cols.append((dc1 * f1 + dd1 * f2)[:, None])
        B = np.hstack(cols)[self.free]
        dUf = -np.asarray(self.lin.solve(A, B)).reshape(self.nfree, -1)
        dU = np.zeros((self.ndof, dUf.shape[1]))
        dU[self.free] = dUf
        rows = (3 * self.surf_node[:, None] + np.arange(3)).ravel()
        return dU[rows]                                  # (3*nsurf, K [+1])


# ----------------------------------------------------------------------------------------------
# optimisation level with analytic Jacobian (replaces run_level)
# ----------------------------------------------------------------------------------------------
def make_run_level(base):
    Stop = base.Stop

    def run_level(L, K, tri_labels, x0, lb, ub, free_nu, nu_fixed, ctx):
        from scipy.optimize import least_squares, minimize
        fem, trk, args = ctx["fem"], ctx["trk"], ctx["args"]
        X_def, X_tgt, corr = ctx["X_def"], ctx["X_tgt"], ctx["corr"]
        pairs = base.region_pairs(tri_labels, ctx["adj"])
        face_labels = np.where(fem.press_face, tri_labels[fem.tri_of_face], -1)
        nu0 = x0[-1] if free_nu else nu_fixed
        fem.build(face_labels, K, nu0)
        trk.new_level()
        analytic = corr and args.optimizer == "lsq"
        print("\n=== Level %d: K=%d regions, %d params, %d adjacency pairs, optimizer=%s, jac=%s ==="
              % (L, K, len(x0), len(pairs), args.optimizer, "analytic" if analytic else "2-point"))

        def unpack(x):
            return (x[:K], x[K]) if free_nu else (x, nu_fixed)

        def geo_residual(Xs):
            if corr:
                r = (Xs - X_tgt)
                return r.ravel(), np.linalg.norm(r, axis=1)
            dpoly = ctx["def_poly"].copy()
            dpoly.points = Xs
            _, cp = ctx["tgt_poly"].find_closest_cell(Xs, return_closest_point=True)
            _, cp2 = dpoly.find_closest_cell(X_tgt, return_closest_point=True)
            d1 = np.linalg.norm(Xs - cp, axis=1)
            d2 = np.linalg.norm(X_tgt - cp2, axis=1)
            return np.r_[d1, d2], np.r_[d1, d2]

        cache = {}

        def evaluate(x):
            trk.check()
            q, nu = unpack(x)
            U = fem.forward(q, nu)
            rreg = args.reg * (q[pairs[:, 0]] - q[pairs[:, 1]]) if len(pairs) else np.zeros(0)
            cache.update(x=np.array(x, copy=True), U=None if U is None else U.copy())
            if U is None:
                trk.n_fail += 1
                trk.log(L, K, np.nan, np.nan, nu, 0)
                base_r = trk.level_best_r if trk.level_best_r is not None else ctx["r_zero"]
                return None, np.r_[2.0 * base_r[:len(ctx["r_zero"])], rreg], np.inf
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
            trk.improve(err, r, state)
            return U, np.r_[r, rreg], err + (np.dot(rreg, rreg) / max(1, len(rreg)))

        Jreg = np.zeros((len(pairs), len(x0)))
        if len(pairs):
            Jreg[np.arange(len(pairs)), pairs[:, 0]] = args.reg
            Jreg[np.arange(len(pairs)), pairs[:, 1]] = -args.reg

        def jac(x):
            if "x" not in cache or not np.array_equal(cache["x"], x):
                evaluate(x)
            q, nu = unpack(x)
            J = None
            if cache.get("U") is not None:
                J = fem.jacobian(cache["U"], q, nu, free_nu)
            if J is None:                                   # failed state: reuse last good Jacobian
                return cache.get("J", np.zeros((len(ctx["r_zero"]) + len(pairs), len(x))))
            cache["J"] = np.vstack([J, Jreg])
            return cache["J"]

        reason = "converged"
        try:
            if args.optimizer == "lsq":
                least_squares(lambda x: evaluate(x)[1], x0, jac=jac if analytic else "2-point",
                              bounds=(lb, ub), method="trf",
                              x_scale=np.maximum(np.abs(x0), 0.1), diff_step=5e-3,
                              ftol=1e-6, xtol=1e-6, gtol=1e-8, max_nfev=100000)
            else:
                minimize(lambda x: evaluate(np.clip(x, lb, ub))[2], x0, method="Nelder-Mead",
                         bounds=list(zip(lb, ub)),
                         options=dict(maxfev=100000, xatol=1e-4, fatol=1e-3, adaptive=True))
        except Stop as s:
            reason = s.reason
        print("  level %d: %d Newton linear solves so far" % (L, fem.n_newton))
        return reason, trk.level_best_err

    return run_level


# ----------------------------------------------------------------------------------------------
# entry point: load the original script as a module and swap the FEM backend
# ----------------------------------------------------------------------------------------------
def load_base(path):
    spec = importlib.util.spec_from_file_location("lung_fit_base", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--base", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                    "6_1_lung_inverse_fem_fit.py"))
    ap.add_argument("--device", default=None, help="warp device: cpu | cuda | cuda:0 (default: best)")
    ap.add_argument("--solver", default="auto", choices=["auto", "cudss", "pardiso", "scipy"])
    own, rest = ap.parse_known_args()
    sys.argv = [sys.argv[0]] + rest

    wp.init()
    base = load_base(own.base)

    orig_parse = base.parse_args

    def parse_args():
        a = orig_parse()
        a.device, a.solver = own.device, own.solver
        return a

    def check_deps():
        missing = []
        for mod in ["numpy", "scipy", "sklearn", "pyvista", "vtk"]:
            try:
                __import__(mod)
            except Exception as e:
                missing.append("%s (%s)" % (mod, e))
        if missing:
            sys.exit("Missing dependencies: " + ", ".join(missing))

    base.parse_args = parse_args
    base.check_deps = check_deps
    base.LungFEM = WarpLungFEM
    base.run_level = make_run_level(base)
    base.main()


if __name__ == "__main__":
    main()
