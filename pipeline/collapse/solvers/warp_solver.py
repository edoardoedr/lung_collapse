"""NVIDIA Warp core: same physics, options and load path as GetFEMSolver (P1), on GPU or CPU.

Kernels from pipeline_codes_v1/lung_inverse_fem_fit_warp.py (validated there: tangent and
Jacobian vs finite differences). Residual convention as the GetFEM model, R(U) = 0 with

    R = f_int(U)                                      Compressible_Neo_Hookean, [c1, d1] = mat_params(nu)
      + pressure_sign * q_k * J F^-T N               per region (P1 flat face: current area vector / 3)
      + (k/2e) (pos(g)^2 - pos(g-e)^2) wallN         wall penalty on the non-clamped faces, g = u.wallN - wallG

u = 0 on the nodes of the clamped faces (eliminated). Everything in float64.

options: the GetFEMSolver ones (order must be 1), plus
  device         Warp device (None = default: CUDA if available, else CPU)
  wall_update    "outer" (as GetFEMSolver): Newton solves with the wall linearisation fixed,
                 alternated with re-linearisations until the gap settles (wall_max_updates,
                 wall_settle_mm). "newton": the wall (normal, gap) is re-linearised at every
                 residual evaluation inside one Newton solve, so the contact is part of Newton
                 and the outer rounds disappear; at convergence it solves the same equations
                 (gap = signed distance - allowed margin), without the wall_settle_mm tolerance
  linear_solver  auto | cudss (nvmath-python) | pardiso (pypardiso) | scipy (SuperLU); None = auto

Known differences from GetFEM, by construction (see scripts/compare_warp_getfem.py):
  - Newton follows GetFEM's classical Newton as documented in its source: convergence when
    min(|R|_1, |dU|_1 / |U|_1) <= newton_tol, "simplest" line search (accept if |R|_1 does not
    exceed 1.5x the current one, else step x 3/5 down to 1e-3). GetFEM divides |R|_1 by an
    "approximate external load" norm when the model provides one; here it is taken as 1.
  - wall term integrated with the IM_TRIANGLE(3) face rule (4 points, negative centre weight),
    assumed to be what IM_TETRAHEDRON(3) uses on faces.
"""

import logging
import time

import numpy as np
import scipy.sparse as sp
import warp as wp

from ..geometry import WallDistance
from .base import ForwardSolver
from .common import DEFAULTS as GETFEM_DEFAULTS
from .common import dmat_params_dnu, mat_params

log = logging.getLogger(__name__)
wp.config.quiet = True


class _NoCudssThreadingWarning(logging.Filter):
    """nvmath logs this on the root logger at every cuDSS factorisation; it only concerns the
    speed of cuDSS host-side planning."""
    def filter(self, record):
        return "No multithreading interface library" not in record.getMessage()


logging.getLogger().addFilter(_NoCudssThreadingWarning())
f64 = wp.float64

DEFAULTS = {**GETFEM_DEFAULTS, "device": None, "linear_solver": None}

# IM_TRIANGLE(3): barycentric points, weights normalised to the face area
FACE_QUAD_B = np.array([[1 / 3, 1 / 3, 1 / 3], [0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6]])
FACE_QUAD_W = np.array([-27.0, 25.0, 25.0, 25.0]) / 48.0

# GetFEM simplest_newton_line_search defaults
LS_MAX_RATIO, LS_MIN_ALPHA, LS_MULT = 1.5, 1e-3, 0.6


# ----------------------------------------------------------------------------------------------
# kernels (float64; float literals in kernels are float32, so constants are written f64(...))
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
        for r in range(3):                         # accumulated: the wall kernel adds to Fb too
            Fb[i * 9 + r * 3 + 0] = Fb[i * 9 + r * 3 + 0] + S0
            Fb[i * 9 + r * 3 + 1] = Fb[i * 9 + r * 3 + 1] + S1
            Fb[i * 9 + r * 3 + 2] = Fb[i * 9 + r * 3 + 2] + S2


@wp.kernel
def wall_kernel(U: wp.array(dtype=wp.vec3d), faces: wp.array(dtype=wp.vec3i),
                active: wp.array(dtype=wp.int32), area0: wp.array(dtype=f64),
                wN: wp.array(dtype=wp.vec3d), wG: wp.array(dtype=f64),
                qb: wp.array(dtype=wp.vec3d), qw: wp.array(dtype=f64), nq: int,
                c: f64, e: f64, compute_K: int,
                f: wp.array(dtype=wp.vec3d), Fb: wp.array(dtype=wp.mat33d)):
    # c (pos(g)^2 - pos(g-e)^2) wallN.du on the reference face, wallN / wallG P1 data;
    # tangent 2c (pos(g) - pos(g-e)) (wallN x wallN) N_r N_s with wallN, wallG held fixed
    i = wp.tid()
    if active[i] == 0:
        return
    fc = faces[i]
    zero = f64(0.0)
    for p in range(nq):
        b = qb[p]
        u = b[0] * U[fc[0]] + b[1] * U[fc[1]] + b[2] * U[fc[2]]
        n = b[0] * wN[fc[0]] + b[1] * wN[fc[1]] + b[2] * wN[fc[2]]
        G = b[0] * wG[fc[0]] + b[1] * wG[fc[1]] + b[2] * wG[fc[2]]
        g = wp.dot(u, n) - G
        pg = wp.max(g, zero)
        pe = wp.max(g - e, zero)
        w = qw[p] * area0[i]
        fv = w * c * (pg * pg - pe * pe)
        for r in range(3):
            wp.atomic_add(f, fc[r], (fv * b[r]) * n)
        if compute_K != 0:
            kn = (w * f64(2.0) * c * (pg - pe)) * wp.outer(n, n)
            for r in range(3):
                for s in range(3):
                    Fb[i * 9 + r * 3 + s] = Fb[i * 9 + r * 3 + s] + (b[r] * b[s]) * kn


# ----------------------------------------------------------------------------------------------
# linear solver
# ----------------------------------------------------------------------------------------------
class LinearSolver:
    """Sparse direct solve A X = B (B: vector or matrix), A in CSR."""

    def __init__(self, backend, device):
        self.backend = backend or "auto"
        if self.backend == "auto":
            self.backend = "scipy"
            try:
                import pypardiso  # noqa: F401
                self.backend = "pardiso"
            except ImportError:
                pass
            if device.is_cuda:
                try:
                    import nvmath  # noqa: F401
                    self.backend = "cudss"
                except ImportError:
                    pass
        if self.backend not in ("cudss", "pardiso", "scipy"):
            raise ValueError("warp linear_solver must be auto, cudss, pardiso or scipy")
        if self.backend == "cudss":                     # fail here, not as a Newton failure later
            if not device.is_cuda:
                raise ValueError("linear_solver cudss needs a CUDA device")
            import nvmath  # noqa: F401
        elif self.backend == "pardiso":
            import pypardiso  # noqa: F401

    def solve(self, A, B):
        if self.backend == "cudss":
            from nvmath.sparse.advanced import DirectSolver
            with DirectSolver(A.tocsr(), np.asfortranarray(B)) as s:
                s.plan()
                s.factorize()
                x = s.solve()
            return np.asarray(x)
        if self.backend == "pardiso":
            import pypardiso
            return pypardiso.spsolve(A.tocsr(), np.ascontiguousarray(B))
        from scipy.sparse.linalg import splu
        return splu(A.tocsc()).solve(B)


# ----------------------------------------------------------------------------------------------
# solver
# ----------------------------------------------------------------------------------------------
class WarpSolver(ForwardSolver):
    name = "warp"
    supports_wall = True

    def __init__(self, problem, options):
        unknown = sorted(set(options) - set(DEFAULTS))
        if unknown:
            raise ValueError("unknown warp solver option(s): %s" % ", ".join(unknown))
        super().__init__(problem, {**DEFAULTS, **options})
        o = self.options
        if o["order"] != 1:
            raise NotImplementedError("the warp solver implements P1 only (solver_options.order = 1)")
        if o["wall_update"] not in ("outer", "newton"):
            raise ValueError("warp solver: wall_update must be 'outer' or 'newton'")
        self.device = wp.get_device(o["device"])
        self.lin = LinearSolver(o["linear_solver"], self.device)

        # mesh, positively oriented tets
        X = np.asarray(problem.nodes, float)
        T = np.asarray(problem.tets, np.int64).copy()
        def edges(T):
            return np.stack([X[T[:, 1]] - X[T[:, 0]], X[T[:, 2]] - X[T[:, 0]], X[T[:, 3]] - X[T[:, 0]]], axis=2)
        neg = np.linalg.det(edges(T)) < 0
        T[neg] = T[neg][:, [0, 2, 1, 3]]
        Dm = edges(T)
        det = np.linalg.det(Dm)
        if np.any(det <= 0):
            raise RuntimeError("degenerate tetrahedra in the volume mesh")
        self.X, self.tets, self.nn = X, T, len(X)
        self.ndof = 3 * self.nn

        # boundary faces = surface triangles, 1:1, oriented outward
        faces = np.asarray(problem.faces, np.int64).copy()
        loc = np.array([[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]])
        allf, allo = T[:, loc].reshape(-1, 3), T[:, [0, 1, 2, 3]].reshape(-1)
        key, inv, cnt = np.unique(np.sort(allf, axis=1), axis=0, return_inverse=True, return_counts=True)
        outer = cnt[inv.ravel()] == 1
        okey = np.sort(allf[outer], axis=1)
        if len(okey) != len(faces) or not np.array_equal(np.unique(okey, axis=0), np.unique(np.sort(faces, axis=1), axis=0)):
            raise RuntimeError("Warp boundary (%d faces) does not match the surface (%d triangles)"
                               % (len(okey), len(faces)))
        opp = dict(zip(map(tuple, okey), allo[outer]))
        oppv = np.array([opp[tuple(k)] for k in np.sort(faces, axis=1)])
        nrm = np.cross(X[faces[:, 1]] - X[faces[:, 0]], X[faces[:, 2]] - X[faces[:, 0]])
        flip = np.einsum("ij,ij->i", nrm, X[faces].mean(1) - X[oppv]) < 0
        faces[flip] = faces[flip][:, [0, 2, 1]]
        self.faces = faces                                       # face i = surface triangle i
        area0 = 0.5 * np.linalg.norm(np.cross(X[faces[:, 1]] - X[faces[:, 0]], X[faces[:, 2]] - X[faces[:, 0]]), axis=1)

        # clamped nodes (all dofs), free dofs
        clamped = np.asarray(problem.clamped_tri, bool)
        fixed_nodes = np.unique(faces[clamped])
        fixed = np.zeros(self.ndof, bool)
        fixed[(3 * fixed_nodes[:, None] + np.arange(3)).ravel()] = True
        self.free = np.where(~fixed)[0]
        self.nfree = len(self.free)
        self.dof2free = -np.ones(self.ndof, np.int64)
        self.dof2free[self.free] = np.arange(self.nfree)

        # surface points are mesh nodes
        self.surf_node = np.asarray(problem.surface_nodes, np.int64)
        self.surf_pts = problem.reference
        if np.abs(X[self.surf_node] - self.surf_pts).max() > 1e-9:
            raise RuntimeError("surface points are not volume mesh nodes")
        self.surf_rows = (3 * self.surf_node[:, None] + np.arange(3)).ravel()

        self._build_sparsity()
        dev = self.device
        self.wX = wp.array(X, dtype=wp.vec3d, device=dev)
        self.wT = wp.array(T.astype(np.int32), dtype=wp.vec4i, device=dev)
        self.wBi = wp.array(np.linalg.inv(Dm), dtype=wp.mat33d, device=dev)
        self.wV = wp.array(det / 6.0, dtype=f64, device=dev)
        self.wF = wp.array(faces.astype(np.int32), dtype=wp.vec3i, device=dev)
        self.wf = wp.zeros(self.nn, dtype=wp.vec3d, device=dev)
        self.wKb = wp.zeros(len(T) * 16, dtype=wp.mat33d, device=dev)
        self.wFb = wp.zeros(len(faces) * 9, dtype=wp.mat33d, device=dev)
        self.wJ = wp.zeros(len(T), dtype=f64, device=dev)
        self.wfq = wp.zeros(len(faces), dtype=f64, device=dev)

        # cavity wall: same data as GetFEMSolver (wallN on the surface nodes, wallG scalar)
        self.wall = None
        if problem.has_wall:
            self.wall = WallDistance(problem.wall_points, problem.wall_tris)
            self.wallN = np.zeros((self.nn, 3))
            self.wallG = np.zeros(self.nn)
            self.wActive = wp.array((~clamped).astype(np.int32), dtype=wp.int32, device=dev)
            self.wArea0 = wp.array(area0, dtype=f64, device=dev)
            self.wQB = wp.array(FACE_QUAD_B, dtype=wp.vec3d, device=dev)
            self.wQW = wp.array(FACE_QUAD_W, dtype=f64, device=dev)
        log.info("Warp: %d nodes, %d tets, %d dofs (P1), %d clamped faces, wall %s, device %s, linear solver %s",
                 self.nn, len(T), self.ndof, clamped.sum(),
                 "on (stiffness %g, update %s)" % (o["wall_stiffness"], o["wall_update"]) if self.wall else "off",
                 dev, self.lin.backend)
        self.wall_in_newton = self.wall is not None and o["wall_update"] == "newton"

        self.labels, self.K = None, 0
        self.fq = np.zeros(len(faces))
        self.U, self.q_last, self.nu_last = None, None, None     # last converged state, its q and nu
        self.Ucur = np.zeros(self.ndof)                          # current Newton state
        self.nu = 0.3
        self.counts = dict(newton_calls=0, newton_iters=0, wall_updates=0, failed_paths=0, jacobians=0)
        self.times = dict(assembly_s=0.0, linear_s=0.0, assemblies=0, linear_solves=0)

    # ---- sparsity of the free-dof tangent (computed once) ----
    def _build_sparsity(self):
        T, Fc = self.tets, self.faces
        ii, jj = np.meshgrid(np.arange(3), np.arange(3), indexing="ij")
        a, b = np.meshgrid(np.arange(4), np.arange(4), indexing="ij")
        re = np.broadcast_to(3 * T[:, a.ravel()][:, :, None, None] + ii[None, None], (len(T), 16, 3, 3)).ravel()
        ce = np.broadcast_to(3 * T[:, b.ravel()][:, :, None, None] + jj[None, None], (len(T), 16, 3, 3)).ravel()
        r3, c3 = np.meshgrid(np.arange(3), np.arange(3), indexing="ij")
        rf = np.broadcast_to(3 * Fc[:, r3.ravel()][:, :, None, None] + ii[None, None], (len(Fc), 9, 3, 3)).ravel()
        cf = np.broadcast_to(3 * Fc[:, c3.ravel()][:, :, None, None] + jj[None, None], (len(Fc), 9, 3, 3)).ravel()
        fr, fc = self.dof2free[np.r_[re, rf]], self.dof2free[np.r_[ce, cf]]
        self.keep = (fr >= 0) & (fc >= 0)
        uniq, inv = np.unique(fr[self.keep] * self.nfree + fc[self.keep], return_inverse=True)
        self.inv = inv.ravel()
        self.indices = (uniq % self.nfree).astype(np.int64)
        self.indptr = np.r_[0, np.cumsum(np.bincount(uniq // self.nfree, minlength=self.nfree))].astype(np.int64)
        self.nnz = len(uniq)

    def _assemble(self, U, compute_K, pressure=True, wall=True, c=None):
        """(R over all dofs, tangent on the free dofs or None); (None, None) if a tet inverts."""
        t0 = time.time()
        c1, d1 = mat_params(self.nu) if c is None else c
        dev = self.device
        wU = wp.array(U.reshape(-1, 3), dtype=wp.vec3d, device=dev)
        self.wf.zero_()
        k = 1 if compute_K else 0
        if compute_K:
            self.wFb.zero_()
        wp.launch(elastic_kernel, dim=len(self.tets), device=dev,
                  inputs=[wU, self.wT, self.wBi, self.wV, f64(c1), f64(d1), k, self.wf, self.wKb, self.wJ])
        if pressure and np.any(self.fq != 0.0):
            self.wfq.assign(wp.array(self.fq, dtype=f64, device=dev))
            wp.launch(pressure_kernel, dim=len(self.faces), device=dev,
                      inputs=[self.wX, wU, self.wF, self.wfq, k, self.wf, self.wFb])
        if wall and self.wall is not None:
            e = float(self.options["wall_eps"])
            wp.launch(wall_kernel, dim=len(self.faces), device=dev,
                      inputs=[wU, self.wF, self.wActive, self.wArea0,
                              wp.array(self.wallN, dtype=wp.vec3d, device=dev),
                              wp.array(self.wallG, dtype=f64, device=dev),
                              self.wQB, self.wQW, len(FACE_QUAD_W),
                              f64(self.options["wall_stiffness"] / (2.0 * e)), f64(e), k, self.wf, self.wFb])
        ok = self.wJ.numpy().min() > 0.0
        R = self.wf.numpy().ravel().copy() if ok else None
        A = None
        if ok and compute_K:
            vals = np.r_[self.wKb.numpy().ravel(), self.wFb.numpy().ravel()][self.keep]
            A = sp.csr_matrix((np.bincount(self.inv, weights=vals, minlength=self.nnz), self.indices, self.indptr),
                              shape=(self.nfree, self.nfree))
        self.times["assembly_s"] += time.time() - t0
        self.times["assemblies"] += 1
        if ok and not np.all(np.isfinite(R)):
            return None, None
        return R, A

    def _linsolve(self, A, B):
        t0 = time.time()
        try:
            return np.asarray(self.lin.solve(A, B))
        finally:
            self.times["linear_s"] += time.time() - t0
            self.times["linear_solves"] += 1

    # ---- model per clustering level ----
    def set_regions(self, labels, n_regions):
        self.labels = np.asarray(labels)
        self.K = n_regions
        self.q_last = None                                       # q of another partition

    def _set_q(self, q):
        m = self.labels >= 0
        self.fq = np.zeros(len(self.faces))
        self.fq[m] = self.options["pressure_sign"] * np.asarray(q, float)[self.labels[m]]

    def _newton(self):
        """GetFEM classical Newton with the simplest line search on self.Ucur; True if converged."""
        o = self.options
        self.counts["newton_calls"] += 1
        f = self.free
        U = self.Ucur
        R, A = self._assemble_at(U, True)
        if R is None:
            return False
        res = np.abs(R[f]).sum()
        crit, it = res, 0
        try:
            while True:
                if crit <= o["newton_tol"]:
                    self.Ucur = U
                    return True
                if it >= o["newton_maxit"]:
                    return False
                try:
                    dx = self._linsolve(A, -R[f]).ravel()
                except Exception:
                    return False
                alpha, n_ls = 1.0, 0
                while True:                                      # simplest line search
                    n_ls += 1
                    conv_alpha = alpha
                    Ut = U.copy()
                    Ut[f] += alpha * dx
                    Rt, _ = self._assemble_at(Ut, False)
                    rt = np.abs(Rt[f]).sum() if Rt is not None else np.inf
                    if (n_ls <= 1 and rt < res) or rt <= LS_MAX_RATIO * res or conv_alpha <= LS_MIN_ALPHA:
                        break
                    alpha *= LS_MULT
                if not np.isfinite(rt):
                    return False
                U, it = Ut, it + 1
                R, A = self._assemble_at(U, True)
                if R is None:
                    return False
                res = np.abs(R[f]).sum()
                crit = min(res, np.abs(dx).sum() / max(1e-25, np.abs(U).sum()))
        finally:
            self.counts["newton_iters"] += it

    def _assemble_at(self, U, compute_K):
        """_assemble, with the wall re-linearised at U first when the contact is part of Newton."""
        if self.wall_in_newton:
            self._update_wall(U)
        return self._assemble(U, compute_K)

    def _update_wall(self, U=None):
        """Linearise the wall at U (default: the current state); returns the max change of the gap
        data [mm]. At the linearisation point g = u.wallN - wallG = signed distance - allowed margin."""
        Us = (self.Ucur if U is None else U).reshape(-1, 3)[self.surf_node]
        phi, n = self.wall(self.surf_pts + Us)
        G = np.einsum("ij,ij->i", n, Us) - phi + self.problem.wall_allow
        self.counts["wall_updates"] += 1
        old = self.wallG[self.surf_node].copy()
        self.wallN[self.surf_node] = n
        self.wallG[self.surf_node] = G
        return float(np.abs(G - old).max())

    def _solve(self):
        """Newton; with a wall, alternated with the wall re-linearisation until the gap settles."""
        if self.wall is None or self.wall_in_newton:
            return self._newton()
        self._update_wall()
        for _ in range(self.options["wall_max_updates"]):
            if not self._newton():
                return False
            if self._update_wall() < self.options["wall_settle_mm"]:
                break
        return True

    def _path(self, U0, q0, q, n):
        """n sub-steps from state (U0, q0) to q; True if all converged."""
        self.Ucur = np.array(U0, dtype=float)
        for s in range(1, n + 1):
            self._set_q(q0 + (q - q0) * s / n)
            if not self._solve():
                self.counts["failed_paths"] += 1
                return False
        return True

    def _full(self, q, nu):
        self.nu = nu
        n0 = self.options["load_steps"]
        if self.U is not None:                                   # from the last converged state
            q0 = q if self.q_last is None else self.q_last
            sub = self.q_last is not None and self.options["warm_substeps"]
            for n in ((1, n0, 3 * n0) if sub else (1,)):
                if self._path(self.U, q0, q, n):
                    return self.Ucur.copy()
        for n in (n0, 3 * n0):                                   # ramp from the reference
            if self._path(np.zeros(self.ndof), np.zeros_like(q), q, n):
                return self.Ucur.copy()
        return None

    def solve(self, q, nu):
        q = np.asarray(q, float)
        U = self._full(q, float(nu))
        if U is None:
            return None
        self.U, self.q_last, self.nu_last = U, q.copy(), float(nu)
        return U.reshape(-1, 3)[self.surf_node].copy()

    def jacobian(self, q, nu, with_nu=False):
        """dU/dp = -K_t^-1 dR/dp at the converged state, one factorisation for all columns.
        dR/dq_k = pressure_sign * (current area vector / 3) on the nodes of region k;
        dR/dnu = dc1/dnu f_int(c1=1, d1=0) + dd1/dnu f_int(c1=0, d1=1) (f_int is linear in c1, d1).
        With a wall, wallN / wallG are held fixed (Gauss-Newton Jacobian, as GetFEMSolver)."""
        q, nu = np.asarray(q, float), float(nu)
        if self.U is None or self.q_last is None or not np.array_equal(q, self.q_last) or nu != self.nu_last:
            return None
        self.nu = nu
        self._set_q(q)
        R, A = self._assemble_at(self.U, True)
        if A is None:
            return None
        x = self.X + self.U.reshape(-1, 3)
        Fc = self.faces
        Avec = 0.5 * np.cross(x[Fc[:, 1]] - x[Fc[:, 0]], x[Fc[:, 2]] - x[Fc[:, 0]]) / 3.0
        G = np.zeros((self.nn, 3, self.K))
        m = self.labels >= 0
        for r in range(3):
            np.add.at(G, (Fc[m, r], slice(None), self.labels[m]), self.options["pressure_sign"] * Avec[m])
        cols = [G.reshape(self.ndof, self.K)]
        if with_nu:
            f1, _ = self._assemble(self.U, False, pressure=False, wall=False, c=(1.0, 0.0))
            f2, _ = self._assemble(self.U, False, pressure=False, wall=False, c=(0.0, 1.0))
            dc1, dd1 = dmat_params_dnu(nu)
            cols.append((dc1 * f1 + dd1 * f2)[:, None])
        B = np.hstack(cols)[self.free]
        dU = np.zeros((self.ndof, B.shape[1]))
        dU[self.free] = -self._linsolve(A, B).reshape(self.nfree, -1)
        self.counts["jacobians"] += 1
        return dU[self.surf_rows]

    def stats(self):
        return dict(self.counts)

    def get_state(self):
        return None if self.U is None else self.U.copy()

    def set_state(self, state):
        self.U = None if state is None else np.asarray(state, dtype=float).copy()
        self.q_last = self.nu_last = None

    def reset(self):
        self.U, self.q_last, self.nu_last = None, None, None

    def export_volume(self, path, state):
        import pyvista as pv
        g = pv.UnstructuredGrid({pv.CellType.TETRA: self.tets}, self.X.copy())
        g.point_data["Displacement"] = np.asarray(state).reshape(-1, 3)
        g.save(str(path), binary=False)
