"""Shared by the P1 cores that assemble the system themselves (warp, torch): everything except the
assembly. Same physics, options and load path as GetFEMSolver; residual convention R(U) = 0 with

    R = f_int(U)                                      hyperelastic material (GetFEM default: common.py)
      + pressure_sign * q_k * J F^-T N               per region (P1 flat face: current area vector / 3)
      + (k/2e) (pos(g)^2 - pos(g-e)^2) wallN         wall penalty on the non-clamped faces, g = u.wallN - wallG

u = 0 on the nodes of the clamped faces (eliminated). Everything in float64.

A core subclasses NodalSolver and implements _assemble (residual over all dofs and tangent blocks,
see _csr for their layout) and _dfint_dnu. Here: mesh preparation, sparsity, linear solvers,
GetFEM-like Newton with the load path and the wall rounds, Jacobian dU/d(q, nu), state, export.

options: the GetFEMSolver ones (order must be 1), plus
  wall_update    "outer" (as GetFEMSolver): Newton solves with the wall linearisation fixed,
                 alternated with re-linearisations until the gap settles (wall_max_updates,
                 wall_settle_mm). "newton": the wall (normal, gap) is re-linearised at every
                 residual evaluation inside one Newton solve, so the contact is part of Newton
                 and the outer rounds disappear; at convergence it solves the same equations
                 (gap = signed distance - allowed margin), without the wall_settle_mm tolerance
  linear_solver  auto | cudss (nvmath-python, CUDA) | pardiso (pypardiso) | scipy (SuperLU); None = auto

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

from ..geometry import WallDistance
from .base import ForwardSolver
from .common import DEFAULTS as GETFEM_DEFAULTS

log = logging.getLogger(__name__)


class _NoCudssThreadingWarning(logging.Filter):
    """nvmath logs this on the root logger at every cuDSS factorisation; it only concerns the
    speed of cuDSS host-side planning."""
    def filter(self, record):
        return "No multithreading interface library" not in record.getMessage()


logging.getLogger().addFilter(_NoCudssThreadingWarning())

DEFAULTS = {**GETFEM_DEFAULTS, "linear_solver": None}

# IM_TRIANGLE(3): barycentric points, weights normalised to the face area
FACE_QUAD_B = np.array([[1 / 3, 1 / 3, 1 / 3], [0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6]])
FACE_QUAD_W = np.array([-27.0, 25.0, 25.0, 25.0]) / 48.0

# GetFEM simplest_newton_line_search defaults
LS_MAX_RATIO, LS_MIN_ALPHA, LS_MULT = 1.5, 1e-3, 0.6


class LinearSolver:
    """Sparse direct solve A X = B (B: vector or matrix), A in CSR, on the host (numpy in and out)."""

    def __init__(self, backend, cuda):
        self.backend = backend or "auto"
        if self.backend == "auto":
            self.backend = "scipy"
            try:
                import pypardiso  # noqa: F401
                self.backend = "pardiso"
            except ImportError:
                pass
            if cuda:
                try:
                    import nvmath  # noqa: F401
                    self.backend = "cudss"
                except ImportError:
                    pass
        if self.backend not in ("cudss", "pardiso", "scipy"):
            raise ValueError("linear_solver must be auto, cudss, pardiso or scipy")
        if self.backend == "cudss":                     # fail here, not as a Newton failure later
            if not cuda:
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


class NodalSolver(ForwardSolver):
    """P1 core skeleton; subclasses set `name`, `extra_defaults` and implement the assembly."""

    supports_wall = True
    extra_defaults = {}

    def __init__(self, problem, options):
        defaults = {**DEFAULTS, **self.extra_defaults}
        unknown = sorted(set(options) - set(defaults))
        if unknown:
            raise ValueError("unknown %s solver option(s): %s" % (self.name, ", ".join(unknown)))
        super().__init__(problem, {**defaults, **options})
        o = self.options
        if o["order"] != 1:
            raise NotImplementedError("the %s solver implements P1 only (solver_options.order = 1)" % self.name)
        if o["wall_update"] not in ("outer", "newton"):
            raise ValueError("%s solver: wall_update must be 'outer' or 'newton'" % self.name)

        # mesh, positively oriented tets
        X = np.asarray(problem.nodes, float)
        T = np.asarray(problem.tets, np.int64).copy()
        def edges(T):
            return np.stack([X[T[:, 1]] - X[T[:, 0]], X[T[:, 2]] - X[T[:, 0]], X[T[:, 3]] - X[T[:, 0]]], axis=2)
        neg = np.linalg.det(edges(T)) < 0
        T[neg] = T[neg][:, [0, 2, 1, 3]]
        self.Dm = edges(T)                                       # columns = edges from node 0
        det = np.linalg.det(self.Dm)
        if np.any(det <= 0):
            raise RuntimeError("degenerate tetrahedra in the volume mesh")
        self.vol0 = det / 6.0
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
            raise RuntimeError("%s boundary (%d faces) does not match the surface (%d triangles)"
                               % (self.name, len(okey), len(faces)))
        opp = dict(zip(map(tuple, okey), allo[outer]))
        oppv = np.array([opp[tuple(k)] for k in np.sort(faces, axis=1)])
        nrm = np.cross(X[faces[:, 1]] - X[faces[:, 0]], X[faces[:, 2]] - X[faces[:, 0]])
        flip = np.einsum("ij,ij->i", nrm, X[faces].mean(1) - X[oppv]) < 0
        faces[flip] = faces[flip][:, [0, 2, 1]]
        self.faces = faces                                       # face i = surface triangle i
        self.area0 = 0.5 * np.linalg.norm(np.cross(X[faces[:, 1]] - X[faces[:, 0]],
                                                   X[faces[:, 2]] - X[faces[:, 0]]), axis=1)

        # clamped nodes (all dofs), free dofs
        self.clamped = np.asarray(problem.clamped_tri, bool)
        fixed_nodes = np.unique(faces[self.clamped])
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

        # cavity wall: same data as GetFEMSolver (wallN on the surface nodes, wallG scalar)
        self.wall = None
        if problem.has_wall:
            self.wall = WallDistance(problem.wall_points, problem.wall_tris)
            self.wallN = np.zeros((self.nn, 3))
            self.wallG = np.zeros(self.nn)
        self.wall_in_newton = self.wall is not None and o["wall_update"] == "newton"

        self.labels, self.K = None, 0
        self.fq = np.zeros(len(faces))
        self.U, self.q_last, self.nu_last = None, None, None     # last converged state, its q and nu
        self.Ucur = np.zeros(self.ndof)                          # current Newton state
        self.nu = 0.3
        self.counts = dict(newton_calls=0, newton_iters=0, wall_updates=0, failed_paths=0, jacobians=0)
        self.times = dict(assembly_s=0.0, linear_s=0.0, assemblies=0, linear_solves=0)

    def _log_setup(self, device, extra=""):
        log.info("%s: %d nodes, %d tets, %d dofs (P1), %d clamped faces, wall %s, device %s, linear solver %s%s",
                 self.name, self.nn, len(self.tets), self.ndof, self.clamped.sum(),
                 "on (stiffness %g, update %s)" % (self.options["wall_stiffness"], self.options["wall_update"])
                 if self.wall else "off", device, self.lin.backend, extra)

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

    def _csr(self, Kb, Fb):
        """Free-dof tangent from the element blocks: Kb (n_tets, 4, 4, 3, 3) with
        Kb[t, a, b, i, j] = dR[node a, i] / dU[node b, j], and Fb (n_faces, 3, 3, 3, 3) the same
        for the face nodes (pressure + wall). Flattened arrays in this order are accepted too."""
        vals = np.r_[np.asarray(Kb).ravel(), np.asarray(Fb).ravel()][self.keep]
        return sp.csr_matrix((np.bincount(self.inv, weights=vals, minlength=self.nnz), self.indices, self.indptr),
                             shape=(self.nfree, self.nfree))

    # ---- what a core implements ----
    def _assemble(self, U, compute_K, pressure=True, wall=True):
        """(R over all dofs, tangent on the free dofs (_csr) or None); (None, None) if a tet
        inverts or R is not finite. Pressures: self.fq per face; wall: self.wallN / self.wallG."""
        raise NotImplementedError

    def _dfint_dnu(self, U):
        """d f_int / d nu at U (all dofs)."""
        raise NotImplementedError

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
        t0 = time.time()
        try:
            return self._assemble(U, compute_K)
        finally:
            self.times["assembly_s"] += time.time() - t0
            self.times["assemblies"] += 1

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
        dR/dnu = d f_int / d nu (_dfint_dnu).
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
            cols.append(np.asarray(self._dfint_dnu(self.U))[:, None])
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
