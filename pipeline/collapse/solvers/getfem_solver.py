"""GetFEM core (port of pipeline_codes_v1/6.1 / 6.2_lung_inverse_fem_fit[_wall].py, class LungFEM).

options (fem_fit.solver_options):
  order             1 | 2, Lagrange order of the displacement (P2 slower, no volumetric locking)
  load_steps        pressure ramp from zero when a warm-started solve fails (retried with 3x the steps)
  warm_substeps     if the direct step from the last converged state fails, first try load_steps and
                    3x load_steps sub-steps from that state before the ramp from zero (as 6.2).
                    Helps with contact, costs failed Newton runs otherwise; false = as 6.1
  newton_tol        Newton residual tolerance
  newton_maxit      Newton iterations per load step
  pressure_sign     +1 / -1, sign of the pressure term (q > 0 must collapse)
  wall_stiffness    contact penalty [E per mm of penetration]; 20 -> ~0.05-0.1 mm residual penetration
  wall_eps          [mm] width of the smooth start of the penalty (no contact / contact chatter)
  wall_max_updates  Newton / wall re-linearisation rounds per solve
  wall_settle_mm    stop the rounds when the gap data changes less than this
  wall_update       "outer": Newton / wall re-linearisation rounds (the only mode here; "newton",
                    the wall updated at every Newton iteration, is implemented by the warp core)
  max_solve_s       None, or seconds after which one solve() gives up (counted as failed; checked
                    before each Newton run, so it can overrun by one Newton run)
  slow_ramp         true: when a solve fails also retry with 3 x load_steps (sub-)steps; false =
                    fail fast (as 6.2's default)
  linear_solver     None = GetFEM's choice (MUMPS if built with it, else SuperLU), or a name passed
                    to md.solve as "lsolver" (e.g. "mumps", "superlu")
"""

import logging
import time

import getfem as gf
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu
from scipy.spatial import cKDTree

from ..geometry import WallDistance
from .base import ForwardSolver
from .common import DEFAULTS, mat_params, ramps

log = logging.getLogger(__name__)

RID_CLAMPED, RID_WALL = 10, 11


def to_scipy(M):
    """gf.Spmat -> scipy CSC."""
    jc, ir = M.csc_ind()
    m, n = M.size()
    return csc_matrix((np.asarray(M.csc_val(), float), np.asarray(ir), np.asarray(jc)), shape=(m, n))


def sparse_solve(A, B):
    """A^-1 B with one factorisation for all the columns of B (pypardiso if installed)."""
    try:
        import pypardiso
        return pypardiso.spsolve(A.tocsr(), np.ascontiguousarray(B))
    except ImportError:
        return splu(A.tocsc()).solve(B)


class GetFEMSolver(ForwardSolver):
    name = "getfem"
    supports_wall = True

    def __init__(self, problem, options):
        unknown = sorted(set(options) - set(DEFAULTS))
        if unknown:
            raise ValueError("unknown getfem solver option(s): %s" % ", ".join(unknown))
        super().__init__(problem, {**DEFAULTS, **options})
        o = self.options
        if o["wall_update"] != "outer":
            raise ValueError("getfem solver: wall_update must be 'outer' ('newton' is for warp and torch)")
        try:
            gf.util_trace_level(0)
            gf.util_warning_level(0)
        except Exception:
            pass

        self.mesh = gf.Mesh("empty", 3)
        pts = problem.nodes[problem.tets]                       # (m, 4, 3)
        self.mesh.add_convex(gf.GeoTrans("GT_PK(3,1)"), np.transpose(pts, (2, 1, 0)))
        self.mfu = gf.MeshFem(self.mesh, 3)
        self.mfu.set_classical_fem(o["order"])
        self.mim = gf.MeshIm(self.mesh, gf.Integ("IM_TETRAHEDRON(%d)" % (3 if o["order"] == 1 else 5)))
        self.ndof = self.mfu.nbdof()

        # GetFEM outer faces -> problem surface triangles (by centroid, the mesh is the same)
        self.of = self.mesh.outer_faces()
        mpts = self.mesh.pts()
        cents = np.array([mpts[:, self.mesh.pid_in_faces(self.of[:, [i]])].mean(axis=1)
                          for i in range(self.of.shape[1])])
        tri_cent = problem.nodes[problem.faces].mean(axis=1)
        d, self.tri_of_face = cKDTree(tri_cent).query(cents)
        if self.of.shape[1] != len(problem.tris) or d.max() > 1e-6:
            raise RuntimeError("GetFEM boundary (%d faces) does not match the surface (%d triangles, "
                               "max centroid distance %.2e mm)" % (self.of.shape[1], len(problem.tris), d.max()))
        clamped = problem.clamped_tri[self.tri_of_face]
        self.mesh.set_region(RID_CLAMPED, self.of[:, clamped])
        self.fixed_dofs = np.asarray(self.mfu.basic_dof_on_region(RID_CLAMPED), dtype=int)
        self.free = np.setdiff1d(np.arange(self.ndof), self.fixed_dofs)
        self._gsign = None

        # surface points -> their 3 vector dofs (vertex dofs sit on the mesh nodes)
        self.surf_pts = problem.reference
        dd, idx = cKDTree(self.mfu.basic_dof_nodes().T).query(self.surf_pts, k=3)
        idx = np.sort(idx, axis=1)
        self.surf_dof = idx
        self.use_interp = not (dd.max() < 1e-4 and np.all(np.diff(idx, axis=1) == 1)
                               and np.all(idx[:, 0] % 3 == 0))
        if self.use_interp:
            log.warning("surface points are not dofs: using compute_interpolate_on")

        # cavity wall: penalty contact on the pressure faces, gap linearised at the current state
        self.wall = None
        if problem.has_wall:
            if self.use_interp:
                raise RuntimeError("the wall needs the surface points on dofs")
            self.wall = WallDistance(problem.wall_points, problem.wall_tris)
            self.mfs = gf.MeshFem(self.mesh, 1)
            self.mfs.set_classical_fem(o["order"])
            self.surf_sdof = idx[:, 0] // 3
            self.mesh.set_region(RID_WALL, self.of[:, ~clamped])
        log.info("GetFEM: %d nodes, %d tets, %d dofs (P%d), %d clamped faces, wall %s",
                 self.mesh.nbpts(), self.mesh.nbcvs(), self.ndof, o["order"], clamped.sum(),
                 "on (stiffness %g)" % o["wall_stiffness"] if self.wall else "off")

        self.md, self.K, self.cur_nu = None, 0, None
        self.U, self.q_last, self.nu_last = None, None, None     # last converged state, its q and nu
        self.n_builds = 0
        self.counts = dict(newton_calls=0, newton_iters=0, wall_updates=0, failed_paths=0, jacobians=0)
        self._deadline = np.inf

    def set_regions(self, labels, n_regions):
        self.n_builds += 1
        base = 1000 * self.n_builds                              # fresh region ids per level
        face_labels = labels[self.tri_of_face]
        md = gf.Model("real")
        md.add_fem_variable("u", self.mfu)
        md.add_initialized_data("params", mat_params(0.3))
        md.add_finite_strain_elasticity_brick(self.mim, "Compressible_Neo_Hookean", "u", "params")
        md.add_Dirichlet_condition_with_simplification("u", RID_CLAMPED)
        F = "(Id(meshdim)+Grad_u)"
        self.base = base
        # pressure term with q = 1: d(residual)/dq_k up to the sign GetFEM gives md.rhs()
        self.g_expr = "(%g)*Det(%s)*((Inv(%s))'*Normal).Test_u" % (self.options["pressure_sign"], F, F)
        self._gsign = None
        for k in range(n_regions):
            self.mesh.set_region(base + k, self.of[:, face_labels == k])
            md.add_initialized_data("q%d" % k, [0.0])
            # follower pressure (Nanson)
            expr = "(%g)*q%d*Det(%s)*((Inv(%s))'*Normal).Test_u" % (self.options["pressure_sign"], k, F, F)
            if hasattr(md, "add_nonlinear_term"):
                md.add_nonlinear_term(self.mim, expr, base + k)
            else:
                md.add_nonlinear_generic_assembly_brick(self.mim, expr, base + k)
        if self.wall is not None:
            md.add_initialized_fem_data("wallN", self.mfu, np.zeros(self.ndof))
            md.add_initialized_fem_data("wallG", self.mfs, np.zeros(self.mfs.nbdof()))
            # penalty k*f(g), g = penetration beyond the allowed position, f a C1 ramp:
            # 0 (g < 0), g^2/(2e) (0 < g < e), g - e/2 (g > e)
            g, e = "(u.wallN - wallG)", self.options["wall_eps"]
            md.add_nonlinear_term(self.mim, "(%g)*(sqr(pos_part(%s)) - sqr(pos_part(%s - %g)))*(wallN.Test_u)"
                                  % (self.options["wall_stiffness"] / (2 * e), g, g, e), RID_WALL)
        self.md, self.K, self.cur_nu = md, n_regions, 0.3
        self.q_last = None                                       # q of another partition

    def _set_q(self, q):
        for k in range(self.K):
            self.md.set_variable("q%d" % k, [float(q[k])])

    def _newton(self):
        o = self.options
        if time.time() > self._deadline:
            self.counts["fail_time"] = self.counts.get("fail_time", 0) + 1
            return False
        self.counts["newton_calls"] += 1
        try:
            args = ("max_iter", o["newton_maxit"], "max_res", o["newton_tol"], "lsearch", "simplest")
            if o["linear_solver"]:
                args += ("lsolver", o["linear_solver"])
            r = self.md.solve(*args)
            conv = bool(r[1]) if isinstance(r, (tuple, list)) and len(r) > 1 else True
        except Exception:
            return False
        try:                                                     # (iterations, converged)
            self.counts["newton_iters"] += int(r[0])
        except (TypeError, ValueError, IndexError):
            pass
        return conv and np.all(np.isfinite(self.md.variable("u")))

    def _update_wall(self):
        """Linearise the wall at the current u; returns the max change of the gap data [mm]."""
        U = self.md.variable("u")
        Us = U[self.surf_dof]
        phi, n = self.wall(self.surf_pts + Us)
        G = np.einsum("ij,ij->i", n, Us) - phi + self.problem.wall_allow
        N = np.zeros(self.ndof)
        N[self.surf_dof] = n
        Gs = np.zeros(self.mfs.nbdof())
        Gs[self.surf_sdof] = G
        self.counts["wall_updates"] += 1
        old = self.md.variable("wallG")[self.surf_sdof]
        self.md.set_variable("wallN", N)
        self.md.set_variable("wallG", Gs)
        return float(np.abs(G - old).max())

    def _solve(self):
        """Newton; with a wall, alternated with the wall re-linearisation until the gap settles."""
        if self.wall is None:
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
        self.md.set_variable("u", U0)
        for s in range(1, n + 1):
            self._set_q(q0 + (q - q0) * s / n)
            if not self._solve():
                self.counts["failed_paths"] += 1
                return False
        return True

    def _full(self, q, nu):
        if nu != self.cur_nu:
            self.md.set_variable("params", mat_params(nu))
            self.cur_nu = nu
        o = self.options
        self._deadline = time.time() + (o["max_solve_s"] or np.inf)
        q0 = q if self.q_last is None else self.q_last
        sub = self.q_last is not None and o["warm_substeps"]
        for warm, n in ramps(o["load_steps"], self.U is not None, sub, o["slow_ramp"]):
            if warm and self._path(self.U, q0, q, n):            # from the last converged state
                return self.md.variable("u").copy()
            if not warm and self._path(np.zeros(self.ndof), np.zeros_like(q), q, n):   # from the reference
                return self.md.variable("u").copy()
        return None

    def solve(self, q, nu):
        q = np.asarray(q, float)
        U = self._full(q, float(nu))
        if U is None:
            return None
        self.U, self.q_last, self.nu_last = U, q.copy(), float(nu)
        if not self.use_interp:
            return U[self.surf_dof]
        return np.asarray(gf.compute_interpolate_on(self.mfu, U, self.surf_pts.T)).reshape(3, -1).T

    def jacobian(self, q, nu, with_nu=False):
        """Sensitivities at the converged state: K_t dU/dp = d(rhs)/dp on the free dofs, with
        one factorisation of the tangent matrix; no nonlinear solve. The rhs is linear in each
        q_k, so its derivative is the pressure term assembled with q_k = 1 (sign calibrated once
        per level against an actual rhs difference). With a wall, wallN / wallG are held fixed
        (Gauss-Newton Jacobian: the gap re-linearisation is not differentiated)."""
        q, nu = np.asarray(q, float), float(nu)
        if (self.use_interp or self.U is None or self.q_last is None
                or not np.array_equal(q, self.q_last) or nu != self.nu_last):
            return None
        md, f = self.md, self.free
        md.set_variable("u", self.U)
        self._set_q(q)
        if nu != self.cur_nu:
            md.set_variable("params", mat_params(nu))
            self.cur_nu = nu
        md.assembly("build_all")
        Kt = to_scipy(md.tangent_matrix())
        rhs0 = np.array(md.rhs(), dtype=float)
        if Kt.shape != (self.ndof, self.ndof) or len(rhs0) != self.ndof:
            raise RuntimeError("tangent %s / rhs %d do not match the %d dofs of u"
                               % (Kt.shape, len(rhs0), self.ndof))
        B = np.column_stack([np.asarray(gf.asm_generic(self.mim, 1, self.g_expr, self.base + k, md),
                                        dtype=float)[:self.ndof] for k in range(self.K)])
        if self._gsign is None:                                  # d(rhs)/dq_0 from a unit step
            e0 = np.zeros(self.K)
            e0[0] = 1.0
            self._set_q(q + e0)
            md.assembly("build_rhs")
            d = np.array(md.rhs(), dtype=float)[f] - rhs0[f]
            self._set_q(q)
            s = 1.0 if d @ B[f, 0] >= 0 else -1.0
            mismatch = np.linalg.norm(d - s * B[f, 0]) / max(np.linalg.norm(d), 1e-300)
            if mismatch > 1e-8:
                raise RuntimeError("pressure column does not match d(rhs)/dq (relative %.2e)" % mismatch)
            self._gsign = s
        B *= self._gsign
        if with_nu:                                              # central difference of the rhs only
            h, r = 1e-6, []
            for dn in (h, -h):
                md.set_variable("params", mat_params(nu + dn))
                md.assembly("build_rhs")
                r.append(np.array(md.rhs(), dtype=float))
            md.set_variable("params", mat_params(nu))
            B = np.column_stack([B, (r[0] - r[1]) / (2 * h)])
        dU = np.zeros((self.ndof, B.shape[1]))
        dU[f] = sparse_solve(Kt.tocsr()[f][:, f], B[f]).reshape(len(f), -1)
        self.counts["jacobians"] += 1
        return dU[self.surf_dof.ravel()]

    def stats(self):
        return dict(self.counts)

    def get_state(self):
        return None if self.U is None else self.U.copy()

    def set_state(self, state):
        self.U = None if state is None else np.asarray(state).copy()
        self.q_last = self.nu_last = None

    def reset(self):
        self.U, self.q_last, self.nu_last = None, None, None

    def export_volume(self, path, state):
        self.mfu.export_to_vtk(str(path), "ascii", self.mfu, state, "Displacement")
