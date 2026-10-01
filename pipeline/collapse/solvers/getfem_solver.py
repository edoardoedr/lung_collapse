"""GetFEM core (port of pipeline_codes_v1/6.1 / 6.2_lung_inverse_fem_fit[_wall].py, class LungFEM).

options (fem_fit.solver_options):
  order             1 | 2, Lagrange order of the displacement (P2 slower, no volumetric locking)
  load_steps        sub-steps when a direct solve fails (retried with 3x the steps)
  newton_tol        Newton residual tolerance
  newton_maxit      Newton iterations per load step
  pressure_sign     +1 / -1, sign of the pressure term (q > 0 must collapse)
  wall_stiffness    contact penalty [E per mm of penetration]; 20 -> ~0.05-0.1 mm residual penetration
  wall_eps          [mm] width of the smooth start of the penalty (no contact / contact chatter)
  wall_max_updates  Newton / wall re-linearisation rounds per solve
  wall_settle_mm    stop the rounds when the gap data changes less than this
"""

import logging

import getfem as gf
import numpy as np
from scipy.spatial import cKDTree

from ..geometry import WallDistance
from .base import ForwardSolver

log = logging.getLogger(__name__)

DEFAULTS = dict(order=1, load_steps=4, newton_tol=1e-7, newton_maxit=30, pressure_sign=1.0,
                wall_stiffness=20.0, wall_eps=0.5, wall_max_updates=6, wall_settle_mm=0.05)
RID_CLAMPED, RID_WALL = 10, 11


def mat_params(nu):
    """GetFEM 'Compressible_Neo_Hookean' takes [c1, d1] = [mu/2, K/2]; here E = 1."""
    return [1.0 / (4.0 * (1.0 + nu)), 1.0 / (6.0 * (1.0 - 2.0 * nu))]


class GetFEMSolver(ForwardSolver):
    name = "getfem"
    supports_wall = True

    def __init__(self, problem, options):
        unknown = sorted(set(options) - set(DEFAULTS))
        if unknown:
            raise ValueError("unknown getfem solver option(s): %s" % ", ".join(unknown))
        super().__init__(problem, {**DEFAULTS, **options})
        o = self.options
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

        # surface points -> their 3 vector dofs (vertex dofs sit on the mesh nodes)
        self.surf_pts = problem.reference
        dd, idx = cKDTree(self.mfu.basic_dof_nodes().T).query(self.surf_pts, k=3)
        idx = np.sort(idx, axis=1)
        self.surf_dof = idx
        self.use_interp = not (dd.max() < 1e-6 and np.all(np.diff(idx, axis=1) == 1)
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
        self.U, self.q_last = None, None                         # last converged state and its q
        self.n_builds = 0

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
        try:
            r = self.md.solve("max_iter", o["newton_maxit"], "max_res", o["newton_tol"], "lsearch", "simplest")
            conv = bool(r[1]) if isinstance(r, (tuple, list)) and len(r) > 1 else True
        except Exception:
            return False
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
                return False
        return True

    def _full(self, q, nu):
        if nu != self.cur_nu:
            self.md.set_variable("params", mat_params(nu))
            self.cur_nu = nu
        n0 = self.options["load_steps"]
        if self.U is not None:                                   # from the last converged state
            q0 = q if self.q_last is None else self.q_last
            for n in ((1,) if self.q_last is None else (1, n0, 3 * n0)):
                if self._path(self.U, q0, q, n):
                    return self.md.variable("u").copy()
        for n in (n0, 3 * n0):                                   # ramp from the reference
            if self._path(np.zeros(self.ndof), np.zeros_like(q), q, n):
                return self.md.variable("u").copy()
        return None

    def solve(self, q, nu):
        q = np.asarray(q, float)
        U = self._full(q, float(nu))
        if U is None:
            return None
        self.U, self.q_last = U, q.copy()
        if not self.use_interp:
            return U[self.surf_dof]
        return np.asarray(gf.compute_interpolate_on(self.mfu, U, self.surf_pts.T)).reshape(3, -1).T

    def get_state(self):
        return None if self.U is None else self.U.copy()

    def set_state(self, state):
        self.U = None if state is None else np.asarray(state).copy()
        self.q_last = None

    def reset(self):
        self.U, self.q_last = None, None

    def export_volume(self, path, state):
        self.mfu.export_to_vtk(str(path), "ascii", self.mfu, state, "Displacement")
