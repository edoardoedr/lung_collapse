"""NVIDIA Warp core: same physics, options and load path as GetFEMSolver (P1), on GPU or CPU.

Kernels from the former pipeline_codes_v1/lung_inverse_fem_fit_warp.py (git history) (validated there: tangent and
Jacobian vs finite differences). Material: GetFEM Compressible_Neo_Hookean only (common.py),
tangent derived by hand (nh_P, nh_dP). Residual, Newton, load path, wall rounds, linear solvers
and Jacobian: nodal.py (shared with the torch core); options documented there, plus
  device         Warp device (None = default: CUDA if available, else CPU)
"""

import logging

import numpy as np
import warp as wp

from .common import dmat_params_dnu, mat_params
from .nodal import LinearSolver, NodalSolver

log = logging.getLogger(__name__)
wp.config.quiet = True
f64 = wp.float64


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


class WarpSolver(NodalSolver):
    name = "warp"
    extra_defaults = {"device": None}

    def __init__(self, problem, options):
        super().__init__(problem, options)
        self.device = dev = wp.get_device(self.options["device"])
        self.lin = LinearSolver(self.options["linear_solver"], dev.is_cuda)
        T, faces = self.tets, self.faces
        self.wX = wp.array(self.X, dtype=wp.vec3d, device=dev)
        self.wT = wp.array(T.astype(np.int32), dtype=wp.vec4i, device=dev)
        self.wBi = wp.array(np.linalg.inv(self.Dm), dtype=wp.mat33d, device=dev)
        self.wV = wp.array(self.vol0, dtype=f64, device=dev)
        self.wF = wp.array(faces.astype(np.int32), dtype=wp.vec3i, device=dev)
        self.wf = wp.zeros(self.nn, dtype=wp.vec3d, device=dev)
        self.wKb = wp.zeros(len(T) * 16, dtype=wp.mat33d, device=dev)
        self.wFb = wp.zeros(len(faces) * 9, dtype=wp.mat33d, device=dev)
        self.wJ = wp.zeros(len(T), dtype=f64, device=dev)
        self.wfq = wp.zeros(len(faces), dtype=f64, device=dev)
        if self.wall is not None:
            self.wActive = wp.array((~self.clamped).astype(np.int32), dtype=wp.int32, device=dev)
            self.wArea0 = wp.array(self.area0, dtype=f64, device=dev)
            self.wQB = wp.array(self.qb, dtype=wp.vec3d, device=dev)
            self.wQW = wp.array(self.qw, dtype=f64, device=dev)
        self._log_setup(dev)

    def _assemble(self, U, compute_K, pressure=True, wall=True, c=None):
        """(R over all dofs, tangent on the free dofs or None); (None, None) if a tet inverts."""
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
                              self.wQB, self.wQW, len(self.qw),
                              f64(self.options["wall_stiffness"] / (2.0 * e)), f64(e), k, self.wf, self.wFb])
        ok = self.wJ.numpy().min() > 0.0
        R = self.wf.numpy().ravel().copy() if ok else None
        A = self._csr(self.wKb.numpy(), self.wFb.numpy()) if ok and compute_K else None
        if ok and not np.all(np.isfinite(R)):
            return None, None
        return R, A

    def _dfint_dnu(self, U):
        """f_int is linear in (c1, d1): dc1/dnu f_int(c1=1, d1=0) + dd1/dnu f_int(c1=0, d1=1)."""
        f1, _ = self._assemble(U, False, pressure=False, wall=False, c=(1.0, 0.0))
        f2, _ = self._assemble(U, False, pressure=False, wall=False, c=(0.0, 1.0))
        dc1, dd1 = dmat_params_dnu(self.nu)
        return dc1 * f1 + dd1 * f2
