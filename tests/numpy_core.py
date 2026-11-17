"""Test-only FEM core in numpy: the same compressible Neo-Hookean P1 model as the torch / Warp
cores (common.mat_params), assembled on the CPU with a complex-step tangent. Slow but exact, and
it needs nothing the Mac lacks, so the shared code of solvers/nodal.py (Newton, load path, wall
potential, Jacobian) can be run locally on a real problem.npz."""

import numpy as np

import stubs  # noqa: F401
from eurelaps_lung.fem.solvers.common import dmat_params_dnu, mat_params
from eurelaps_lung.fem.solvers.nodal import LinearSolver, NodalSolver


def _skew(v):
    z = np.zeros(len(v))
    return np.stack([np.stack([z, -v[:, 2], v[:, 1]], 1), np.stack([v[:, 2], z, -v[:, 0]], 1),
                     np.stack([-v[:, 1], v[:, 0], z], 1)], 1)


class NumpySolver(NodalSolver):
    name = "numpy"

    def __init__(self, problem, options):
        super().__init__(problem, options)
        self.lin = LinearSolver("scipy", False)
        self.Bm = np.linalg.inv(self.Dm)
        # shape-function gradients: nodes 1..3 = rows of Dm^-1, node 0 = minus their sum
        G = np.zeros((len(self.tets), 4, 3))
        G[:, 1:] = self.Bm
        G[:, 0] = -self.Bm.sum(1)
        self.G = G

    def _F(self, U):
        x = self.X + U.reshape(-1, 3)
        return np.einsum("tai,taK->tiK", x[self.tets], self.G)

    @staticmethod
    def _P(F, c1, d1):
        J = np.linalg.det(F)
        FiT = np.linalg.inv(F).transpose(0, 2, 1)
        I1 = (F * F).sum((1, 2))
        j = J ** (-2.0 / 3.0)
        return (c1 * j)[:, None, None] * (2.0 * F - (2.0 / 3.0) * I1[:, None, None] * FiT) \
            + (2.0 * d1 * (J - 1.0) * J)[:, None, None] * FiT

    def _fint(self, F, c):
        f = np.zeros((self.nn, 3))
        np.add.at(f, self.tets, np.einsum("t,tiK,taK->tai", self.vol0, self._P(F, *c), self.G))
        return f

    def _assemble(self, U, compute_K, pressure=True, wall=True, c=None):
        if wall and self.wall is not None:
            raise NotImplementedError("the numpy test core has only the potential wall")
        F = self._F(U)
        if not np.all(np.linalg.det(F) > 0):
            return None, None
        c = mat_params(self.nu) if c is None else c
        f = self._fint(F, c)
        x = self.X + U.reshape(-1, 3)
        Fc = self.faces
        a, b = x[Fc[:, 1]] - x[Fc[:, 0]], x[Fc[:, 2]] - x[Fc[:, 0]]
        if pressure:
            np.add.at(f, Fc, (self.fq[:, None] * np.cross(a, b) / 6.0)[:, None, :].repeat(3, 1))
        R = f.ravel()
        if not np.all(np.isfinite(R)):
            return None, None
        if not compute_K:
            return R, None
        dP = np.zeros((len(F), 3, 3, 3, 3))                        # dP[i, K, j, L]
        e = 1e-30
        for j in range(3):
            for L in range(3):
                Fc_ = F.astype(complex)
                Fc_[:, j, L] += 1j * e
                dP[:, :, :, j, L] = self._P(Fc_, *c).imag / e
        Kb = np.einsum("t,tiKjL,taK,tbL->tabij", self.vol0, dP, self.G, self.G)
        Fb = np.zeros((len(Fc), 3, 3, 3, 3))
        if pressure:
            dn = [_skew(b) - _skew(a), -_skew(b), _skew(a)]       # d(a x b) / d x_s
            for s in range(3):
                Fb[:, :, s] = (self.fq[:, None, None] / 6.0 * dn[s])[:, None]
        return R, self._csr(Kb, Fb)

    def _dfint_dnu(self, U):
        F = self._F(U)
        d = dmat_params_dnu(self.nu)
        return (d[0] * self._fint(F, (1.0, 0.0)) + d[1] * self._fint(F, (0.0, 1.0))).ravel()
