"""PyTorch core: same physics, options and load path as GetFEMSolver / WarpSolver (P1), with the
material given only as a strain-energy density (materials.py): stress and tangent come from
automatic differentiation (torch.func), so a new material needs no derivation.

Residual, Newton, load path, wall rounds, linear solvers and Jacobian: nodal.py (shared with the
Warp core); options documented there, plus
  device           "cpu" | "cuda" | "cuda:N" | None (= cuda if available, else cpu). The solve
                   needs float64 and a sparse direct solver, so "mps" (Apple GPU, float32 only)
                   is refused here; TorchFEM below runs on mps for residual-only uses
  material         name in materials.MATERIALS (default "neo_hookean" = GetFEM / Warp)
  material_params  dict of the material's fixed parameters (e.g. {"c01_fraction": 0.3})

TorchFEM is the assembly alone, on any device and dtype and differentiable with respect to U:
e.g. the residual of a GNN prediction as a physics loss (no linear solve, float32 / mps fine).
"""

import functools
import logging

import numpy as np
import torch
from torch.func import grad, jacfwd, jvp, vmap

from .materials import MATERIALS, det3
from .nodal import LinearSolver, NodalSolver

log = logging.getLogger(__name__)


def get_material(name, params=None):
    if name not in MATERIALS:
        raise ValueError("unknown material '%s' (available: %s)" % (name, ", ".join(MATERIALS)))
    return functools.partial(MATERIALS[name], **(params or {}))


def _skew(v):
    z = torch.zeros_like(v[..., 0])
    return torch.stack([torch.stack([z, -v[..., 2], v[..., 1]], -1),
                        torch.stack([v[..., 2], z, -v[..., 0]], -1),
                        torch.stack([-v[..., 1], v[..., 0], z], -1)], -2)


class TorchFEM:
    """P1 hyperelastic assembly on a fixed tetrahedral mesh (no boundary conditions, no wall).

    nodes (N, 3), tets (E, 4) positively oriented, faces (F, 3) outward; U (N, 3) displacement
    tensor. residual(U, fq, nu) = f_int + follower pressure forces, (N, 3), differentiable."""

    def __init__(self, nodes, tets, faces, material="neo_hookean", material_params=None,
                 device="cpu", dtype=torch.float64):
        self.device, self.dtype = torch.device(device), dtype
        t = functools.partial(torch.as_tensor, device=self.device)
        X = np.asarray(nodes, float)
        T = np.asarray(tets, np.int64)
        Dm = np.stack([X[T[:, 1]] - X[T[:, 0]], X[T[:, 2]] - X[T[:, 0]], X[T[:, 3]] - X[T[:, 0]]], axis=2)
        Bi = np.linalg.inv(Dm)                                  # rows = gradients of N1..N3
        Gr = np.concatenate([-Bi.sum(axis=1, keepdims=True), Bi], axis=1)   # (E, 4, 3), N0 = 1 - N1 - N2 - N3
        self.X = t(X, dtype=dtype)
        self.T = t(T)
        self.faces = t(np.asarray(faces, np.int64))
        self.G = t(Gr, dtype=dtype)
        self.V = t(np.linalg.det(Dm) / 6.0, dtype=dtype)
        self.nn = len(X)
        self.W = get_material(material, material_params)
        self._P = vmap(grad(self.W), in_dims=(0, None))                  # dW/dF
        self._C = vmap(jacfwd(grad(self.W)), in_dims=(0, None))          # d2W/dF2

    def tensor(self, a):
        return torch.as_tensor(a, dtype=self.dtype, device=self.device)

    def deformation_gradient(self, U):
        """F (E, 3, 3) and J (E,)."""
        F = torch.eye(3, dtype=self.dtype, device=self.device) + torch.einsum("eai,eaj->eij", U[self.T], self.G)
        return F, vmap(det3)(F)

    def _scatter(self, idx, vals):
        return torch.zeros(self.nn, 3, dtype=vals.dtype, device=self.device).index_add_(
            0, idx.reshape(-1), vals.reshape(-1, 3))

    def internal_forces(self, U, nu, F=None):
        """f_int (N, 3) = sum_e V_e P_e grad N_a."""
        if F is None:
            F, _ = self.deformation_gradient(U)
        P = self._P(F, self.tensor(nu))
        return self._scatter(self.T, self.V[:, None, None] * torch.einsum("eij,eaj->eai", P, self.G))

    def pressure_forces(self, U, fq):
        """fq (F,) pressure per face (sign included): fq/3 * current area vector on each face node."""
        x = (self.X + U)[self.faces]
        A = 0.5 * torch.linalg.cross(x[:, 1] - x[:, 0], x[:, 2] - x[:, 0], dim=-1)
        w = (self.tensor(fq) / 3.0)[:, None, None]
        return self._scatter(self.faces, (w * A[:, None, :]).expand(-1, 3, -1))

    def residual(self, U, fq, nu):
        return self.internal_forces(U, nu) + self.pressure_forces(U, fq)

    # ---- tangent blocks (solver use) ----
    def elastic_blocks(self, F, nu):
        """Kb (E, 4, 4, 3, 3): V sum_jl dP_ij/dF_kl gradN_a,j gradN_b,l."""
        C = self._C(F, self.tensor(nu))
        return self.V[:, None, None, None, None] * torch.einsum("eijkl,eaj,ebl->eabik", C, self.G, self.G)

    def pressure_blocks(self, U, fq):
        """(F, 3, 3, 3, 3): d(force on node r)/d(x of node s), the same for the three r."""
        x = (self.X + U)[self.faces]
        w = (self.tensor(fq) / 3.0)[:, None, None]
        S = torch.stack([_skew(0.5 * (x[:, 2] - x[:, 1])), _skew(0.5 * (x[:, 0] - x[:, 2])),
                         _skew(0.5 * (x[:, 1] - x[:, 0]))], 1) * w[..., None]       # (F, s, 3, 3)
        return S[:, None].expand(-1, 3, -1, -1, -1)


class TorchSolver(NodalSolver):
    name = "torch"
    extra_defaults = {"device": None, "material": "neo_hookean", "material_params": {}}

    def __init__(self, problem, options):
        super().__init__(problem, options)
        o = self.options
        dev = o["device"] or ("cuda" if torch.cuda.is_available() else "cpu")
        if torch.device(dev).type not in ("cpu", "cuda"):
            raise ValueError("torch solver: device must be cpu or cuda, not '%s' (the solve needs float64 "
                             "and a sparse direct solver; mps has neither)" % dev)
        self.fem = TorchFEM(self.X, self.tets, self.faces, o["material"], o["material_params"], dev)
        self.lin = LinearSolver(o["linear_solver"], self.fem.device.type == "cuda")
        fe = self.fem
        if self.wall is not None:
            self.tActive = fe.tensor(~self.clamped)
            self.tArea0 = fe.tensor(self.area0)
            self.tQB, self.tQW = fe.tensor(self.qb), fe.tensor(self.qw)
        self._log_setup(fe.device, ", material %s %s" % (o["material"], o["material_params"] or ""))

    def _wall(self, U, compute_K):
        """c (pos(g)^2 - pos(g-e)^2) wallN.du on the reference faces (wall_quadrature), wallN / wallG
        held fixed; tangent 2c (pos(g) - pos(g-e)) (wallN x wallN) N_r N_s."""
        fe, faces = self.fem, self.fem.faces
        e = float(self.options["wall_eps"])
        c = self.options["wall_stiffness"] / (2.0 * e)
        b = self.tQB                                                     # (P, 3)
        u = torch.einsum("pr,frk->fpk", b, U[faces])
        n = torch.einsum("pr,frk->fpk", b, fe.tensor(self.wallN)[faces])
        G = torch.einsum("pr,fr->fp", b, fe.tensor(self.wallG)[faces])
        g = (u * n).sum(-1) - G
        pg, pe = g.clamp(min=0.0), (g - e).clamp(min=0.0)
        w = self.tQW[None] * (self.tArea0 * self.tActive)[:, None]       # (F, P)
        f = fe._scatter(faces, torch.einsum("fp,pr,fpk->frk", w * c * (pg ** 2 - pe ** 2), b, n))
        if not compute_K:
            return f, None
        kn = (w * 2.0 * c * (pg - pe))[..., None, None] * n[..., :, None] * n[..., None, :]
        return f, torch.einsum("pr,ps,fpij->frsij", b, b, kn)

    def _assemble(self, U, compute_K, pressure=True, wall=True):
        fe = self.fem
        Ut = fe.tensor(U.reshape(-1, 3))
        F, J = fe.deformation_gradient(Ut)
        if not bool((J > 0).all()):
            return None, None
        f = fe.internal_forces(Ut, self.nu, F)
        Fb = torch.zeros(len(self.faces), 3, 3, 3, 3, dtype=fe.dtype, device=fe.device) if compute_K else None
        if pressure and np.any(self.fq != 0.0):
            f = f + fe.pressure_forces(Ut, self.fq)
            if compute_K:
                Fb = Fb + fe.pressure_blocks(Ut, self.fq)
        if wall and self.wall is not None:
            fw, Fw = self._wall(Ut, compute_K)
            f = f + fw
            if compute_K:
                Fb = Fb + Fw
        R = f.reshape(-1).cpu().numpy()
        if not np.all(np.isfinite(R)):
            return None, None
        A = self._csr(fe.elastic_blocks(F, self.nu).cpu().numpy(), Fb.cpu().numpy()) if compute_K else None
        return R, A

    def _dfint_dnu(self, U):
        """Forward-mode derivative of f_int with respect to nu (any material)."""
        fe = self.fem
        Ut = fe.tensor(U.reshape(-1, 3))
        nu = fe.tensor(self.nu)
        _, d = jvp(lambda n: fe.internal_forces(Ut, n), (nu,), (torch.ones_like(nu),))
        return d.reshape(-1).cpu().numpy()
