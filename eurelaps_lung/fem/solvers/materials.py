"""Hyperelastic materials of the torch core, as strain-energy densities.

A material is a function W(F, nu, **params) -> scalar for ONE deformation gradient F (3, 3):
energy per unit reference volume, non-dimensional with E = 1 (the fit identifies q = p / E, so
every stiffness must scale with E: at small strain the material must have Young's modulus 1 and
Poisson's ratio nu). nu may be a tensor (the Jacobian differentiates W with respect to it);
params are fixed numbers from solver_options.material_params.

The torch core gets the stress P = dW/dF and the tangent dP/dF by automatic differentiation, so
a new material is only its W: write the function, add it to MATERIALS, select it with
solver_options {"material": "<name>", "material_params": {...}}. Use only operations that work
under torch.func.vmap (no Python if on tensor values, no .item()); J > 0 is guaranteed by the
caller (inverted elements are rejected before W is evaluated).

GetFEM and Warp implement only "neo_hookean" (GetFEM Compressible_Neo_Hookean).
"""

from .common import mat_params


def det3(F):
    return (F[0, 0] * (F[1, 1] * F[2, 2] - F[1, 2] * F[2, 1])
            - F[0, 1] * (F[1, 0] * F[2, 2] - F[1, 2] * F[2, 0])
            + F[0, 2] * (F[1, 0] * F[2, 1] - F[1, 1] * F[2, 0]))


def neo_hookean(F, nu):
    """GetFEM Compressible_Neo_Hookean: W = c1 (J^-2/3 I1 - 3) + d1 (J - 1)^2, [c1, d1] = [mu/2, K/2]."""
    c1, d1 = mat_params(nu)
    J = det3(F)
    I1 = (F * F).sum()
    return c1 * (J ** (-2.0 / 3.0) * I1 - 3.0) + d1 * (J - 1.0) ** 2


def mooney_rivlin(F, nu, c01_fraction=0.0):
    """Compressible Mooney-Rivlin, W = c10 (J^-2/3 I1 - 3) + c01 (J^-4/3 I2 - 3) + d1 (J - 1)^2,
    with c10 + c01 = mu/2 (same small-strain E and nu as neo_hookean) and c01 = c01_fraction * mu/2.
    c01_fraction = 0 is exactly neo_hookean. Example of a second material, not validated against
    another implementation."""
    c1, d1 = mat_params(nu)
    J = det3(F)
    C = F.T @ F
    I1 = C.diagonal().sum()
    I2 = 0.5 * (I1 * I1 - (C * C).sum())
    c01 = c01_fraction * c1
    return ((c1 - c01) * (J ** (-2.0 / 3.0) * I1 - 3.0) + c01 * (J ** (-4.0 / 3.0) * I2 - 3.0)
            + d1 * (J - 1.0) ** 2)


MATERIALS = {
    "neo_hookean": neo_hookean,
    "mooney_rivlin": mooney_rivlin,
}
