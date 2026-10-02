"""Shared by the FEM cores (no FEM library imported here): solver options and the material,
GetFEM 'Compressible_Neo_Hookean', non-dimensional (E = 1):

    W = c1 (J^-2/3 I1 - 3) + d1 (J - 1)^2,   [c1, d1] = [mu/2, K/2]
"""

# solver_options of the GetFEM-equivalent cores (documented in getfem_solver.py)
DEFAULTS = dict(order=1, load_steps=4, newton_tol=1e-7, newton_maxit=30, pressure_sign=1.0, warm_substeps=False,
                wall_stiffness=20.0, wall_eps=0.5, wall_max_updates=6, wall_settle_mm=0.05, linear_solver=None)


def mat_params(nu):
    """[c1, d1] = [mu/2, K/2] with E = 1."""
    return [1.0 / (4.0 * (1.0 + nu)), 1.0 / (6.0 * (1.0 - 2.0 * nu))]


def dmat_params_dnu(nu):
    """d[c1, d1]/dnu."""
    return [-1.0 / (4.0 * (1.0 + nu) ** 2), 1.0 / (3.0 * (1.0 - 2.0 * nu) ** 2)]
