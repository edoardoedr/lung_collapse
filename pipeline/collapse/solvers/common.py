"""Shared by the FEM cores (no FEM library imported here): solver options and the material,
GetFEM 'Compressible_Neo_Hookean', non-dimensional (E = 1):

    W = c1 (J^-2/3 I1 - 3) + d1 (J - 1)^2,   [c1, d1] = [mu/2, K/2]
"""

# solver_options of the GetFEM-equivalent cores (documented in getfem_solver.py)
DEFAULTS = dict(order=1, load_steps=4, newton_tol=1e-7, newton_maxit=30, pressure_sign=1.0, warm_substeps=False,
                wall_stiffness=20.0, wall_eps=0.5, wall_max_updates=6, wall_settle_mm=0.05, wall_update="outer",
                linear_solver=None, max_solve_s=None, slow_ramp=True)


def ramps(n0, warm, substeps, slow_ramp):
    """Load paths tried by solve(): (from the last converged state, number of sub-steps) in order.
    warm: a converged state exists; substeps: also sub-steps from it (warm_substeps, same partition);
    slow_ramp: also the 3 x load_steps variants (off = fail fast)."""
    out = []
    if warm:
        out += [(True, n) for n in ((1, n0, 3 * n0) if substeps else (1,)) if slow_ramp or n != 3 * n0]
    out += [(False, n) for n in ((n0, 3 * n0) if slow_ramp else (n0,))]
    return out


def mat_params(nu):
    """[c1, d1] = [mu/2, K/2] with E = 1."""
    return [1.0 / (4.0 * (1.0 + nu)), 1.0 / (6.0 * (1.0 - 2.0 * nu))]


def dmat_params_dnu(nu):
    """d[c1, d1]/dnu."""
    return [-1.0 / (4.0 * (1.0 + nu) ** 2), 1.0 / (3.0 * (1.0 - 2.0 * nu) ** 2)]
