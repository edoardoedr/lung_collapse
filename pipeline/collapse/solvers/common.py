"""Shared by the FEM cores (no FEM library imported here): solver options and the material,
GetFEM 'Compressible_Neo_Hookean', non-dimensional (E = 1):

    W = c1 (J^-2/3 I1 - 3) + d1 (J - 1)^2,   [c1, d1] = [mu/2, K/2]
"""

# solver_options of the GetFEM-equivalent cores (documented in getfem_solver.py)
DEFAULTS = dict(order=1, load_steps=4, newton_tol=1e-7, newton_maxit=30, pressure_sign=1.0, warm_substeps=False,
                wall_stiffness=20.0, wall_eps=0.5, wall_max_updates=6, wall_settle_mm=0.05, wall_update="outer",
                linear_solver=None, max_solve_s=None, slow_ramp=True,
                wall_quadrature="face", adaptive_steps=False, min_load_step=1.0 / 64)
WALL_QUADRATURES = ("face", "nodal")


WALL_NEAR_MM = 1.0       # points closer to the wall than this (gap > -1 mm) count for wall_settle_mm


def wall_gap_change(Us, n_old, G_old, phi, allow):
    """Settling measure of the wall rounds [mm]: how far the previous linearisation's gap
    (u.n_old - G_old) is from the true gap (phi - allow) at the current state, over the points near
    the wall. Not the change of G itself: G = n.u - phi + allow moves with the normal times the
    displacement, and far from the wall (near the cavity's medial axis) the closest wall point can
    jump between two sides at every round, changing G by mm while the contact does not change."""
    import numpy as np
    g_old = np.einsum("ij,ij->i", n_old, Us) - G_old
    g_new = phi - allow
    near = np.maximum(g_old, g_new) > -WALL_NEAR_MM
    return float(np.abs(g_old - g_new)[near].max()) if near.any() else 0.0


def ramps(n0, warm, substeps, slow_ramp, adaptive=False):
    """Load paths tried by solve(): (from the last converged state, number of sub-steps) in order.
    warm: a converged state exists; substeps: also sub-steps from it (warm_substeps, same partition);
    slow_ramp: also the 3 x load_steps variants (off = fail fast). adaptive: the steps adapt
    (adaptive_path), so one path from the last state (first step: all of it) and one from the
    reference (first step: 1 / load_steps)."""
    if adaptive:
        return ([(True, 1)] if warm else []) + [(False, n0)]
    out = []
    if warm:
        out += [(True, n) for n in ((1, n0, 3 * n0) if substeps else (1,)) if slow_ramp or n != 3 * n0]
    out += [(False, n) for n in ((n0, 3 * n0) if slow_ramp else (n0,))]
    return out


def adaptive_path(get_state, set_state, set_q, solve_step, q0, q, first_step, min_step, on_step=None):
    """From the current state (load q0) to q in load steps that halve when a step fails (the
    state is restored) and double after a success; False when a step smaller than min_step (a
    fraction of the whole path) fails. With a wall a large step lets the points slide far along
    the linearised (tangent-plane) wall and the rounds diverge, while small steps converge in 1-2
    rounds (karl04: p/E steps of 0.02 fine, 0.05-0.1 failing)."""
    s, h = 0.0, float(first_step)
    while s < 1.0 - 1e-12:
        h = min(h, 1.0 - s)
        U = get_state()
        set_q(q0 + (q - q0) * (s + h))
        if on_step is not None:
            on_step(s + h, h)
        if solve_step():
            s, h = s + h, 2.0 * h
        else:
            set_state(U)
            h *= 0.5
            if h < min_step:
                return False
    return True


def mat_params(nu):
    """[c1, d1] = [mu/2, K/2] with E = 1."""
    return [1.0 / (4.0 * (1.0 + nu)), 1.0 / (6.0 * (1.0 - 2.0 * nu))]


def dmat_params_dnu(nu):
    """d[c1, d1]/dnu."""
    return [-1.0 / (4.0 * (1.0 + nu) ** 2), 1.0 / (3.0 * (1.0 - 2.0 * nu) ** 2)]
