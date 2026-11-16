"""wall_update "potential" in solvers/nodal.py, on a real problem (karl04 with the reference wall)
with the numpy test core. Skipped when results/ has no local copy of the fem_setup output.

    python tests/test_wall_potential.py        (first run: ~2 min to build the distance grid)
"""

import dataclasses

import numpy as np

import stubs
from pipeline.collapse.problem import CollapseProblem

SETUP = stubs.ROOT / "results/karl04_noreg_hilum_wall/fem/setup/problem.npz"
CACHE = stubs.ROOT / "tests/.cache/karl04_wall_grid.npz"
SPACING, MARGIN, SMOOTH, TOL = 4.0, 16.0, 3.0, 5.0
OPTS = dict(wall_update="potential", adaptive_steps=True, warm_substeps=True, slow_ramp=False, newton_tol=1e-9)
_cache = {}


def problem():
    """karl04 with the wall field added (built here in numpy instead of fem_setup's VTK)."""
    if "p" in _cache:
        return _cache["p"]
    if not SETUP.exists():
        return None
    p = CollapseProblem.load(SETUP)
    if not CACHE.exists():
        import sdf_numpy
        g, origin = sdf_numpy.grid(p.wall_points, p.wall_tris, SPACING, MARGIN, SMOOTH)
        CACHE.parent.mkdir(exist_ok=True)
        np.savez_compressed(CACHE, grid=g, origin=origin)
    d = np.load(CACHE)
    p = dataclasses.replace(p, wall_grid=d["grid"], wall_origin=d["origin"], wall_spacing=np.array(SPACING),
                            wall_smoothing_mm=np.array(SMOOTH))
    p.wall_field_allow = TOL + np.maximum(p.wall_field()(p.reference, 0), 0.0)
    _cache["p"] = p
    return p


def solver(p, **opts):
    from numpy_core import NumpySolver
    s = NumpySolver(p, {**OPTS, **opts})
    free = ~p.clamped_tri
    labels = -np.ones(len(p.tris), np.int64)
    labels[free] = np.arange(free.sum())
    s.set_regions(labels, int(free.sum()))
    return s


def load(p, q0, gamma):
    """p/E per free triangle: uniform + gradient along the anterior direction (supine)."""
    cen = p.reference[p.tris].mean(1)[~p.clamped_tri]
    h = -cen[:, 1]
    return q0 + gamma * (h - h.mean())


def test_tangent_vs_fd():
    """Wall force and tangent blocks against central differences of the residual, in contact."""
    p = problem()
    if p is None:
        return "skipped"
    s = solver(p)
    rng = np.random.default_rng(0)
    n = np.zeros((s.nn, 3))
    n[p.surface_nodes] = p.reference - p.reference.mean(0)
    U = (7.0 * n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9) + rng.normal(0, 0.2, n.shape)).ravel()
    U[np.setdiff1d(np.arange(s.ndof), s.free)] = 0.0
    s.fq[:] = 0.0
    s.nu = 0.4
    R, A = s._assemble_at(U, True)
    R0, A0 = s._assemble(U, True, wall=False)
    contact = int((s.field(s.surf_pts + U.reshape(-1, 3)[s.surf_node], 0) - s.field_allow > 0).sum())
    assert contact > 50, contact
    W = (A - A0).toarray()                                       # the wall part of the tangent
    assert np.abs(W - W.T).max() < 1e-12 * np.abs(W).max()
    err = 0.0
    for _ in range(12):
        d = rng.normal(size=s.nfree)
        d /= np.linalg.norm(d)
        e, Up, Um = 1e-5, U.copy(), U.copy()
        Up[s.free] += e * d
        Um[s.free] -= e * d
        fd = ((s._assemble_at(Up, False)[0] - s._assemble(Up, False, wall=False)[0])
              - (s._assemble_at(Um, False)[0] - s._assemble(Um, False, wall=False)[0]))[s.free] / (2 * e)
        err = max(err, np.linalg.norm(W @ d - fd) / np.linalg.norm(fd))
    assert err < 1e-6, err
    return "%d nodes in contact, wall tangent vs FD rel %.1e" % (contact, err)


def test_solve_and_jacobian():
    """Collapse with a gravity-like gradient against the wall: one Newton per load step, small
    penetration, and the Jacobian (now exact with a wall) against finite differences."""
    p = problem()
    if p is None:
        return "skipped"
    s = solver(p)
    q = load(p, 0.25, 1.4e-3)
    Us = s.solve(q, 0.4)
    assert Us is not None, s.stats()
    g = s.field(p.reference + Us, 0) - s.field_allow
    c = s.stats()
    assert c["wall_updates"] == 0 and c["fail_unsettled"] == 0
    assert (g > 0).sum() > 5 and g.max() < 0.5, ((g > 0).sum(), g.max())
    J = s.jacobian(q, 0.4, with_nu=True)
    state, fq = s.get_state(), s.fq.copy()
    rng = np.random.default_rng(3)
    err = 0.0
    for k in range(4):                                          # 3 load directions + nu
        d = np.zeros(len(q) + 1)
        if k < 3:
            d[:-1] = [np.ones(len(q)), q - q.mean(), rng.normal(size=len(q))][k]
        else:
            d[-1] = 1.0
        d /= np.linalg.norm(d)
        e, out = 1e-5, []
        for sgn in (1.0, -1.0):
            s.set_state(state, np.where(p.clamped_tri, np.nan, 0.0))
            s.fq_last = fq.copy()
            out.append(s.solve(q + sgn * e * d[:-1], 0.4 + sgn * e * d[-1]))
        fd = ((out[0] - out[1]) / (2 * e)).ravel()
        err = max(err, np.linalg.norm(J @ d - fd) / np.linalg.norm(fd))
    assert err < 1e-5, err
    return ("%d points in contact, max gap %.3f mm, %d Newton runs / %d iterations, 0 wall rounds; "
            "Jacobian vs FD rel %.1e" % ((g > 0).sum(), g.max(), c["newton_calls"], c["newton_iters"], err))


if __name__ == "__main__":
    stubs.run_tests(globals())
