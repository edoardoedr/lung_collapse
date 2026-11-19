"""Compression barrier (solvers/nodal.py barrier_terms, METHODOLOGY.md 3.4.1): one tetrahedron
against finite differences, then karl04 with the numpy core (skipped without the local setup).

    python tests/test_barrier.py
"""

import json

import numpy as np

import stubs
from eurelaps_lung.fem.solvers.nodal import barrier_terms
from test_wall_potential import load, problem, solver

PHI, KAPPA, ONSET = 0.15, 1.0, 0.1


def _tet(J_target, seed=0):
    """A random tet (shape-function gradients G, volume V) and nodal displacements u giving J ~ J_target."""
    rng = np.random.default_rng(seed)
    Dm = np.eye(3) * 10.0 + rng.normal(0, 1.5, (3, 3))
    if np.linalg.det(Dm) < 0:
        Dm[:, [0, 1]] = Dm[:, [1, 0]]
    Bi = np.linalg.inv(Dm)
    G = np.concatenate([-Bi.sum(0, keepdims=True), Bi], 0)[None]          # (1, 4, 3)
    V = np.array([np.linalg.det(Dm) / 6.0])
    X = np.vstack([np.zeros(3), Dm.T])                                     # nodes
    Fd = np.eye(3) + rng.normal(0, 0.08, (3, 3))                          # some shear too
    Fd *= (J_target / np.linalg.det(Fd)) ** (1 / 3)                       # det = J_target exactly
    u = X @ Fd.T - X
    u -= u[0]
    return G, V, u


def _F(G, u):
    return np.eye(3)[None] + np.einsum("ai,eaj->eij", u, G)


def _parts(G, V, u, K=False):
    return barrier_terms(_F(G, u), G, V, PHI, KAPPA, ONSET, K)


def test_zero_above_onset():
    G, V, u = _tet(0.8)
    J, e, f, K = _parts(G, V, u, True)
    assert J[0] > PHI + ONSET and e[0] == 0.0 and not f.any() and not K.any()
    return "J %.2f: no energy, force or tangent" % J[0]


def test_force_and_tangent_vs_fd():
    out = []
    for Jt in (0.24, 0.2, 0.17):
        G, V, u = _tet(Jt, seed=int(100 * Jt))
        J, e, f, K = _parts(G, V, u, True)
        assert PHI < J[0] < PHI + ONSET, J
        h, ef, eK = 1e-6, 0.0, 0.0
        Kmat = K[0].transpose(0, 2, 1, 3).reshape(12, 12)                  # [a i, b k]
        fd_f, fd_K = np.zeros(12), np.zeros((12, 12))
        for c in range(12):
            up, um = u.copy(), u.copy()
            up.flat[c] += h
            um.flat[c] -= h
            _, ep, fp, _ = _parts(G, V, up)
            _, em, fm, _ = _parts(G, V, um)
            fd_f[c] = V[0] * (ep[0] - em[0]) / (2 * h)
            fd_K[:, c] = (fp[0] - fm[0]).ravel() / (2 * h)
        ef = np.abs(f[0].ravel() - fd_f).max() / np.abs(fd_f).max()
        eK = np.abs(Kmat - fd_K).max() / np.abs(fd_K).max()
        assert ef < 1e-6 and eK < 1e-5, (Jt, ef, eK)
        assert np.abs(Kmat - Kmat.T).max() < 1e-9 * np.abs(Kmat).max()       # hyperelastic: symmetric
        out.append("J %.3f: force %.0e, tangent %.0e" % (J[0], ef, eK))
    return "; ".join(out)


def test_c2_at_onset_and_unbounded_at_phi():
    s = np.array([1e-3, 1e-4])
    d = ONSET - s
    b, b1 = s ** 3 / d, -3 * s ** 2 / d - s ** 3 / d ** 2
    b2 = 6 * s / d + 6 * s ** 2 / d ** 2 + 2 * s ** 3 / d ** 3
    assert b[1] < 1e-10 and abs(b1[1]) < 1e-6 and b2[1] < 1e-2                # -> 0 at J_on
    G, V, u = _tet(PHI + 1e-4, seed=5)
    J, e, f, _ = _parts(G, V, u)
    assert J[0] - PHI < 1e-3 and np.abs(f).max() > 1e3 * V[0], (J, np.abs(f).max())
    u2 = _tet(PHI * 0.9, seed=5)[2]
    assert _parts(G, V, u2)[2] is None                                     # J <= phi: rejected
    return "b, b', b'' -> 0 at J_on; force %.1e at J - phi = %.0e; J < phi rejected" % (np.abs(f).max(), J[0] - PHI)


def _solve(p, q0, gamma, barrier, max_s=600):
    s = solver(p, barrier=barrier, max_solve_s=max_s, newton_maxit=50)
    Us = s.solve(load(p, q0, gamma), 0.30)
    return s, Us


def test_inactive_barrier_changes_nothing():
    p = problem()
    if p is None:
        return "skipped"
    s0, U0 = _solve(p, 0.2, 1.4e-3, None)
    s1, U1 = _solve(p, 0.2, 1.4e-3, dict(phi=0.10, stiffness=KAPPA, onset=ONSET))
    assert U0 is not None and U1 is not None
    rep = s1.barrier_report()
    assert rep["tets_active"] == 0, rep
    d = np.abs(U0 - U1).max()
    assert d < 1e-9, d
    return "q0 0.2: J min %.2f > J_on 0.20, max difference %.1e mm" % (rep["J_min"], d)


def test_beyond_KE_converges_with_barrier():
    """q0 = 1.2 > K/E = 0.83 (nu 0.30, ~12 mmHg at E 1.36 kPa): Neo-Hookean has no equilibrium,
    with the barrier the lung stops above phi."""
    p = problem()
    if p is None:
        return "skipped"
    q0 = 1.2
    s0, U0 = _solve(p, q0, 1.4e-3, None, max_s=300)
    s1, U1 = _solve(p, q0, 1.4e-3, dict(phi=PHI, stiffness=KAPPA, onset=ONSET))
    assert U1 is not None, s1.stats()
    rep = s1.barrier_report()
    from eurelaps_lung.fem.geometry import signed_volume
    r = signed_volume(p.reference + U1, p.tris) / signed_volume(p.reference, p.tris)
    assert rep["J_min"] > PHI and rep["tets_active"] > 0, rep
    assert PHI < r < 0.4, r
    # tangent with the barrier active, against finite differences of the residual
    U = s1.U.copy()
    R, A = s1._assemble_at(U, True)
    rng = np.random.default_rng(1)
    err = 0.0
    for _ in range(4):
        dv = rng.normal(size=s1.nfree)
        dv /= np.linalg.norm(dv)
        e, Up, Um = 1e-6, U.copy(), U.copy()
        Up[s1.free] += e * dv
        Um[s1.free] -= e * dv
        fd = (s1._assemble_at(Up, False)[0] - s1._assemble_at(Um, False)[0])[s1.free] / (2 * e)
        err = max(err, np.linalg.norm(A @ dv - fd) / np.linalg.norm(fd))
    assert err < 1e-5, err
    nh = "failed"
    if U0 is not None:                                                     # clamped: shear can hold it
        r0 = signed_volume(p.reference + U0, p.tris) / signed_volume(p.reference, p.tris)
        r0p = s0.barrier_report()
        nh = "converged: volume %.1f %%, J min %.3f, %.0f %% of the volume below phi" % (
            100 * r0, r0p["J_min"], 100 * float((s0.vol0 * (_J(s0) < PHI)).sum() / s0.vol0.sum()))
    return ("q0 1.2: Neo-Hookean %s; with the barrier volume %.1f %% of the inflated one, J min %.3f "
            "(phi %.2f), %d tets / %.0f %% of the volume below J_on, %d Newton iterations; tangent vs FD %.0e"
            % (nh, 100 * r, rep["J_min"], PHI, rep["tets_active"],
               100 * rep["volume_fraction_active"], s1.stats()["newton_iters"], err))


def _J(s):
    F = np.eye(3) + np.einsum("eai,eaj->eij", s.U.reshape(-1, 3)[s.tets], s.Gr)
    return np.linalg.det(F)


def test_config_phi_from_ct():
    from eurelaps_lung.config import load_config
    cfg = load_config(stubs.ROOT / "configs/karl08.json",
                      ['physics.barrier={"phi": "ct"}', "solver.core=torch"])
    b = cfg.physics.barrier
    ph = json.loads((stubs.ROOT / "configs/karl08.json").read_text())["physics"]
    expect = ph["ct_density"] / 1050 + 0.54 * ph["ct_emphysema"] / 100
    assert abs(b["phi"] - expect) < 1e-12, (b, expect)
    no = load_config(stubs.ROOT / "configs/karl08.json",
                     ['physics.barrier={"phi": "ct"}', "physics.emphysema=false"]).physics.barrier
    assert abs(no["phi"] - ph["ct_density"] / 1050) < 1e-12 and no["phi_trapped"] == 0.0
    try:
        load_config(stubs.ROOT / "configs/karl08.json", ['physics.barrier={"phi": "ct"}', "physics.emphysema=null"])
        raise AssertionError("barrier phi ct accepted without physics.emphysema")
    except ValueError:
        pass
    o = cfg.solver_options()
    assert o["barrier"] == dict(phi=b["phi"], stiffness=1.0, onset=0.1)
    try:
        cfg.solver_options("getfem")
        raise AssertionError("getfem accepted the barrier")
    except ValueError:
        pass
    return "karl08: phi = %.3f (tissue %.3f + trapped %.3f); getfem refuses it" % (
        b["phi"], b["phi_tissue"], b["phi_trapped"])


if __name__ == "__main__":
    stubs.run_tests(globals())
