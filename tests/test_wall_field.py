"""Smooth wall field (eurelaps_lung/fem/wall_field.py): numpy / scipy only.

    python tests/test_wall_field.py        (or pytest tests/)
"""

import numpy as np

import stubs  # noqa: F401
from eurelaps_lung.fem.wall_field import WallField, penalty, penalty_energy

R = 40.0


def sphere_field(h=1.0):
    ax = np.arange(-60.0, 60.0 + h / 2, h)
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
    return WallField(np.sqrt(X ** 2 + Y ** 2 + Z ** 2) - R, [ax[0]] * 3, h)


def points(n=400, seed=0):
    rng = np.random.default_rng(seed)
    d = rng.normal(size=(n, 3))
    return d / np.linalg.norm(d, axis=1, keepdims=True) * rng.uniform(25.0, 52.0, (n, 1))


def test_sphere_values():
    """phi, gradient and Hessian of an analytic sphere distance."""
    out = []
    for h in (2.0, 1.0):
        f, X = sphere_field(h), points()
        r = np.linalg.norm(X, axis=1)
        n = X / r[:, None]
        phi, g, H = f(X)
        Hx = (np.eye(3) - n[:, :, None] * n[:, None, :]) / r[:, None, None]
        out.append((np.abs(phi - (r - R)).max(), np.abs(g - n).max(), np.abs(H - Hx).max()))
    assert out[1][0] < 1e-5 and out[1][1] < 1e-5 and out[1][2] < 1e-4, out
    assert out[1][0] < out[0][0] / 8, out                        # 4th order in the spacing
    return "h=1 mm: phi %.1e mm, gradient %.1e, Hessian %.1e /mm" % out[1]


def test_derivatives_vs_fd():
    """Gradient and Hessian against central differences, also exactly on cell boundaries."""
    rng = np.random.default_rng(1)
    grid = rng.normal(size=(24, 22, 20))
    f = WallField(grid, [3.0, -2.0, 5.0], 1.5)
    X = np.array([3.0, -2.0, 5.0]) + 1.5 * rng.uniform(2.0, 17.0, (300, 3))
    X[:100] = np.array([3.0, -2.0, 5.0]) + 1.5 * np.round(rng.uniform(2.0, 17.0, (100, 3)))   # grid points
    e = 1e-5
    phi, g, H = f(X)
    gf, Hf = np.zeros_like(g), np.zeros_like(H)
    for k in range(3):
        d = np.zeros(3)
        d[k] = e
        gf[:, k] = (f(X + d, 0) - f(X - d, 0)) / (2 * e)
        Hf[:, :, k] = (f(X + d, 1)[1] - f(X - d, 1)[1]) / (2 * e)
    eg = np.abs(g - gf).max() / np.abs(g).max()
    eh = np.abs(H - Hf)[100:].max() / np.abs(H).max()             # inside the cells
    # on a cell boundary the Hessian is continuous with a kink (C2 spline), so the difference
    # quotient across it is only first order in the step
    eb = np.abs(H - Hf)[:100].max() / np.abs(H).max()
    assert eg < 1e-7 and eh < 1e-7 and eb < 1e-4, (eg, eh, eb)
    assert np.abs(H - H.transpose(0, 2, 1)).max() == 0.0
    return "gradient rel %.1e, Hessian rel %.1e (on cell boundaries %.1e)" % (eg, eh, eb)


def test_interpolates_grid():
    rng = np.random.default_rng(2)
    grid = rng.normal(size=(12, 13, 14))
    f = WallField(grid, [0.0, 0.0, 0.0], 2.0)
    i = np.stack(np.meshgrid(np.arange(1, 10), np.arange(1, 11), np.arange(1, 12), indexing="ij"), -1).reshape(-1, 3)
    err = np.abs(f(2.0 * i, 0) - grid[i[:, 0], i[:, 1], i[:, 2]]).max()
    assert err < 1e-12, err


def test_outside_grid():
    f = sphere_field(2.0)
    assert not f.inside(np.array([[0.0, 0.0, 59.5]]))[0] and f.inside(np.array([[0.0, 0.0, 50.0]]))[0]
    try:
        f(np.array([[0.0, 0.0, 70.0]]))
    except ValueError:
        return
    raise AssertionError("no error outside the grid")


def test_penalty():
    """Force law = derivative of the energy, continuous slope, linear beyond e."""
    k, e = 20.0, 0.5
    g = np.linspace(-1.0, 3.0, 4001)
    w, dw = penalty(g, k, e)
    d = 1e-6
    assert np.abs((penalty_energy(g + d, k, e) - penalty_energy(g - d, k, e)) / (2 * d) - w).max() < 1e-6
    assert np.abs((penalty(g + d, k, e)[0] - penalty(g - d, k, e)[0]) / (2 * d) - dw)[np.abs(g * (g - e)) > 1e-3].max() < 1e-6
    assert np.all(w[g <= 0] == 0) and np.allclose(dw[g > e], k) and np.allclose(w[g > e], k * (g[g > e] - e / 2))


if __name__ == "__main__":
    stubs.run_tests(globals())
