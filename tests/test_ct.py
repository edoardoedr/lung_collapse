"""eurelaps_lung/ct.py on synthetic CT arrays: density, emphysema with the noise filter, trachea air.

    python tests/test_ct.py
"""

import numpy as np

import stubs
from eurelaps_lung import ct

SP = (1.0, 1.0, 1.0)


def phantom(emph=0.0, noise=0.0, seed=0):
    """A 40^3 box of lung at -850 HU (0.15 g/mL) with a block of emphysema at -990 HU of fraction emph,
    in air at -1000 HU, plus Gaussian noise."""
    rng = np.random.default_rng(seed)
    hu = np.full((60, 60, 60), -1000.0, np.float32)
    lung = np.zeros(hu.shape, bool)
    lung[10:50, 10:50, 10:50] = True
    hu[lung] = -850.0
    n = int(round(emph * 40))
    if n:
        hu[10:50, 10:50, 10:10 + n] = -990.0
    return hu + rng.normal(0, noise, hu.shape).astype(np.float32), lung


def test_density():
    hu, lung = phantom()
    r = ct.inflated_lung(hu, lung, SP, hu_air=-1000.0)
    assert abs(r["rho_bar_kg_m3"] - 150.0) < 1e-3 and abs(r["volume_mL"] - 64.0) < 1e-9
    assert abs(r["phi_tissue"] - 150 / 1050) < 1e-6 and r["laa950"] == 0.0
    return "rho 150 kg/m3, phi tissue %.3f" % r["phi_tissue"]


def test_emphysema_and_noise():
    hu, lung = phantom(emph=0.25)
    e = ct.emphysema(hu, lung, -1000.0, SP)
    core = 36 ** 3
    expect = 100.0 * 36 * 36 * 8 / core                     # the block inside the 2 mm border
    assert abs(e["laa950"] - expect) < 1.0, (e, expect)
    noisy, _ = phantom(emph=0.0, noise=90.0)
    raw = 100 * ((noisy[lung] < -950).mean())
    filt = ct.emphysema(noisy, lung, -1000.0, SP)["laa950"]
    assert raw > 10.0 and filt < 2.0, (raw, filt)
    return "emphysema %.1f %% (expected %.1f); noise SD 90 HU: %.1f %% raw, %.1f %% filtered" % (
        e["laa950"], expect, raw, filt)


def test_trachea_air_and_phi():
    hu, lung = phantom()
    aw = np.zeros(hu.shape, bool)
    aw[40:58, 26:34, 26:34] = True                          # a tube along k (superior = +k here)
    hu[aw] = -980.0
    M = np.eye(4)
    assert ct.air_hu_trachea(hu, aw, M, SP) == -980.0
    assert abs(ct.barrier_phi(150.0, True, 20.0) - (150 / 1050 + 0.54 * 0.2)) < 1e-12
    assert ct.barrier_phi(150.0, False, 20.0) == 150 / 1050
    return "trachea air -980 HU; phi with / without emphysema"


if __name__ == "__main__":
    stubs.run_tests(globals())
