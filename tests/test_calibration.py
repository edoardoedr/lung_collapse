"""Calibration rules (eurelaps_lung/calibration.py) and the parameters main_predict.py puts in a
config: numpy only, no problem.npz needed.

    python tests/test_calibration.py
"""

import json
import math
import tempfile
from pathlib import Path

import stubs
import main_predict
from eurelaps_lung import calibration as C
from eurelaps_lung.config import load_config


def synthetic(a=3.6, b=-0.0135, rho=200.0, n=8, c=None):
    """Fit summaries that follow log10 E = a + b * collapse % exactly (with c: log10 s_g law)."""
    out, dens = {}, {}
    for i in range(n):
        pct, L, rm = 20.0 + 8.0 * i, 180.0 + 5.0 * i, 80.0 + 7.0 * i
        if c is None:
            E = 10 ** (a + b * pct)
        else:
            E = rho * C.G / 1000 * L / 10 ** (a + b * pct + c * math.log10(rm))
        out["p%d" % i] = {"target_volume_ratio": 1 - pct / 100, "assd_mm": 3.0,
                          "physics": {"E_Pa": E, "nu": 0.3, "q0": 0.4, "tissue_density_kg_m3": rho,
                                      "gravity": {"extent_mm": L, "slope_pE": C.slope_of(E, rho, L)}}}
        dens["p%d" % i] = rm
    return out, dens


def test_rule_E():
    s, _ = synthetic()
    r = C.fit_rule(s, "E")
    assert abs(r["a"] - 3.6) < 1e-9 and abs(r["b"] + 0.0135) < 1e-11 and r["residual_sd"] < 1e-9
    E, rho = C.apply_rule(r, 50.0)
    assert abs(E - 10 ** (3.6 - 0.0135 * 50)) < 1e-6 and rho == 200.0 and r["nu"] == 0.3
    s250, _ = synthetic(rho=250.0)                              # fits at 250: E rescaled to 200, same slope
    r250 = C.fit_rule(s250, "E")
    assert abs(r250["a"] - (3.6 + math.log10(0.8))) < 1e-9 and r250["tissue_density"] == 200.0
    return "a %.4f, b %.5f; E at 50 %% %.0f Pa" % (r["a"], r["b"], E)


def test_rule_E_rho_and_sg_rho():
    s, dens = synthetic()
    r = C.fit_rule(s, "E_rho", dens)
    E, rho = C.apply_rule(r, 30.0, rho_meas=100.0)
    assert rho == 100.0 and E > 0
    s, dens = synthetic(a=-0.5, b=0.004, c=0.3)
    r = C.fit_rule(s, "sg_rho", dens)
    assert abs(r["c"] - 0.3) < 1e-9 and abs(r["a"] + 0.5) < 1e-9
    p = s["p3"]
    E, rho = C.apply_rule(r, C.collapse_pct(p), dens["p3"], C.extent(p))
    assert abs(E - p["physics"]["E_Pa"]) < 1e-6 * E and rho == 200.0
    return "sg_rho recovers c %.3f and the patients' E" % r["c"]


def test_in_bounds():
    s, _ = synthetic()
    s["p0"]["physics"]["E_Pa"] = 15500.0
    s["p1"]["physics"]["E_Pa"] = 170.0
    ok = [p for p in s if C.in_bounds(s[p])]
    assert "p0" not in ok and "p1" not in ok and len(ok) == len(s) - 2
    old, _ = synthetic(rho=250.0)                               # fits at 250 with the old bounds 200-20000
    old["p0"]["physics"]["E_Pa"] = 19500.0                      # at the bound: 15600 at 200
    old["p1"]["physics"]["E_Pa"] = 1500.0                       # 1200 at 200: inside
    assert not C.in_bounds(old["p0"]) and C.in_bounds(old["p1"])


def test_fit_overrides():
    cfg = config()
    sets = C.fit_overrides(cfg, ["physics.E_Pa", "nu"])
    cfg2 = load_config(cfg.source, sets)
    ph = cfg2.physics
    assert ph.E_Pa.fit and ph.nu.fit and ph.q0.fit and ph.E_Pa.bounds == (160.0, 16000.0)
    assert ph.E_Pa.value == 1360.0 and ph.q0.value == 0.3
    return " ".join(sets)


def config(**physics):
    d = Path(tempfile.mkdtemp(prefix="lung_calib_"))
    raw = {"patient": "x", "data_dir": ".", "output_dir": "out", "inputs": {"inflated": "lung.vtk"},
           "physics": {"E_Pa": 1360.0, "material": {"nu": 0.3}, "ct_density": 90.0, **physics}}
    (d / "cfg.json").write_text(json.dumps(raw))
    return load_config(d / "cfg.json")


def test_predict_overrides():
    """main_predict: the rule file and --collapse give a config with E from the rule, volume_ratio."""
    s, _ = synthetic()
    r = C.fit_rule(s, "E")
    cfg = config()
    sets = C.rule_overrides(r, 40.0, cfg.physics.ct_density) + main_predict.collapse_overrides(40.0)
    ph = load_config(cfg.source, sets).physics
    assert abs(ph.volume_ratio - 0.6) < 1e-12 and ph.tissue_density == 200.0 and ph.nu.value == 0.3
    assert abs(ph.E_Pa.value - C.apply_rule(r, 40.0)[0]) < 1e-6 and ph.E_rule["a"] == r["a"]
    fit = {"physics": {"E_Pa": 800.0, "nu": 0.35, "q0": 0.42, "tissue_density_kg_m3": 200.0}}
    ph = load_config(cfg.source, main_predict.fit_run_overrides(fit, with_q0=True)).physics
    assert ph.E_Pa.value == 800.0 and ph.nu.value == 0.35 and ph.q0.value == 0.42 and ph.tissue_density == 200.0
    return "E %.0f Pa at 40 %%" % ph.E_Pa.value


def test_rule_on_local_fits():
    """The rule of the 12 fitE_nu30 fits (density 250) = the built-in one at 200 (config.E_RULE_A, E_RULE_B)."""
    root = Path(stubs.ROOT) / "results"
    s = {f.parts[-4]: json.loads(f.read_text()) for f in root.glob("*/runs/fitE_nu30/summary.json")}
    if not s:
        return "skipped"
    r = C.fit_rule({p: v for p, v in s.items() if C.in_bounds(v)}, "E")
    from eurelaps_lung.config import E_RULE_A, E_RULE_B
    assert r["n"] == 12 and abs(r["a"] - E_RULE_A) < 1e-4 and abs(r["b"] - E_RULE_B) < 1e-6
    return "a %.4f, b %.6f, %d patients" % (r["a"], r["b"], r["n"])


if __name__ == "__main__":
    stubs.run_tests(globals())
