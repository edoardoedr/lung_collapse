"""The collapse (forward) and fit steps end to end on karl04 with the numpy test core and numpy
stand-ins for VTK (fakes.py). Skipped without a local copy of the fem_setup output.

    python tests/test_steps.py
"""

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

import stubs
import fakes  # noqa: F401
import test_wall_potential as W
from pipeline.collapse import fit, forward
from pipeline.collapse.load import LoadModel
from pipeline.config import load_config

BASE = {
    "patient": "karl04", "data_dir": ".", "inputs": {"inflated": "lung.vtk", "collapsed": "lung_collapsed.vtk"},
    "solver": {"core": "numpy_core:NumpySolver", "wall_update": "potential", "adaptive_steps": True,
               "warm_substeps": True, "newton_maxit": 50, "load_steps": 4, "max_solve_s": 60},
    "fit": {"time_budget_min": 20},
}
_tmp = {}


def config(overrides=(), **top):
    """A config in a temporary output folder that holds karl04's problem.npz (with the wall field)."""
    p = W.problem()
    if p is None:
        return None
    if "dir" not in _tmp:
        _tmp["dir"] = Path(tempfile.mkdtemp(prefix="lung_steps_"))
        (_tmp["dir"] / "out/fem/setup").mkdir(parents=True)
        p.save(_tmp["dir"] / "out/fem/setup/problem.npz")
        shutil.copy(W.SETUP.parent / "setup_summary.json", _tmp["dir"] / "out/fem/setup/setup_summary.json")
    path = _tmp["dir"] / "cfg.json"
    path.write_text(json.dumps({**BASE, "output_dir": "out", **top}))
    return load_config(path, list(overrides))


def summary(cfg):
    return json.loads((cfg.run_dir / "summary.json").read_text())


def test_load_model():
    """Levels: q and its Jacobian pieces, gravity slope, positions."""
    cfg = config(['physics.E_Pa={"fit":true,"init":1700}', 'physics.gravity={"reference":"supine","collapse":"left_lateral"}'])
    if cfg is None:
        return "skipped"
    p = W.problem()
    m = LoadModel(cfg.physics, p)
    lv = m.fit_levels(cfg.fit)[0]
    assert lv.names == ["q0", "gravity_slope"], lv.names
    x = np.array([0.4, 0.3])
    q = lv.q_tri(x)
    free = ~p.clamped_tri
    h = -p.reference[p.tris].mean(1)[:, 1]                      # supine: anterior = up
    slope = np.polyfit(h[free], q[free], 1)[0] * np.ptp(h[free])
    assert abs(slope - 0.3) < 1e-9 and abs(np.average(q[free], weights=m.area[free]) - 0.4) < 1e-12
    assert np.allclose(lv.q_solver(x)[lv.labels[free]], q[free]) and abs(lv.body(x) - 0.3) < 1e-15
    # resultant of the load = the weight along the collapse position: gamma V e_B
    tri_n = np.cross(p.reference[p.tris[:, 1]] - p.reference[p.tris[:, 0]], p.reference[p.tris[:, 2]] - p.reference[p.tris[:, 0]]) / 2
    hc = -(p.reference[p.tris].mean(1) @ np.array([0.0, 1.0, 0.0]))
    surf = -(0.3 / m.extent * (hc - hc.mean()))[:, None] * tri_n   # the gravity term on the whole closed surface
    from pipeline.collapse.geometry import signed_volume
    V = signed_volume(p.reference, p.tris)
    total = surf.sum(0) + 0.3 * m.body_direction * V
    gamma = 0.3 / m.extent
    assert np.allclose(total, gamma * V * np.array([1.0, 0.0, 0.0]), rtol=1e-9, atol=1e-9 * gamma * V), total / (gamma * V)
    E = m.E_of_slope(0.3)
    d = m.describe(lv, x)
    assert abs(d["E_Pa"] - E) < 1e-9 and abs(d["p0_Pa"] - 0.4 * E) < 1e-9 and d["gravity"]["body_force"]
    x0 = lv.x0_from(m.start_q_tri(), 0.4, m.area)
    assert np.allclose(x0, [0.5, m.slope_of_E(1700.0)])
    return "E of slope 0.3: %.0f Pa; resultant = weight along the collapse position" % E


def test_forward_volume_ratio():
    """Forward run to a given volume ratio, then the same q0 given directly."""
    cfg = config(['physics.collapse={"volume_ratio":0.8}', "run_name=vol80"])
    if cfg is None:
        return "skipped"
    forward.run(cfg)
    s = summary(cfg)
    assert abs(s["volume_ratio"] - 0.8) <= cfg.collapse.volume_tol and s["physics"]["fitted"] == []
    q0 = s["physics"]["q0"]
    cfg2 = config(['physics.collapse={"q0":%r}' % q0, "run_name=q0"])
    forward.run(cfg2)
    s2 = summary(cfg2)
    assert abs(s2["volume_ratio"] - s["volume_ratio"]) < 1e-6
    assert s["assd_mm"] > 0 and s["wall_field_max_penetration_mm"] < 0.5
    return ("volume 80 %% at q0 %.4f in %d solves (p0 %.0f Pa at E %.0f), ASSD vs the real collapsed lung %.2f mm"
            % (q0, s["solves"], s["physics"]["p0_Pa"], s["physics"]["E_Pa"], s["assd_mm"]))


def test_fit_recovers_parameters():
    """Synthetic target = a forward run; the fit of q0 and E must find the parameters back."""
    cfg = config(['physics.collapse={"q0":0.35}', "physics.E_Pa=1200", "run_name=truth"])
    if cfg is None:
        return "skipped"
    forward.run(cfg)
    truth = np.load(cfg.run_dir / "state.npz")["surface_displacement"]
    import dataclasses
    p = W.problem()
    synth = dataclasses.replace(p, target_points=p.reference + truth, target_tris=p.tris.copy())
    synth.save(cfg.setup_dir / "problem.npz")
    try:
        cfg = config(['physics.collapse={"q0":{"fit":true,"init":0.5}}', 'physics.E_Pa={"fit":true,"init":2500}',
                      "run_name=refit", "fit.target_error_mm=0.001"])
        fit.run(cfg)
        s = summary(cfg)
    finally:
        p.save(cfg.setup_dir / "problem.npz")
    ph = s["physics"]
    assert abs(ph["q0"] - 0.35) < 2e-3 and abs(ph["E_Pa"] - 1200) < 15, ph
    assert s["assd_mm"] < 0.05 and set(ph["fitted"]) == {"q0", "gravity_slope"}
    return "q0 %.4f (0.35), E %.0f Pa (1200), ASSD %.4f mm, %d solves (%d failed)" % (
        ph["q0"], ph["E_Pa"], s["assd_mm"], s["evals"], s["failed_solves"])


def test_fit_real_target():
    """Fit of q0 on the real collapsed lung, default physics (gravity supine, E 1.7 kPa)."""
    cfg = config(["run_name=real"])
    if cfg is None:
        return "skipped"
    fit.run(cfg)
    s = summary(cfg)
    assert s["physics"]["fitted"] == ["q0"] and s["failed_solves"] == 0 and s["assd_mm"] < s["baseline_assd_mm"]
    return "ASSD %.2f mm (no deformation %.2f), p0 %.0f Pa, collapse %.0f %%, %d solves" % (
        s["assd_mm"], s["baseline_assd_mm"], s["physics"]["p0_Pa"], s["collapse_percent"], s["evals"])


if __name__ == "__main__":
    stubs.run_tests(globals())
