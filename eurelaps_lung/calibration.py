"""Population parameters from the per-patient fits (main_calibrate.py), and their use in a prediction
(main_predict.py). numpy only: works on the summary.json of the fit runs.

A rule gives the gravity load of a new patient from its collapse % (and, optionally, its lung density
measured on the CT); the uniform pressure q0 then comes from the volume (forward volume_ratio):

  E       log10 E = a + b * collapse %, E at the model's density RHO_REF (200 kg/m3, the pleural gradient
          of the literature): a fit at another density rho_fit enters as E_fit * RHO_REF / rho_fit (only
          density / E is identified: same load, same slope)
  E_rho   the same on E_true = E_fit * rho_meas / rho_fit; the forward uses the patient's rho_meas
          (physics.ct_density) as tissue_density: a material E with the measured weight
  sg_rho  log10 s_g = a + b * collapse % + c * log10 rho_meas; the forward uses E = rho_fit g L / s_g
          (L = the lung's extent along gravity): the density as a free predictor of the slope

The forward run uses tissue_density RHO_REF (E, sg_rho) or rho_meas (E_rho); q0 comes from the collapse %
(forward volume_ratio search) or from the pressure p0 given (q0 = p0 / E).

Every rule file also has a q0 rule, log10 q0 = a + b * collapse % (q0 = p0 / E of the same fits): not used
by the run, it gives the estimates shown before one (estimate(): the p0 of a collapse %, the collapse % of
a p0 and E).

The default rules are the file eurelaps_lung/data/rules.json (main_calibrate.py --make-default writes it);
another file for a whole process: the environment variable EURELAPS_RULES (load_rules()).

Only the patients whose fitted E is not at a bound enter the regression (in_bounds). The other
fitted parameters (e.g. nu) get the median over those patients.

Rule file (JSON): {rule, coef, a, b[, c], tissue_density, nu, patients, n, residual_sd, collapse_range,
fit_run, E_bounds, created, q0: {a, b, residual_sd, r, n}}.
"""

import json
import math
import os
import statistics
import time
from importlib import resources
from pathlib import Path

import numpy as np

G = 9.81
RHO_REF = 200.0          # model density [kg/m3] of the rules E and sg_rho (config default physics.tissue_density)
RULES = ("E", "E_rho", "sg_rho")
RULES_FILE = "rules.json"                 # the default rules, in eurelaps_lung/data/
RULES_ENV = "EURELAPS_RULES"             # environment variable: another rule file for the whole process

# fittable parameters: short name -> (config key, default bounds of the calibration fits). The E bounds
# are at the model density RHO_REF: a fit at another tissue_density gets them scaled by tissue_density /
# RHO_REF (same slope range, s_g = rho g L / E). 160-16000 = the 200-20000 of the fits at 250 (2026-11-17).
FIT_PARAMS = {
    "E": ("physics.E_Pa", (160.0, 16000.0)),
    "nu": ("physics.material.nu", (0.10, 0.45)),
    "q0": ("physics.collapse.q0", (-1.0, 3.0)),
}
ALIASES = {"E_Pa": "E", "physics.E_Pa": "E", "physics.nu": "nu", "physics.material.nu": "nu",
           "physics.collapse.q0": "q0"}


def param_name(name):
    """'E' | 'physics.E_Pa' | 'nu' | ... -> the short name of FIT_PARAMS."""
    n = ALIASES.get(name, name)
    if n not in FIT_PARAMS:
        raise ValueError("cannot fit '%s': one of %s" % (name, ", ".join(FIT_PARAMS)))
    return n


def fit_overrides(cfg, names, bounds=None, rho_ref=RHO_REF):
    """--set items that make the parameters `names` free in the fit step, starting from the config's
    value (clipped into the bounds). q0 is always free. bounds: {name: (lo, hi)} instead of the defaults;
    the E bounds are at the density rho_ref and scaled to the config's tissue_density."""
    names = {param_name(n) for n in names} | {"q0"}
    ph = cfg.physics
    start = {"E": ph.E_Pa.value, "nu": ph.nu.value, "q0": ph.q0.value if ph.q0 is not None else 0.3}
    out = []
    for n in sorted(names):
        key, b = FIT_PARAMS[n]
        lo, hi = (bounds or {}).get(n, b)
        if n == "E":
            k = ph.tissue_density / rho_ref
            lo, hi = lo * k, hi * k
        init = min(max(start[n], lo), hi)
        spec = {"fit": True, "init": init, "bounds": [lo, hi]}
        if n == "q0":                                            # replaces p0_Pa / volume_ratio
            out.append("physics.collapse=%s" % json.dumps({"q0": spec}))
        else:
            out.append("%s=%s" % (key, json.dumps(spec)))
    return out


# ------------------------------------------------------------------------------------------------
# quantities of a fit run's summary.json
# ------------------------------------------------------------------------------------------------
def collapse_pct(s):
    """Collapse % of the patient's real collapsed lung."""
    return 100.0 * (1.0 - s["target_volume_ratio"])


def rho_fit(s):
    return s["physics"].get("tissue_density_kg_m3", 250.0)


def extent(s):
    return s["physics"]["gravity"]["extent_mm"]


def slope(s):
    return s["physics"]["gravity"]["slope_pE"]


def E_at(s, rho_ref=RHO_REF):
    """The fitted E at the density rho_ref (same slope): E_fit * rho_ref / rho_fit."""
    return s["physics"]["E_Pa"] * rho_ref / rho_fit(s)


def in_bounds(s, E_bounds=FIT_PARAMS["E"][1], tol=0.10, rho_ref=RHO_REF):
    """The fitted E is not at a bound (relative distance > tol). E_bounds are at the density rho_ref and
    the fit's E is compared at that density, so fits at any tissue_density (fit_overrides scales the
    bounds) are judged alike."""
    lo, hi = E_bounds
    return lo * (1 + tol) < E_at(s, rho_ref) < hi * (1 - tol)


def features(rule, s, rho=None):
    if rule == "sg_rho":
        return [1.0, collapse_pct(s), math.log10(rho)]
    return [1.0, collapse_pct(s)]


def target(rule, s, rho=None, rho_ref=RHO_REF):
    if rule == "E":
        return math.log10(E_at(s, rho_ref))
    if rule == "E_rho":
        return math.log10(s["physics"]["E_Pa"] * rho / rho_fit(s))
    return math.log10(slope(s))


# ------------------------------------------------------------------------------------------------
# the rule
# ------------------------------------------------------------------------------------------------
def fit_rule(summaries, rule="E", rho_meas=None, rho_ref=RHO_REF, **info):
    """Rule fitted on {patient: summary}; rho_meas {patient: kg/m3} for E_rho / sg_rho; rho_ref: the
    model density of the rules E / sg_rho. info: extra keys of the record (fit_run, E_bounds, ...)."""
    if rule not in RULES:
        raise ValueError("rule: one of %s" % ", ".join(RULES))
    if rule != "E" and not rho_meas:
        raise ValueError("rule %s needs the measured densities" % rule)
    pats = sorted(summaries)
    n_coef = 3 if rule == "sg_rho" else 2
    if len(pats) < n_coef:
        raise ValueError("rule %s: %d patients, at least %d needed" % (rule, len(pats), n_coef))
    rho = rho_meas or {}
    A = np.array([features(rule, summaries[p], rho.get(p)) for p in pats])
    y = np.array([target(rule, summaries[p], rho.get(p), rho_ref) for p in pats])
    coef = np.linalg.lstsq(A, y, rcond=None)[0]
    res = y - A @ coef
    dof = max(len(pats) - len(coef), 1)
    rec = dict(rule=rule, coef=[float(c) for c in coef], a=float(coef[0]), b=float(coef[1]),
               tissue_density=float(rho_ref), fit_tissue_density=sorted({rho_fit(summaries[p]) for p in pats}),
               nu=statistics.median(summaries[p]["physics"]["nu"] for p in pats),
               patients=pats, n=len(pats), residual_sd=float(math.sqrt(float(res @ res) / dof)),
               collapse_range=[float(A[:, 1].min()), float(A[:, 1].max())], created=time.strftime("%Y-%m-%d %H:%M"), **info)
    if rule == "sg_rho":
        rec["c"] = float(coef[2])
    rec["q0"] = fit_q0_rule(summaries)
    return rec


def fit_q0_rule(summaries):
    """log10 q0 = a + b * collapse % on {patient: summary} (q0 = p0 / E of the fits): the estimate of the
    pressure of a collapse % (and back), shown before a run."""
    pats = sorted(summaries)
    c = np.array([collapse_pct(summaries[p]) for p in pats])
    q = np.array([summaries[p]["physics"]["q0"] for p in pats], dtype=float)
    if len(pats) < 2 or np.any(q <= 0):
        return None
    A = np.stack([np.ones_like(c), c], axis=1)
    y = np.log10(q)
    coef = np.linalg.lstsq(A, y, rcond=None)[0]
    res = y - A @ coef
    r = float(np.corrcoef(c, y)[0, 1]) if np.std(c) > 0 and np.std(y) > 0 else 0.0
    return dict(a=float(coef[0]), b=float(coef[1]), residual_sd=float(math.sqrt(float(res @ res) / max(len(pats) - 2, 1))),
                r=r, n=len(pats))


def apply_rule(rule, collapse_percent, rho_meas=None, L=None):
    """(E [Pa], tissue_density [kg/m3]) of the forward run. rho_meas: the patient's CT density (E_rho,
    sg_rho); L [mm]: its extent along gravity (sg_rho)."""
    a, b = rule["a"], rule["b"]
    if rule["rule"] == "E":
        return 10.0 ** (a + b * collapse_percent), rule["tissue_density"]
    if rho_meas is None:
        raise ValueError("rule %s needs the patient's CT density (physics.ct_density)" % rule["rule"])
    if rule["rule"] == "E_rho":
        return 10.0 ** (a + b * collapse_percent), rho_meas
    if L is None:
        raise ValueError("rule sg_rho needs the lung's extent along gravity")
    sg = 10.0 ** (a + b * collapse_percent + rule["c"] * math.log10(rho_meas))
    rho = rule["tissue_density"]
    return rho * G / 1000.0 * L / sg, rho


def rule_overrides(rule, collapse_percent, rho_meas=None, L=None):
    """--set items that put the rule's parameters in a config (E, tissue_density, nu). Rules E and E_rho
    use physics.E_Pa {"rule": "collapse", a, b} so that the summary records them."""
    E, rho = apply_rule(rule, collapse_percent, rho_meas, L)
    out = ["physics.tissue_density=%.6g" % rho]
    if rule["rule"] in ("E", "E_rho"):
        out.append("physics.E_Pa=%s" % json.dumps({"rule": "collapse", "a": rule["a"], "b": rule["b"]}))
    else:
        out.append("physics.E_Pa=%.6g" % E)
    if rule.get("nu") is not None:
        out.append("physics.material.nu=%.6g" % rule["nu"])
    return out


def slope_of(E, rho, L):
    """Gravity slope s_g = rho g L / E (p/E across the lung)."""
    return rho * G / 1000.0 * L / E


def load_rule(path):
    with open(path) as f:
        r = json.load(f)
    if r.get("rule") not in RULES or "a" not in r or "b" not in r:
        raise ValueError("%s is not a rule file of main_calibrate.py" % path)
    r.setdefault("source", str(path))
    return r


def default_rules_path():
    """The rule file in use: $EURELAPS_RULES when set, else the packaged eurelaps_lung/data/rules.json."""
    env = os.environ.get(RULES_ENV)
    if env:
        return Path(env)
    return Path(str(resources.files("eurelaps_lung").joinpath("data", RULES_FILE)))


def load_rules(path=None):
    """The rules of a file (a path), or the ones in use (None: default_rules_path())."""
    return load_rule(path or default_rules_path())


def write_default_rules(rule, note=None):
    """Make `rule` (fit_rule's record) the packaged default (eurelaps_lung/data/rules.json): the built-in E
    rule of {"rule": "collapse"} and the q0 rule of estimate(). The package needs a new version after it."""
    path = Path(str(resources.files("eurelaps_lung").joinpath("data", RULES_FILE)))
    rec = {"_how": note or ("The prediction rules of eurelaps_lung (the default): written by main_calibrate.py "
                            "--make-default, read by config.py ({\"rule\": \"collapse\"}) and predict.estimate()")}
    rec.update({k: v for k, v in rule.items() if k not in ("_how", "source")})
    path.write_text(json.dumps(rec, indent=2) + "\n")
    return path


def estimate(rule, collapse_percent=None, p0=None, E=None):
    """What a run should give, from the rules (before running it). Either
      collapse_percent [, E]:  E from the E rule (unless given), q0 from the q0 rule, p0 = q0 E;
      p0 and E:                q0 = p0 / E, the collapse % from the q0 rule inverted.
    Returns {collapse_percent, E_Pa, p0_Pa, q0, from: "collapse" | "p0", in_range: the collapse % inside the
    range the rules were fitted on}. The E rule needs its predictors: only rule E gives E alone."""
    q = rule.get("q0")
    if q is None:
        raise ValueError("the rule file has no q0 rule (made before 0.1.1): run main_calibrate.py --collect again")
    if (collapse_percent is None) == (p0 is None):
        raise ValueError("estimate: give the collapse %, or p0 and E")
    if collapse_percent is not None:
        c = float(collapse_percent)
        if E is None:
            if rule["rule"] != "E":
                raise ValueError("rule %s needs the patient's density (and extent): give E" % rule["rule"])
            E = apply_rule(rule, c)[0]
        q0 = 10.0 ** (q["a"] + q["b"] * c)
        out = dict(collapse_percent=c, E_Pa=float(E), q0=q0, p0_Pa=q0 * float(E), **{"from": "collapse"})
    else:
        if E is None:
            raise ValueError("estimate from p0 needs E: the E rule needs the collapse %, which is unknown here")
        q0 = float(p0) / float(E)
        if q0 <= 0:
            raise ValueError("p0 and E must be > 0")
        c = (math.log10(q0) - q["a"]) / q["b"]
        out = dict(collapse_percent=c, E_Pa=float(E), q0=q0, p0_Pa=float(p0), **{"from": "p0"})
    lo, hi = rule.get("collapse_range", (-math.inf, math.inf))
    out["in_range"] = bool(lo <= out["collapse_percent"] <= hi)
    return out
