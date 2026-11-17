"""Leave-one-out test of the transferable model: predict each patient's collapsed lung from its
collapse % (and, optionally, its lung density measured on the CT), with the gravity load from a
rule fitted on the OTHER patients' fits.

    python scripts/leave_one_out.py --configs configs/*.json [--fit-run fitE_nu30] [--rule E|E_rho|sg_rho]
        [--density analysis/ct_density.csv] [--run NAME] [--jobs 4] [--frames 30] [--collect] [--set KEY=VALUE ...]

Prediction setting: inputs = inflated lung, collapse %, (density rho_bar); nu as in the config; q0
from the volume (forward `volume_ratio`); the gravity slope s_g = rho_bar g L / E from a rule:

  E       log10 E = a + b * collapse %, rho_bar = the config's tissue_density for all (default run loo)
  E_rho   the same on E_true = E_fit * rho_meas / rho_fit, forward with physics.tissue_density = the
          patient's rho_meas (needs --density; run loo_rho): a material E with the measured weight
  sg_rho  log10 s_g = a + b * collapse % + c * log10 rho_meas, forward with E = rho_fit g L / s_g
          (needs --density; run loo_sgrho): the density as a free predictor of the slope

The rule is fitted on the other patients whose `--fit-run` has E inside --E-bounds; each patient
gets one forward run (logs in logs_cohort/). Table: the predicted slope against the fitted one,
ASSD of the prediction against the patient's own fit. --density: csv of scripts/ct_density.py
(patient, rho_bar_inflated in g/mL).
"""

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.config import load_config                              # noqa: E402

G = 9.81


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--fit-run", default="fitE_nu30", help="run with E fitted per patient")
    ap.add_argument("--rule", default="E", choices=["E", "E_rho", "sg_rho"])
    ap.add_argument("--density", help="csv of scripts/ct_density.py (rules E_rho, sg_rho)")
    ap.add_argument("--run", help="run name of the predictions (default loo / loo_rho / loo_sgrho)")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--frames", type=int, default=0, help="> 0: also write the collapse sequence")
    ap.add_argument("--E-bounds", nargs=2, type=float, default=[200.0, 20000.0], help="E bounds of the fit run [Pa]")
    ap.add_argument("--bound-tol", type=float, default=0.10, help="relative distance from an E bound = at the bound")
    ap.add_argument("--set", nargs="+", metavar="KEY=VALUE", default=[], help="more overrides for every run")
    ap.add_argument("--collect", action="store_true", help="do not run, only make the table")
    args = ap.parse_args()
    run_name = args.run or {"E": "loo", "E_rho": "loo_rho", "sg_rho": "loo_sgrho"}[args.rule]
    rho_meas = {}
    if args.rule != "E":
        if not args.density:
            sys.exit("--rule %s needs --density (csv of scripts/ct_density.py)" % args.rule)
        with open(args.density) as f:
            rho_meas = {r["patient"]: 1000.0 * float(r["rho_bar_inflated"]) for r in csv.DictReader(f)}

    fits = {}                                          # patient -> (config, summary of the fit run)
    for c in args.configs:
        cfg = load_config(c, ["run_name=%s" % args.fit_run])
        f = cfg.run_dir / "summary.json"
        if not f.is_file():
            print("%s: no %s run, skipped" % (cfg.patient, args.fit_run))
            continue
        if rho_meas and cfg.patient not in rho_meas:
            print("%s: no density in %s, skipped" % (cfg.patient, args.density))
            continue
        fits[cfg.patient] = (c, json.loads(f.read_text()))

    def in_bounds(s):
        lo, hi = args.E_bounds
        E = s["physics"]["E_Pa"]
        return lo * (1 + args.bound_tol) < E < hi * (1 - args.bound_tol)

    usable = sorted(p for p, (_, s) in fits.items() if in_bounds(s))
    print("rule %s, run %s; fits with E inside the bounds (%d): %s" % (args.rule, run_name, len(usable), ", ".join(usable)))
    print("left out of the regression (E at a bound): %s" % ", ".join(sorted(set(fits) - set(usable))))

    ph = lambda p: fits[p][1]["physics"]                                     # noqa: E731
    tc = lambda p: 100.0 * (1.0 - fits[p][1]["target_volume_ratio"])         # noqa: E731
    rho_fit = lambda p: ph(p).get("tissue_density_kg_m3", 250.0)             # noqa: E731
    L = lambda p: ph(p)["gravity"]["extent_mm"]                              # noqa: E731
    sg = lambda p: ph(p)["gravity"]["slope_pE"]                              # noqa: E731

    def features(p):
        if args.rule == "sg_rho":
            return [1.0, tc(p), math.log10(rho_meas[p])]
        return [1.0, tc(p)]

    def target(p):
        if args.rule == "E":
            return math.log10(ph(p)["E_Pa"])
        if args.rule == "E_rho":
            return math.log10(ph(p)["E_Pa"] * rho_meas[p] / rho_fit(p))
        return math.log10(sg(p))

    jobs = []
    for p, (c, s) in fits.items():
        others = [q for q in usable if q != p]
        coef = np.linalg.lstsq(np.array([features(q) for q in others]), np.array([target(q) for q in others]),
                               rcond=None)[0]
        y = float(np.dot(features(p), coef))
        sets = ['physics.collapse={"volume_ratio": %.5f}' % s["target_volume_ratio"]]
        if args.rule == "E":
            E_pred, rho = 10 ** y, rho_fit(p)
        elif args.rule == "E_rho":
            E_pred, rho = 10 ** y, rho_meas[p]
            sets.append("physics.tissue_density=%.1f" % rho)
        else:
            rho = rho_fit(p)
            E_pred = rho * G / 1000.0 * L(p) / 10 ** y
        sets.append("physics.E_Pa=%.1f" % E_pred)
        sg_pred = rho * G / 1000.0 * L(p) / E_pred
        jobs.append(dict(p=p, c=c, s=s, E_pred=E_pred, rho=rho, sg_pred=sg_pred, sets=sets, coef=coef))

    logs = ROOT / "logs_cohort"
    logs.mkdir(exist_ok=True)

    def run(j):
        cmd = [sys.executable, str(ROOT / "main.py"), "--config", j["c"], "--steps", "collapse", "--set", *j["sets"],
               "collapse.sequence_frames=%d" % args.frames, "run_name=%s" % run_name, *args.set]
        with open(logs / ("%s_%s.txt" % (j["p"], run_name)), "w") as f:
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT).returncode
        print("%s %s: exit %d (E_pred %.0f Pa, rho %.0f, slope %.3f)" % (j["p"], run_name, rc, j["E_pred"], j["rho"],
                                                                         j["sg_pred"]), flush=True)

    if not args.collect:
        with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
            list(ex.map(run, jobs))

    print("\n%-11s %7s %7s %6s %8s %8s %8s %8s %8s %8s %7s" % (
        "patient", "target%", "pred%", "rho", "E_pred", "sg_fit", "sg_pred", "sg_ratio", "ASSDfit", "ASSDloo", "diff"))
    diffs, loo_all, fit_all, lr = [], [], [], []
    for j in jobs:
        p, s = j["p"], j["s"]
        f = load_config(j["c"], ["run_name=%s" % run_name]).run_dir / "summary.json"
        r = json.loads(f.read_text()) if f.is_file() else None
        mark = "" if p in usable else " *"
        ratio = j["sg_pred"] / sg(p)
        if p in usable:
            lr.append(math.log10(ratio))
        if r is None:
            print("%-11s %7.1f %7s %6.0f %8.0f %8.3f %8.3f %8.2f %8.2f %8s%s" % (
                p, tc(p), "-", j["rho"], j["E_pred"], sg(p), j["sg_pred"], ratio, s["assd_mm"], "missing", mark))
            continue
        d = r["assd_mm"] - s["assd_mm"]
        if p in usable:
            diffs.append(d)
            loo_all.append(r["assd_mm"])
            fit_all.append(s["assd_mm"])
        print("%-11s %7.1f %7.1f %6.0f %8.0f %8.3f %8.3f %8.2f %8.2f %8.2f %+7.2f%s" % (
            p, tc(p), r["collapse_percent"], j["rho"], j["E_pred"], sg(p), j["sg_pred"], ratio, s["assd_mm"],
            r["assd_mm"], d, mark))
    if lr:
        print("\nslope error (patients in bounds): rms of log10 %.3f (x%.2f)" % (
            math.sqrt(statistics.mean(x * x for x in lr)), 10 ** math.sqrt(statistics.mean(x * x for x in lr))))
    if diffs:
        print("%d patients in bounds: ASSD fit median %.2f / mean %.2f; prediction median %.2f / mean %.2f; "
              "difference median %+.2f / mean %+.2f / max %+.2f mm" % (
                  len(diffs), statistics.median(fit_all), statistics.mean(fit_all), statistics.median(loo_all),
                  statistics.mean(loo_all), statistics.median(diffs), statistics.mean(diffs), max(diffs)))
    print("* = E of the fit at a bound (not used in the rule nor in the summary)")


if __name__ == "__main__":
    main()
