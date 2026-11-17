"""Calibrate the model on a cohort: the population parameters that predict a new patient's collapse
from its collapse % alone.

    python main_calibrate.py --configs configs/*.json --fit E [nu] [--rule E|E_rho|sg_rho]
        [--density analysis/ct_density.csv] [--fit-run fitE] [--jobs 4] [--skip-fit] [--no-loo]
        [--loo-run loo] [--frames 0] [--out calibration/E_fitE.json] [--collect] [--set KEY=VALUE ...]

Three phases (eurelaps_lung/calibration.py has the rules):

  1. fit     every patient with a collapsed lung: `main_fit.py --steps fit` with the parameters of --fit
             free (start = the config's value, bounds calibration.FIT_PARAMS; q0 always free), run
             name --fit-run. --skip-fit: use the existing runs.
  2. rule    the rule (log10 E or log10 s_g against the collapse %) on the patients whose E is not at
             a bound (--E-bounds, --bound-tol), E at the model density --tissue-density (200; fits at
             250 rescaled by 0.8, bounds compared at that density); the other fitted parameters (nu)
             get their median.
             Written to --out: the --params file of main_predict.py.
  3. loo     leave-one-out: every patient predicted (`main_predict.py`, its own collapse %) with the rule
             fitted on the OTHER patients; table of the predicted against the fitted slope and of the
             ASSD of the prediction against the patient's fit. --no-loo: skip.

--density: csv of scripts/ct_density.py (patient, rho_bar_inflated in g/mL), for the rules E_rho and
sg_rho. --collect: no runs, only the rule and the tables from the existing runs. Logs of the runs in
logs_cohort/<patient>_<run>.txt. Keep --jobs small (<= 4): parallel runs share one GPU.
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

from eurelaps_lung import calibration as C
from eurelaps_lung.config import load_config

ROOT = Path(__file__).resolve().parent
LOGS = ROOT / "logs_cohort"


def run_jobs(jobs, n):
    """jobs: [(label, log name, command)] run in parallel, n at a time."""
    LOGS.mkdir(exist_ok=True)

    def run(job):
        label, name, cmd = job
        with open(LOGS / ("%s.txt" % name), "w") as f:
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT).returncode
        print("%s: exit %d" % (label, rc), flush=True)

    with ThreadPoolExecutor(max_workers=max(1, n)) as ex:
        list(ex.map(run, jobs))


def read_density(path):
    with open(path) as f:
        return {r["patient"]: 1000.0 * float(r["rho_bar_inflated"]) for r in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--fit", nargs="+", default=["E"], metavar="PARAM",
                    help="parameters fitted per patient besides q0: %s" % ", ".join(n for n in C.FIT_PARAMS if n != "q0"))
    ap.add_argument("--fit-run", help="run name of the per-patient fits (default: fit + the parameters, e.g. fitE)")
    ap.add_argument("--rule", default="E", choices=C.RULES)
    ap.add_argument("--density", help="csv of scripts/ct_density.py (rules E_rho, sg_rho)")
    ap.add_argument("--E-bounds", nargs=2, type=float, default=list(C.FIT_PARAMS["E"][1]),
                    help="E bounds of the fits [Pa] at --tissue-density (scaled to each config's tissue_density)")
    ap.add_argument("--tissue-density", type=float, default=C.RHO_REF,
                    help="model density of the rules E / sg_rho [kg/m3]; fits at another density are rescaled")
    ap.add_argument("--bound-tol", type=float, default=0.10, help="relative distance from an E bound = at the bound")
    ap.add_argument("--skip-fit", action="store_true", help="use the existing fit runs")
    ap.add_argument("--no-loo", action="store_true", help="skip the leave-one-out")
    ap.add_argument("--loo-run", help="run name of the leave-one-out predictions (default loo / loo_rho / loo_sgrho)")
    ap.add_argument("--frames", type=int, default=0, help="> 0: the leave-one-out runs also write the sequence")
    ap.add_argument("--out", help="rule file (default calibration/<rule>_<fit-run>.json)")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--collect", action="store_true", help="no runs: rule and tables from the existing runs")
    ap.add_argument("--set", nargs="+", metavar="KEY=VALUE", default=[], help="more overrides for every run")
    args = ap.parse_args()

    names = sorted({C.param_name(n) for n in args.fit} - {"q0"})
    if "E" not in names:
        sys.exit("the rules predict E: --fit must include E")
    fit_run = args.fit_run or "fit" + "".join(names)
    loo_run = args.loo_run or {"E": "loo", "E_rho": "loo_rho", "sg_rho": "loo_sgrho"}[args.rule]
    out = Path(args.out) if args.out else ROOT / "calibration" / ("%s_%s.json" % (args.rule, fit_run))
    rho_meas = {}
    if args.rule != "E":
        if not args.density:
            sys.exit("--rule %s needs --density (csv of scripts/ct_density.py)" % args.rule)
        rho_meas = read_density(args.density)

    cfgs = {}                                                   # patient -> config path
    for c in args.configs:
        cfg = load_config(c, ["run_name=%s" % fit_run] + args.set)
        if cfg.collapsed is None:
            print("%s: no collapsed lung, skipped" % cfg.patient)
            continue
        cfgs[cfg.patient] = (c, cfg)

    # ---- 1. per-patient fits
    if not (args.skip_fit or args.collect):
        bounds = {"E": tuple(args.E_bounds)}
        jobs = [("%s %s" % (p, fit_run), "%s_%s" % (p, fit_run),
                 [sys.executable, str(ROOT / "main_fit.py"), "--config", c, "--steps", "fit", "--set",
                  *C.fit_overrides(cfg, names, bounds, args.tissue_density), "run_name=%s" % fit_run, *args.set])
                for p, (c, cfg) in cfgs.items()]
        print("phase 1: %d fits (%s free), run %s" % (len(jobs), ", ".join(["q0"] + names), fit_run))
        run_jobs(jobs, args.jobs)

    fits = {}
    for p, (c, cfg) in cfgs.items():
        f = cfg.run_dir / "summary.json"
        if not f.is_file():
            print("%s: no %s run, skipped" % (p, fit_run))
            continue
        if rho_meas and p not in rho_meas:
            print("%s: no density in %s, skipped" % (p, args.density))
            continue
        fits[p] = json.loads(f.read_text())

    # ---- 2. the rule on every usable patient
    usable = sorted(p for p, s in fits.items() if C.in_bounds(s, args.E_bounds, args.bound_tol, args.tissue_density))
    print("\nphase 2: rule %s on the fits %s with E inside the bounds (%d): %s" % (
        args.rule, fit_run, len(usable), ", ".join(usable)))
    print("left out (E at a bound): %s" % (", ".join(sorted(set(fits) - set(usable))) or "-"))
    if len(usable) < 3:
        sys.exit("too few fits for a rule (%d usable)" % len(usable))
    rule = C.fit_rule({p: fits[p] for p in usable}, args.rule, rho_meas, args.tissue_density,
                      fit_run=fit_run, fitted=["q0"] + names, E_bounds=list(args.E_bounds),
                      configs=[str(Path(cfgs[p][0])) for p in usable])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rule, indent=2))
    print("coefficients %s, residual sd %.3f (log10), tissue_density %.0f, nu %.3f, collapse %.0f-%.0f %%" % (
        ", ".join("%.5g" % c for c in rule["coef"]), rule["residual_sd"], rule["tissue_density"], rule["nu"],
        *rule["collapse_range"]))
    print("rule file: %s  (main_predict.py --params %s)" % (out, out))
    if args.no_loo:
        return

    # ---- 3. leave-one-out
    jobs, rows = [], []
    for p, s in sorted(fits.items()):
        others = {q: fits[q] for q in usable if q != p}
        r = C.fit_rule(others, args.rule, rho_meas, args.tissue_density)
        rho_p = rho_meas.get(p)
        E_pred, rho = C.apply_rule(r, C.collapse_pct(s), rho_p, C.extent(s))
        sets = C.rule_overrides(r, C.collapse_pct(s), rho_p, C.extent(s))
        rows.append(dict(p=p, s=s, E_pred=E_pred, rho=rho, sg_pred=C.slope_of(E_pred, rho, C.extent(s))))
        c = cfgs[p][0]
        jobs.append(("%s %s (E_pred %.0f Pa, rho %.0f)" % (p, loo_run, E_pred, rho), "%s_%s" % (p, loo_run),
                     [sys.executable, str(ROOT / "main_predict.py"), "--config", c, "--collapse",
                      "%.6f" % C.collapse_pct(s), "--run", loo_run, "--frames", str(args.frames),
                      "--set", *sets, *args.set]))
    if not args.collect:
        print("\nphase 3: leave-one-out, %d predictions, run %s" % (len(jobs), loo_run))
        run_jobs(jobs, args.jobs)
    loo_table(rows, usable, cfgs, loo_run)


def loo_table(rows, usable, cfgs, loo_run):
    print("\n%-11s %7s %7s %6s %8s %8s %8s %8s %8s %8s %7s" % (
        "patient", "target%", "pred%", "rho", "E_pred", "sg_fit", "sg_pred", "sg_ratio", "ASSDfit", "ASSDloo", "diff"))
    diffs, loo_all, fit_all, lr = [], [], [], []
    for j in rows:
        p, s = j["p"], j["s"]
        f = load_config(cfgs[p][0], ["run_name=%s" % loo_run]).run_dir / "summary.json"
        r = json.loads(f.read_text()) if f.is_file() else None
        mark = "" if p in usable else " *"
        ratio = j["sg_pred"] / C.slope(s)
        if p in usable:
            lr.append(math.log10(ratio))
        head = "%-11s %7.1f" % (p, C.collapse_pct(s))
        if r is None:
            print("%s %7s %6.0f %8.0f %8.3f %8.3f %8.2f %8.2f %8s%s" % (
                head, "-", j["rho"], j["E_pred"], C.slope(s), j["sg_pred"], ratio, s["assd_mm"], "missing", mark))
            continue
        d = r["assd_mm"] - s["assd_mm"]
        if p in usable:
            diffs.append(d)
            loo_all.append(r["assd_mm"])
            fit_all.append(s["assd_mm"])
        print("%s %7.1f %6.0f %8.0f %8.3f %8.3f %8.2f %8.2f %8.2f %+7.2f%s" % (
            head, r["collapse_percent"], j["rho"], j["E_pred"], C.slope(s), j["sg_pred"], ratio, s["assd_mm"],
            r["assd_mm"], d, mark))
    if lr:
        rms = math.sqrt(statistics.mean(x * x for x in lr))
        print("\nslope error (patients in bounds): rms of log10 %.3f (x%.2f)" % (rms, 10 ** rms))
    if diffs:
        print("%d patients in bounds: ASSD fit median %.2f / mean %.2f; prediction median %.2f / mean %.2f; "
              "difference median %+.2f / mean %+.2f / max %+.2f mm" % (
                  len(diffs), statistics.median(fit_all), statistics.mean(fit_all), statistics.median(loo_all),
                  statistics.mean(loo_all), statistics.median(diffs), statistics.mean(diffs), max(diffs)))
    print("* = E of the fit at a bound (not used in the rule nor in the summary)")


if __name__ == "__main__":
    main()
