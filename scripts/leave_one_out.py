"""Leave-one-out test of the transferable model: predict each patient's collapsed lung from its
collapse % alone, with E from the E(collapse %) regression of the OTHER patients' fits.

    python scripts/leave_one_out.py --configs configs/*.json [--fit-run fitE] [--run loo] [--jobs 4]
        [--frames 30] [--collect] [--set KEY=VALUE ...]

For each patient: fit log10 E = a + b * collapse % on the other patients whose `--fit-run` has E
inside its bounds (not at a bound), E_pred from the patient's own target collapse %, then one
forward run `python main.py --config <c> --steps collapse --set physics.E_Pa=<E_pred>
'physics.collapse={"volume_ratio": <target ratio>}' run_name=<run>` (logs in logs_cohort/).
Table: E_pred vs E fitted, ASSD of the prediction vs the patient's own fit.
"""

import argparse
import json
import math
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.config import load_config                              # noqa: E402


def fit_line(xs, ys):
    mx, my = statistics.mean(xs), statistics.mean(ys)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    return my - b * mx, b


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--fit-run", default="fitE", help="run with E fitted per patient")
    ap.add_argument("--run", default="loo", help="run name of the predictions")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--frames", type=int, default=0, help="> 0: also write the collapse sequence")
    ap.add_argument("--E-bounds", nargs=2, type=float, default=[200.0, 20000.0], help="E bounds of the fit run [Pa]")
    ap.add_argument("--bound-tol", type=float, default=0.05, help="relative distance from an E bound = at the bound")
    ap.add_argument("--set", nargs="+", metavar="KEY=VALUE", default=[], help="more overrides for every run")
    ap.add_argument("--collect", action="store_true", help="do not run, only make the table")
    args = ap.parse_args()

    fits = {}                                          # patient -> (config, summary of the fit run)
    for c in args.configs:
        cfg = load_config(c, ["run_name=%s" % args.fit_run])
        f = cfg.run_dir / "summary.json"
        if not f.is_file():
            print("%s: no %s run, skipped" % (cfg.patient, args.fit_run))
            continue
        fits[cfg.patient] = (c, json.loads(f.read_text()))

    def in_bounds(s):
        lo, hi = args.E_bounds
        E = s["physics"]["E_Pa"]
        return E > lo * (1 + args.bound_tol) and E < hi * (1 - args.bound_tol)

    usable = {p: s for p, (_, s) in fits.items() if in_bounds(s)}
    print("fits with E inside the bounds (%d): %s" % (len(usable), ", ".join(sorted(usable))))
    print("left out of the regression (E at a bound): %s" % ", ".join(sorted(set(fits) - set(usable))))
    tc = lambda s: 100 * (1 - s["target_volume_ratio"])               # noqa: E731

    jobs = []
    for p, (c, s) in fits.items():
        others = [q for q in usable if q != p]
        a, b = fit_line([tc(usable[q]) for q in others], [math.log10(usable[q]["physics"]["E_Pa"]) for q in others])
        E_pred = 10 ** (a + b * tc(s))
        jobs.append((p, c, s, E_pred, a, b))

    logs = ROOT / "logs_cohort"
    logs.mkdir(exist_ok=True)

    def run(job):
        p, c, s, E_pred, _, _ = job
        cmd = [sys.executable, str(ROOT / "main.py"), "--config", c, "--steps", "collapse", "--set",
               "physics.E_Pa=%.1f" % E_pred,
               'physics.collapse={"volume_ratio": %.5f}' % s["target_volume_ratio"],
               "collapse.sequence_frames=%d" % args.frames, "run_name=%s" % args.run, *args.set]
        with open(logs / ("%s_%s.txt" % (p, args.run)), "w") as f:
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT).returncode
        print("%s %s: exit %d (E_pred %.0f Pa)" % (p, args.run, rc, E_pred), flush=True)

    if not args.collect:
        with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
            list(ex.map(run, jobs))

    print("\n%-11s %7s %7s %8s %8s %7s %7s %8s %8s %7s" % (
        "patient", "target%", "pred%", "E_fit", "E_pred", "a", "b", "ASSDfit", "ASSDloo", "diff"))
    diffs, loo_all, fit_all = [], [], []
    for p, c, s, E_pred, a, b in jobs:
        f = load_config(c, ["run_name=%s" % args.run]).run_dir / "summary.json"
        r = json.loads(f.read_text()) if f.is_file() else None
        mark = "" if p in usable else " *"
        if r is None:
            print("%-11s %7.1f %7s %8.0f %8.0f %7.3f %7.4f %8.2f %8s%s" % (
                p, tc(s), "-", s["physics"]["E_Pa"], E_pred, a, b, s["assd_mm"], "missing", mark))
            continue
        d = r["assd_mm"] - s["assd_mm"]
        diffs.append(d)
        loo_all.append(r["assd_mm"])
        fit_all.append(s["assd_mm"])
        print("%-11s %7.1f %7.1f %8.0f %8.0f %7.3f %7.4f %8.2f %8.2f %+7.2f%s" % (
            p, tc(s), r["collapse_percent"], s["physics"]["E_Pa"], E_pred, a, b, s["assd_mm"], r["assd_mm"], d, mark))
    if diffs:
        print("\n%d patients: ASSD fit median %.2f / mean %.2f; prediction median %.2f / mean %.2f; "
              "difference median %+.2f / mean %+.2f mm" % (
                  len(diffs), statistics.median(fit_all), statistics.mean(fit_all), statistics.median(loo_all),
                  statistics.mean(loo_all), statistics.median(diffs), statistics.mean(diffs)))
    print("* = E of the fit at a bound (not used in any regression)")


if __name__ == "__main__":
    main()
