"""Fits of several patients with one parameter shared by all of them, scanned over given values
(e.g. a common E): for each value, the fit step of every config with that value fixed.

    python scripts/fit_cohort.py --configs configs/karl*.json --param physics.E_Pa --values 1000 1700 3000 \
        [--jobs 4] [--prefix gE] [--set KEY=VALUE ...]

Each run is `python main_fit.py --config <c> --steps fit --set <param>=<value> run_name=<prefix><value>`
(logs in logs_cohort/); then one table: ASSD per patient and value, the median / mean per value
over the patients that finished all values, and the failed solves. Keep --jobs small (<= 4): many
parallel runs share one GPU and slow every solve.
"""

import argparse
import json
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eurelaps_lung.config import load_config                              # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--param", required=True, help="config key shared by all patients, e.g. physics.E_Pa")
    ap.add_argument("--values", nargs="+", required=True)
    ap.add_argument("--prefix", default=None, help="run name = prefix + value (default: last part of --param + '_')")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--set", nargs="+", metavar="KEY=VALUE", default=[], help="more overrides for every run")
    ap.add_argument("--collect", action="store_true", help="do not run, only make the table from the existing runs")
    args = ap.parse_args()
    prefix = args.prefix if args.prefix is not None else args.param.split(".")[-1] + "_"
    logs = ROOT / "logs_cohort"
    logs.mkdir(exist_ok=True)
    jobs = [(c, v) for v in args.values for c in args.configs]

    def run(job):
        c, v = job
        name = "%s%s" % (prefix, v)
        cmd = [sys.executable, str(ROOT / "main_fit.py"), "--config", c, "--steps", "fit", "--set",
               "%s=%s" % (args.param, v), "run_name=%s" % name, *args.set]
        with open(logs / ("%s_%s.txt" % (Path(c).stem, name)), "w") as f:
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT).returncode
        print("%s %s: exit %d" % (Path(c).stem, name, rc), flush=True)

    if not args.collect:
        with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
            list(ex.map(run, jobs))

    table = {}
    for c, v in jobs:
        cfg = load_config(c, ["run_name=%s%s" % (prefix, v)])
        f = cfg.run_dir / "summary.json"
        table.setdefault(cfg.patient, {})[v] = json.loads(f.read_text()) if f.is_file() else None
    print("\nASSD [mm] (p0 [Pa], failed / solves) per %s" % args.param)
    print("%-12s " % "patient" + " ".join("%24s" % v for v in args.values))
    for p, r in table.items():
        print("%-12s " % p + " ".join(
            "%24s" % ("not finished" if r[v] is None else "%.2f (%.0f, %d/%d)" % (
                r[v]["assd_mm"], r[v]["physics"]["p0_Pa"], r[v]["failed_solves"], r[v]["evals"])) for v in args.values))
    done = [p for p, r in table.items() if all(r[v] is not None for v in args.values)]
    if done:
        print("%-12s " % ("median (%d)" % len(done)) + " ".join(
            "%24.2f" % statistics.median(table[p][v]["assd_mm"] for p in done) for v in args.values))
        print("%-12s " % "mean" + " ".join(
            "%24.2f" % statistics.mean(table[p][v]["assd_mm"] for p in done) for v in args.values))
        best = {p: min(args.values, key=lambda v: table[p][v]["assd_mm"]) for p in done}
        print("best value per patient: " + ", ".join("%s %s" % kv for kv in best.items()))


if __name__ == "__main__":
    main()
