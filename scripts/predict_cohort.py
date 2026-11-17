"""The model's prediction for every patient: the collapsed lung from its collapse % alone.

    python scripts/predict_cohort.py --configs configs/*.json [--params calibration/E_fitE.json] [--run pred]
        [--frames 30] [--jobs 4] [--collect] [--set KEY=VALUE ...]

For each config: the collapse % of the patient's collapsed lung (V_collapsed / V_inflated of
fem/setup/problem.npz), then one forward run with E from the collapse % and q0 searched to reach
that volume:

    python main_predict.py --config <c> --collapse <the patient's %> [--params <rule file>] --frames <frames> --run <run>

E from --params (rule file of main_calibrate.py), else from the built-in rule (physics.E_Pa
{"rule": "collapse"}, eurelaps_lung/config.py). Logs in logs_cohort/<patient>_<run>.txt; then the
table: collapse target / reached, E, p0 (and with the CT density), ASSD against the real collapsed
lung, rigid rotation, wall.
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

from eurelaps_lung.fem.geometry import signed_volume                 # noqa: E402
from eurelaps_lung.fem.problem import CollapseProblem                 # noqa: E402
from eurelaps_lung.config import load_config                               # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--params", help="rule file of main_calibrate.py (default: the built-in rule)")
    ap.add_argument("--run", default="pred")
    ap.add_argument("--frames", type=int, default=30, help="collapse sequence frames (0 = none)")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--set", nargs="+", metavar="KEY=VALUE", default=[], help="more overrides for every run")
    ap.add_argument("--collect", action="store_true", help="do not run, only make the table")
    args = ap.parse_args()

    jobs = []
    for c in args.configs:
        cfg = load_config(c, ["run_name=%s" % args.run])
        f = cfg.setup_dir / "problem.npz"
        if not f.is_file():
            print("%s: no %s (run geometry, hilum, fem_setup first), skipped" % (cfg.patient, f))
            continue
        p = CollapseProblem.load(f)
        if not p.has_target:
            print("%s: no collapsed lung in problem.npz, skipped" % cfg.patient)
            continue
        ratio = signed_volume(*p.target_surface) / signed_volume(p.reference, p.tris)
        jobs.append((cfg.patient, c, ratio, cfg.run_dir))

    logs = ROOT / "logs_cohort"
    logs.mkdir(exist_ok=True)

    def run(job):
        patient, c, ratio, _ = job
        cmd = [sys.executable, str(ROOT / "main_predict.py"), "--config", c, "--collapse", "%.6f" % (100 * (1 - ratio)),
               "--frames", str(args.frames), "--run", args.run]
        sets = list(args.set)
        if args.params:
            cmd += ["--params", args.params]
        else:
            sets.insert(0, 'physics.E_Pa={"rule": "collapse"}')
        cmd += ["--set", *sets]
        with open(logs / ("%s_%s.txt" % (patient, args.run)), "w") as f:
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT).returncode
        print("%s %s: exit %d (collapse %.1f %%)" % (patient, args.run, rc, 100 * (1 - ratio)), flush=True)

    if not args.collect:
        with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
            list(ex.map(run, jobs))

    print("\n%-11s %7s %7s %7s %7s %6s %7s %7s %7s %7s %6s %6s" % (
        "patient", "target%", "pred%", "E", "p0", "rhoCT", "E_ct", "p0_ct", "ASSD", "rot", "wall", "solves"))
    assd = []
    for patient, _, ratio, run_dir in jobs:
        f = run_dir / "summary.json"
        if not f.is_file():
            print("%-11s %7.1f  missing (see logs_cohort/%s_%s.txt)" % (patient, 100 * (1 - ratio), patient, args.run))
            continue
        s = json.loads(f.read_text())
        ph = s["physics"]
        ct = ph.get("ct_density_kg_m3")
        assd.append(s["assd_mm"])
        print("%-11s %7.1f %7.1f %7.0f %7.0f %6s %7s %7s %7.2f %7.1f %6.2f %6d" % (
            patient, 100 * (1 - ratio), s["collapse_percent"], ph["E_Pa"], ph["p0_Pa"],
            "%.0f" % ct if ct else "-", "%.0f" % ph["E_Pa_ct"] if ct else "-", "%.0f" % ph["p0_Pa_ct"] if ct else "-",
            s["assd_mm"], s["rigid_rotation_deg"], s.get("wall_max_penetration_mm", 0.0), s["solves"]))
    if assd:
        print("\n%d patients: ASSD median %.2f, mean %.2f mm" % (len(assd), statistics.median(assd), statistics.mean(assd)))


if __name__ == "__main__":
    main()
