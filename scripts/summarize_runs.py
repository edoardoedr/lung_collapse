"""One table of the runs found under result folders (output_dir/runs/<run_name>/summary.json).

    python scripts/summarize_runs.py results/ [more folders] [--runs fit gE1000] [--out analysis/runs_v2.csv]

json / csv only (runs on the Mac). Per run: patient, run, kind (collapse / fit), the physical
parameters (E, p0, q0, nu, gravity slope, what was fitted), collapse %, ASSD and errors against
the collapsed lung, wall penetration, solves and failures. The old layout (fem/<run>/
result_summary.json, from main) is read by scripts/summarize_fits.py.
"""

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

COLS = ("patient", "case", "run", "kind", "E_Pa", "p0_Pa", "q0", "nu", "gravity_slope", "fitted", "collapse_percent",
        "target_collapse_percent", "assd_mm", "baseline_assd_mm", "mean_err_mm", "target_to_fitted_mm", "max_err_mm",
        "rigid_rotation_deg", "wall_max_penetration_mm", "wall_field_max_penetration_mm", "evals", "failed_solves",
        "elapsed_min", "stop_reason", "fail_maxit", "fail_time", "fail_unsettled")


def row(path):
    s = json.loads(path.read_text())
    ph = s["physics"]
    g = ph.get("gravity") or {}
    st = s.get("solver_stats", {})
    r = dict(patient=s.get("patient"), case=path.parents[2].name, run=path.parent.name, kind=s.get("kind"),
             E_Pa=ph["E_Pa"], p0_Pa=ph["p0_Pa"], q0=ph["q0"], nu=ph["nu"], gravity_slope=g.get("slope_pE"),
             fitted="+".join(ph.get("fitted", [])), evals=s.get("evals", s.get("solves")),
             target_collapse_percent=100 * (1 - s["target_volume_ratio"]) if "target_volume_ratio" in s else None)
    for k in COLS:
        if k not in r:
            r[k] = s.get(k, st.get(k))
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--runs", nargs="*", help="only these run names")
    ap.add_argument("--out", help="also write the table as csv")
    args = ap.parse_args()
    rows = []
    for folder in args.folders:
        for p in sorted(Path(folder).rglob("runs/*/summary.json")):
            if args.runs and p.parent.name not in args.runs:
                continue
            try:
                rows.append(row(p))
            except (KeyError, json.JSONDecodeError) as e:
                print("skipped %s (%s)" % (p, e), file=sys.stderr)
    if not rows:
        sys.exit("no runs/<run_name>/summary.json found")
    rows.sort(key=lambda r: (r["run"], r["patient"] or ""))
    show = ("patient", "run", "E_Pa", "p0_Pa", "q0", "collapse_percent", "target_collapse_percent", "assd_mm",
            "evals", "failed_solves", "elapsed_min", "wall_max_penetration_mm", "fitted")
    fmt = lambda v: "-" if v is None else ("%.3g" % v if isinstance(v, float) else str(v))   # noqa: E731
    print(" | ".join(show))
    for r in rows:
        print(" | ".join(fmt(r[k]) for k in show))
    print()
    for run in sorted({r["run"] for r in rows}):
        a = [r["assd_mm"] for r in rows if r["run"] == run and r["assd_mm"] is not None]
        if a:
            print("%-20s %2d runs: ASSD median %.2f, mean %.2f mm; failed solves %d of %d" % (
                run, len(a), statistics.median(a), statistics.mean(a),
                sum(r["failed_solves"] or 0 for r in rows if r["run"] == run),
                sum(r["evals"] or 0 for r in rows if r["run"] == run)))
    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLS)
            w.writeheader()
            w.writerows(rows)
        print("wrote %s" % args.out)


if __name__ == "__main__":
    main()
