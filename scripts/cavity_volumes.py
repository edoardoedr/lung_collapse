"""Is the pleural cavity (the model's "inflated lung") larger than the lung that filled it?

    python scripts/cavity_volumes.py --configs configs/*.json [--fit-run fitE_nu30] [--out analysis/cavity.csv]

There is one CT per patient, with the pneumothorax: the reference is its pleural cavity, which the
loss of the pleural pressure enlarges (chest wall out, mediastinum to the other side, diaphragm
down). Per patient, from the input surfaces only (data_dir):
  cavity, collapsed and contralateral volumes (the contralateral lung = the union of its lobes,
  lung_*_lobe_<other side>.vtk, summed);
  enlargement = (cavity / contralateral) / the normal ratio (right / left = --rl-ratio, default 1.15):
  1 = a cavity of normal size, 1.2 = 20 % larger than expected;
  collapse % against the cavity, and the collapse % if the lung had filled only the expected volume
  (contralateral x normal ratio);
  trachea shift: x of the trachea (top 15 mm of the airway tree) against the midline of the chest
  (middle of the lateral extents of the two sides), positive = away from the pneumothorax. A proxy
  of the mediastinal shift (the heart is not segmented).
With --fit-run, E and the ASSD of that run, and the correlations of log10 E with the enlargement,
the collapse % and the trachea shift. Needs VTK (server, or the Mac env lungct).
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eurelaps_lung import calibration as C                               # noqa: E402
from eurelaps_lung.config import load_config                              # noqa: E402
from eurelaps_lung import vtk_mesh as vm                                  # noqa: E402
from eurelaps_lung.data_io import read_surface                            # noqa: E402


def volume(mesh):
    m = vm.clean(mesh)                         # clean() turns degenerate triangles into lines: dropped
    return abs(m.volume) / 1000.0, vm.boundary_edges(m)             # mL


def corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    r = float(np.corrcoef(x, y)[0, 1])
    rank = lambda v: np.argsort(np.argsort(v))                       # noqa: E731
    rs = float(np.corrcoef(rank(x), rank(y))[0, 1])
    return r, rs


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--fit-run", default=None, help="run with E fitted (e.g. fitE_nu30)")
    ap.add_argument("--rl-ratio", type=float, default=1.15, help="normal right / left lung volume ratio")
    ap.add_argument("--out", help="also write the table as csv")
    args = ap.parse_args()

    rows = []
    for c in args.configs:
        cfg = load_config(c, ["run_name=%s" % args.fit_run] if args.fit_run else None)
        name = cfg.inflated.name
        side = "left" if "left" in name else "right" if "right" in name else None
        if side is None or cfg.collapsed is None or not cfg.inflated.is_file():
            print("%s: inputs not usable (%s), skipped" % (cfg.patient, name))
            continue
        other = "right" if side == "left" else "left"
        lobes = sorted(cfg.data_dir.glob("lung_*_lobe_%s.vtk" % other))
        if not lobes:
            print("%s: no lung_*_lobe_%s.vtk, skipped" % (cfg.patient, other))
            continue
        cav_m, col_m = read_surface(cfg.inflated), read_surface(cfg.collapsed)
        contra_m = [read_surface(f) for f in lobes]
        V_cav, open_cav = volume(cav_m)
        V_col, open_col = volume(col_m)
        vc = [volume(m) for m in contra_m]
        V_con, open_con = sum(v for v, _ in vc), sum(o for _, o in vc)
        expected = V_con * (args.rl_ratio if side == "right" else 1.0 / args.rl_ratio)

        # trachea shift: airway points within 15 mm of the top, against the chest midline (LPS: +x = left)
        shift = None
        aw = cfg.structures.get("airways")
        if aw is not None and aw.is_file():
            P = np.asarray(read_surface(aw).points)
            x_tr = P[P[:, 2] > P[:, 2].max() - 15.0, 0].mean()
            Pc = np.asarray(cav_m.points)
            Po = np.vstack([np.asarray(m.points) for m in contra_m])
            left_x, right_x = (Pc[:, 0].max(), Po[:, 0].min()) if side == "left" else (Po[:, 0].max(), Pc[:, 0].min())
            mid = 0.5 * (left_x + right_x)
            shift = float((mid - x_tr) if side == "left" else (x_tr - mid))   # > 0: away from the pneumothorax

        r = dict(patient=cfg.patient, side=side, V_cavity_mL=V_cav, V_collapsed_mL=V_col, V_contra_mL=V_con,
                 n_lobes=len(lobes), enlargement=V_cav / expected, collapse_vs_cavity=100 * (1 - V_col / V_cav),
                 collapse_vs_expected=100 * (1 - V_col / expected), trachea_shift_mm=shift,
                 open_edges=open_cav + open_col + open_con, E_Pa=None, assd_mm=None)
        if args.fit_run:
            f = cfg.run_dir / "summary.json"
            if f.is_file():
                s = json.loads(f.read_text())
                r["E_Pa"], r["assd_mm"] = C.E_at(s), s["assd_mm"]       # E at the model density 200
                r["E_in_bounds"] = C.in_bounds(s)
        rows.append(r)

    hdr = ("patient", "side", "V_cavity_mL", "V_collapsed_mL", "V_contra_mL", "enlargement", "collapse_vs_cavity",
           "collapse_vs_expected", "trachea_shift_mm", "E_Pa", "assd_mm", "open_edges")
    fmt = lambda v: "-" if v is None else ("%.3g" % v if isinstance(v, float) else str(v))   # noqa: E731
    print(" | ".join(hdr))
    for r in rows:
        print(" | ".join(fmt(r[k]) for k in hdr))
    if any(r["open_edges"] for r in rows):
        print("open_edges > 0: a surface is not closed, its volume is approximate")

    en = [r["enlargement"] for r in rows]
    print("\nenlargement: median %.3f, range %.3f-%.3f (1 = normal right/left ratio %.2f)" % (
        float(np.median(en)), min(en), max(en), args.rl_ratio))
    print("enlargement vs collapse %% (vs cavity): r %.2f, Spearman %.2f" % corr(
        en, [r["collapse_vs_cavity"] for r in rows]))
    if args.fit_run:
        ok = [r for r in rows if r.get("E_in_bounds")]               # drop the patients with E at a bound
        if len(ok) >= 4:
            lE = [math.log10(r["E_Pa"]) for r in ok]
            print("\nlog10 E, %d patients with E inside the bounds:" % len(ok))
            for k in ("enlargement", "collapse_vs_cavity", "collapse_vs_expected", "trachea_shift_mm"):
                pts = [(r[k], e) for r, e in zip(ok, lE) if r[k] is not None]
                if len(pts) >= 4:
                    print("  vs %-22s r %+.2f, Spearman %+.2f" % ((k,) + corr(*zip(*pts))))
    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print("wrote %s" % args.out)


if __name__ == "__main__":
    main()
