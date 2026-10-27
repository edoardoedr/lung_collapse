"""Analyse every fit found under one or more result folders: one table row per fit, paired
comparisons (registered vs not, wall vs not, regions vs gradient) and the pressure pattern.

numpy / matplotlib only (runs on the Mac): it reads fem/setup/problem.npz, setup_summary.json and
fem/<run>/{best_state.npz, result_summary.json}, nothing else.

    python scripts/summarize_fits.py results_server/            # [more folders] [--out analysis]

A case folder is any folder with fem/setup/problem.npz; the patient is the karlNN / patient_NN at the
start of its name, the variant (registered or not, clamp with still points or hilum only, wall)
comes from the setup itself. Output in --out: runs.csv, compare.csv, report.md, *.png.

All shape errors are measured here in the same way for every fit, against the collapsed lung
surface (the step-2 mesh: the same surface with and without registration, since align is none):
  fit_to_tgt   distance of each free fitted point to the target surface (mean / p95 / max)
  tgt_to_fit   distance of each target point to the fitted surface (finds uncovered parts)
  assd         mean of the two
The summary's own mean_err_mm is point-to-point for registered fits and point-to-surface without
registration, so it is reported but not compared.

Pressure pattern (p/E per free triangle, area-weighted): least squares q = c + Σ b_a ξ_a on the
normalised axes of fit.axis_coords (ap + = anterior, si + = superior, ml + = lateral), R² of that
fit, and the mean p/E of the anterior / posterior and apical / basal thirds over the mean.
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

AXES = ("ap", "si", "ml")


# ----------------------------------------------------------------------------------------------
# geometry (numpy)
# ----------------------------------------------------------------------------------------------
def tri_area(X, tris):
    a, b, c = X[tris[:, 0]], X[tris[:, 1]], X[tris[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)


def volume_ml(X, tris):
    a, b, c = X[tris[:, 0]], X[tris[:, 1]], X[tris[:, 2]]
    return abs(np.einsum("ij,ij->", a, np.cross(b, c))) / 6.0 / 1000.0


def _seg_dist(P, A, B):
    """P (n, 1, 3), segments A-B (1, T, 3) -> (n, T)."""
    ab = B - A
    t = np.clip(np.einsum("ntk,ntk->nt", P - A, np.broadcast_to(ab, (P.shape[0],) + ab.shape[1:]))
                / np.maximum(np.einsum("tk,tk->t", ab[0], ab[0]), 1e-30), 0.0, 1.0)
    return np.linalg.norm(P - (A + t[..., None] * ab), axis=2)


def point_surface_distance(P, X, tris, chunk=128):
    """Exact distance of each point P (N, 3) to the triangle surface (X, tris)."""
    A, B, C = X[tris[:, 0]][None], X[tris[:, 1]][None], X[tris[:, 2]][None]
    v0, v1 = B - A, C - A
    d00, d01, d11 = (np.einsum("ntk,ntk->nt", u, w) for u, w in ((v0, v0), (v0, v1), (v1, v1)))
    den = np.maximum(d00 * d11 - d01 ** 2, 1e-30)
    nrm = np.cross(v0, v1)
    nrm = nrm / np.maximum(np.linalg.norm(nrm, axis=2, keepdims=True), 1e-30)
    out = np.empty(len(P))
    for s in range(0, len(P), chunk):
        p = P[s:s + chunk, None, :]
        w = p - A
        d20, d21 = np.einsum("ntk,ntk->nt", w, np.broadcast_to(v0, w.shape)), \
            np.einsum("ntk,ntk->nt", w, np.broadcast_to(v1, w.shape))
        v = (d11 * d20 - d01 * d21) / den
        u = (d00 * d21 - d01 * d20) / den
        inside = (v >= 0) & (u >= 0) & (v + u <= 1)
        plane = np.abs(np.einsum("ntk,ntk->nt", w, np.broadcast_to(nrm, w.shape)))
        edge = np.minimum(np.minimum(_seg_dist(p, A, B), _seg_dist(p, B, C)), _seg_dist(p, C, A))
        out[s:s + chunk] = np.where(inside, plane, edge).min(axis=1)
    return out


def axis_coords(X_ref, tris, clamped, area, hilum):
    """Copy of pipeline.collapse.fit.axis_coords (that module needs pyvista): normalised coordinate
    of each triangle centroid, LPS reference; ap + = anterior (-y), si + = superior (+z),
    ml + = lateral (away from the hilum in x)."""
    C = X_ref[tris].mean(1)
    side = np.sign(C[~clamped, 0].mean() - hilum[0])
    raw = dict(ap=-C[:, 1], si=C[:, 2], ml=side * C[:, 0])
    free = ~clamped
    return {a: (v - np.average(v[free], weights=area[free])) / np.ptp(v[free]) for a, v in raw.items()}


def third_ratio(coord, q, w):
    """Mean q of the upper / lower area-weighted third of coord, over the mean q."""
    o = np.argsort(coord)
    cw = np.cumsum(w[o]) / w.sum()
    m = np.average(q, weights=w)
    lo, hi = o[cw <= 1 / 3], o[cw >= 2 / 3]
    return np.average(q[hi], weights=w[hi]) / m, np.average(q[lo], weights=w[lo]) / m


# ----------------------------------------------------------------------------------------------
# one fit
# ----------------------------------------------------------------------------------------------
def patient_of(name):
    """Patient id at the start of a case folder name (karl01_noreg_hilum_wall -> karl01)."""
    m = re.match(r"(karl\d+|patient_\d+)", name)
    return m.group(1) if m else name


def analyse_run(case_dir, run_dir, prob, setup):
    s = json.loads((run_dir / "result_summary.json").read_text())
    b = np.load(run_dir / "best_state.npz")
    tris, X_ref = prob["tris"], prob["nodes"][prob["surface_nodes"]]
    closest = "target_tris" in prob.files and len(prob["target_tris"]) > 0
    # variant from the setup itself, not from the folder name
    patient, reg, wall = patient_of(case_dir.name), "noreg" if closest else "reg", len(prob["wall_tris"]) > 0
    clamp = "still %g" % setup["clamp_still_mm"] if setup.get("clamp_still_mm") else "hilum only"
    T_pts, T_tris = (prob["target_points"], prob["target_tris"]) if closest else (prob["target"], tris)
    Xs = X_ref + b["surface_displacement"]
    clamped_tri = prob["clamped_tri"].astype(bool)
    free_pt = np.ones(len(X_ref), bool)
    free_pt[np.unique(tris[clamped_tri])] = False
    area = tri_area(X_ref, tris)
    hilum = np.array(setup["anchor_center_lps"][0])

    d_ft = point_surface_distance(Xs[free_pt], T_pts, T_tris)
    d_tf = point_surface_distance(T_pts, Xs, tris)
    d0 = point_surface_distance(X_ref[free_pt], T_pts, T_tris)
    V_ref, V_tgt, V_fit = volume_ml(X_ref, tris), volume_ml(T_pts, T_tris), volume_ml(Xs, tris)
    C_ref = X_ref[tris].mean(1)
    side = "left" if C_ref[~clamped_tri, 0].mean() > hilum[0] else "right"   # LPS: +x = left

    labels, q = b["tri_labels"], b["q"]
    q_tri = np.where(labels >= 0, q[np.maximum(labels, 0)], np.nan)
    f = ~clamped_tri & np.isfinite(q_tri)
    w, qf = area[f], q_tri[f]
    xi = axis_coords(X_ref, tris, clamped_tri, area, hilum)
    M = np.column_stack([np.ones(f.sum())] + [xi[a][f] for a in AXES])
    sw = np.sqrt(w)
    coef = np.linalg.lstsq(M * sw[:, None], qf * sw, rcond=None)[0]
    mean_q = np.average(qf, weights=w)
    var = np.average((qf - mean_q) ** 2, weights=w)
    r2 = 1 - np.average((qf - M @ coef) ** 2, weights=w) / var if var > 1e-12 else np.nan
    ant, post = third_ratio(xi["ap"][f], qf, w)
    apex, base = third_ratio(xi["si"][f], qf, w)

    row = dict(case=case_dir.name, patient=patient, side=side, registration=reg, clamp=clamp, wall=wall,
               run=run_dir.name, model=s.get("pressure_model", "regions"), K=s.get("K"),
               stop=s.get("stop_reason"), evals=s.get("evals"), failed=s.get("failed_solves"),
               minutes=s.get("elapsed_min"),
               surface_points=len(X_ref), clamped_area_frac=area[clamped_tri].sum() / area.sum(),
               still_tris=setup.get("still_triangles"),
               V_inflated_ml=V_ref, V_collapsed_ml=V_tgt, V_fit_ml=V_fit,
               collapse=1 - V_tgt / V_ref, collapse_fit=1 - V_fit / V_ref,
               volume_err_pct=100 * (V_fit - V_tgt) / V_tgt,
               baseline_mm=d0.mean(),
               fit_to_tgt_mm=d_ft.mean(), fit_to_tgt_p95_mm=np.percentile(d_ft, 95), fit_to_tgt_max_mm=d_ft.max(),
               tgt_to_fit_mm=d_tf.mean(), tgt_to_fit_p95_mm=np.percentile(d_tf, 95),
               assd_mm=0.5 * (d_ft.mean() + d_tf.mean()), hausdorff_mm=max(d_ft.max(), d_tf.max()),
               summary_mean_err_mm=s.get("mean_err_mm"), summary_normal_err_mm=s.get("mean_normal_err_mm"),
               rotation_fit_deg=(s.get("rigid_rotation_deg") or {}).get("fitted"),
               wall_pen_max_mm=s.get("wall_max_penetration_mm"), wall_points_beyond=s.get("wall_points_beyond"),
               pE_mean=mean_q, pE_std=np.sqrt(var), pE_min=qf.min(), pE_max=qf.max(),
               pE_r2_linear=r2, anterior_ratio=ant, posterior_ratio=post, apex_ratio=apex, base_ratio=base)
    for a, c in zip(AXES, coef[1:]):
        row["slope_" + a] = c
        row["slope_rel_" + a] = c / mean_q
    return row


def collect(roots):
    rows = []
    for root in roots:
        for pf in sorted(Path(root).rglob("fem/setup/problem.npz")):
            case_dir = pf.parents[2]
            if "old" in case_dir.relative_to(root).parts:
                continue
            ss = pf.parent / "setup_summary.json"
            if not ss.is_file():
                print("skip %s: no setup_summary.json" % case_dir, file=sys.stderr)
                continue
            prob, setup = np.load(pf), json.loads(ss.read_text())
            for run_dir in sorted((case_dir / "fem").iterdir()):
                if run_dir.name == "setup" or not (run_dir / "best_state.npz").is_file() \
                        or not (run_dir / "result_summary.json").is_file():
                    continue
                try:
                    rows.append(analyse_run(case_dir, run_dir, prob, setup))
                    print("  %-22s %-14s assd %5.2f mm" % (case_dir.name, run_dir.name, rows[-1]["assd_mm"]))
                except Exception as e:                       # one broken run must not stop the rest
                    print("skip %s/%s: %s" % (case_dir.name, run_dir.name, e), file=sys.stderr)
    return rows


# ----------------------------------------------------------------------------------------------
# tables
# ----------------------------------------------------------------------------------------------
COMPARE = ["assd_mm", "fit_to_tgt_mm", "tgt_to_fit_mm", "fit_to_tgt_p95_mm", "volume_err_pct",
           "rotation_fit_deg", "pE_mean", "slope_ap", "slope_si", "failed", "minutes"]
PAIRS = [("registration", "reg", "noreg"), ("clamp", "still 5", "hilum only"), ("wall", False, True),
         ("model", "regions", "gradient")]
DIMS = ("patient", "registration", "clamp", "wall", "model")


def variant(r):
    return "%s / %s / %s / %s" % (r["registration"], r["clamp"], "wall" if r["wall"] else "no wall", r["model"])


def compare(rows):
    out = []
    dims = DIMS
    for dim, a, b in PAIRS:
        idx = {}
        for r in rows:
            idx.setdefault(tuple(r[d] for d in dims if d != dim) + (r[dim],), []).append(r)
        for r in rows:
            if r[dim] != a:
                continue
            key = tuple(r[d] for d in dims if d != dim)
            for o in idx.get(key + (b,), []):
                row = dict(compare=dim, A=a, B=b, **{d: r[d] for d in dims if d != dim},
                           run_A=r["run"], run_B=o["run"])
                for m in COMPARE:
                    va, vb = r.get(m), o.get(m)
                    row[m + "_A"], row[m + "_B"] = va, vb
                    row[m + "_delta"] = vb - va if va is not None and vb is not None else None
                out.append(row)
    return out


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, keys)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: ("%.4g" % v if isinstance(v, float) else v) for k, v in r.items()})


def fmt(v, p=2):
    return "–" if v is None or (isinstance(v, float) and not np.isfinite(v)) else \
        ("%.*f" % (p, v) if isinstance(v, (float, np.floating)) else str(v))


def md_table(head, rows):
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
                     + ["| " + " | ".join(r) + " |" for r in rows]) + "\n"


def report(rows, cmp, out):
    L = ["# Fit analysis\n", "%d fits, %d patients. Shape errors measured by this script against the "
         "collapsed lung surface, free points only (see the docstring).\n" % (len(rows), len({r["patient"] for r in rows}))]

    L.append("\n## Per fit\n")
    rs = sorted(rows, key=lambda r: tuple(str(r[d]) for d in DIMS) + (r["run"],))
    L.append(md_table(
        ["case", "run", "side", "K", "collapse", "fit coll.", "ASSD", "fit→tgt", "tgt→fit", "p95", "rot °",
         "mean p/E", "ap", "si", "ml", "R²", "failed", "min", "stop"],
        [[r["case"], r["run"], r["side"], fmt(r["K"]), fmt(100 * r["collapse"], 0) + "%",
          fmt(100 * r["collapse_fit"], 0) + "%", fmt(r["assd_mm"]), fmt(r["fit_to_tgt_mm"]),
          fmt(r["tgt_to_fit_mm"]), fmt(r["fit_to_tgt_p95_mm"]), fmt(r["rotation_fit_deg"], 1),
          fmt(r["pE_mean"]), fmt(r["slope_ap"]), fmt(r["slope_si"]), fmt(r["slope_ml"]),
          fmt(r["pE_r2_linear"]), fmt(r["failed"]), fmt(r["minutes"], 1), str(r["stop"])] for r in rs]))

    L.append("\n## Per variant (median over patients)\n")
    groups = {}
    for r in rows:
        groups.setdefault(variant(r), []).append(r)
    trow = []
    for v, g in sorted(groups.items()):
        med = lambda k: float(np.median([r[k] for r in g if r[k] is not None]))
        c, m = np.array([r["collapse"] for r in g]), np.array([r["pE_mean"] for r in g])
        lin = "–"
        if len(g) >= 3 and np.ptp(c) > 0:
            k, q0 = np.polyfit(c, m, 1)
            res = m - (q0 + k * c)
            lin = "%.2f + %.2f·c (max res %.2f)" % (q0, k, np.abs(res).max())
        trow.append([v, str(len(g)), fmt(med("assd_mm")), fmt(med("fit_to_tgt_mm")), fmt(med("tgt_to_fit_mm")),
                     fmt(med("volume_err_pct"), 1), fmt(med("rotation_fit_deg"), 1),
                     "%d/%d" % (sum(r["slope_ap"] > 0 for r in g), len(g)),
                     "%d/%d" % (sum(r["slope_si"] < 0 for r in g), len(g)),
                     fmt(med("slope_rel_ap")), fmt(med("slope_rel_si")), lin, fmt(med("minutes"), 1)])
    L.append(md_table(["variant", "n", "ASSD", "fit→tgt", "tgt→fit", "vol err %", "rot °", "ap > 0",
                       "si < 0", "rel ap", "rel si", "mean p/E vs collapse c", "min"], trow))

    L.append("\n## Paired comparisons (B − A, median over patients; negative error = B better)\n")
    prow = []
    for dim, a, b in PAIRS:
        fix = lambda c: tuple(c.get(d) for d in DIMS[1:])           # the dimensions held fixed (None = compared)
        for key in sorted({fix(c) for c in cmp if c["compare"] == dim}, key=str):
            g = [c for c in cmp if c["compare"] == dim and fix(c) == key]
            if not g:
                continue
            med = lambda m: float(np.median([c[m + "_delta"] for c in g if c[m + "_delta"] is not None] or [np.nan]))
            fixed = ", ".join(str(x) for x in key if x is not None)
            better = sum(c["assd_mm_delta"] < 0 for c in g)
            prow.append(["%s: %s → %s" % (dim, a, b), fixed.replace("True", "wall").replace("False", "no wall"),
                         str(len(g)), "%d/%d" % (better, len(g)), fmt(med("assd_mm")), fmt(med("fit_to_tgt_mm")),
                         fmt(med("tgt_to_fit_mm")), fmt(med("rotation_fit_deg"), 1), fmt(med("pE_mean")),
                         fmt(med("slope_ap")), fmt(med("failed"), 0), fmt(med("minutes"), 1)])
    L.append(md_table(["comparison", "fixed", "n", "B better (ASSD)", "Δ ASSD", "Δ fit→tgt", "Δ tgt→fit",
                       "Δ rot °", "Δ mean p/E", "Δ ap", "Δ failed", "Δ min"], prow))

    L.append("\n## Pressure pattern vs collapse and clamp (Pearson r over patients, per variant)\n")
    crow = []
    for v, g in sorted(groups.items()):
        if len(g) < 4:
            continue
        c = np.array([r["collapse"] for r in g])
        cl = np.array([r["clamped_area_frac"] for r in g])
        cr = lambda k, x: fmt(float(np.corrcoef(x, [r[k] for r in g])[0, 1]))
        crow.append([v, str(len(g)), cr("slope_rel_ap", c), cr("slope_rel_si", c), cr("slope_rel_ap", cl),
                     cr("slope_rel_si", cl), fmt(float(np.corrcoef(c, cl)[0, 1]))])
    L.append(md_table(["variant", "n", "rel ap ~ collapse", "rel si ~ collapse", "rel ap ~ clamp",
                       "rel si ~ clamp", "collapse ~ clamp"], crow))
    (out / "report.md").write_text("\n".join(L))


def plots(rows, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib missing: no figures", file=sys.stderr)
        return
    groups = {}
    for r in rows:
        groups.setdefault(variant(r), []).append(r)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    for v, g in sorted(groups.items()):
        ax[0].scatter([100 * r["collapse"] for r in g], [r["pE_mean"] for r in g], label=v)
        ax[1].scatter([r["slope_rel_si"] for r in g], [r["slope_rel_ap"] for r in g], label=v)
        ax[2].scatter([100 * r["clamped_area_frac"] for r in g], [r["slope_rel_ap"] for r in g], label=v)
    ax[0].set(xlabel="collapse %", ylabel="mean p/E", title="pressure level vs collapse")
    ax[1].axhline(0, c="k", lw=0.5), ax[1].axvline(0, c="k", lw=0.5)
    ax[1].set(xlabel="relative slope si (+ = apex pushed more)", ylabel="relative slope ap (+ = anterior more)",
              title="pressure gradient direction")
    ax[2].set(xlabel="clamped area %", ylabel="relative slope ap", title="gradient vs clamp")
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "pressure.png", dpi=130)

    patients = sorted({r["patient"] for r in rows})
    vs = sorted(groups)
    fig, ax = plt.subplots(figsize=(max(8, len(patients) * 0.9), 4.5))
    wbar = 0.8 / max(len(vs), 1)
    for j, v in enumerate(vs):
        val = {r["patient"]: r["assd_mm"] for r in groups[v]}
        ax.bar(np.arange(len(patients)) + j * wbar, [val.get(p, np.nan) for p in patients], wbar, label=v)
    ax.set_xticks(np.arange(len(patients)) + 0.4 - wbar / 2, patients, rotation=45)
    ax.set(ylabel="ASSD to the collapsed lung [mm]", title="shape error per patient and variant")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "assd.png", dpi=130)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("roots", nargs="+", help="folders searched recursively for <case>/fem/setup/problem.npz")
    ap.add_argument("--out", default="analysis", help="output folder")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = collect(a.roots)
    if not rows:
        raise SystemExit("no fits found under %s" % ", ".join(a.roots))
    cmp = compare(rows)
    write_csv(out / "runs.csv", rows)
    write_csv(out / "compare.csv", cmp)
    report(rows, cmp, out)
    plots(rows, out)
    print("\n%d fits, %d paired comparisons -> %s/{runs.csv, compare.csv, report.md, pressure.png, assd.png}"
          % (len(rows), len(cmp), out))


if __name__ == "__main__":
    main()
