"""Lung mass and mean density from the CT of the pneumothorax, to fix the density of the model.

    python scripts/ct_density.py [--ct-dir Input_CT] [--data-dir Input_Data] [--results results]
        [--fit-run fitE_nu30] [--patients karl01 ...] [--out analysis/ct_density.csv]

Only the pneumothorax CT exists (one per patient): the model's "inflated lung" is its pleural
cavity (lung_<side>.vtk), the collapsed lung lung_<side>_collapsed.vtk. Tissue and blood do not
leave the lung when it collapses, only air, so

    mass = sum over the collapsed-lung voxels of rho(HU) * voxel volume
    rho_bar (inflated) = mass / V_cavity

with rho(HU) = (HU - HU_air) / (0 - HU_air) g/mL (water 1.0), clipped to [0, 1.1], and HU_air the
median of the pleural air (cavity minus collapsed lung, eroded by 2 mm, away from the borders).
Checks of the alignment of CT and surfaces: the HU of the pleural air (should be ~ -1000) and
of the collapsed lung. The contralateral lung (lung_*_lobe_<other side>.vtk, same CT, not collapsed)
gives the density the lung had inflated, so V_expected = mass / rho_contra is the volume it filled
before the pneumothorax and enlargement_mass = V_cavity / V_expected how much larger the cavity is.
Also the density of the collapsed lung by thirds of height (anterior,
middle, posterior: the gravity axis of a supine patient), and, with --fit-run, E rescaled to the
measured density: E_rho = E_fit * rho_bar / tissue_density of the fit (only rho_bar / E is fitted).

One patient from its config (also a new patient, CT of the INFLATED lung, no collapsed lung):

    python scripts/ct_density.py --config configs/<p>.json --ct <ct file> [--write-config]

With a collapsed lung in the config: as above. Without: rho_bar = the mean density inside the
inflated lung, rho(HU) calibrated on the air in the trachea (top 30 mm of the airway tree, eroded
by 2 mm; -1000 HU if there is no airway surface). --write-config sets physics.ct_density
[kg/m3] in the config (report only: E and p0 are then also given at this density).

Needs SimpleITK and VTK (Mac env `lungct`, or the server). Images: <ct-dir>/<patient>.nii[.gz] (or
.nrrd / .mha), read in LPS like the surfaces.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eurelaps_lung.data_io import read_surface                             # noqa: E402


def mask_from_surface(surf, img):
    """Inside of a closed LPS surface as a boolean array (z, y, x), on the grid of a SimpleITK image."""
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
    D = np.array(img.GetDirection()).reshape(3, 3)
    o, sp, n = np.array(img.GetOrigin()), np.array(img.GetSpacing()), img.GetSize()
    s = surf.copy()
    s.points = np.linalg.solve(D, (np.asarray(surf.points) - o).T).T / sp      # continuous index
    st = vtk.vtkPolyDataToImageStencil()
    st.SetInputData(s)
    st.SetOutputOrigin(0, 0, 0)
    st.SetOutputSpacing(1, 1, 1)
    st.SetOutputWholeExtent(0, n[0] - 1, 0, n[1] - 1, 0, n[2] - 1)
    im = vtk.vtkImageData()
    im.SetExtent(0, n[0] - 1, 0, n[1] - 1, 0, n[2] - 1)
    im.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
    im.GetPointData().GetScalars().Fill(1)
    sten = vtk.vtkImageStencil()
    sten.SetInputData(im)
    sten.SetStencilConnection(st.GetOutputPort())
    sten.SetBackgroundValue(0)
    sten.Update()
    return vtk_to_numpy(sten.GetOutput().GetPointData().GetScalars()).reshape(n[2], n[1], n[0]).astype(bool)


def find_ct(ct_dir, p):
    for ext in (".nii.gz", ".nii", ".nrrd", ".mha", ".mhd"):
        f = ct_dir / (p + ext)
        if f.is_file() and f.stat().st_size > 0:
            return f
    return None


def analyse(p, ct, surf_dir):
    import SimpleITK as sitk
    from scipy.ndimage import binary_erosion
    side = "left" if (surf_dir / "lung_left.vtk").is_file() else "right" if (surf_dir / "lung_right.vtk").is_file() else None
    if side is None or not (surf_dir / ("lung_%s_collapsed.vtk" % side)).is_file():
        return None, "no cavity / collapsed surface"
    img = sitk.ReadImage(str(ct))
    a = sitk.GetArrayViewFromImage(img)
    vox = float(np.prod(img.GetSpacing())) / 1000.0                       # mL
    cav = mask_from_surface(read_surface(surf_dir / ("lung_%s.vtk" % side)), img)
    col = mask_from_surface(read_surface(surf_dir / ("lung_%s_collapsed.vtk" % side)), img)
    if not col.any() or not cav.any():
        return None, "surfaces outside the image"
    air = cav & ~col
    r_vox = max(1, int(round(2.0 / min(img.GetSpacing()))))
    air_core = binary_erosion(air, iterations=r_vox)
    hu_air = float(np.median(a[air_core])) if air_core.sum() > 1000 else float(np.median(a[air]))
    h = a[col].astype(np.float32)
    rho = np.clip((h - hu_air) / (0.0 - hu_air), 0.0, 1.1)                # g/mL
    mass = float(rho.sum() * vox)
    V_cav, V_col = cav.sum() * vox, col.sum() * vox
    # density of the collapsed lung by thirds of the AP extent (LPS y: + posterior) -- direction-aware
    # contralateral lung (union of its lobes, same CT, not collapsed): its density is what the
    # collapsed lung would have when inflated -> the volume the lung filled before the pneumothorax
    other = "right" if side == "left" else "left"
    con = np.zeros_like(col)
    for f in sorted(surf_dir.glob("lung_*_lobe_%s.vtk" % other)):
        con |= mask_from_surface(read_surface(f), img)
    rho_con = V_exp = None
    if con.any():
        rho_con = float(np.clip((a[con].astype(np.float32) - hu_air) / (0.0 - hu_air), 0.0, 1.1).mean())
        V_exp = mass / rho_con
    D = np.array(img.GetDirection()).reshape(3, 3)
    zz, yy, xx = np.nonzero(col)
    y_lps = (np.array([xx, yy, zz]).T * np.array(img.GetSpacing())) @ D.T[:, 1] + img.GetOrigin()[1]
    t = np.quantile(y_lps, [1 / 3, 2 / 3])
    thirds = [float(rho[sel].mean()) for sel in (y_lps < t[0], (y_lps >= t[0]) & (y_lps < t[1]), y_lps >= t[1])]
    return dict(patient=p, side=side, V_cavity_mL=V_cav, V_collapsed_mL=V_col,
                outside_cavity_mL=float((col & ~cav).sum() * vox), hu_air=hu_air,
                hu_pleural_air_mean=float(a[air].mean()), hu_collapsed_mean=float(h.mean()),
                hu_collapsed_median=float(np.median(h)), mass_g=mass, rho_collapsed=mass / V_col,
                rho_bar_inflated=mass / V_cav, rho_contra=rho_con, V_expected_mL=V_exp,
                enlargement_mass=V_cav / V_exp if V_exp else None,
                collapse_vs_cavity=100 * (1 - V_col / V_cav),
                collapse_vs_expected=100 * (1 - V_col / V_exp) if V_exp else None,
                rho_ant=thirds[0], rho_mid=thirds[1], rho_post=thirds[2]), None


def analyse_inflated(p, ct, inflated, airways=None):
    """rho_bar of an inflated lung = its mean density on the CT (no pneumothorax)."""
    import SimpleITK as sitk
    from scipy.ndimage import binary_erosion
    img = sitk.ReadImage(str(ct))
    a = sitk.GetArrayViewFromImage(img)
    vox = float(np.prod(img.GetSpacing())) / 1000.0
    lung = mask_from_surface(read_surface(inflated), img)
    if not lung.any():
        return None, "surface outside the image"
    hu_air, air_src = -1000.0, "nominal"
    if airways is not None and Path(airways).is_file():
        aw = mask_from_surface(read_surface(airways), img)
        if aw.any():
            D = np.array(img.GetDirection()).reshape(3, 3)
            zz, yy, xx = np.nonzero(aw)
            z_lps = (np.array([xx, yy, zz]).T * np.array(img.GetSpacing())) @ D.T[:, 2] + img.GetOrigin()[2]
            top = np.zeros_like(aw)
            sel = z_lps > z_lps.max() - 30.0                           # trachea
            top[zz[sel], yy[sel], xx[sel]] = True
            core = binary_erosion(top, iterations=max(1, int(round(2.0 / min(img.GetSpacing())))))
            if core.sum() > 200:
                hu_air, air_src = float(np.median(a[core])), "trachea"
    h = a[lung].astype(np.float32)
    rho = np.clip((h - hu_air) / (0.0 - hu_air), 0.0, 1.1)
    V = lung.sum() * vox
    return dict(patient=p, mode="inflated", V_inflated_mL=V, hu_air=hu_air, hu_air_source=air_src,
                hu_lung_mean=float(h.mean()), hu_lung_median=float(np.median(h)), mass_g=float(rho.sum() * vox),
                rho_bar_inflated=float(rho.mean())), None


def one_config(args):
    """--config mode: one patient, from its config; optionally writes physics.ct_density."""
    from eurelaps_lung.config import load_config
    cfg = load_config(args.config)
    ct = Path(args.ct) if args.ct else find_ct(Path(args.ct_dir), cfg.patient)
    if ct is None or not ct.is_file():
        sys.exit("no CT: give --ct <file>")
    if cfg.collapsed is not None and cfg.collapsed.is_file():
        r, why = analyse(cfg.patient, ct, cfg.data_dir)
    else:
        r, why = analyse_inflated(cfg.patient, ct, cfg.inflated, cfg.structures.get("airways"))
    if r is None:
        sys.exit("%s: %s" % (cfg.patient, why))
    rho = 1000.0 * r["rho_bar_inflated"]
    if r.get("mode") == "inflated":
        print("%s (inflated lung): volume %.0f mL, lung %.0f HU (median %.0f), air %.0f HU (%s), mass %.0f g -> "
              "rho_bar %.1f kg/m3" % (cfg.patient, r["V_inflated_mL"], r["hu_lung_mean"], r["hu_lung_median"],
                                      r["hu_air"], r["hu_air_source"], r["mass_g"], rho))
    else:
        print("%s (cavity + collapsed lung): mass %.0f g / cavity %.0f mL -> rho_bar %.1f kg/m3" % (
            cfg.patient, r["mass_g"], r["V_cavity_mL"], rho))
    if args.write_config:
        raw = json.loads(Path(args.config).read_text())
        ph = {}
        for k, v in raw["physics"].items():
            if k == "ct_density":
                continue
            ph[k] = v
            if k == "tissue_density":
                ph["ct_density"] = round(rho, 1)
        ph.setdefault("ct_density", round(rho, 1))
        raw["physics"] = ph
        Path(args.config).write_text(json.dumps(raw, indent=2, ensure_ascii=False) + "\n")
        print("physics.ct_density = %.1f written to %s" % (rho, args.config))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ct-dir", default=str(ROOT / "Input_CT"))
    ap.add_argument("--data-dir", default=str(ROOT / "Input_Data"), help="surfaces, one folder per patient")
    ap.add_argument("--results", default=str(ROOT / "results"), help="output folders <p>/runs/<run>/summary.json")
    ap.add_argument("--fit-run", default=None)
    ap.add_argument("--patients", nargs="*")
    ap.add_argument("--out")
    ap.add_argument("--config", help="one patient from its config (also without a collapsed lung)")
    ap.add_argument("--ct", help="--config mode: the CT file (default <ct-dir>/<patient>.nii[.gz])")
    ap.add_argument("--write-config", action="store_true", help="--config mode: set physics.ct_density in it")
    args = ap.parse_args()
    if args.config:
        return one_config(args)
    ct_dir, data_dir = Path(args.ct_dir), Path(args.data_dir)
    pats = args.patients or sorted(d.name for d in data_dir.iterdir() if d.is_dir())
    rows = []
    for p in pats:
        ct = find_ct(ct_dir, p)
        if ct is None:
            print("%s: no CT" % p, flush=True)
            continue
        try:
            r, why = analyse(p, ct, data_dir / p)
        except Exception as e:                                           # truncated download, unreadable file
            r, why = None, "%s: %s" % (type(e).__name__, e)
        if r is None:
            print("%s: skipped (%s)" % (p, why), flush=True)
            continue
        if args.fit_run:
            f = Path(args.results) / p / "runs" / args.fit_run / "summary.json"
            if f.is_file():
                s = json.loads(f.read_text())
                rho_fit = s["physics"].get("tissue_density_kg_m3", 250.0)
                r["E_fit"] = s["physics"]["E_Pa"]
                r["E_rho"] = s["physics"]["E_Pa"] * 1000.0 * r["rho_bar_inflated"] / rho_fit
        rows.append(r)
        print("%-11s %-5s cavity %5.0f mL  collapsed %5.0f mL  air %6.0f HU (mean %5.0f)  lung %5.0f HU  mass %4.0f g"
              "  rho collapsed %.3f  rho_bar %.0f kg/m3  contra %.0f kg/m3  V_exp %s mL  enl %s"
              "  ant/mid/post %.2f/%.2f/%.2f%s" % (
                  p, r["side"], r["V_cavity_mL"], r["V_collapsed_mL"], r["hu_air"], r["hu_pleural_air_mean"],
                  r["hu_collapsed_mean"], r["mass_g"], r["rho_collapsed"], 1000 * r["rho_bar_inflated"],
                  1000 * (r["rho_contra"] or 0), "%.0f" % r["V_expected_mL"] if r["V_expected_mL"] else "-",
                  "%.2f" % r["enlargement_mass"] if r["enlargement_mass"] else "-",
                  r["rho_ant"], r["rho_mid"], r["rho_post"],
                  "  E %.0f -> %.0f Pa" % (r["E_fit"], r["E_rho"]) if "E_rho" in r else ""), flush=True)
    if rows:
        rb = [1000 * r["rho_bar_inflated"] for r in rows]
        print("\nrho_bar (inflated): median %.0f, range %.0f-%.0f kg/m3 (%d patients)" % (np.median(rb), min(rb), max(rb), len(rows)))
    if args.out and rows:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print("wrote %s" % args.out)


if __name__ == "__main__":
    main()
