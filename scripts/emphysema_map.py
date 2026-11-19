"""Emphysema map for a visual check in 3D Slicer: where the CT is below -950 HU inside the lungs.

    python scripts/emphysema_map.py [patient ...]        (Mac env lungct; Input_CT/, Input_Data/ -> results/<p>/)

The same measure as scripts/ct_density.py (HU calibrated on the pleural air, 3x3x3 median filter
against the noise, 2 mm inside the lung border, airway lumen excluded). Writes
results/<p>/emphysema.nrrd, a labelmap on the CT grid:
  1 = low attenuation (emphysema) in the contralateral lung (inflated, the lobe surfaces)
  2 = low attenuation in the collapsed lung (air still trapped in it)
and prints LAA%-950 of both. Load it in Slicer over the CT: scripts/slicer_emphysema.py.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from ct_density import airway_mask, find_ct, mask_from_surface      # noqa: E402
from eurelaps_lung.data_io import read_surface                       # noqa: E402


def laa_mask(a, mask, hu_air, r_vox, exclude):
    """Voxels of the eroded mask (minus exclude) whose median-filtered, calibrated HU is < -950."""
    from scipy.ndimage import binary_erosion, median_filter
    core = binary_erosion(mask, iterations=r_vox)
    if exclude is not None:
        core &= ~exclude
    out = np.zeros(mask.shape, bool)
    if core.sum() < 100:
        return out, core
    zz, yy, xx = np.nonzero(core)
    lo = np.maximum(np.array([zz.min(), yy.min(), xx.min()]) - 1, 0)
    hi = np.array([zz.max(), yy.max(), xx.max()]) + 2
    box = tuple(slice(l, h) for l, h in zip(lo, hi))
    med = median_filter(np.asarray(a[box], np.float32), size=3)
    cal = (med - hu_air) / (0.0 - hu_air) * 1000.0 - 1000.0
    out[box] = (cal < -950.0) & core[box]
    return out, core


def one(p):
    import SimpleITK as sitk
    from scipy.ndimage import binary_erosion
    d = ROOT / "Input_Data" / p
    side = "left" if (d / "lung_left.vtk").is_file() else "right" if (d / "lung_right.vtk").is_file() else None
    ct = find_ct(ROOT / "Input_CT", p)
    if side is None or ct is None or not (d / ("lung_%s_collapsed.vtk" % side)).is_file():
        print("%s: skipped (no CT / cavity / collapsed lung)" % p)
        return
    other = "right" if side == "left" else "left"
    img = sitk.ReadImage(str(ct))
    a = sitk.GetArrayViewFromImage(img)
    r = max(1, int(round(2.0 / min(img.GetSpacing()))))
    cav = mask_from_surface(read_surface(d / ("lung_%s.vtk" % side)), img)
    col = mask_from_surface(read_surface(d / ("lung_%s_collapsed.vtk" % side)), img)
    air = binary_erosion(cav & ~col, iterations=r)
    hu_air = float(np.median(a[air])) if air.sum() > 1000 else float(np.median(a[cav & ~col]))
    con = np.zeros_like(col)
    for f in sorted(d.glob("lung_*_lobe_%s.vtk" % other)):
        con |= mask_from_surface(read_surface(f), img)
    aw = airway_mask(d, img)
    lab = np.zeros(col.shape, np.uint8)
    out = []
    for value, m in ((1, con), (2, col)):
        if not m.any():
            out.append("-")
            continue
        laa, core = laa_mask(a, m, hu_air, r, aw)
        lab[laa] = value
        out.append("%.1f %%" % (100.0 * laa.sum() / max(core.sum(), 1)))
    li = sitk.GetImageFromArray(lab)
    li.CopyInformation(img)
    dst = ROOT / "results" / p
    dst.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(li, str(dst / "emphysema.nrrd"), True)
    print("%-11s %-5s LAA%%-950 contralateral %s, collapsed %s -> %s" % (p, side, out[0], out[1],
                                                                     dst / "emphysema.nrrd"), flush=True)


if __name__ == "__main__":
    pats = sys.argv[1:] or sorted(x.name for x in (ROOT / "Input_Data").iterdir() if x.is_dir())
    for p in pats:
        one(p)
