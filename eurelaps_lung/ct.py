"""Densities and emphysema from the CT, on arrays (numpy + scipy only: no SimpleITK, no VTK), for
scripts/ct_density.py and for the 3D Slicer extension (which has the CT as an array and the lung as a
labelmap). METHODOLOGY.md section 2 and 3.4.1.

Arrays are indexed [k, j, i] (slice, row, column), as SimpleITK GetArrayFromImage and Slicer
arrayFromVolume give them; `spacing` is (si, sj, sk) in mm (SimpleITK GetSpacing order); masks are
boolean arrays of the same shape.

    density_scale(hu, hu_air)               HU -> g/mL, air 0, water 1
    air_hu(hu, air_mask, spacing)           HU of the air (median, 2 mm inside the mask)
    air_hu_trachea(hu, airway_mask, ijk_to_lps, spacing)   ... in the top 30 mm of the airway tree
    emphysema(hu, lung_mask, hu_air, spacing, exclude)     LAA%-950, LAA%-910, Perc15 (median-filtered)
    inflated_lung(hu, lung_mask, spacing, ...)             a new patient's inflated lung: density and emphysema
    barrier_phi(ct_density, emphysema, ct_emphysema)       the compression limit phi of physics.barrier "ct"

The emphysema yes / no is not decided here: below ~5 % the index does not separate emphysema from
none (analysis/emphysema_review.md); the user reads the CT (physics.emphysema).
"""

import numpy as np

TISSUE_DENSITY = 1050.0        # airless lung (tissue + blood) [kg/m3]
TRAPPED_FRACTION = 0.54        # share of the emphysematous air that stays in (cohort, analysis/results.md 5j)
BORDER_MM = 2.0                # partial volume at the lung border


def _r_vox(spacing, mm=BORDER_MM):
    return max(1, int(round(mm / min(spacing))))


def density_scale(hu, hu_air):
    """Density [g/mL] of HU values, air (hu_air) 0 and water (0 HU) 1, clipped to [0, 1.1]."""
    return np.clip((np.asarray(hu, np.float32) - hu_air) / (0.0 - hu_air), 0.0, 1.1)


def calibrated_hu(hu, hu_air):
    """HU rescaled so that the scan's air is -1000 and water 0."""
    return (np.asarray(hu, np.float32) - hu_air) / (0.0 - hu_air) * 1000.0 - 1000.0


def air_hu(hu, air_mask, spacing):
    """Median HU of an air region, 2 mm inside it (all of it if that leaves < 1000 voxels)."""
    from scipy.ndimage import binary_erosion
    core = binary_erosion(air_mask, iterations=_r_vox(spacing))
    sel = core if core.sum() > 1000 else air_mask
    return float(np.median(hu[sel]))


def air_hu_trachea(hu, airway_mask, ijk_to_lps, spacing, top_mm=30.0):
    """Median HU of the air in the top `top_mm` of the airway tree (the trachea), 2 mm inside it; None if
    too few voxels. ijk_to_lps: 4x4 matrix from voxel (i, j, k) to LPS mm."""
    from scipy.ndimage import binary_erosion
    kk, jj, ii = np.nonzero(airway_mask)
    if len(kk) == 0:
        return None
    M = np.asarray(ijk_to_lps, float)
    z = M[2, 0] * ii + M[2, 1] * jj + M[2, 2] * kk + M[2, 3]           # superior coordinate
    sel = z > z.max() - top_mm
    top = np.zeros(airway_mask.shape, bool)
    top[kk[sel], jj[sel], ii[sel]] = True
    core = binary_erosion(top, iterations=_r_vox(spacing))
    return float(np.median(hu[core])) if core.sum() > 200 else None


def _median_filtered(hu, mask):
    """3x3x3 median filter of hu over the bounding box of mask; the values on mask (against the noise:
    on thin-slice scans the noise alone puts normal lung below -950 HU)."""
    from scipy.ndimage import median_filter
    kk, jj, ii = np.nonzero(mask)
    lo = np.maximum(np.array([kk.min(), jj.min(), ii.min()]) - 1, 0)
    hi = np.array([kk.max(), jj.max(), ii.max()]) + 2
    box = tuple(slice(a, b) for a, b in zip(lo, hi))
    return median_filter(np.asarray(hu[box], np.float32), size=3)[mask[box]]


def emphysema(hu, lung_mask, hu_air, spacing, exclude=None):
    """Emphysema indices of a lung on the calibrated HU scale, after a 3x3x3 median filter, 2 mm inside
    the border and outside `exclude` (the airway lumen): {laa950, laa910 [%], perc15 [HU], n_vox}."""
    from scipy.ndimage import binary_erosion
    core = binary_erosion(lung_mask, iterations=_r_vox(spacing))
    if exclude is not None:
        core &= ~exclude
    if core.sum() < 100:
        return dict(laa950=None, laa910=None, perc15=None, n_vox=int(core.sum()))
    h = calibrated_hu(_median_filtered(hu, core), hu_air)
    return dict(laa950=float(100 * (h < -950).mean()), laa910=float(100 * (h < -910).mean()),
                perc15=float(np.percentile(h, 15)), n_vox=int(core.sum()))


def low_attenuation_volume(hu, mask, hu_air, spacing):
    """Volume [mL] of the median-filtered voxels of mask below -950 HU (calibrated): air still in a
    collapsed lung."""
    vox = float(np.prod(spacing)) / 1000.0
    return float((_median_filtered(hu, mask) < hu_air + 0.05 * (0.0 - hu_air)).sum() * vox)


def inflated_lung(hu, lung_mask, spacing, hu_air=None, airway_mask=None, ijk_to_lps=None):
    """A new patient's inflated lung: {rho_bar_kg_m3, mass_g, volume_mL, laa950, laa910, perc15, hu_air,
    hu_air_source, phi_tissue}. The air HU: given, else the trachea (airway_mask + ijk_to_lps), else -1000.
    rho_bar goes to physics.ct_density, laa950 to physics.ct_emphysema."""
    src = "given"
    if hu_air is None and airway_mask is not None and ijk_to_lps is not None:
        hu_air, src = air_hu_trachea(hu, airway_mask, ijk_to_lps, spacing), "trachea"
    if hu_air is None:
        hu_air, src = -1000.0, "nominal"
    vox = float(np.prod(spacing)) / 1000.0
    rho = density_scale(hu[lung_mask], hu_air)
    exclude = None
    if airway_mask is not None:
        from scipy.ndimage import binary_dilation
        exclude = binary_dilation(airway_mask, iterations=1)
    e = emphysema(hu, lung_mask, hu_air, spacing, exclude)
    rho_bar = 1000.0 * float(rho.mean())
    return dict(rho_bar_kg_m3=rho_bar, mass_g=float(rho.sum() * vox), volume_mL=float(lung_mask.sum() * vox),
                laa950=e["laa950"], laa910=e["laa910"], perc15=e["perc15"], hu_air=float(hu_air), hu_air_source=src,
                phi_tissue=rho_bar / TISSUE_DENSITY)


def barrier_phi(ct_density, emphysema_present, ct_emphysema=None, tissue_density=TISSUE_DENSITY,
                trapped_fraction=TRAPPED_FRACTION):
    """phi = ct_density / tissue_density [+ trapped_fraction * ct_emphysema / 100 if emphysema_present]:
    the smallest volume fraction the lung can reach (config.py physics.barrier "ct", the same formula)."""
    phi = ct_density / tissue_density
    if emphysema_present:
        if ct_emphysema is None:
            raise ValueError("emphysema present: its volume ct_emphysema (LAA%-950, %) is needed")
        phi += trapped_fraction * ct_emphysema / 100.0
    return phi


# ---------------------------------------------------------------------------------------------------
# files: the `ct` step of the pipeline (reads the CT and the surfaces; needs SimpleITK or itk, and VTK)
# ---------------------------------------------------------------------------------------------------

def read_ct(path):
    """CT file (.nii[.gz], .nrrd, .mha, ...) -> (HU array [k, j, i] float32, spacing (si, sj, sk),
    4x4 ijk -> LPS). SimpleITK if installed, else itk (both read in LPS)."""
    try:
        import SimpleITK as sitk
        img = sitk.ReadImage(str(path))
        a = sitk.GetArrayFromImage(img).astype(np.float32)
        D, sp, o = np.array(img.GetDirection()).reshape(3, 3), np.array(img.GetSpacing()), np.array(img.GetOrigin())
    except ImportError:
        import itk
        img = itk.imread(str(path))
        a = itk.array_from_image(img).astype(np.float32)
        D = np.array(img.GetDirection()).reshape(3, 3)
        sp, o = np.array(img.GetSpacing()), np.array(img.GetOrigin())
    M = np.eye(4)
    M[:3, :3] = D * sp
    M[:3, 3] = o
    return a, tuple(float(s) for s in sp), M


def mask_from_surface(surface, shape, ijk_to_lps):
    """Inside of a closed LPS surface (a file or a data_io surface) as a boolean array [k, j, i] on the
    grid of a CT (shape, ijk_to_lps). VTK stencil, the voxel centres inside."""
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    from .data_io import read_surface
    surf = read_surface(surface) if not hasattr(surface, "points") else surface
    s = surf.copy()
    M = np.asarray(ijk_to_lps, float)
    s.points = np.linalg.solve(M[:3, :3], (np.asarray(surf.points) - M[:3, 3]).T).T     # continuous index
    nk, nj, ni = shape
    st = vtk.vtkPolyDataToImageStencil()
    st.SetInputData(s.to_vtk())
    st.SetOutputOrigin(0, 0, 0)
    st.SetOutputSpacing(1, 1, 1)
    st.SetOutputWholeExtent(0, ni - 1, 0, nj - 1, 0, nk - 1)
    im = vtk.vtkImageData()
    im.SetExtent(0, ni - 1, 0, nj - 1, 0, nk - 1)
    im.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
    im.GetPointData().GetScalars().Fill(1)
    sten = vtk.vtkImageStencil()
    sten.SetInputData(im)
    sten.SetStencilConnection(st.GetOutputPort())
    sten.SetBackgroundValue(0)
    sten.Update()
    return vtk_to_numpy(sten.GetOutput().GetPointData().GetScalars()).reshape(nk, nj, ni).astype(bool)


def measure(ct_path, inflated, collapsed=None, contralateral=(), airways=None):
    """The CT values of one patient, from its files (LPS surfaces):
      with the collapsed lung (pneumothorax CT, the cohort): ct_density = mass of the collapsed lung /
        volume of the inflated surface (the cavity), HU calibrated on the pleural air; ct_emphysema =
        LAA%-950 of the contralateral lung (the operated one is seen only collapsed), or of the collapsed
        lung when no contralateral surface is given;
      without (the CT of the inflated lung, a new patient): density and LAA%-950 of the inflated lung,
        HU calibrated in the trachea (airways).
    Returns a dict with ct_density [kg/m3], ct_emphysema [%] and the details."""
    from scipy.ndimage import binary_dilation
    hu, sp, M = read_ct(ct_path)
    vox = float(np.prod(sp)) / 1000.0
    lung = mask_from_surface(inflated, hu.shape, M)
    if not lung.any():
        raise ValueError("the inflated lung surface lies outside the CT %s" % ct_path)
    aw = mask_from_surface(airways, hu.shape, M) if airways else None
    aw = aw if aw is not None and aw.any() else None
    if collapsed is None:
        r = inflated_lung(hu, lung, sp, airway_mask=aw, ijk_to_lps=M)
        return dict(mode="inflated", ct_density=r["rho_bar_kg_m3"], ct_emphysema=r["laa950"],
                    emphysema_measured_on="the inflated lung", volume_mL=r["volume_mL"], mass_g=r["mass_g"],
                    hu_air=r["hu_air"], hu_air_source=r["hu_air_source"], laa910=r["laa910"], perc15=r["perc15"],
                    ct=str(ct_path))
    col = mask_from_surface(collapsed, hu.shape, M)
    if not col.any():
        raise ValueError("the collapsed lung surface lies outside the CT %s" % ct_path)
    hu_air = air_hu(hu, lung & ~col, sp)                         # the pleural air
    mass = float(density_scale(hu[col], hu_air).sum() * vox)
    V_cav, V_col = float(lung.sum() * vox), float(col.sum() * vox)
    excl = binary_dilation(aw, iterations=1) if aw is not None else None
    con = np.zeros(hu.shape, bool)
    for f in contralateral or ():
        con |= mask_from_surface(f, hu.shape, M)
    e_col = emphysema(hu, col, hu_air, sp, excl)
    e_con = emphysema(hu, con, hu_air, sp, excl) if con.any() else None
    e = e_con or e_col
    col_air = col & ~excl if excl is not None else col
    return dict(mode="pneumothorax", ct_density=1000.0 * mass / V_cav, ct_emphysema=e["laa950"],
                emphysema_measured_on="the contralateral lung" if e_con else "the collapsed lung",
                laa950_collapsed=e_col["laa950"], mass_g=mass, volume_cavity_mL=V_cav, volume_collapsed_mL=V_col,
                trapped_air_mL=low_attenuation_volume(hu, col_air, hu_air, sp), hu_air=hu_air,
                laa910=e["laa910"], perc15=e["perc15"], ct=str(ct_path))


VALUES_FILE = "ct/ct_values.json"


def run(cfg):
    """`ct` step: the CT values of the patient (measure) -> output_dir/ct/ct_values.json, read by the
    config loader for physics.ct_density / ct_emphysema when the config does not set them. Skipped
    (with a message) when inputs.ct is not given."""
    import json
    import logging
    log = logging.getLogger(__name__)
    if cfg.ct is None:
        log.info("no CT (inputs.ct): skipped; physics.ct_density / ct_emphysema must then be given if the "
                 "barrier is used")
        return None
    v = measure(cfg.ct, cfg.inflated, cfg.collapsed, cfg.contralateral, cfg.structures.get("airways"))
    out = cfg.output_dir / VALUES_FILE
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(v, indent=2))
    from .config import resolve_barrier
    ph = cfg.physics                                               # this run's config: the values, unless given
    if ph.ct_density is None:
        ph.ct_density = round(float(v["ct_density"]), 3)
    if ph.ct_emphysema is None and v.get("ct_emphysema") is not None:
        ph.ct_emphysema = round(float(v["ct_emphysema"]), 3)
    resolve_barrier(ph)
    log.info("CT %s (%s): density %.1f kg/m3, emphysema volume LAA%%-950 %.1f %% (%s)%s; -> %s",
             cfg.ct.name, v["mode"], v["ct_density"], v["ct_emphysema"] or 0.0, v["emphysema_measured_on"],
             "; emphysema yes / no is the user's (physics.emphysema = %s)" % cfg.physics.emphysema, out)
    return v
