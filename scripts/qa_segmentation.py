"""Segmentation check from the CT: where the collapsed lung / cavity surfaces disagree with the image.

    python scripts/qa_segmentation.py [patient ...]        (Mac env lungct; Input_CT/, Input_Data/, results/)

Per patient: label 1 = soft-tissue density (HU > -100) inside the collapsed lung, 2 mm from its border,
blobs >= 5 mL (normal: the hilar vessels and some dependent atelectasis, ~3-11 %% of the lung; more =
heart / mediastinum inside the segmentation, or an airless lung); label 2 = tissue or fluid (HU > -500)
between the collapsed lung and the cavity, blobs >= 5 mL (lung left out of the segmentation, pleural
effusion, or diaphragm / abdomen / heart inside the cavity). Writes results/<p>/qa_segmentation.nrrd
(load it in Slicer as a labelmap over the CT) and prints the volumes.
"""
import sys, glob, os
sys.path.insert(0, '.')
import numpy as np, SimpleITK as sitk
from scipy import ndimage as ndi
from scripts.ct_density import mask_from_surface, find_ct
from eurelaps_lung.data_io import read_surface
from pathlib import Path
pats = sys.argv[1:] or sorted(os.path.basename(d) for d in glob.glob('results/*'))
for p in pats:
    d = Path('Input_Data') / p
    side = 'left' if (d / 'lung_left.vtk').is_file() else 'right'
    ct = find_ct(Path('Input_CT'), p)
    img = sitk.ReadImage(str(ct)); a = sitk.GetArrayViewFromImage(img); sp = np.array(img.GetSpacing()); vox = sp.prod() / 1000
    cav = mask_from_surface(read_surface(d / f'lung_{side}.vtk'), img)
    col = mask_from_surface(read_surface(d / f'lung_{side}_collapsed.vtk'), img)
    # 1: dense inside the collapsed lung, away from its border (2 mm), big blobs only (>= 5 mL)
    inner = ndi.binary_erosion(col, iterations=max(1, int(round(2 / sp.min()))))
    dense = inner & (a > -100)
    dense = ndi.binary_opening(dense, iterations=1)
    lab, n = ndi.label(dense); sizes = ndi.sum(dense, lab, range(1, n + 1)) * vox
    keep1 = np.isin(lab, 1 + np.nonzero(sizes >= 5)[0])
    # 2: tissue between collapsed lung and cavity (HU > -500), big blobs only
    gap = cav & ~col
    tis = ndi.binary_opening(gap & (a > -500), iterations=1)
    lab2, n2 = ndi.label(tis); s2 = ndi.sum(tis, lab2, range(1, n2 + 1)) * vox
    keep2 = np.isin(lab2, 1 + np.nonzero(s2 >= 5)[0])
    hu_fluid = float(np.median(a[keep2])) if keep2.any() else float('nan')
    print("%-10s collapsed %5.0f mL: dense inside %5.0f mL (%4.1f %%, largest %4.0f mL) | tissue in the gap %5.0f mL (largest %4.0f mL, median %4.0f HU)" % (
        p, col.sum() * vox, keep1.sum() * vox, 100 * keep1.sum() / col.sum(), sizes.max() if n else 0, keep2.sum() * vox, s2.max() if n2 else 0, hu_fluid), flush=True)
    labels = np.zeros(a.shape, np.uint8); labels[keep1] = 1; labels[keep2] = 2
    L = sitk.GetImageFromArray(labels); L.CopyInformation(img)
    sitk.WriteImage(L, f'results/{p}/qa_segmentation.nrrd', useCompression=True)
