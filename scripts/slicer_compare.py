"""3D Slicer: one patient's CT, cavity, real collapsed lung, fit and prediction together.

In Slicer's Python console (Mac), with the results downloaded in results/<patient>/:

    PATIENT = "karl23"; p = r"/Users/<you>/.../lung_collapse/scripts/slicer_compare.py"; exec(open(p).read(), {"__file__": p, "PATIENT": PATIENT})

optional, in the same dict: "RUNS": ["fit_final", "pred"], "CT": False (no CT), "SEQUENCE": "pred" (also load the
collapse sequence of that run; it replaces any sequence loaded before).

Loads (nodes named <patient>_...): the CT Input_CT/<patient>.nii[.gz] (lung window), the cavity
(results/<p>/lung_inflated_mesh.vtp, white, transparent), the real collapsed lung (lung_collapsed_mesh.vtp,
grey wireframe), and every run's final surface results/<p>/runs/<run>/surface.vtp coloured by Error_mm
(distance to the real collapsed lung, 0-15 mm). Toggle them in the Data module. Surfaces are LPS (tag in
the files), the CT is read in LPS by Slicer: they overlap without any transform.
"""
import glob
import os

import slicer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(globals().get("__file__", "scripts/slicer_compare.py"))))
PATIENT = globals().get("PATIENT") or "karl06"
RUNS = globals().get("RUNS") or ["fit_final", "pred"]
LOAD_CT = globals().get("CT", True)
SEQUENCE = globals().get("SEQUENCE")
COLORS = {"fit_final": (0.2, 0.6, 1.0), "pred": (1.0, 0.55, 0.1)}


def _clear(prefix):
    for node in list(slicer.util.getNodes(prefix + "*").values()):
        slicer.mrmlScene.RemoveNode(node)


def _model(path, name, color, opacity=1.0, wire=False, scalar=None, smax=15.0):
    if not os.path.isfile(path):
        print("missing: %s" % path)
        return None
    n = slicer.util.loadModel(path)
    n.SetName(name)
    d = n.GetDisplayNode()
    d.SetColor(*color)
    d.SetOpacity(opacity)
    d.SetVisibility2D(True)                                     # outline in the slice views too
    if wire:
        d.SetRepresentation(d.WireframeRepresentation)
    if scalar:
        try:
            d.SetActiveScalarName(scalar)
            d.SetAndObserveColorNodeID("vtkMRMLColorTableNodeFileColdToHotRainbow.txt")
            d.SetScalarRangeFlag(slicer.vtkMRMLDisplayNode.UseManualScalarRange)
            d.SetScalarRange(0.0, smax)
            d.SetScalarVisibility(True)
        except Exception as e:                                  # older Slicer: plain colour
            print("no scalar colouring (%s)" % e)
    return n


def compare():
    res = os.path.join(ROOT, "results", PATIENT)
    _clear(PATIENT + "_")
    if LOAD_CT:
        ct = (glob.glob(os.path.join(ROOT, "Input_CT", PATIENT + ".nii.gz")) +
              glob.glob(os.path.join(ROOT, "Input_CT", PATIENT + ".nii")))
        if ct:
            v = slicer.util.loadVolume(ct[0])
            v.SetName(PATIENT + "_CT")
            vd = v.GetDisplayNode()
            vd.SetAutoWindowLevel(False)
            vd.SetWindowLevel(1500, -600)                       # lung window
            slicer.util.setSliceViewerLayers(background=v)
        else:
            print("no CT in Input_CT/ for %s" % PATIENT)
    _model(os.path.join(res, "lung_inflated_mesh.vtp"), PATIENT + "_cavity", (1, 1, 1), opacity=0.15)
    _model(os.path.join(res, "lung_collapsed_mesh.vtp"), PATIENT + "_real_collapsed", (0.5, 0.5, 0.5), wire=True)
    for run in RUNS:
        _model(os.path.join(res, "runs", run, "surface.vtp"), "%s_%s" % (PATIENT, run),
               COLORS.get(run, (0.3, 0.9, 0.3)), scalar="Error_mm")
    if SEQUENCE:
        p = os.path.join(res, "runs", SEQUENCE, "sequence", "load_collapse_in_slicer.py")
        if os.path.isfile(p):
            exec(open(p).read(), {"__file__": p})
        else:
            print("no sequence: %s" % p)
    lm = slicer.app.layoutManager()
    lm.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)
    lm.threeDWidget(0).threeDView().resetFocalPoint()
    print("%s: loaded CT, cavity, real collapsed lung and %s (coloured by Error_mm, 0-15 mm)" % (PATIENT, ", ".join(RUNS)))


compare()
