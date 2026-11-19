"""3D Slicer: one patient's CT in the lung window with the emphysema map on top (visual check).

First, on the Mac (env lungct): python scripts/emphysema_map.py   (writes results/<p>/emphysema.nrrd)
Then in Slicer's Python console:

    PATIENT = "patient_10"; p = r"/Users/<you>/.../lung_collapse/scripts/slicer_emphysema.py"; exec(open(p).read(), {"__file__": p, "PATIENT": PATIENT})

Loads the CT Input_CT/<patient>.nii[.gz] (window -1000 / -500 HU: emphysema is black, normal lung
dark grey) and results/<p>/emphysema.nrrd as a labelmap: yellow = below -950 HU in the contralateral
lung, red = in the collapsed lung (air still trapped). Scroll the axial view: true emphysema is
clusters / holes without walls, mostly in the upper lobes; noise is scattered single dots.
Changing PATIENT and running again replaces the previous patient. Record the verdict in
analysis/emphysema_review.md.
"""
import glob
import os

import slicer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(globals().get("__file__", "scripts/slicer_emphysema.py"))))
PATIENT = globals().get("PATIENT") or "patient_10"


def show():
    for node in list(slicer.util.getNodes("emph_*").values()):
        slicer.mrmlScene.RemoveNode(node)
    cts = [f for f in glob.glob(os.path.join(ROOT, "Input_CT", PATIENT + ".*")) if os.path.getsize(f) > 0]
    if not cts:
        print("no CT for %s in Input_CT/" % PATIENT)
        return
    vol = slicer.util.loadVolume(cts[0])
    vol.SetName("emph_%s_CT" % PATIENT)
    vd = vol.GetDisplayNode()
    vd.AutoWindowLevelOff()
    vd.SetWindowLevel(500, -750)                                 # -1000 .. -500 HU
    lab_path = os.path.join(ROOT, "results", PATIENT, "emphysema.nrrd")
    lab = None
    if os.path.isfile(lab_path):
        lab = slicer.util.loadLabelVolume(lab_path)
        lab.SetName("emph_%s_map" % PATIENT)
        ct = slicer.vtkMRMLColorTableNode()
        ct.SetTypeToUser()
        ct.SetNumberOfColors(3)
        ct.SetColor(0, "background", 0, 0, 0, 0)
        ct.SetColor(1, "emphysema_contralateral", 1.0, 0.9, 0.0, 1)
        ct.SetColor(2, "trapped_in_collapsed", 1.0, 0.15, 0.1, 1)
        ct.SetName("emph_colors")
        slicer.mrmlScene.AddNode(ct)
        lab.GetDisplayNode().SetAndObserveColorNodeID(ct.GetID())
    else:
        print("missing %s: run python scripts/emphysema_map.py %s on the Mac" % (lab_path, PATIENT))
    slicer.util.setSliceViewerLayers(background=vol, label=lab, labelOpacity=0.5)
    slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)
    slicer.util.resetSliceViews()
    print("%s: CT window -1000 / -500 HU; yellow = emphysema in the contralateral lung, red = in the collapsed "
          "lung. Toggle the map: Ctrl/Cmd+click the label opacity, or the Data module." % PATIENT)


show()
