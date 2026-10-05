"""Load several collapse sequences side by side in 3D Slicer, on one shared time slider.

Each sequence folder is one written by scripts/collapse_sequence.py (frame_*.vtp, target_aligned.vtp),
e.g. results/karl04/sequence_fit_torch and results/karl04/sequence_fit_torch_plane. In Slicer's
Python console:

    exec(open('/path/to/lung_collapse/scripts/slicer_compare_sequences.py').read())
    load('/path/to/results/karl04/sequence_fit_torch', '/path/to/results/karl04/sequence_fit_torch_plane')

Each sequence gets its own colour; the target of the first one is shown as a grey wireframe. Play
or scrub with the Sequences toolbar. (Ported from the former pipeline_codes_v1/8_Load_sequence_3DSlicer.py.)
For a single sequence, the load_collapse_in_slicer.py written next to its frames does more
(colour by displacement, hilum spheres, autoplay).
"""

import os

import slicer

COLORS = [(0.9, 0.5, 0.4), (0.4, 0.6, 0.9), (0.5, 0.8, 0.4), (0.9, 0.8, 0.3)]
CNAMES = ["orange", "blue", "green", "yellow"]


def load(*folders):
    if not folders:
        print("usage: load('<sequence folder>', '<another sequence folder>', ...)")
        return
    browser = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSequenceBrowserNode")
    loaded, first = [], None
    for j, folder in enumerate(folders):
        folder = folder.rstrip("/")
        frames = sorted(f for f in os.listdir(folder) if f.startswith("frame_") and f.endswith(".vtp")) \
            if os.path.isdir(folder) else []
        if not frames:
            print("no frame_*.vtp in %s (run scripts/collapse_sequence.py first)" % folder)
            continue
        first = first or folder
        name = "%s_%s" % (os.path.basename(os.path.dirname(folder)), os.path.basename(folder))
        seq = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSequenceNode", name + "_seq")
        seq.SetIndexName("load")
        seq.SetIndexUnit("%")
        for i, f in enumerate(frames):
            node = slicer.util.loadModel(os.path.join(folder, f))
            seq.SetDataNodeAtValue(node, "%.1f" % (100.0 * i / max(len(frames) - 1, 1)))
            slicer.mrmlScene.RemoveNode(node)
        browser.AddSynchronizedSequenceNode(seq)
        loaded.append((seq, name, j))
    if not loaded:
        print("nothing loaded")
        return
    browser.SetName("Collapse_" + "_vs_".join(n for _, n, _ in loaded))
    browser.SetSelectedItemNumber(0)
    slicer.modules.sequences.logic().UpdateProxyNodesFromSequences(browser)
    for seq, name, j in loaded:
        proxy = browser.GetProxyNode(seq)
        proxy.SetName(name)
        proxy.CreateDefaultDisplayNodes()
        proxy.GetDisplayNode().SetColor(*COLORS[j % len(COLORS)])
        proxy.GetDisplayNode().SetScalarVisibility(False)
        print("  %-40s %s, %d frames" % (name, CNAMES[j % len(CNAMES)], seq.GetNumberOfDataNodes()))
    target = os.path.join(first, "target_aligned.vtp")
    if os.path.isfile(target):
        tgt = slicer.util.loadModel(target)
        tgt.SetName("target")
        tgt.GetDisplayNode().SetRepresentation(1)
        tgt.GetDisplayNode().SetColor(0.6, 0.6, 0.6)
    slicer.modules.sequences.setToolBarActiveBrowserNode(browser)
    slicer.modules.sequences.showSequenceBrowser(browser)
    print("Target = grey wireframe. Press play in the Sequences toolbar.")


print("slicer_compare_sequences: call load('<sequence folder>', ...)")
