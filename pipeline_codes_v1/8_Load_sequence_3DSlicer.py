# load_seq.py -- load one or more collapse sequences into 3D Slicer
# In Slicer's Python console:
#   exec(open('/home/pr502/IBT_pr502_riggio_HIWI/load_seq.py').read())
# It loads RUNS below. To load a different run later, without editing the file:
#   load('/home/pr502/IBT_pr502_riggio_HIWI/results/<run folder>')

import os
import slicer

RUNS = [
    "/home/pr502/IBT_pr502_riggio_HIWI/results/fem_fit_K90_nowall",
    # "/home/pr502/IBT_pr502_riggio_HIWI/results/fem_fit_K90_wall_selfbound",
]

COLORS = [(0.9, 0.5, 0.4), (0.4, 0.6, 0.9), (0.5, 0.8, 0.4), (0.9, 0.8, 0.3)]
CNAMES = ["orange", "blue", "green", "yellow"]


def load(*runs):
    runs = runs or RUNS
    browser = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSequenceBrowserNode")
    loaded, first = [], None
    for j, run in enumerate(runs):
        run = run.rstrip("/")
        folder = run if os.path.basename(run) == "sequence" else os.path.join(run, "sequence")
        if not os.path.isdir(folder):
            print("no sequence folder: %s  (run make_collapse_sequence first)" % folder)
            continue
        frames = sorted(f for f in os.listdir(folder)
                        if f.startswith("frame_") and f.endswith(".vtp"))
        if not frames:
            print("no frames in %s" % folder)
            continue
        first = first or folder
        name = os.path.basename(os.path.dirname(folder))
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
        print("  %-32s %s, %d frames" % (name, CNAMES[j % 4], seq.GetNumberOfDataNodes()))
    tgt = slicer.util.loadModel(os.path.join(first, "target_used.vtp"))
    tgt.SetName("target")
    tgt.GetDisplayNode().SetRepresentation(1)
    tgt.GetDisplayNode().SetColor(0.6, 0.6, 0.6)
    slicer.modules.sequences.setToolBarActiveBrowserNode(browser)
    slicer.modules.sequences.showSequenceBrowser(browser)
    print("Target = grey wireframe. Press play in the Sequences toolbar.")


load()