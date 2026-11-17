"""Collapse sequence of a run, to look at in 3D Slicer (or ParaView): the state of
output_dir/runs/<run_name>/ re-solved with its load ramped 0 -> 100 %% (quasi-static load path,
each frame warm-started from the previous one, same core and options), or, mode "linear", a
straight morph from the inflated to the final shape (no FEM library needed).

Output, output_dir/runs/<run_name>/sequence/ (self-contained: copy the folder to the machine with Slicer):
  frame_000.vtp ...           surface per frame: Displacement_mm, Disp_mag_mm, Pressure_Pa
                              (+ Error_mm with a collapsed lung, WallDistance_mm with a wall)
  volume_000.vtk ...          volume frames (volume=True)
  collapse.pvd                ParaView: open, Apply, Play (time = load fraction)
  target.vtp, hilum_anchor.mrk.json
  load_collapse_in_slicer.py  then, on the Slicer machine, one command:
                                  Slicer --python-script <folder>/load_collapse_in_slicer.py
                              (or in Slicer's Python console: p = r'<that file>'; exec(open(p).read(), {'__file__': p}))
                              loads everything, colours by displacement and starts playing
"""

import json
import logging
import shutil
import time

import numpy as np

from ..data_io import write_surface
from .geometry import WallDistance, polydata
from .problem import CollapseProblem

log = logging.getLogger(__name__)


def make_sequence(cfg, frames=30, mode="fem", volume=False, fps=8.0, solver=None):
    """solver: a core already built on this problem (the forward step passes its own)."""
    problem = CollapseProblem.load(cfg.setup_dir / "problem.npz")
    state_path = cfg.run_dir / "state.npz"
    if not state_path.is_file():
        raise SystemExit("%s not found: run the collapse or fit step of this config first" % state_path)
    with np.load(state_path) as b:
        q, nu, labels, Us_best = b["q"], float(b["nu"]), b["tri_labels"], b["surface_displacement"]
        body, body_dir, sign = float(b["body"]), b["body_direction"], float(b["pressure_sign"])
    E = json.loads((cfg.run_dir / "summary.json").read_text())["physics"]["E_Pa"]
    X_ref, tris = problem.reference, problem.tris
    target_surf = WallDistance(*problem.target_surface) if problem.has_target else None
    out = cfg.run_dir / "sequence"
    out.mkdir(parents=True, exist_ok=True)
    for old in list(out.glob("frame_*.vtp")) + list(out.glob("volume_*.vtk")):
        old.unlink()
    lams = np.linspace(0.0, 1.0, frames + 1)
    log.info("%s: %d frames, mode %s, nu=%.3f, mean p=%.0f Pa", cfg.patient, len(lams), mode, nu, q.mean() * E)

    # displacement per frame
    seq, n_lin, t0 = [], 0, time.time()
    if mode == "fem":
        if solver is None:
            from .fit import make_solver
            solver = make_solver(cfg, problem)
        solver.reset()
        solver.set_regions(labels, len(q))
        if body:
            solver.set_body(body_dir)
    else:
        solver = None
    for i, lam in enumerate(lams):
        Us, state = Us_best * lam, None
        if solver is not None and lam > 0.0:
            r = solver.solve(sign * q * lam, nu, body * lam) if body else solver.solve(sign * q * lam, nu)
            if r is None:
                n_lin += 1
                log.warning("frame %d (%.0f%%): Newton failed, linear morph used for this frame", i, 100 * lam)
            else:
                Us, state = r, solver.get_state()
        seq.append((Us, state))
        if i % 5 == 0 or i == len(lams) - 1:
            log.info("frame %3d/%d  load %5.1f%%  (%.0f s)", i, len(lams) - 1, 100 * lam, time.time() - t0)
    d_end = np.linalg.norm(seq[-1][0] - Us_best, axis=1)
    log.info("last frame vs the run's solution: mean %.2e, max %.2e mm", d_end.mean(), d_end.max())
    if d_end.max() > 1e-3:
        log.warning("the load ramp reached a different equilibrium; the last frame is replaced by the run's solution")
        seq[-1] = (Us_best, None)

    # write
    wall = WallDistance(problem.wall_points, problem.wall_tris) if problem.has_wall else None
    q_tri = np.where(labels >= 0, q[np.maximum(labels, 0)], 0.0) * E
    dmax = max(np.linalg.norm(Us, axis=1).max() for Us, _ in seq)
    names = []
    for i, (lam, (Us, state)) in enumerate(zip(lams, seq)):
        s = polydata(X_ref + Us, tris)
        s.point_data["Displacement_mm"] = Us
        s.point_data["Disp_mag_mm"] = np.linalg.norm(Us, axis=1)
        if target_surf is not None:
            s.point_data["Error_mm"] = np.linalg.norm(X_ref + Us - target_surf.closest(X_ref + Us)[0], axis=1)
        if wall is not None:
            s.point_data["WallDistance_mm"] = wall.distance(X_ref + Us)   # > 0 outside the cavity
        s.cell_data["Pressure_Pa"] = q_tri * lam
        s.field_data["LoadFraction"] = np.array([lam])
        names.append("frame_%03d.vtp" % i)
        write_surface(s, out / names[-1], "LPS")
        if volume and solver is not None and state is not None:
            try:
                solver.export_volume(out / ("volume_%03d.vtk" % i), state)
            except NotImplementedError:
                pass

    with open(out / "collapse.pvd", "w") as f:
        f.write('<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1">\n <Collection>\n')
        for lam, n in zip(lams, names):
            f.write('  <DataSet timestep="%.4f" group="" part="0" file="%s"/>\n' % (lam, n))
        f.write(" </Collection>\n</VTKFile>\n")
    if problem.has_target:
        write_surface(polydata(*problem.target_surface), out / "target.vtp", "LPS")
    if cfg.hilum.output.is_file():
        shutil.copy(cfg.hilum.output, out / "hilum_anchor.mrk.json")
    script = out / "load_collapse_in_slicer.py"
    script.write_text(SLICER_SCRIPT % dict(folder=str(out), n=len(names), dmax=float(np.ceil(dmax)), fps=fps))

    vols = [abs(polydata(X_ref + seq[j][0], tris).volume) / 1000 for j in (0, len(seq) // 2, -1)]
    log.info("volume: start %.0f mL -> half load %.0f mL -> end %.0f mL; %d frame(s) by linear morph", *vols, n_lin)
    log.info("sequence written to %s. Slicer: Slicer --python-script %s (or in the Python console: "
             "p = r'%s'; exec(open(p).read(), {'__file__': p})); ParaView: open %s, Apply, Play",
             out, script, script, out / "collapse.pvd")
    return out


SLICER_SCRIPT = r'''"""Lung collapse sequence for 3D Slicer, written by the lung_collapse pipeline.

    Slicer --python-script load_collapse_in_slicer.py
or, in Slicer's Python console:
    p = r'<this folder>/load_collapse_in_slicer.py'; exec(open(p).read(), {'__file__': p})
(plain exec(open(...).read()) does not tell the script where it is)

Loads the frames as a sequence (coloured by displacement), the target as a grey wireframe
("LungCollapse_target") and the hilum spheres, switches to the 3D view and starts playing. Files are looked up next to this
script, so the folder can be copied anywhere.
"""
import os
import sys

import slicer

FALLBACK = %(folder)r
N_FRAMES = %(n)d
DISP_MAX_MM = %(dmax).1f
FPS = %(fps)g


def _folder():
    for p in (globals().get("__file__"), sys.argv[0] if sys.argv and sys.argv[0].endswith(".py") else None):
        if p and os.path.isfile(os.path.join(os.path.dirname(os.path.abspath(p)), "frame_000.vtp")):
            return os.path.dirname(os.path.abspath(p))
    if os.path.isfile(os.path.join(FALLBACK, "frame_000.vtp")):
        return FALLBACK
    raise RuntimeError("frames not found next to the script nor in %%s; in the Python console run it as "
                       "p = r'<path>/load_collapse_in_slicer.py'; exec(open(p).read(), {'__file__': p})" %% FALLBACK)


def load_collapse():
    folder = _folder()
    scene = slicer.mrmlScene                                    # reload cleanly: only our own nodes
    for node in [scene.GetNthNode(i) for i in range(scene.GetNumberOfNodes())]:
        if node is not None and ((node.GetName() or "").startswith("LungCollapse")
                                 or node.GetAttribute("LungCollapse") == "1"):
            scene.RemoveNode(node)

    seq = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSequenceNode", "LungCollapse_seq")
    seq.SetIndexName("load")
    seq.SetIndexUnit("%%")
    for i in range(N_FRAMES):
        node = slicer.util.loadModel(os.path.join(folder, "frame_%%03d.vtp" %% i))   # SPACE=LPS in the file
        seq.SetDataNodeAtValue(node, "%%.1f" %% (100.0 * i / max(N_FRAMES - 1, 1)))
        slicer.mrmlScene.RemoveNode(node)

    browser = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSequenceBrowserNode", "LungCollapse_browser")
    browser.AddSynchronizedSequenceNode(seq)
    browser.SetSelectedItemNumber(0)
    slicer.modules.sequences.logic().UpdateProxyNodesFromSequences(browser)
    proxy = browser.GetProxyNode(seq)
    proxy.SetName("LungCollapse")
    proxy.CreateDefaultDisplayNodes()
    d = proxy.GetDisplayNode()
    d.SetColor(0.9, 0.5, 0.4)
    try:                                                        # colour by displacement, fixed range
        d.SetActiveScalarName("Disp_mag_mm")
        d.SetAndObserveColorNodeID("vtkMRMLColorTableNodeFileColdToHotRainbow.txt")
        d.SetScalarRangeFlag(slicer.vtkMRMLDisplayNode.UseManualScalarRange)
        d.SetScalarRange(0.0, DISP_MAX_MM)
        d.SetScalarVisibility(True)
    except Exception as e:
        print("displacement colouring not set (%%s)" %% e)

    if os.path.isfile(os.path.join(folder, "target.vtp")):
        tgt = slicer.util.loadModel(os.path.join(folder, "target.vtp"))
        tgt.SetName("LungCollapse_target")
        tgt.GetDisplayNode().SetRepresentation(1)               # wireframe
        tgt.GetDisplayNode().SetColor(0.6, 0.6, 0.6)
    hil = os.path.join(folder, "hilum_anchor.mrk.json")
    if os.path.isfile(hil):
        before = set(n.GetID() for n in slicer.util.getNodesByClass("vtkMRMLMarkupsNode"))
        slicer.util.loadMarkups(hil)
        for n in slicer.util.getNodesByClass("vtkMRMLMarkupsNode"):
            if n.GetID() not in before:
                n.SetAttribute("LungCollapse", "1")

    lm = slicer.app.layoutManager()
    lm.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutOneUp3DView)
    lm.threeDWidget(0).threeDView().resetFocalPoint()
    slicer.modules.sequences.setToolBarActiveBrowserNode(browser)
    slicer.modules.sequences.showSequenceBrowser(browser)
    browser.SetPlaybackRateFps(FPS)
    browser.SetPlaybackLooped(True)
    browser.SetPlaybackActive(True)
    print("Loaded %%d frames from %%s: colour = displacement 0-%%.0f mm, grey wireframe = target. "
          "Pause / scrub with the Sequences toolbar." %% (seq.GetNumberOfDataNodes(), folder, DISP_MAX_MM))


load_collapse()
'''
