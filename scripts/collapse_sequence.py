"""Collapse sequence of one patient, to look at in 3D Slicer (or ParaView). Run it when needed:

    python scripts/collapse_sequence.py --config configs/karl04.json
    python scripts/collapse_sequence.py --config configs/karl04.json --frames 40 --volume
    python scripts/collapse_sequence.py --config configs/karl04.json --mode linear   # no FEM library needed

Needs the fem_setup and fem_fit outputs of the config. Mode "fem" re-solves the forward FEM with
the best fitted pressures ramped 0 -> 100 %% (quasi-static load path, each frame warm-started from
the previous one) with the same core and solver options as fem_fit; mode "linear" is a straight
morph from the inflated to the fitted shape.

Output, output_dir/sequence/ (sequence_<run_name>/ for a fem_fit.run_name other than "fit"; self-contained: copy the folder to the machine with Slicer):
  frame_000.vtp ...           surface per frame: Displacement_mm, Disp_mag_mm, Error_mm,
                              Pressure_Pa, PressureRegion (+ WallDistance_mm with a wall)
  volume_000.vtk ...          volume frames (--volume)
  collapse.pvd                ParaView: open, Apply, Play (time = load fraction)
  target_aligned.vtp, hilum_anchor.mrk.json
  load_collapse_in_slicer.py  then, on the Slicer machine, one command:
                                  Slicer --python-script <folder>/load_collapse_in_slicer.py
                              (or in Slicer's Python console: p = r'<that file>'; exec(open(p).read(), {'__file__': p}))
                              loads everything, colours by displacement and starts playing
"""

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.collapse.geometry import WallDistance, polydata   # noqa: E402
from pipeline.collapse.problem import CollapseProblem             # noqa: E402
from pipeline.collapse.solvers import get_solver                  # noqa: E402
from pipeline.config import load_config                           # noqa: E402
from pipeline.data_io import write_surface                        # noqa: E402

log = logging.getLogger("collapse_sequence")


def make_sequence(cfg, frames, mode, volume, fps):
    fit = cfg.fem_fit
    problem = CollapseProblem.load(fit.setup / "problem.npz")
    best_path = fit.workdir / "best_state.npz"
    if not best_path.is_file():
        raise SystemExit("%s not found: run fem_fit for this config first" % best_path)
    with np.load(best_path) as b:
        q, nu, labels, Us_best = b["q"], float(b["nu"]), b["tri_labels"], b["surface_displacement"]
    summary = fit.workdir / "result_summary.json"
    sign = json.loads(summary.read_text()).get("pressure_sign", 1.0) if summary.is_file() else 1.0
    K = len(q)
    X_ref, X_tgt, tris = problem.reference, problem.target, problem.tris
    out = cfg.output_dir / ("sequence" if fit.run_name == "fit" else "sequence_" + fit.run_name)
    out.mkdir(parents=True, exist_ok=True)
    for old in list(out.glob("frame_*.vtp")) + list(out.glob("volume_*.vtk")):
        old.unlink()
    lams = np.linspace(0.0, 1.0, frames + 1)
    log.info("%s: %d frames, mode %s, K=%d, nu=%.3f, mean p=%.0f Pa%s", cfg.patient, len(lams), mode, K, nu,
             q.mean() * fit.E_Pa, ", solver %s" % fit.solver if mode == "fem" else "")

    # displacement per frame
    solver, seq, n_lin, t0 = None, [], 0, time.time()
    if mode == "fem":
        solver = get_solver(fit.solver)(problem, fit.solver_options)
        solver.set_regions(labels, K)
    for i, lam in enumerate(lams):
        Us, state = Us_best * lam, None
        if solver is not None and lam > 0.0:
            r = solver.solve(sign * q * lam, nu)              # warm start: previous frame
            if r is None:
                n_lin += 1
                log.warning("frame %d (%.0f%%): Newton failed, linear morph used for this frame", i, 100 * lam)
            else:
                Us, state = r, solver.get_state()
        seq.append((Us, state))
        if i % 5 == 0 or i == len(lams) - 1:
            log.info("frame %3d/%d  load %5.1f%%  (%.0f s)", i, len(lams) - 1, 100 * lam, time.time() - t0)
    d_end = np.linalg.norm(seq[-1][0] - Us_best, axis=1)
    log.info("last frame vs fitted solution: mean %.2e, max %.2e mm", d_end.mean(), d_end.max())
    if d_end.max() > 1e-3:
        log.warning("the load ramp reached a different equilibrium; the last frame is replaced by the fit")
        seq[-1] = (Us_best, None)

    # write
    wall = WallDistance(problem.wall_points, problem.wall_tris) if problem.has_wall else None
    q_tri = np.where(labels >= 0, q[np.maximum(labels, 0)], 0.0) * fit.E_Pa
    dmax = max(np.linalg.norm(Us, axis=1).max() for Us, _ in seq)
    names = []
    for i, (lam, (Us, state)) in enumerate(zip(lams, seq)):
        s = polydata(X_ref + Us, tris)
        s.point_data["Displacement_mm"] = Us
        s.point_data["Disp_mag_mm"] = np.linalg.norm(Us, axis=1)
        s.point_data["Error_mm"] = np.linalg.norm(X_ref + Us - X_tgt, axis=1)
        if wall is not None:
            s.point_data["WallDistance_mm"] = wall.distance(X_ref + Us)   # > 0 outside the cavity
        s.cell_data["Pressure_Pa"] = q_tri * lam
        s.cell_data["PressureRegion"] = labels
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
    write_surface(polydata(X_tgt, tris), out / "target_aligned.vtp", "LPS")
    anchor = cfg.hilum.output if cfg.hilum is not None else None
    if anchor is not None and anchor.is_file():
        shutil.copy(anchor, out / "hilum_anchor.mrk.json")
    script = out / "load_collapse_in_slicer.py"
    script.write_text(SLICER_SCRIPT % dict(folder=str(out), n=len(names), dmax=float(np.ceil(dmax)), fps=fps))

    vols = [abs(polydata(X_ref + seq[j][0], tris).volume) / 1000 for j in (0, len(seq) // 2, -1)]
    log.info("volume: start %.0f mL -> half load %.0f mL -> end %.0f mL (target %.0f mL); %d frame(s) "
             "by linear morph", *vols, abs(polydata(X_tgt, tris).volume) / 1000, n_lin)
    print("\nwritten to %s\n  Slicer  : Slicer --python-script %s\n            (or in the Python console: "
          "p = r'%s'; exec(open(p).read(), {'__file__': p}))\n  ParaView: open %s, Apply, Play" % (out, script, script, out / "collapse.pvd"))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True, help="patient config (uses its fem_setup / fem_fit outputs)")
    ap.add_argument("--frames", type=int, default=30, help="load steps after the initial frame")
    ap.add_argument("--mode", default="fem", choices=["fem", "linear"],
                    help="fem = re-solve along the pressure ramp (physical); linear = straight morph")
    ap.add_argument("--volume", action="store_true", help="also write volume frames (fem mode)")
    ap.add_argument("--fps", type=float, default=8.0, help="playback speed in Slicer")
    ap.add_argument("--set", nargs="+", metavar="KEY=VALUE", default=[],
                    help="config overrides as in main.py, e.g. fem_fit.run_name=fit_torch_plane")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    cfg = load_config(args.config, args.set)
    if cfg.fem_fit is None:
        raise SystemExit("%s has no fem_fit section" % args.config)
    make_sequence(cfg, args.frames, args.mode, args.volume, args.fps)


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

    tgt = slicer.util.loadModel(os.path.join(folder, "target_aligned.vtp"))
    tgt.SetName("LungCollapse_target")
    tgt.GetDisplayNode().SetRepresentation(1)                   # wireframe
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


if __name__ == "__main__":
    main()
