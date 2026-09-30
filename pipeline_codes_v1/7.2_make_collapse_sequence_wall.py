#!/usr/bin/env python3
"""
make_collapse_sequence.py -- turn a lung_inverse_fem_fit.py result into an animation

Re-solves the forward FEM with the BEST fitted pressures, ramped 0 -> 100 % over N frames
(quasi-static load path, each frame warm-started from the previous one). This is the
physical collapse path, not a straight-line morph. The last frame is checked against the
saved best solution.

Needs lung_inverse_fem_fit.py in the same folder, and a finished fem_fit_* run folder.

Usage:
  /usr/bin/python3 make_collapse_sequence.py --run-dir results/fem_fit_20260910_190609
  options: --frames 30   --mode fem|linear   --volume  (also write tetra volume frames)

Output (inside the run folder, sub-folder 'sequence/'):
  collapse.pvd                    -> ParaView: File > Open, Apply, press Play
  frame_000.vtp ... frame_NNN.vtp -> surface per frame (Displacement, Error_mm, Pressure_Pa)
  load_sequence_in_slicer.py      -> Slicer: exec(open('.../load_sequence_in_slicer.py').read())
"""
import os
import sys
import json
import time
import types
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    import lung_inverse_fem_fit as F
except ImportError:
    sys.exit("ERROR: put make_collapse_sequence.py next to lung_inverse_fem_fit.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="fem_fit_* folder")
    ap.add_argument("--frames", type=int, default=30, help="number of steps after the start frame")
    ap.add_argument("--mode", default="fem", choices=["fem", "linear"],
                    help="fem = re-solve along the pressure ramp (physical); linear = straight morph")
    ap.add_argument("--volume", action="store_true", help="also export volume frames (.vtk)")
    ap.add_argument("--wall", default=None,
                    help="cavity surface (e.g. inflated left lung) the lung may not leave during collapse")
    ap.add_argument("--wall-stiffness", type=float, default=20.0)
    ap.add_argument("--wall-tol", type=float, default=2.0, help="allowed outward motion [mm] (like the extension margin)")
    args = ap.parse_args()

    import pyvista as pv
    rd = os.path.abspath(args.run_dir)
    need = ["clean_field.npz", "best_state.npz", "hilum_tri_mask.npy", "deformable_volume.msh",
            "target_used.vtp", "run_args.json"]
    miss = [f for f in need if not os.path.isfile(os.path.join(rd, f))]
    if miss:
        sys.exit("ERROR: missing in run folder: %s" % ", ".join(miss))

    run = json.load(open(os.path.join(rd, "run_args.json")))
    if args.wall is None and run.get("wall"):            # fit was done with a wall -> replay with it
        args.wall = run["wall"]
        args.wall_stiffness = float(run.get("wall_stiffness", args.wall_stiffness))
        args.wall_tol = float(run.get("wall_tol", args.wall_tol))
        print("using the wall from the fit: %s" % args.wall)
    cf = np.load(os.path.join(rd, "clean_field.npz"))
    X_def, tris = cf["pts"], cf["tris"]
    best = np.load(os.path.join(rd, "best_state.npz"))
    q, nu, tri_labels, U_best = best["q"], float(best["nu"]), best["tri_labels"], best["U"]
    K = len(q)
    hilum_tri = np.load(os.path.join(rd, "hilum_tri_mask.npy"))
    tgt = pv.read(os.path.join(rd, "target_used.vtp"))
    X_tgt = np.asarray(tgt.points)
    corr = len(X_tgt) == len(X_def)
    E = float(run.get("E", 3000.0))
    print("run: %s\n  K=%d regions, nu=%.3f, mean p=%.0f Pa, %d surface pts"
          % (rd, K, nu, q.mean() * E, len(X_def)))

    outd = os.path.join(rd, "sequence")
    os.makedirs(outd, exist_ok=True)

    faces_vtk = np.hstack([np.full((len(tris), 1), 3), tris]).ravel()
    base = pv.PolyData(X_def.copy(), faces_vtk)
    q_tri = np.where(tri_labels >= 0, q[np.maximum(tri_labels, 0)], 0.0) * E

    # ---------------- displacement per frame ----------------
    lams = np.linspace(0.0, 1.0, args.frames + 1)
    fem = None
    if args.mode == "fem":
        cen, _, _ = F.tri_geometry(X_def, tris)
        fargs = types.SimpleNamespace(newton_maxit=int(run.get("newton_maxit", 30)),
                                      newton_tol=float(run.get("newton_tol", 1e-7)),
                                      load_steps=int(run.get("load_steps", 4)),
                                      wall_tol=args.wall_tol)
        fem = F.LungFEM(os.path.join(rd, "deformable_volume.msh"), int(run.get("order", 1)),
                        X_def, cen, hilum_tri, fargs)
        if fem.ndof != len(U_best):
            sys.exit("ERROR: mesh/dof mismatch with best_state.npz (different run folder?)")
        if args.wall:
            wpoly, wsp = F.read_surface(args.wall)
            if (wsp or "LPS") == "RAS":
                wpoly.points[:, :2] *= -1.0
            wpoly = F.orient(wpoly)
            fem.set_wall(F.Wall(wpoly), args.wall_stiffness, args.wall_tol)
        fem.build(np.where(fem.press_face, tri_labels[fem.tri_of_face], -1), K, nu)

    frames_U, t0 = [], time.time()
    for i, lam in enumerate(lams):
        if args.mode == "linear" or lam == 0.0:
            U = U_best * lam
        else:
            U = fem.forward(q * lam, nu)       # warm-started from previous frame
            if U is None:
                print("  frame %d (%.0f%%): Newton failed -> using linear morph for this frame"
                      % (i, 100 * lam))
                U = U_best * lam
            else:
                U = U.copy()
        frames_U.append(U)
        if i % 5 == 0 or i == len(lams) - 1:
            print("  frame %3d/%d  load %5.1f%%  (%.0f s)" % (i, len(lams) - 1, 100 * lam, time.time() - t0))

    if args.mode == "fem" and fem.wall is not None:
        d_end = np.linalg.norm(fem.surface_disp(frames_U[-1]) - fem.surface_disp(U_best), axis=1)
        print("  with wall: end state differs from the (wall-free) fit by mean %.2f / max %.2f mm"
              % (d_end.mean(), d_end.max()))
        if corr:
            e = np.linalg.norm(X_def + fem.surface_disp(frames_U[-1]) - X_tgt, axis=1)
            print("  with wall: end-state mean error to target %.2f mm" % e.mean())
    elif args.mode == "fem":
        dmax = np.abs(frames_U[-1] - U_best).max()
        print("  check: last frame vs saved best solution, max |dU| = %.2e mm %s"
              % (dmax, "(OK)" if dmax < 1e-3 else "(WARNING: differs, see note below)"))
        if dmax >= 1e-3:
            print("  -> the ramp found a slightly different equilibrium; the last frame is replaced "
                  "by the saved best so the end state matches the fit exactly.")
            frames_U[-1] = U_best.copy()

    # ---------------- write frames ----------------
    diag_wall = fem.wall if (fem is not None and fem.wall is not None) else None
    names, surf, pen_max = [], [], []
    Us_best_lin = None if fem is not None else best_surface_disp(rd, X_def)
    for i, (lam, U) in enumerate(zip(lams, frames_U)):
        Us = fem.surface_disp(U) if fem is not None else Us_best_lin * lam
        surf.append(Us)
        m = base.copy()
        m.points = X_def + Us
        m.point_data["Displacement"] = Us
        m.point_data["Disp_mag_mm"] = np.linalg.norm(Us, axis=1)
        if corr:
            m.point_data["Error_mm"] = np.linalg.norm(m.points - X_tgt, axis=1)
        m.cell_data["Pressure_Pa"] = q_tri * lam
        m.cell_data["PressureRegion"] = tri_labels
        m.field_data["LoadFraction"] = np.array([lam])
        if diag_wall is not None:
            phi, _ = diag_wall.eval(m.points)
            m.point_data["WallDist_mm"] = phi          # >0 = outside the cavity
            pen_max.append(float(phi.max()))
        name = "frame_%03d.vtp" % i
        F.write_vtp_lps(m, os.path.join(outd, name))
        names.append(name)
        if args.volume and fem is not None:
            fem.mfu.export_to_vtk(os.path.join(outd, "volume_%03d.vtk" % i), "ascii",
                                  fem.mfu, U, "Displacement")

    # ParaView collection (timestep = load fraction 0..1)
    with open(os.path.join(outd, "collapse.pvd"), "w") as f:
        f.write('<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1">\n <Collection>\n')
        for lam, n in zip(lams, names):
            f.write('  <DataSet timestep="%.4f" group="" part="0" file="%s"/>\n' % (lam, n))
        f.write(" </Collection>\n</VTKFile>\n")
    # target for reference in the same folder
    F.write_vtp_lps(tgt, os.path.join(outd, "target_used.vtp"))

    # Slicer loader
    slicer_py = os.path.join(outd, "load_sequence_in_slicer.py")
    with open(slicer_py, "w") as f:
        f.write(SLICER_TEMPLATE % dict(folder=outd, n=len(names)))

    vols = [abs(pv.PolyData(X_def + surf[j], faces_vtk).volume) / 1000
            for j in (0, len(surf) // 2, len(surf) - 1)]
    print("\nvolume: start %.0f mL -> mid %.0f mL -> end %.0f mL (target %.0f mL)"
          % (vols[0], vols[1], vols[2], abs(tgt.volume) / 1000))
    if pen_max:
        print("max distance OUTSIDE the wall per frame [mm]: " +
              " ".join("%.1f" % v for v in pen_max[::max(1, len(pen_max) // 10)]) +
              "  (last %.2f)" % pen_max[-1])
    print("\nwritten to %s" % outd)
    print("  ParaView : open collapse.pvd -> Apply -> Play (timestep = load fraction)")
    print("  Slicer   : in the Python console run\n"
          "             exec(open('%s').read())" % slicer_py)


def best_surface_disp(rd, X_def):
    import pyvista as pv
    p = os.path.join(rd, "deformed_best.vtp")
    if not os.path.isfile(p):
        sys.exit("ERROR: linear mode needs deformed_best.vtp in the run folder")
    return np.asarray(pv.read(p).points) - X_def


SLICER_TEMPLATE = r'''# Load the collapse frames into a Slicer sequence (run in Slicer's Python console)
import os, slicer
folder = %(folder)r
n = %(n)d
try:
    seq = slicer.util.getNode("LungCollapse_seq")          # reuse if already loaded
except slicer.util.MRMLNodeNotFoundException:
    seq = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSequenceNode", "LungCollapse_seq")
    seq.SetIndexName("load")
    seq.SetIndexUnit("%%")
    for i in range(n):
        node = slicer.util.loadModel(os.path.join(folder, "frame_%%03d.vtp" %% i))   # VTP carries SPACE=LPS
        seq.SetDataNodeAtValue(node, "%%.1f" %% (100.0 * i / (n - 1)))
        slicer.mrmlScene.RemoveNode(node)
browser = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSequenceBrowserNode", "LungCollapse_browser")
browser.AddSynchronizedSequenceNode(seq)
browser.SetSelectedItemNumber(0)
slicer.modules.sequences.logic().UpdateProxyNodesFromSequences(browser)
proxy = browser.GetProxyNode(seq)
proxy.SetName("LungCollapse")
proxy.CreateDefaultDisplayNodes()
proxy.GetDisplayNode().SetColor(0.9, 0.5, 0.4)
tgt = slicer.util.loadModel(os.path.join(folder, "target_used.vtp"))
tgt.GetDisplayNode().SetRepresentation(1)   # wireframe target for comparison
tgt.GetDisplayNode().SetColor(0.6, 0.6, 0.6)
slicer.modules.sequences.setToolBarActiveBrowserNode(browser)
slicer.modules.sequences.showSequenceBrowser(browser)
print("Loaded %%d frames. Use the Sequence toolbar (play button) to animate." %% seq.GetNumberOfDataNodes())
'''

if __name__ == "__main__":
    main()
