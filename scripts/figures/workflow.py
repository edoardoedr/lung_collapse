from common import *

W, H = 680, 900
o = [header(W, H, "Lung collapse pipeline",
            "Steps of the pipeline: segmentation in 3D Slicer, input check, surface meshes, hilum anchor, model geometry "
            "(clamp, chest wall, volume mesh), then the physical model with its parameters given or fitted, run forward "
            "(collapse) or inverse (fit); results per run and the GNN surrogate they feed.")]
y = 24
steps = [
    (GREY, "Segmentation", ["inflated lung, airways, arteries, veins (+ collapsed lung, optional)"], "3D Slicer (manual)"),
    (GREEN, "Input check", ["closed surfaces, same scan, containment, notches"], "check_inputs  (needs the collapsed lung)"),
    (GREEN, "Surface meshes", ["inflated lung = the model, collapsed lung = the target; ~480 nodes each (ACVD)"], "geometry"),
    (GREEN, "Hilum anchor", ["rings where airways and vessels enter the lung; hilum = their centroid"], "hilum"),
    (ORANGE, "Model geometry", ["clamp at the hilum, chest wall as a smooth field, tetrahedral mesh"], "fem_setup  →  fem/setup/problem.npz"),
]
for pal, t, lines, code in steps:
    o.append(box(80, y, 520, 62, pal, t, lines, code))
    o.append(arrow(340, y + 62, 340, y + 82))
    y += 84
# physics
o.append(box(80, y, 520, 96, PURPLE, "Physical model: parameters given or fitted",
             ["stiffness E, compressibility ν, density ρ̄, patient position (CT → collapse),",
              "uniform pressure q₀ = p₀ / E, or the volume ratio to reach",
              "pressure / E  =  q₀ + γ (h − h̄),   γ = ρ̄ g / E,   inside the chest wall"], "physics"))
py = y + 96
o.append(path("M240 %g L240 %g" % (py, py + 34)))
o.append(path("M440 %g L440 %g" % (py, py + 34)))
y = py + 36
o.append(box(60, y, 270, 96, BLUE, "Forward", ["all parameters given", "one FEM solve along the load path", "→ collapsed lung (no collapsed CT needed)"], "collapse"))
o.append(box(350, y, 270, 96, BLUE, "Inverse", ["parameters marked “fit” are found by least", "squares on the distance to the collapsed CT", "→ E, p₀, ν of the patient"], "fit"))
o.append(path("M195 %g L195 %g L300 %g L300 %g" % (y + 96, y + 116, y + 116, y + 132)))
o.append(path("M485 %g L485 %g L380 %g L380 %g" % (y + 96, y + 116, y + 116, y + 132)))
y += 134
o.append(box(80, y, 520, 80, AMBER, "Results of a run",
             ["collapsed surface, physical parameters, collapse %, errors, wall penetration", "collapse sequence for 3D Slicer / ParaView"],
             "runs/&lt;run_name&gt;/  surface.vtp  summary.json  state.npz  sequence/"))
o.append(arrow(340, y + 80, 340, y + 100))
y += 102
o.append(box(80, y, 520, 62, GREY, "GNN surrogate (planned)", ["trained on forward runs: geometry + load → displacement; tested on the real collapses"], None, dash=True))
o.append("</svg>\n")
open("collapsed_lung_project_workflow.svg", "w").write("\n".join(o))
print("height used", y + 62 + 24)
