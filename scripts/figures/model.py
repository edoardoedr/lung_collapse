import math
from common import *

W, H = 760, 876
LUNG, LUNG_S = "#f6ddd5", "#993c1d"
BL = "#185fa5"
o = [header(W, H, "Physical model of the lung collapse",
            "Left: in vivo the inflated lung fills the chest cavity, held open by the pleural pressure, which is lower at "
            "the top; its gradient carries the weight of the lung. Right: in the pneumothorax that pressure is lost; the "
            "load is an inward pressure, uniform part plus gradient, the lung shrinks towards the clamped hilum and rests "
            "on the dependent chest wall. Below: patient positions and the direction of gravity in the reference CT and "
            "during the collapse, and the table of the parameters.")]


def ellipse_pt(cx, cy, rx, ry, deg):
    a = math.radians(deg)
    return cx + rx * math.cos(a), cy - ry * math.sin(a)          # deg counter-clockwise from +x, y up


def normal(rx, ry, deg):
    a = math.radians(deg)
    nx, ny = math.cos(a) / rx, -math.sin(a) / ry
    n = math.hypot(nx, ny)
    return nx / n, ny / n


def arc(cx, cy, rx, ry, d0, d1, n=24):
    pts = [ellipse_pt(cx, cy, rx, ry, d0 + (d1 - d0) * i / n) for i in range(n + 1)]
    return "M" + " L".join("%.1f %.1f" % p for p in pts)


def panel(x0, title):
    o.append('<rect x="%g" y="20" width="350" height="330" rx="12" class="grp"/>' % x0)
    o.append(text(x0 + 14, 38, title, "h", "#444441"))


RX, RY = 112, 84
# ---------------- A: in vivo ----------------
panel(20, "In vivo: the inflated lung (reference)")
cx, cy = 185, 200
o.append('<ellipse cx="%g" cy="%g" rx="%g" ry="%g" fill="%s" stroke="%s" stroke-width="1.2"/>' % (cx, cy, RX, RY, LUNG, LUNG_S))
o.append('<path d="%s" fill="none" stroke="#444441" stroke-width="5" stroke-linecap="round"/>' % arc(cx, cy, RX, RY, 158, 202))
o.append(text(cx - RX + 16, cy, "hilum", "l", "#444441"))
o.append(text(cx - RX + 16, cy + 14, "u = 0", "m", "#444441"))
for deg in range(-135, 136, 27):                                 # outward pull, longer at the top
    x, y = ellipse_pt(cx, cy, RX, RY, deg)
    nx, ny = normal(RX, RY, deg)
    L = 8 + 20 * (cy + RY - y) / (2 * RY)
    o.append(arrow(x + 3 * nx, y + 3 * ny, x + (3 + L) * nx, y + (3 + L) * ny, "ap", "a", ' style="stroke:%s"' % BL))
o.append(text(cx + 22, cy - 16, "inflated lung", "s", LUNG_S))
o.append(text(cx + 22, cy, sub("stress-free reference Ω_0"), "s", LUNG_S))
o.append(text(cx + 22, cy + 16, "density ρ̄, stiffness E, ν", "s", LUNG_S))
o.append(text(30, 62, "pleural under-pressure, larger at the top:", "l", BL))
o.append(text(30, 77, sub("p_{atm} − p_{pl} = p_0 + ρ̄g (h − h̄)"), "m", BL))
o.append(text(30, 314, "its gradient ρ̄g ≈ 2 Pa/mm carries the weight (Archimedes):", "l"))
o.append(text(30, 333, sub("−∮ p_{pl} n dA = ρ̄ g V e_h,   e_h = −e_A"), "m"))
# gravity and height axis
gx = 350
o.append(arrow(gx, 150, gx, 250, "ag", "a", ' style="stroke:#993c1d;stroke-width:2"'))
o.append(text(gx, 262, "g", "m", "#993c1d", "middle"))
o.append(text(gx, 278, sub("e_A"), "m", "#993c1d", "middle"))
o.append(arrow(gx - 20, 250, gx - 20, 150, "ak", "a", ' style="stroke:#444441"'))
o.append(text(gx - 20, 139, "h", "m", "#444441", "middle"))
o.append(text(gx - 18, 123, "anterior", "k", None, "middle"))
o.append(text(gx - 40, 296, "posterior", "k", None, "middle"))

# ---------------- B: pneumothorax ----------------
panel(390, "Pneumothorax: load and collapsed lung")
cx2 = 555
o.append('<ellipse cx="%g" cy="%g" rx="%g" ry="%g" fill="none" stroke="#5f5e5a" stroke-width="1.2" stroke-dasharray="6 4"/>' % (cx2, cy, RX, RY))
# collapsed lung: attached to the hilum, resting on the dependent wall
ccx, ccy, crx, cry = cx2 - 34, cy + 27, 76, 56
o.append('<ellipse cx="%g" cy="%g" rx="%g" ry="%g" fill="%s" stroke="%s" stroke-width="1.2"/>' % (ccx, ccy, crx, cry, LUNG, LUNG_S))
o.append('<path d="%s" fill="none" stroke="#444441" stroke-width="5" stroke-linecap="round"/>' % arc(cx2, cy, RX, RY, 158, 202))
o.append(text(ccx + 4, ccy - 8, "collapsed lung", "s", LUNG_S))
o.append(text(398, 252, "hilum", "l", "#444441"))
o.append(text(398, 267, "u = 0", "m", "#444441"))
o.append(text(ccx + 4, ccy + 9, "x = X + u", "m", LUNG_S, "middle"))
for deg in range(-54, 136, 27):                                  # inward load, larger at the top
    x, y = ellipse_pt(cx2, cy, RX, RY, deg)
    nx, ny = normal(RX, RY, deg)
    L = 8 + 20 * (cy + RY - y) / (2 * RY)
    o.append(arrow(x + (L + 16) * nx, y + (L + 16) * ny, x + 16 * nx, y + 16 * ny, "ap", "a", ' style="stroke:%s"' % BL))
# contact on the dependent wall
o.append('<path d="%s" fill="none" stroke="#854f0b" stroke-width="4" stroke-linecap="round"/>' % arc(cx2, cy, RX, RY, -128, -92))
for deg in (-122, -110, -98):
    x, y = ellipse_pt(cx2, cy, RX, RY, deg)
    nx, ny = normal(RX, RY, deg)
    o.append(arrow(x + 15 * nx, y + 15 * ny, x + 3 * nx, y + 3 * ny, "ak", "a", ' style="stroke:#854f0b"'))
o.append(text(400, 62, "load = the pleural pressure that is lost, on the free surface:", "l", BL))
o.append(text(400, 77, sub("Δp(X) = p_0 + ρ̄g (h(X) − h̄)"), "m", BL))
o.append(text(400, 315, "chest wall (dashed): margin A, rigid, frictionless;", "l", "#854f0b"))
o.append(text(400, 330, "solid brown: contact, the wall now carries the weight", "l", "#854f0b"))
o.append(text(562, 296, "contact", "l", "#854f0b"))
o.append(text(cx2 + 62, cy - 34, "air at p_atm".replace("p_atm", sub("p_{atm}")), "s", "#5f5e5a"))

# ---------------- equation ----------------
o.append('<rect x="20" y="364" width="720" height="62" rx="8" fill="%s" stroke="%s" stroke-width="0.8"/>' % (PURPLE["fill"], PURPLE["stroke"]))
o.append(text(380, 381, "Non-dimensional load (only ratios to E matter for the shape)", "t", PURPLE["title"]))
o.append(text(380, 407, sub("q(X) = Δp / E = q_0 + γ (h(X) − h̄),  q_0 = p_0 / E,  γ = ρ̄ g / E,  h = −e_A · X"),
              "m", PURPLE["title"], "middle", ' style="font-size:15px"'))

# ---------------- C: positions ----------------
o.append('<rect x="20" y="440" width="720" height="166" rx="12" class="grp"/>')
o.append(text(34, 458, sub("Patient position: gravity e_A in the inflated CT, e_B during the collapse (LPS axes)"), "h", "#444441"))


def torso(cx, cy, rot, label, vec):
    """Axial section of the chest: torso, two lungs, spine dot at the back; rot in degrees."""
    g = ['<g transform="translate(%g %g) rotate(%g)">' % (cx, cy, rot),
         '<ellipse cx="0" cy="0" rx="44" ry="27" fill="#f1efe8" stroke="#5f5e5a" stroke-width="1"/>',
         '<ellipse cx="-19" cy="-2" rx="15" ry="16" fill="%s" stroke="%s" stroke-width="0.8"/>' % (LUNG, LUNG_S),
         '<ellipse cx="19" cy="-2" rx="15" ry="16" fill="%s" stroke="%s" stroke-width="0.8"/>' % (LUNG, LUNG_S),
         '<circle cx="0" cy="19" r="4.5" fill="#5f5e5a"/>', '</g>']
    o.append("\n".join(g))
    o.append('<line x1="%g" y1="562" x2="%g" y2="562" stroke="#5f5e5a" stroke-width="2.5" stroke-linecap="round"/>' % (cx - 56, cx + 56))
    o.append(arrow(cx + 64, 506, cx + 64, 552, "ag", "a", ' style="stroke:#993c1d;stroke-width:2"'))
    o.append(text(cx + 73, 528, "g", "m", "#993c1d"))
    o.append(text(cx, 577, label, "s", "#444441"))
    o.append(text(cx, 592, vec, "k", None, "middle"))


torso(100, 533, 0, "supine", "e = +y (posterior)")
torso(270, 516, 90, "left lateral decubitus", "e = +x (left)")
torso(440, 533, 180, "prone", "e = −y (anterior)")
o.append('<circle cx="540" cy="482" r="4.5" fill="#5f5e5a"/>')
o.append(text(550, 482, "spine (posterior)", "k"))
o.append(text(532, 506, sub("same position (e_A = e_B):"), "l", "#444441"))
o.append(text(532, 522, "surface load only", "l"))
o.append(text(532, 546, "different positions: + body force", "l", "#444441"))
o.append(text(532, 564, sub("b = ρ̄ g (e_B − e_A)"), "m", "#993c1d"))
o.append(text(532, 582, sub("resultant = weight along e_B"), "l"))

# ---------------- D: parameters ----------------
o.append('<rect x="20" y="620" width="720" height="242" rx="12" class="grp"/>')
o.append(text(34, 638, "Parameters: each one given (a number) or fitted to the collapsed CT", "h", "#444441"))
cols = (34, 110, 392, 566)
for x, t in zip(cols, ("symbol", "meaning", "config key", "default")):
    o.append(text(x, 660, t, "l", "#444441", None, ' style="font-weight:600"'))
o.append('<line x1="34" y1="670" x2="726" y2="670" stroke="#b4b2ab" stroke-width="0.8"/>')
rows = [(sub("q_0 = p_0 / E"), "uniform pressure released: sets the collapse %", "physics.collapse", "fitted, start 0.3"),
        ("E", "effective stiffness of the lung tissue", "physics.E_Pa", "1360 Pa, or E(collapse %)"),
        ("ν", "compressibility (the lung loses air)", "physics.material.nu", "0.30"),
        ("ρ̄", "mean density of the inflated lung (model)", "physics.tissue_density", "200 kg/m³ (ρ̄g 1.96 Pa/mm)"),
        (sub("ρ̄_{CT}"), "the same measured on the CT (report only)", "physics.ct_density", "per patient, median 163 kg/m³"),
        (sub("e_A, e_B"), "patient position: reference CT, collapse", "physics.gravity", "supine, supine"),
        ("A, σ", "chest wall: margin, smoothing of the field", "fem_setup.wall", "5 mm, 3 mm"),
        ("k", "chest wall: penalty stiffness", "physics.wall", "20 E/mm")]
for i, (sym, meaning, key, default) in enumerate(rows):
    y = 684 + 20 * i
    o.append(text(cols[0], y, sym, "m", "#3c3489"))
    o.append(text(cols[1], y, meaning, "l"))
    o.append(text(cols[2], y, key, "k"))
    o.append(text(cols[3], y, default, "l"))
o.append(text(34, 846, sub("E rule (12 fitted patients, ν 0.30, ρ̄ 200 kg/m³):  log_{10} E [Pa] = 3.53 − 0.0135 × collapse %"
                           "   →  ≈ 1.6 kPa at 25 %, 0.73 kPa at 50 %"), "l", "#3c3489"))
o.append("</svg>\n")
open("collapse_model.svg", "w").write("\n".join(o))
