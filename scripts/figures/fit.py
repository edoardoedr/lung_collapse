from common import *

W, H = 792, 800
o = [header(W, H, "Forward and inverse problem",
            "The model geometry and the physical parameters define the load; a finite-element solve by Newton's method "
            "along the load path gives the collapsed lung. Forward: the parameters are given, optionally the uniform "
            "pressure is searched for a target volume. Inverse: the parameters marked fit are updated by bounded least "
            "squares on the two-way distance to the collapsed CT, with analytic sensitivities, until a stop criterion.")]
# inputs
o.append(box(20, 20, 350, 78, ORANGE, "Model geometry", ["inflated lung: tetrahedral mesh Ω₀, clamp Γ_D at the hilum".replace("Ω₀", sub("Ω_0")).replace("Γ_D", sub("Γ_D")),
                                                         "chest wall as a smooth distance field φ̃, margin A"], "fem/setup/problem.npz"))
o.append(box(390, 20, 350, 78, PURPLE, "Physical parameters θ", [sub("q_0 (or p_0, or volume ratio), E, ν, ρ̄, positions e_A → e_B, φ"),
                                                                 "each one given, or marked “fit”"], "physics"))
o.append(path("M195 98 L195 114 L330 114 L330 128"))
o.append(path("M565 98 L565 114 L430 114 L430 128"))
# load
o.append(box(120, 130, 520, 78, PURPLE, "Load", [sub("surface pressure  q(X) = q_0 + γ (h(X) − h̄),   γ = ρ̄ g / E,   h = −e_A · X"),
                                                 sub("body force  b = γ (e_B − e_A)  (only when the position changes)")], "fem/load.py"))
o.append(arrow(380, 208, 380, 226))
# solve
o.append('<rect x="20" y="228" width="720" height="186" rx="12" class="grp"/>')
o.append(text(34, 244, "Finite-element solve  R(U; θ) = 0   (PyTorch / Warp cores, P1 tetrahedra)", "h", BLUE["title"]))
o.append(box(34, 258, 220, 64, BLUE, "Elasticity", ["Neo-Hookean (+ barrier J ≥ φ)", "∫ P(F) : ∇δu dV"]))
o.append(box(270, 258, 220, 64, BLUE, "Pleural pressure", ["follows the surface (Nanson)", sub("∫ q J F^{−T}N · δu dA").replace("^{−T}", "⁻ᵀ")]))
o.append(box(506, 258, 220, 64, BLUE, "Chest wall", ["penalty on the smooth field φ̃", "exact tangent, inside Newton"]))
o.append(box(34, 336, 340, 64, BLUE, "Newton's method", ["exact tangent K = ∂R/∂U, sparse direct solver,".replace("K =", sub("K_t =")), "line search; ≤ 50 iterations"]))
o.append(box(386, 336, 340, 64, BLUE, "Load path", ["load applied proportionally, λ: 0 → 1;", "failed step halved, successful step doubled"]))
o.append(arrow(380, 414, 380, 432))
o.append(box(240, 434, 280, 50, AMBER, "Simulated collapsed lung", ["x = X + u(θ),  volume V(θ)"]))
o.append(path("M240 459 L130 459 L130 506"))
o.append(path("M520 459 L560 459 L560 506"))
# forward
o.append('<rect x="20" y="508" width="230" height="200" rx="12" class="grp"/>')
o.append(text(34, 524, "Forward  (collapse)", "h", GREEN["title"]))
o.append(box(34, 540, 202, 64, GREEN, "Parameters given", [sub("q_0, or p_0 (12 mmHg with the"), "barrier): one solve"]))
o.append(box(34, 618, 202, 76, GREEN, "Volume ratio given", [sub("search q_0 with V(q_0) / V_0 = r:"), "secant steps in a bracket,", "3–6 solves"]))
# inverse
o.append('<rect x="266" y="508" width="474" height="200" rx="12" class="grp"/>')
o.append(text(280, 524, "Inverse  (fit)  with the collapsed CT surface", "h", GREEN["title"]))
o.append(box(280, 540, 220, 76, GREEN, "Distance, both directions", ["simulated → target, along the", "target normal; target → simulated", "(clamped points left out)"]))
o.append(box(514, 540, 212, 76, GREEN, "Sensitivities", [sub("K_t ∂U/∂θ = −∂R/∂θ"), "one linear solve per parameter,", "exact with the smooth wall"]))
o.append(box(280, 630, 220, 64, GREEN, "Bounded least squares", ["trust region (TRF), parameters", "within their bounds"]))
o.append(box(514, 630, 212, 64, GREEN, "Stop", ["tolerances, plateau, target error", "2.5 mm, time budget"]))
o.append(arrow(500, 578, 512, 578))
o.append(path("M620 616 L620 622 L390 622 L390 628"))
o.append(arrow(500, 662, 512, 662))
# loop back: new parameters
o.append(path("M740 662 L764 662 L764 59 L742 59"))
o.append(text(778, 360, "next θ", "l", "#52514e", "middle", ' transform="rotate(-90 778 360)"'))
o.append('<line x1="726" y1="662" x2="740" y2="662" class="a"/>')
# results
o.append(path("M135 708 L135 724 L330 724 L330 738"))
o.append(path("M503 708 L503 724 L430 724 L430 738"))
o.append(box(120, 740, 520, 48, AMBER, "Result", [sub("collapsed surface, E, p_0 = q_0 E, ν, collapse %, ASSD, wall penetration") + "  →  runs/&lt;run_name&gt;/"]))
o.append("</svg>\n")
open("inverse_fem_pressure_fit_detail.svg", "w").write("\n".join(o))
