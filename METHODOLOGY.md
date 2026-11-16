# Methodology

How the collapsed-lung fits are made: pipeline, physical model, equations, pressure models, inverse problem and contact treatment. The results, the evidence behind the design choices and the problems found are in [analysis/results.md](analysis/results.md); the plan and the to-do list in [NEXT.md](NEXT.md); every config key in [configs/README.md](configs/README.md).

## 1. Goal

Training data for a GNN surrogate of lung collapse in VATS. The starting problem is inverse: given the inflated lung (reference) and the collapsed lung (target) from CT, find the regional pleural pressures that collapse the reference onto the target in a FEM model. Plan:

1. **Regions fit** on the real patients: shows that the model reproduces real collapses and gives their pressures.
2. **Common rules** (gradient fit, pressure–collapse rule): a few numbers that transfer to a new lung.
3. **Collapse of an inflated-only lung:** choose the collapse %, find the mean p/E that gives it, one forward solve.
4. **Synthetic dataset + GNN:** geometry + pressure field → displacement. The GNN replaces the FEM solve; it does not find the patient's pressure by itself.

## 2. Pipeline

`check_inputs` → `surface_mesh` → `hilum` → `registration` → `fem_setup` → `fem_fit`. All surfaces are in LPS.

| Step | What it does |
|---|---|
| `check_inputs` | integrity, containment, notches and volume ratio of the segmentations |
| `surface_mesh` | clean-up and ACVD remesh of the collapsed surface, ~480 nodes |
| `hilum` | largest intersection ring of airways, arteries and veins with the lung surface; hilum = centroid of the three ring centres |
| `registration` | `method: "none"`: the inflated surface is only remeshed with the same settings; reference and target are matched by closest point |
| `fem_setup` | clamp, pressure regions, optional wall, Gmsh tetrahedra (14 mm) → `problem.npz` |
| `fem_fit` | inverse problem (section 5) |

**Current configuration** (`configs/<p>_torch.json`, `<p>_wall_torch.json`):

| Item | Choice |
|---|---|
| Registration / target alignment | none / none |
| Clamp | hilum region only (union of the four anchor balls), `clamp_still_mm` 0 |
| Material | compressible Neo-Hookean, ν = 0.40 fixed |
| Unknowns | q = p/E, bounds [−1, 3]; `E_Pa` = 3000 only converts to Pa |
| Loss | closest-point `plane`, symmetric term weight 1 |
| Levels | K = 1, 4, 8, 14, 25, 40 regions |
| Core | torch, analytic Jacobian |
| Wall (`_wall` configs) | the inflated reference surface, allowance 5 mm, `outer` rounds, `nodal` quadrature |

Why these choices (evidence in results, section 7):

- **No registration:** same shape error, less rotation; mask registration gave a wrong reference volume on some patients.
- **No alignment, `plane` loss:** mask-based correspondence slides tangentially along the surface; the normal residual ignores that slip, and the rigid alignment made the fit chase it.
- **Hilum-only clamp:** without registration, "still" points counted every point that slid onto the target as fixed.
- **Symmetric loss:** with the one-sided term the lung shrank onto part of the target.

## 3. FEM model

Notation: $q = p/E$, $X_i$ reference points, $x_i = X_i + u_i$ fitted points, $\theta$ fit parameters. Non-dimensional with E = 1, P1 tetrahedra, u = 0 on the clamped faces.

**Material:**

$$
W(F) = c_1\left(J^{-2/3} I_1 - 3\right) + d_1 (J - 1)^2,
\qquad c_1 = \frac{\mu}{2} = \frac{1}{4(1+\nu)},
\qquad d_1 = \frac{K}{2} = \frac{1}{6(1-2\nu)}
$$

Compressible Neo-Hookean as in GetFEM (`Compressible_Neo_Hookean`), $I_1 = \operatorname{tr}(F^T F)$, $J = \det F$, $P = \partial W / \partial F$. At ν = 0.40: $c_1 = 0.179$, $d_1 = 0.833$, $K = 1.67$. The first term is isochoric (shape change only); the volumetric term $(J-1)^2$ is one of several possible choices (e.g. $\ln^2 J$).

In hydrostatic compression ($F = \lambda I$) the isochoric term vanishes and, at any strain,

$$
\frac{p}{E} = \frac{K}{E}(1 - J)
$$

so the material has **no barrier** against crushing ($W$ stays finite as $J \to 0$) and a uniform pressure above $K/E$ has no equilibrium.

**Equilibrium:** for given $q$, find $u$ ($u = 0$ on the clamp) such that for every $\delta u$

$$
\underbrace{\int_{\Omega_0} P : \nabla \delta u \, dV}_{\text{elasticity}}
+ \underbrace{s \sum_k q_k \int_{\Gamma_k} J F^{-T} N \cdot \delta u \, dA}_{\text{follower pressure}}
+ \underbrace{\int_{\Gamma_f} w(g) \, n_w \cdot \delta u \, dA}_{\text{wall}}
+ \underbrace{\sum_i k_s a_i \left[(I - n_i n_i^T) u_i\right] \cdot \delta u_i}_{\text{tangential support}} = 0
$$

- **Pressure:** Nanson's formula (current area vector, / 3 per node on P1 faces). The sign $s = \pm 1$ is set by a sign check so that $q > 0$ pushes inward.
- **Wall** (only with a wall), penalty with $\langle\cdot\rangle_+$ the positive part:

$$
w(g) = \frac{k}{2e}\left(\langle g \rangle_+^2 - \langle g - e \rangle_+^2\right),
\qquad g = u \cdot n_w - G,
\qquad G = n_w \cdot U_s - \phi + a
$$

  $\phi$ = signed distance to the wall at the linearisation state $U_s$, $a$ = allowance (`wall_tol_mm` plus how far the point already lies outside in the reference), $n_w$ = wall normal (area-weighted vertex normals interpolated at the closest point). $k = 20$, $e = 0.5$ mm.
- **Tangential support** (`support_k` = $k_s$, default 0, nodal cores only): springs on the surface nodes, $a_i$ and $n_i$ the reference nodal area and normal. A smooth, path-independent substitute for friction against the near-mechanism of a small clamp.

## 4. Pressure models

On a free surface triangle $t$, $q = \Phi\,\theta$:

$$
\text{regions: } q_t = \theta_{k(t)}
\qquad\qquad
\text{gradient: } q_t = \theta_0 + \sum_{a} \theta_a \, \xi_a(t),
\qquad \xi_a(t) = \frac{c_a(t) - \bar c_a}{\max c_a - \min c_a}
$$

- **Regions** (`pressure_model: "regions"`): connectivity-constrained Ward clusters of the target displacement with the rigid part removed, for each level K. They come from the patient's own target, so they do not exist for a new lung.
- **Gradient** (`"gradient"`): $c_a(t)$ = centroid coordinate along axis $a$ (ap + = anterior, si + = superior, ml + = lateral; patient supine), $\bar c_a$ its area-weighted mean over the free triangles. A slope $\theta_a$ is the p/E difference between the two ends of the lung. Levels: uniform, then mean + slopes. Each free triangle is one core region, so no core changed.

## 5. Inverse problem

Bounded least squares (scipy `least_squares`, TRF; bounds $-1 \le \theta \le 3$ on regions and on the gradient mean, $|\theta_a| \le 4$ on slopes):

$$
\min_{\theta} \;
\sum_{i \in \text{free}} \|r_i\|^2
+ w_s \frac{n_f}{n_t} \sum_{j=1}^{n_t} \|b_j\|^2
+ \lambda^2 \sum_{(a,b)\ \text{adjacent}} (\theta_a - \theta_b)^2
$$

with $n_f$ free points, $n_t$ target points, $w_s = 1$, $\lambda = 1$ (regions only).

- **Fitted → target:** $c_i$ = closest point of $x_i$ on the target, $n_i$ the normal there, both recomputed at every evaluation; $w = 0$ in the configs:

$$
r_i = \left[n_i n_i^T + w\,(I - n_i n_i^T)\right](x_i - c_i)
$$

- **Target → fitted:** $z_j = \sum_m \beta_{jm} x_{v_m}$ = closest point of target point $y_j$ on the fitted surface (barycentric $\beta$); $b_j = z_j - y_j$, set to 0 when $z_j$ lies on a clamped triangle.
- Clamped points are left out of the loss and of all reported errors (their error is a constant).

**Reported error** (not the square root of the cost; second mean over the non-zero $b_j$):

$$
e = \frac{\operatorname{mean}_i \|r_i\| + w_s \operatorname{mean}_j \|b_j\|}{1 + w_s}
$$

**Jacobian** (Gauss-Newton: $c_i$, $n_i$, triangle and $\beta$ held fixed), one factorisation of the tangent $K_t$ at equilibrium, on the free dofs:

$$
K_t \frac{\partial U}{\partial q} = -\frac{\partial R}{\partial q},
\qquad \frac{\partial U}{\partial \theta} = \frac{\partial U}{\partial q}\,\Phi,
\qquad \frac{\partial r_i}{\partial \theta} = \left[n_i n_i^T + w\,(I - n_i n_i^T)\right]\frac{\partial u_i}{\partial \theta},
\qquad \frac{\partial b_j}{\partial \theta} = \sum_m \beta_{jm} \frac{\partial u_{v_m}}{\partial \theta}
$$

**Fit loop:**

- Levels coarse → fine, each warm-started from the best state so far (area-weighted projection of its p/E onto the new model).
- Before each solve the core is re-anchored to the state of the lowest cost in the level, so a shrinking trust region means a shrinking load step.
- A failed solve gets twice the level's best residual (a rejected trial). If the first solve of a level fails, the level stops and the fit moves on.
- Stops: per level on the lsq tolerances or a plateau (`level_patience`, `level_min_improve`); per fit on `target_error_mm` (2.5), `time_budget_min`, `fit_patience` or the end of the levels. The best state is checkpointed at every improvement.
- $p = q\,E$ only for output: the fit identifies p/E, so E does not change the shape.

## 6. Nonlinear solve and contact

- **Newton:** replica of GetFEM's (criterion $\min(\|R\|_1, \|\Delta U\|_1 / \|U\|_1)$, "simplest" line search). Load path on the face pressures from the last converged state; `adaptive_steps` halves the increment on failure and doubles it on success; `warm_substeps` retries in sub-steps before ramping from zero.
- **Wall, `outer` rounds:** Newton with $n_w$ and $G$ fixed, then re-linearisation, up to `wall_max_updates` (15) rounds until the gap error (previous linearised gap vs true gap, over points with gap > −1 mm) is below `wall_settle_mm` (0.5). A solve that does not settle fails (`unsettled`). A settled state keeps the linearisation it is in equilibrium with (also used by the Jacobian), remembered for the last 256 states, so a warm start from it begins in equilibrium.
- **Wall quadrature `nodal`:** vertex rule (weights 1/3) instead of the IM_TRIANGLE(3) face rule, whose negative centre weight makes the contact tangent indefinite.
- **Time limit:** `max_solve_s` (120 s) with `slow_ramp: false`, so a hopeless solve fails fast.
- `wall_update: "newton"` (re-linearise at every residual) exists for torch / Warp but is not used: it oscillates.
- **Wall as a smooth field, `wall_update: "potential"`** (torch / Warp, 2026-11-16; being validated on the server). `fem_setup` samples the signed distance to the wall on a grid (`wall_grid_mm` 1.5), smooths it (Gaussian σ = `wall_smoothing_mm` 3) and stores it; $\tilde\varphi$ is its tricubic B-spline interpolation (C²). Per free surface node, with reference nodal area $a_i$ and margin $A_i$ = `wall_tol_mm` + $\langle\tilde\varphi(X_i)\rangle_+$:

$$
g_i = \tilde\varphi(x_i) - A_i,
\qquad f_i = a_i\, w(g_i)\, \nabla\tilde\varphi(x_i),
\qquad \frac{\partial f_i}{\partial x_i} = a_i \left[ w'(g_i)\, \nabla\tilde\varphi\, \nabla\tilde\varphi^T + w(g_i)\, \nabla^2\tilde\varphi \right]
$$

  with the same force law $w$ as above (same $k$, $e$), evaluated at the current position inside Newton. No rounds and no linearisation to keep; the tangent is exact, so the Jacobian of section 5 is exact with a wall. The corners of the wall are rounded by a few mm.

## 7. Code and verification

| File (`pipeline/collapse/solvers/`) | Role |
|---|---|
| `base.py` | `ForwardSolver` interface (`set_regions`, `solve`, `jacobian`, state); `fit.py` talks only to it |
| `getfem_solver.py` | reference core (MUMPS, P1/P2); wall only `outer`, no `support_k` |
| `nodal.py` | everything except assembly for P1 cores: Newton, load path, wall rounds, support, Jacobian, linear solvers (cuDSS, pardiso, scipy) |
| `warp_solver.py` | Warp kernels, hand-derived stress and tangent |
| `torch_solver.py` | PyTorch assembly with autodiff; `TorchFEM` is differentiable on any device (physics loss for the GNN) |
| `materials.py` | materials as W(F, ν): a new material is a new function |

Verification (karl04):

- Torch / Warp vs GetFEM: relative difference ≤ 2e-14 without wall, ≤ 1e-11 with wall (`outer`), same penetrations.
- Jacobian vs finite differences ~1e-8; symmetric loss gradient vs FD 2e-10.
- Forward solve: GetFEM 70–107 s, Warp + cuDSS 3.2 s, torch + cuDSS 5.7 s. Full no-wall fit: GetFEM 111 min (finite differences) / 7.9 min (analytic), torch < 1 min.

## 8. Physical load model (proposed 2026-11-12, partly implemented)

Replaces the regional pressures with the physics of the pneumothorax. Status and tests in NEXT.md. Implemented so far (2026-11-16): the fixed-gravity gradient, `fem_fit.gravity_E_Pa` / `tissue_density` (supine, e_g = +y): q = q0 + γ (h − h̄) with γ = ρ̄ g / E given, only q0 (and any non-`ap` slope) fitted, for scans over a common E. Not yet: E or γ as a fitted / shared parameter, other e_g, the barrier material.

**Load.** In vivo the pleural pressure is not uniform: it is lower at the top, and its vertical gradient carries the lung's weight. A pressure linear in height on a closed surface gives, by the divergence theorem, exactly Archimedes' force:

$$
\oint_{\Gamma} \bar\rho g\, h \; n \, dA = \bar\rho g V \, e_h
$$

so the inflated lung "floats" in the pleural pressure field, with $\bar\rho$ its mean density (~250 kg/m³). In the pneumothorax the pleural space fills with air at uniform pressure (the air's own buoyancy is ~1/200 of the lung's weight, negligible). From the stress-free inflated reference the load is therefore the removed pleural pressure:

$$
\Delta p(X) = p_{atm} - p_{pl}(X) = p_0 + \bar\rho g \left(h(X) - \bar h\right),
\qquad h(X) = -\,e_g \cdot X
$$

- $p_0$: uniform elastic recoil (transpulmonary pressure released); $\bar h$ only shifts $p_0$.
- $e_g$: direction of gravity; supine = posterior = $+y$ in LPS, so $h$ grows anteriorly. Lateral decubitus (the VATS position) only changes $e_g$.
- $h$ is taken at the **reference** position $X$: the load is the release of the in-vivo state at each material point.
- **No body force** is added: in the reference the weight is already balanced by the pleural gradient, and adding $\rho g$ in the volume would count it twice. The weight, no longer balanced, is carried by the wall: the **wall is part of the model**, not an option.
- With a heterogeneous density only the mean $\bar\rho = M / V$ enters the gradient (it is what balances the weight); the local density enters through the material (below).

**Equilibrium.** The pressure term of section 3 becomes, in non-dimensional form ($E = 1$):

$$
s \int_{\Gamma_f} q(X)\, J F^{-T} N \cdot \delta u \, dA,
\qquad q(X) = q_0 + \gamma \left(h(X) - \bar h\right),
\qquad q_0 = \frac{p_0}{E}, \quad \gamma = \frac{\bar\rho g}{E}\ [\text{mm}^{-1}]
$$

with the elasticity, wall and support terms unchanged. It is the ap gradient model of section 4 with a physical slope: $\theta_{ap} = \gamma L = \bar\rho g L / E$ ($L$ = extent along $e_g$), so the fitted slope gives $E = \bar\rho g L / \theta_{ap}$. Check: $\bar\rho g$ = 2.45 Pa/mm, with E = 3 kPa and L = 150 mm → $\theta_{ap}$ = 0.12, the order of the fitted slopes. Unknowns per patient: $(q_0, \gamma)$, i.e. $(p_0/E, E)$; $\gamma \ge 0$. If the theory holds, E is similar across patients, and a common E leaves one parameter per patient ($q_0$, set by the collapse %).

**Heterogeneous tissue** (needs the HU of the inflated CT, which we do not have: without them φ_t is uniform or an assumed function of height, bounded by φ_t ≤ V_collapsed / V_inflated, 0.14 for patient_2). Tissue fraction per element $\phi_t(X) \approx 1 + \mathrm{HU}/1000$ (air −1000, tissue ~0); the supine dorsal lung is denser. An element cannot lose more than its air: $J \ge \phi_t$. Candidate volumetric term with a barrier at $J = \phi_t$, stress-free at $J = 1$:

$$
W_{vol}(J) = d_1 (J - 1)^2 + \kappa \left[\frac{1 - \phi_t}{J - \phi_t} + \ln\frac{J - \phi_t}{1 - \phi_t} - 1\right]
$$

($W_{vol}(1) = W_{vol}'(1) = 0$, $W_{vol} \to \infty$ as $J \to \phi_t$, convex for $J < 2 - \phi_t$; it adds $\kappa / (1 - \phi_t)^2$ to the bulk modulus). It removes the crushed elements and makes the air-rich anterior lung able to lose more volume. Torch core only (`materials.py`, per-element parameter). A local stiffness $E(\phi_t)$ is possible but less founded. Mass check: $V_{collapsed} \ge \int \phi_t \, dV$.

**Wall in the physical model.** The wall carries the weight, so it is part of the model. Expected effect on the convergence: the near-mechanism mostly goes (smooth load, resultant = the weight, vertical, pressing the lung into the concave dorsal wall: a stable minimum), the pressures at the bound and the crushed elements too; the corner points of the wall (analysis/results.md section 7) become central instead, since in a supine patient the paravertebral gutter is the lowest point. Two changes, in this order:

*(a) Wall as a smooth potential (numerical; implemented 2026-11-16 as `wall_update: "potential"`, section 6, with the force law of section 3 instead of the plain quadratic below).* The signed distance to the cavity is computed once on a grid and smoothed, so corners become rounded with a radius of a few mm; $\tilde\varphi$ is its interpolation, at least C² (cubic B-spline; trilinear has a discontinuous gradient). The contact energy is evaluated at the current position inside Newton:

$$
E_{wall} = \sum_i a_i\, \frac{k}{2}\,\langle \tilde\varphi(x_i) - a_i^{allow} \rangle_+^2,
\qquad
\frac{\partial^2 E_{wall}}{\partial x_i^2} = a_i k \left(\nabla\tilde\varphi\,\nabla\tilde\varphi^T + (\tilde\varphi - a_i^{allow})\,\nabla^2\tilde\varphi\right) \text{ where } \tilde\varphi > a_i^{allow}
$$

with the exact Hessian (torch autodiff). No outer rounds: contact is part of one minimisation. The current `wall_update: "newton"` oscillated because it missed the curvature term $\nabla^2\tilde\varphi$ and its normal jumped; a smooth field has neither problem.

*(b) Deformable wall (physical, later).* The cavity is not rigid everywhere: the ribs nearly are, the diaphragm and the mediastinum move in a pneumothorax. A physical stiffness per part of the wall (high on the ribs, low on the diaphragm and the mediastinum, told apart by position and normal on the inflated surface: medial = mediastinum, inferior = diaphragm) makes the penetration the deformation of those organs instead of a numerical error, and a softer wall also converges more easily. Cost: 1–2 more parameters.

Rejected: a bilateral sliding constraint on the dorsal points (it needs the contact set, i.e. the target, so it cannot predict); friction (lubricated pleura). With gravity the tangential support `support_k` is probably no longer needed: it was an artificial substitute.
