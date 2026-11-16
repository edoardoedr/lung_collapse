# Methodology

The collapse model: what is modelled, the equations, how it is solved and how its parameters are fitted. Results, the evidence behind the design choices and the problems found are in [analysis/results.md](analysis/results.md); the plan and the to-do list in [NEXT.md](NEXT.md); every config key in [configs/README.md](configs/README.md). The earlier regions-fit methodology is in git (`git show main:METHODOLOGY.md`).

## 1. Goal

Training data for a GNN surrogate of lung collapse in VATS. The model must collapse **any inflated lung** from a few physical parameters, so that the surrogate can be trained on many simulated collapses and tested on the real ones.

1. **Physical model** (this document): elastic lung, loss of the pleural pressure, gravity, chest wall. Parameters: E, ν, density, patient position, the uniform pressure that sets the collapse %.
2. **Calibration on the real patients** (inflated + collapsed CT): fit the parameters, find which can be common to all patients (E) and how the rest follows from the collapse %.
3. **Prediction:** an inflated lung + the collapse % (+ position) → one forward run.
4. **Synthetic dataset + GNN:** geometry + load → displacement.

## 2. What is modelled

**In vivo (reference).** The inflated lung is held open by the negative pleural pressure against its elastic recoil. The pleural pressure is not uniform: it is lower at the top, and its vertical gradient is what carries the lung's weight. A pressure linear in height on a closed surface gives, by the divergence theorem, exactly Archimedes' force,

$$
\oint_{\Gamma} \bar\rho g\, h \; n \, dA = \bar\rho g V \, e_h ,
$$

so the lung "floats" in the pleural pressure field, with $\bar\rho$ its mean density (~250 kg/m³: tissue, blood and air). The measured pleural gradient, ~0.25 cmH₂O/cm ≈ 2.5 Pa/mm, is $\bar\rho g$.

**Pneumothorax.** The pleural space fills with air at uniform (atmospheric) pressure. The lung loses the mean under-pressure, so it shrinks by its recoil, and the gradient, so its weight is no longer carried: it settles on the chest wall. (The buoyancy of the air itself is ~1/200 of the weight: neglected.)

**Model.** The inflated lung is the stress-free reference, clamped at the hilum. The load is the pleural pressure that is removed, and the chest wall is a contact surface:

$$
\Delta p(X) = p_0 + \bar\rho g \left(h(X) - \bar h\right),
\qquad h(X) = -\,e_A \cdot X
$$

- $p_0$: the uniform recoil pressure released; it sets the collapse %. $\bar h$ (area-weighted mean over the free surface) only shifts $p_0$.
- $e_A$: direction of gravity in the reference position (supine: posterior = $+y$ in LPS), so $h$ is the height against gravity. It is taken at the **reference** position $X$: the load is the release of the in-vivo state of each material point.
- **No body force** is added when the collapse happens in the same position: in the reference the weight is already balanced by the pleural gradient. The weight, no longer balanced, is carried by the wall, which is therefore part of the model.
- **Different positions.** If the inflated CT is taken in position A and the collapse happens in position B, the reference loads are the weight along $e_A$ and the pleural gradient along $e_A$; the final ones are the weight along $e_B$. The difference is the surface term above plus a body force

$$
b = \bar\rho g \,(e_B - e_A) \quad \text{per unit reference volume.}
$$

  The resultant of the two is $\bar\rho g V e_B$, the weight along the collapse position, whatever A is. The small motion of the lung while the patient is turned with the chest closed is ignored (the model goes straight from the reference to the collapse).

**Approximations.**

- The reference is taken stress-free, but in vivo the lung is prestressed (recoil, weight). The rigorous version needs the unloaded geometry (one more inverse problem); to first order it is the same.
- Uniform material: there are no CT values of the inflated lung, so no per-element density or air fraction. The Neo-Hookean volumetric term has **no barrier** against crushing: a uniform p/E above K/E = 1.67 has no equilibrium (a barrier at the tissue fraction is a possible extension, NEXT.md).
- Frictionless wall (lubricated pleura); rigid wall.

## 3. Equations

Notation: $q = \Delta p/E$, $X_i$ reference points, $x_i = X_i + u_i$. Non-dimensional with E = 1, P1 tetrahedra, $u = 0$ on the clamped faces. Find $u$ such that for every $\delta u$

$$
\underbrace{\int_{\Omega_0} P : \nabla \delta u \, dV}_{\text{elasticity}}
+ \underbrace{s \int_{\Gamma_f} q(X)\, J F^{-T} N \cdot \delta u \, dA}_{\text{follower pressure}}
- \underbrace{\int_{\Omega_0} \gamma\,(e_B - e_A) \cdot \delta u \, dV}_{\text{body force}}
+ \underbrace{\sum_i a_i\, w(g_i)\, \nabla\tilde\varphi(x_i) \cdot \delta u_i}_{\text{wall}} = 0
$$

$$
q(X) = q_0 + \gamma \left(h(X) - \bar h\right) + \delta q(X),
\qquad q_0 = \frac{p_0}{E},
\qquad \gamma = \frac{\bar\rho g}{E}\ [\text{mm}^{-1}]
$$

Only ratios to E enter: the shape fixes $q_0$ and $\gamma$, i.e. $p_0/E$ and $\bar\rho/E$. With $\bar\rho$ assumed, $\gamma$ gives E. In the code the gravity term is written with the slope across the lung, $s_g = \gamma L$ ($L$ = extent of $h$ over the free surface): $q = q_0 + s_g\,(h - \bar h)/L$, $E = \bar\rho g L / s_g$.

**Material** (compressible Neo-Hookean, as GetFEM's `Compressible_Neo_Hookean`; $I_1 = \operatorname{tr}(F^T F)$, $J = \det F$, $P = \partial W/\partial F$):

$$
W(F) = c_1\left(J^{-2/3} I_1 - 3\right) + d_1 (J - 1)^2,
\qquad c_1 = \frac{\mu}{2} = \frac{1}{4(1+\nu)},
\qquad d_1 = \frac{K}{2} = \frac{1}{6(1-2\nu)}
$$

At ν = 0.40: $c_1 = 0.179$, $d_1 = 0.833$, $K = 1.67$. In hydrostatic compression $p/E = (K/E)(1 - J)$ at any strain. The torch core takes any strain energy $W(F, \nu)$ (`solvers/materials.py`: stress and tangent by automatic differentiation).

**Pressure:** Nanson's formula (current area vector, / 3 per node on P1 faces). The sign $s = \pm 1$ is set by a start-up check so that $q > 0$ pushes inward.

**Body force:** constant, so each tetrahedron gives $b V_t / 4$ to its nodes; no tangent. Zero when $e_A = e_B$.

**Wall** (smooth field, `wall_update: "potential"`). `fem_setup` samples the signed distance to the wall (> 0 outside) on a grid (1.5 mm), smooths it (Gaussian σ = 3 mm) and stores it; $\tilde\varphi$ is its tricubic B-spline interpolation (C²). Per free surface node, with reference nodal area $a_i$ and margin $A_i$ = `tol_mm` + $\langle\tilde\varphi(X_i)\rangle_+$:

$$
g_i = \tilde\varphi(x_i) - A_i,
\qquad w(g) = \frac{k}{2e}\left(\langle g \rangle_+^2 - \langle g - e \rangle_+^2\right),
\qquad \frac{\partial f_i}{\partial x_i} = a_i \left[ w'(g_i)\, \nabla\tilde\varphi\, \nabla\tilde\varphi^T + w(g_i)\, \nabla^2\tilde\varphi \right]
$$

($k = 20$, $e = 0.5$ mm: zero inside, quadratic over $e$, then linear.) The field is evaluated at the current position inside Newton, with the exact tangent (gradient and curvature of the field): no contact rounds, and the Jacobian of section 5 is exact. The default wall is the inflated surface itself, so every point starts on it; the smoothing rounds its corners by a few mm (the points can leave the true wall by ~2.5 mm more than the margin).

The earlier treatment, `wall_update: "outer"`, is kept for comparison and for GetFEM: the wall linearised at a state (normal $n_w$ = interpolated vertex normals at the closest point, gap along it), Newton with it fixed, then re-linearisation, up to `wall_max_updates` rounds until the gap settles (`wall_settle_mm`). Its rounds alternate at the corners of the wall (results, sections 7 and 7b).

**Perturbation $\delta q$** (diagnostic, fit only): free slopes along anatomical axes, $\sum_a s_a \xi_a$ with $\xi_a$ the normalised centroid coordinate (ap + = anterior, si + = superior, ml + = lateral), or one p/E per Ward region of the target displacement (connectivity-constrained clusters; it then replaces $q_0$). The regions come from the patient's own collapse, so they cannot predict: they give the best shape this FEM can reach, and show where the physical load is missing something.

## 4. Forward run (`collapse` step)

All parameters given. One solve along the proportional load ramp (section 6) gives the collapsed lung. The uniform part is given as $q_0$, as $p_0$ in Pa, or as the volume ratio $V_{collapsed}/V_{inflated}$ to reach: the volume decreases monotonically with $q_0$, so $q_0$ is found by secant steps inside a bracket (first guess from the hydrostatic response $1 - V/V_0 = q_0 / (K/E)$), each solve warm-started from the last one; 3–6 solves for 0.5 % in volume.

## 5. Inverse run (`fit` step)

The parameters marked `fit` ($q_0$, E through $s_g$, ν) and the perturbation are fitted to the collapsed surface by bounded least squares (scipy `least_squares`, TRF):

$$
\min_{\theta} \;
\sum_{i \in \text{free}} \|r_i\|^2
+ w_s \frac{n_f}{n_t} \sum_{j=1}^{n_t} \|b_j\|^2
+ \lambda^2 \sum_{(a,b)\ \text{adjacent regions}} (\theta_a - \theta_b)^2
$$

with $n_f$ free points, $n_t$ target points, $w_s = 1$, $\lambda = 1$ (regions only).

- **Fitted → target:** $c_i$ = closest point of $x_i$ on the target surface, $n_i$ the normal there (interpolated vertex normals), both recomputed at every evaluation: $r_i = n_i n_i^T (x_i - c_i)$, the distance along the normal, so sliding along the surface costs nothing. There is no node correspondence between the two surfaces.
- **Target → fitted:** $z_j = \sum_m \beta_{jm} x_{v_m}$ = closest point of target point $y_j$ on the fitted surface (barycentric $\beta$); $b_j = z_j - y_j$, set to 0 when $z_j$ lies on a clamped triangle. Without this term a lung that shrinks onto part of the target scores well.
- Clamped points are left out of the loss and of all reported errors (their error is a constant).

**Reported loss error** (second mean over the non-zero $b_j$):

$$
e = \frac{\operatorname{mean}_i \|r_i\| + w_s \operatorname{mean}_j \|b_j\|}{1 + w_s}
$$

The result is the state of lowest rms (`best_by`: what the optimiser minimises), and the shape error is reported as ASSD (symmetric mean point-to-surface distance).

**Jacobian** (Gauss-Newton: $c_i$, $n_i$, triangle and $\beta$ held fixed). The load is linear in the parameters, $q = \Phi\,\theta + q_c$, body magnitude $= \beta_c + \beta^T\theta$, so with one factorisation of the tangent $K_t$ at equilibrium, on the free dofs:

$$
K_t \frac{\partial U}{\partial q} = -\frac{\partial R}{\partial q},
\qquad \frac{\partial U}{\partial \theta} = \frac{\partial U}{\partial q}\,\Phi + \frac{\partial U}{\partial \text{body}}\,\beta^T,
\qquad \frac{\partial r_i}{\partial \theta} = n_i n_i^T \frac{\partial u_i}{\partial \theta},
\qquad \frac{\partial b_j}{\partial \theta} = \sum_m \beta_{jm} \frac{\partial u_{v_m}}{\partial \theta}
$$

The column of $q_0$ is $\Phi = 1$, of the gravity slope $(h - \bar h)/L$ (plus the body column); ν has its own column ($\partial f_{int}/\partial\nu$).

**Fit loop.**

- Levels: one for the physical parameters; with a gradient perturbation a second one adding the slopes; with regions one per region count, coarse → fine. Each level starts from the best state so far (area-weighted projection of its p/E onto the new parameters).
- Before each solve the core is re-anchored to the state of the lowest cost in the level, so a shrinking trust region means a shrinking load step.
- A failed solve gets twice the level's best residual (a rejected trial). If the first solve of a level fails, the level stops.
- Stops: per level on the optimiser's tolerances or a plateau (`level_patience`, `level_min_improve`); per fit on `target_error_mm`, `time_budget_min`, `fit_patience` or the end of the levels. The best state is checkpointed at every improvement; a watchdog kills a hung run.

## 6. Nonlinear solve

- **Newton:** replica of GetFEM's (criterion $\min(\|R\|_1, \|\Delta U\|_1 / \|U\|_1)$, "simplest" line search).
- **Load path:** every solve ramps the whole load (face pressures and body force) proportionally from the last converged state, or from rest; `adaptive_steps` halves the increment on failure and doubles it on success. This is the physical path: the lung collapses as the pressure is released. There is no continuation in the gravity term (it was needed only with the outer wall rounds).
- **Time limit:** `max_solve_s` (120 s): a hopeless solve fails fast and the fit treats it as a rejected trial.
- A Newton trial that leaves the wall grid is rejected like an inverted element.

## 7. Code and verification

| File (`pipeline/collapse/`) | Role |
|---|---|
| `setup.py` | clamp, wall (surface + smooth field), optional Ward regions, Gmsh mesh → `problem.npz` |
| `load.py` | the load as a function of the parameters: levels, q per triangle, body force, chain rule, physical units |
| `forward.py`, `fit.py` | the forward and the inverse run; `results.py` writes them; `sequence.py` the collapse sequence |
| `wall_field.py` | smoothed signed-distance grid and its tricubic B-spline (value, gradient, Hessian) |
| `solvers/base.py` | `ForwardSolver` interface (`set_regions`, `set_body`, `solve`, `jacobian`, state); the runs talk only to it |
| `solvers/nodal.py` | everything except assembly for the P1 cores: Newton, load path, wall, body force, Jacobian, linear solvers (cuDSS, pardiso, scipy) |
| `solvers/torch_solver.py` | PyTorch assembly with autodiff; `TorchFEM` is differentiable on any device (physics loss for the GNN) |
| `solvers/warp_solver.py` | Warp kernels, hand-derived stress and tangent |
| `solvers/getfem_solver.py` | reference core for validation (MUMPS): no smooth-field wall, no body force |
| `solvers/materials.py` | materials as W(F, ν): a new material is a new function |

Verification (numbers in analysis/results.md):

- Torch / Warp vs GetFEM (karl04): relative difference ≤ 2e-14 without wall, ≤ 1e-11 with the `outer` wall; Jacobian vs finite differences ~1e-8.
- Smooth-field wall: tangent vs finite differences 1e-7; Jacobian with contact vs finite differences 9e-7 on the server (torch), 3e-8 with the numpy test core.
- Body force: resultant = weight along the collapse position; Jacobian column vs finite differences 1e-7 (numpy test core).
- The rewritten fit reproduces the previous code: karl04, E 1.7 kPa, q0 fitted: ASSD 9.00 mm, p0 1283 Pa, 11 solves with the numpy test core on the Mac, as the torch fit on the server (9.00 mm, 1282 Pa, 11 solves).
- A fit of (q0, E) on a synthetic target made by a forward run finds the parameters back (q0 to 1e-4, E within 0.1 %).
- `tests/` runs these checks on the Mac (numpy only).
