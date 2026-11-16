"""The load of the collapse model, as a function of the physical parameters.

p / E on a free surface triangle t (METHODOLOGY.md):

    q_t = q0 + s_g hn_t + sum_a s_a xi_a(t) + delta_k(t)

  q0      uniform part: the released recoil pressure / E (sets the collapse %)
  s_g hn  gravity: the pleural gradient that carried the lung's weight in the reference position,
          hn = (h - h_mean) / L with h = -e_A . X the height against gravity of the triangle
          centroid in the reference, L its extent over the free triangles, and the slope
          s_g = gamma L = rho g L / E (p/E difference between the top and the bottom of the lung)
  s_a xi  optional free slopes along anatomical axes (diagnostic), xi normalised like hn
  delta_k optional p/E per Ward region (diagnostic); it then replaces q0

and, when the collapse position e_B differs from the reference one e_A, the body force
gamma (e_B - e_A) per unit reference volume = s_g (e_B - e_A) / L: the solver gets the direction
(e_B - e_A) / L once (body_direction) and the magnitude s_g at every solve.

A Level is one parametrisation x -> (q per triangle, body magnitude, nu): the parameters marked
"fit" in the config, plus the perturbation. The fit runs the levels in order (coarse -> fine).
Everything is linear in x, so the Jacobian of the surface displacement is the core's dU/d(q, nu,
body) times constant matrices (Level.chain).
"""

import numpy as np

from .geometry import face_adjacency, tri_geometry
from .regions import region_pairs

RHO_G = 9.81e-3            # g in Pa per mm of height per kg/m3 of density


def axis_coords(C, free, area, hilum):
    """Normalised coordinate of each triangle centroid C along the anatomical axes (LPS): ap + =
    anterior (-y), si + = superior (+z), ml + = lateral (away from the hilum in x); centred on the
    area-weighted mean of the free triangles and divided by their extent."""
    side = np.sign(C[free, 0].mean() - hilum[0]) if hilum is not None else np.nan
    raw = dict(ap=-C[:, 1], si=C[:, 2], ml=side * C[:, 0])
    out = {}
    for a, v in raw.items():
        if np.all(np.isfinite(v)):
            out[a] = (v - np.average(v[free], weights=area[free])) / np.ptp(v[free])
    return out


class Level:
    """x (P,) -> q per triangle (T,), body magnitude, nu. Phi (T, P) maps x to q (zero columns for
    nu), const (T,) is the fixed part; the core sees `labels` / `n_solver` regions with pressures
    M x + const_solver."""

    def __init__(self, names, Phi, const, labels, lb, ub, body_const=0.0, body_coef=None, nu=0.4,
                 nu_index=None, pairs=None, reg=0.0, kind="physical"):
        self.names, self.Phi, self.const, self.labels = list(names), Phi, const, labels
        self.P = len(self.names)
        self.lb, self.ub = np.asarray(lb, float), np.asarray(ub, float)
        self.body_const = float(body_const)
        self.body_coef = np.zeros(self.P) if body_coef is None else np.asarray(body_coef, float)
        self.nu_const, self.nu_index = float(nu), nu_index
        self.pairs = np.zeros((0, 2), dtype=int) if pairs is None else pairs
        self.reg, self.kind = float(reg), kind
        act = labels >= 0
        self.n_solver = int(labels.max()) + 1 if act.any() else 0
        first = np.zeros(self.n_solver, dtype=int)             # one triangle of each core region
        first[labels[act]] = np.nonzero(act)[0]
        self.M, self.const_solver = Phi[first], const[first]
        if not np.allclose(Phi[act], self.M[labels[act]]) or not np.allclose(const[act], self.const_solver[labels[act]]):
            raise ValueError("load level: the core regions do not resolve the pressure field")

    def q_tri(self, x):
        """p/E per triangle, NaN where no pressure (clamped)."""
        return np.where(self.labels >= 0, self.Phi @ x + self.const, np.nan)

    def q_solver(self, x):
        return self.M @ x + self.const_solver

    def body(self, x):
        return self.body_const + float(self.body_coef @ x)

    def nu(self, x):
        return self.nu_const if self.nu_index is None else float(x[self.nu_index])

    def reg_residual(self, x):
        return self.reg * (x[self.pairs[:, 0]] - x[self.pairs[:, 1]])

    def reg_jacobian(self):
        R = np.zeros((len(self.pairs), self.P))
        R[np.arange(len(self.pairs)), self.pairs[:, 0]] = self.reg
        R[np.arange(len(self.pairs)), self.pairs[:, 1]] = -self.reg
        return R

    def chain(self, Jq, Jnu=None, Jbody=None):
        """dU/dx (rows, P) from the core's dU/dq_solver (rows, n_solver; for +q, i.e. the sign
        convention already applied), dU/dnu and dU/dbody (rows,)."""
        J = Jq @ self.M
        if Jbody is not None:
            J = J + np.outer(Jbody, self.body_coef)
        if self.nu_index is not None:
            J[:, self.nu_index] = Jnu
        return J

    def x0_from(self, q_tri, nu, area):
        """Start of the level: area-weighted least-squares projection of a p/E per triangle (the
        best state so far) onto the pressure parameters, nu carried over; clipped into the bounds."""
        m = (self.labels >= 0) & np.isfinite(q_tri)
        cols = [j for j in range(self.P) if j != self.nu_index]
        x = np.zeros(self.P)
        if cols:
            w = np.sqrt(area[m])
            x[cols] = np.linalg.lstsq(self.Phi[m][:, cols] * w[:, None], (q_tri[m] - self.const[m]) * w, rcond=None)[0]
        if self.nu_index is not None:
            x[self.nu_index] = nu
        return np.clip(x, self.lb + 1e-6, self.ub - 1e-6)


class LoadModel:
    """Builds the levels of a physics config on a problem. hilum: (3,) LPS position of the hilum
    (only for the "ml" axis)."""

    def __init__(self, physics, problem, hilum=None):
        self.ph = ph = physics
        self.tris, self.clamped = problem.tris, np.asarray(problem.clamped_tri, bool)
        self.free = ~self.clamped
        X = problem.reference
        self.C, _, self.area = tri_geometry(X, self.tris)
        T = len(self.tris)
        self.per_tri = np.full(T, -1)
        self.per_tri[self.free] = np.arange(self.free.sum())
        self.one_region = np.where(self.free, 0, -1)
        self.problem = problem

        # gravity
        self.hn, self.extent, self.body_direction = np.zeros(T), None, np.zeros(3)
        self.rho_g = ph.tissue_density * RHO_G                    # Pa per mm
        if ph.gravity is not None:
            eA, eB = np.array(ph.gravity["reference"]), np.array(ph.gravity["collapse"])
            h = -self.C @ eA
            h = h - np.average(h[self.free], weights=self.area[self.free])
            self.extent = float(np.ptp(h[self.free]))
            self.hn = np.where(self.free, h / self.extent, 0.0)
            self.body_direction = (eB - eA) / self.extent
        self.has_body = bool(np.any(self.body_direction != 0.0))

        # free slopes
        self.xi = {}
        if ph.gradient_axes:
            coords = axis_coords(self.C, self.free, self.area, hilum)
            missing = [a for a in ph.gradient_axes if a not in coords]
            if missing:
                raise ValueError("gradient axis %s needs the hilum position (setup_summary.json)" % missing)
            self.xi = {a: np.where(self.free, coords[a], 0.0) for a in ph.gradient_axes}

    # ---- physical quantities ----
    def slope_of_E(self, E):
        """Gravity slope s_g = rho g L / E (0 without gravity)."""
        return 0.0 if self.extent is None else self.rho_g * self.extent / E

    def E_of_slope(self, s):
        return self.rho_g * self.extent / s

    def start_q_tri(self):
        """p/E per triangle the fit starts from: the config's q0 and E."""
        q0 = self.ph.q0.value if self.ph.q0 is not None else 0.5
        return np.where(self.free, q0 + self.slope_of_E(self.ph.E_Pa.value) * self.hn, np.nan)

    # ---- levels ----
    def _level(self, q0_free, slopes=(), regions=None, nu_free=None, q0_value=None, fit=None):
        """q0_free: q0 is a parameter; q0_value: fixed q0 otherwise; regions: (K, labels) or None;
        fit: FitConfig (slope bound)."""
        ph, T = self.ph, len(self.tris)
        names, cols, lb, ub, body_coef = [], [], [], [], []
        const = np.zeros(T)
        varying = ph.gravity is not None or bool(slopes)
        pairs = None
        if regions is not None:
            K, lab = regions
            for k in range(K):
                names.append("region%d" % k)
                cols.append((lab == k).astype(float))
            b = ph.q0.bounds if ph.q0 is not None else (-1.0, 3.0)
            lb += [b[0]] * K
            ub += [b[1]] * K
            body_coef += [0.0] * K
            pairs = region_pairs(lab, face_adjacency(self.tris)[0])
        elif q0_free:
            names.append("q0")
            cols.append(self.free.astype(float))
            lb.append(ph.q0.bounds[0])
            ub.append(ph.q0.bounds[1])
            body_coef.append(0.0)
        else:
            const = const + np.where(self.free, q0_value, 0.0)
        body_const = 0.0
        if ph.gravity is not None:
            if ph.E_Pa.fit:
                names.append("gravity_slope")
                cols.append(self.hn)
                lb.append(self.slope_of_E(ph.E_Pa.bounds[1]))
                ub.append(self.slope_of_E(ph.E_Pa.bounds[0]))
                body_coef.append(1.0 if self.has_body else 0.0)
            else:
                s = self.slope_of_E(ph.E_Pa.value)
                const = const + s * self.hn
                body_const = s if self.has_body else 0.0
        for a in slopes:
            names.append("slope_" + a)
            cols.append(self.xi[a])
            bound = fit.slope_bound if fit is not None else 4.0
            lb.append(-bound)
            ub.append(bound)
            body_coef.append(0.0)
        nu_index = None
        if nu_free if nu_free is not None else ph.nu.fit:
            nu_index = len(names)
            names.append("nu")
            cols.append(np.zeros(T))
            lb.append(ph.nu.bounds[0])
            ub.append(ph.nu.bounds[1])
            body_coef.append(0.0)
        Phi = np.column_stack(cols) if cols else np.zeros((T, 0))
        Phi[self.clamped] = 0.0
        const = np.where(self.free, const, 0.0)
        if varying:
            labels = self.per_tri
        elif regions is not None:
            labels = np.where(self.free, regions[1], -1)
        else:
            labels = self.one_region
        return Level(names, Phi, const, labels, lb, ub, body_const, body_coef, ph.nu.value, nu_index, pairs,
                     ph.regions_reg if regions is not None else 0.0,
                     "regions" if regions is not None else "physical")

    def fit_levels(self, fit):
        """The levels of the fit step, coarse -> fine."""
        ph = self.ph
        if ph.q0 is None:
            raise ValueError("the fit needs physics.collapse.q0 (a number, or a fit spec), not p0_Pa / volume_ratio")
        if ph.regions:
            pr = self.problem
            have = pr.levels.tolist()
            missing = sorted(set(ph.regions) - set(have))
            if missing:
                raise ValueError("physics.perturbation.regions %s not among the fem_setup regions %s (set "
                                 "fem_setup.regions and rerun fem_setup)" % (missing, have))
            return [self._level(False, regions=(int(K), lab), fit=fit)
                    for K, lab in zip(pr.levels, pr.regions) if int(K) in ph.regions]
        base = self._level(ph.q0.fit, q0_value=ph.q0.value, fit=fit)
        levels = [base] if base.P else []
        if ph.gradient_axes:
            levels.append(self._level(ph.q0.fit, slopes=ph.gradient_axes, q0_value=ph.q0.value, fit=fit))
        if not levels:
            raise ValueError("nothing to fit: mark a physics parameter with {\"fit\": true, ...} (collapse.q0, "
                             "E_Pa, material.nu) or add a perturbation")
        return levels

    def forward_level(self):
        """One level with q0 as its only parameter and everything else fixed: the forward run
        evaluates it at the given q0, or searches q0 for a volume ratio."""
        ph = self.ph
        if ph.E_Pa.fit or ph.nu.fit or (ph.q0 is not None and ph.q0.fit) or ph.gradient_axes or ph.regions:
            raise ValueError("the collapse (forward) step needs every physics parameter given: no \"fit\" and no "
                             "perturbation (use the fit step)")
        lv = self._level(True, nu_free=False) if ph.q0 is not None else None
        if lv is None:                                           # p0_Pa / volume_ratio: q0 has no bounds here
            names, Phi = ["q0"], self.free.astype(float)[:, None]
            s = self.slope_of_E(ph.E_Pa.value)
            lv = Level(names, Phi, s * self.hn, self.per_tri if ph.gravity is not None else self.one_region,
                       [-np.inf], [np.inf], s if self.has_body else 0.0, None, ph.nu.value)
        return lv

    # ---- report ----
    def describe(self, level, x):
        """The physical parameters of state x of a level, in physical units, and which were fitted."""
        ph = self.ph
        vals = dict(zip(level.names, (float(v) for v in x)))
        q = level.q_tri(x)[self.free]
        a = self.area[self.free]
        E = ph.E_Pa.value
        if "gravity_slope" in vals:
            E = self.E_of_slope(vals["gravity_slope"])
        q0 = vals.get("q0", float(np.average(q, weights=a)))
        out = dict(E_Pa=float(E), q0=q0, p0_Pa=q0 * E, nu=level.nu(x), tissue_density_kg_m3=ph.tissue_density,
                   fitted=[n for n in level.names if not n.startswith("region")] + (["regions"] if level.kind == "regions" else []),
                   pE_min=float(q.min()), pE_max=float(q.max()), material=ph.material)
        if ph.gravity is not None:
            s = vals.get("gravity_slope", self.slope_of_E(ph.E_Pa.value))
            out["gravity"] = dict(reference=list(ph.gravity["reference"]), collapse=list(ph.gravity["collapse"]),
                                  positions=list(ph.gravity["names"]), extent_mm=self.extent,
                                  gamma_pE_per_mm=s / self.extent, slope_pE=s, rho_g_Pa_per_mm=self.rho_g,
                                  body_force=self.has_body)
        else:
            out["gravity"] = None
        slopes = {n[6:]: v for n, v in vals.items() if n.startswith("slope_")}
        if slopes:
            out["slopes_pE"] = slopes
        if level.kind == "regions":
            out["regions"] = int(sum(n.startswith("region") for n in level.names))
        return out
