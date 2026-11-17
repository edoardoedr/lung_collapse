"""Interface every FEM core implements. The collapse and fit steps only talk to this class."""

from abc import ABC, abstractmethod


class ForwardSolver(ABC):
    """Hyperelastic lung under regional follower pleural pressure.

    Contract:
      - reference configuration = problem.nodes / problem.tets (mm, LPS);
      - hyperelastic (compressible Neo-Hookean unless the core offers others), non-dimensional with E = 1: the unknowns are q = p / E per
        region and Poisson's ratio nu (only p / E is identifiable from shapes);
      - u = 0 on the triangles with problem.clamped_tri;
      - every other surface triangle carries the follower pressure (Nanson: J F^-T N) of its
        region; positive q pushes INWARD (collapse). The steps check the sign once at start;
      - solve() returns the displacement of the surface points, in problem.reference order;
      - if problem.has_wall: the wall is rigid and the surface points may not go farther out of
        it than problem.wall_allow (contact on the pressure faces). A core with supports_wall =
        False refuses such a problem.

    A core may keep its last converged solution to warm-start the next solve.
    """

    name = None
    supports_wall = False       # True if the core enforces problem.wall_*
    supports_body = False       # True if the core has the body force (set_body, solve(..., body))

    @classmethod
    def accepted_options(cls):
        """Names of the solver_options this core accepts; None = any (no check)."""
        return None

    def __init__(self, problem, options):
        self.problem = problem
        self.options = options

    @abstractmethod
    def set_regions(self, labels, n_regions):
        """labels: (T,) region of each surface triangle, -1 = no pressure (clamped)."""

    def set_body(self, direction):
        """Direction (3,) of a constant body force per unit reference volume; its magnitude (relative
        to E, per mm) is the `body` of solve(). Only cores with supports_body."""
        raise NotImplementedError("solver '%s' has no body force" % self.name)

    @abstractmethod
    def solve(self, q, nu, body=0.0):
        """(N, 3) surface displacement for pressures q (n_regions,) and body force magnitude `body`,
        or None if not converged."""

    def get_state(self):
        """Copy of the last converged full solution (warm start, export); None if not kept."""
        return None

    def set_state(self, state, load=None, body=0.0):
        """Warm-start the next solve from `state` (from get_state). load: optional p/E per surface
        triangle (as passed to solve, sign included; ignored where clamped) that `state` is in
        equilibrium with, and its body force magnitude, so that the next solve can ramp from it
        instead of jumping to its load."""

    def reset(self):
        """Forget the warm start: the next solve starts from the reference configuration."""

    def stats(self):
        """Cumulative counters for the log (e.g. Newton iterations), {} if not kept."""
        return {}

    def export_volume(self, path, state):
        """Write the volume solution `state` for ParaView."""
        raise NotImplementedError

    def jacobian(self, q, nu, with_nu=False, with_body=False):
        """d(surface displacement)/d(q[, nu][, body]) at the last converged solve, which must be at
        exactly (q, nu): array (3N, n_regions [+1] [+1]) with rows ordered as solve(q, nu).ravel().
        None = not available."""
        return None
