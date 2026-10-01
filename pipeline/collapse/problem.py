"""Solver-independent description of the collapse problem: written by fem_setup, read by fem_fit.

Reference configuration = inflated lung (registration output), target = collapsed lung with the
same surface points. Plain numpy arrays, LPS, mm, so that any FEM core can read them.
"""

from dataclasses import dataclass, field, fields
from pathlib import Path

import numpy as np


@dataclass
class CollapseProblem:
    nodes: np.ndarray          # (n, 3) volume mesh nodes, reference (inflated) configuration
    tets: np.ndarray           # (m, 4) linear tetrahedra, positive volume
    surface_nodes: np.ndarray  # (N,) volume node of each surface point
    tris: np.ndarray           # (T, 3) surface triangles (surface point indices), outward
    clamped_tri: np.ndarray    # (T,) bool, hilum triangles with u = 0 (Dirichlet)
    target: np.ndarray         # (N, 3) target position of each surface point (aligned)
    levels: np.ndarray         # (L,) number of pressure regions per level, coarse -> fine
    regions: np.ndarray        # (L, T) region of each triangle per level, -1 on clamped triangles
    clean_field: np.ndarray    # (N, 3) displacement the regions were clustered on
    # optional rigid cavity wall (empty arrays = no wall): surface point i must stay where the
    # signed distance to the wall (> 0 outside) is <= wall_allow[i]
    wall_points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    wall_tris: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.int64))
    wall_allow: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @property
    def has_wall(self):
        return len(self.wall_tris) > 0

    @property
    def reference(self):
        """(N, 3) surface points in the reference configuration."""
        return self.nodes[self.surface_nodes]

    @property
    def faces(self):
        """(T, 3) surface triangles as volume node indices."""
        return self.surface_nodes[self.tris]

    @property
    def clamped_points(self):
        """Surface point indices with u = 0."""
        return np.unique(self.tris[self.clamped_tri])

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **{f.name: getattr(self, f.name) for f in fields(self)})
        return path

    @classmethod
    def load(cls, path):
        with np.load(path) as d:
            return cls(**{f.name: d[f.name] for f in fields(cls) if f.name in d})
