"""The collapse model: an elastic lung clamped at the hilum, loaded by the pleural pressure it
loses in the pneumothorax (uniform part + the gradient that carried its weight), inside the
chest wall. METHODOLOGY.md has the equations.

  setup.py    fem_setup step: clamp, wall, volume mesh -> output_dir/fem/setup/problem.npz
  load.py     the load as a function of the physical parameters (given or fitted)
  forward.py  collapse step: parameters given -> collapsed lung
  fit.py      fit step: the parameters marked "fit" fitted to the collapsed lung
  results.py, sequence.py   what a run writes; the collapse sequence for Slicer

Only solvers/ knows about a specific FEM library: every core implements
solvers.base.ForwardSolver and receives the same CollapseProblem (problem.py).
"""
