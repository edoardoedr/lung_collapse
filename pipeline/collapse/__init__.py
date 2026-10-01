"""Step 5 - inverse FEM of the collapse, split into two pipeline steps.

  fem_setup (setup.py)  solver-independent preparation, written to output_dir/fem/setup:
                        clamped hilum region, rigid alignment of the target, pressure regions
                        for every coarse-to-fine level, tetrahedral volume mesh.
  fem_fit   (fit.py)    regional pleural pressures fitted so that the inflated lung deforms
                        onto the collapsed one; the FEM core is chosen in the config
                        (solvers/, e.g. "getfem").

Only solvers/ knows about a specific FEM library: every core implements
solvers.base.ForwardSolver and receives the same CollapseProblem (problem.py).
"""
