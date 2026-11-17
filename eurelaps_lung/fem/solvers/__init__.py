"""FEM cores, imported only when selected (GetFEM & co. are not needed by the rest of the pipeline).

A core is chosen by its name below, or as "package.module:ClassName" for one outside this folder.
"""

import importlib

SOLVERS = {
    "getfem": "pipeline.collapse.solvers.getfem_solver:GetFEMSolver",
    "warp": "pipeline.collapse.solvers.warp_solver:WarpSolver",
    "torch": "pipeline.collapse.solvers.torch_solver:TorchSolver",
}


def get_solver(name):
    target = SOLVERS.get(name, name)
    if ":" not in target:
        raise ValueError("unknown solver '%s' (available: %s, or 'module:Class')"
                         % (name, ", ".join(SOLVERS)))
    module, cls = target.split(":")
    try:
        return getattr(importlib.import_module(module), cls)
    except ImportError as e:
        raise RuntimeError("solver '%s' cannot be imported (%s): is its library installed in this "
                           "environment? See README, step 5." % (name, e)) from e


def options_for(name, options, from_solver=None):
    """solver_options written for core `from_solver` (e.g. a torch config), passed to core `name`
    (validation scripts that switch core): options `name` does not accept are dropped (torch's
    device / material for GetFEM, ...), and linear_solver too when switching to or from GetFEM
    (its names, mumps ..., mean nothing to Warp / torch, whose cudss / pardiso / scipy are shared)."""
    keys = get_solver(name).accepted_options()
    out = {k: v for k, v in options.items() if keys is None or k in keys}
    if from_solver is not None and from_solver != name and "getfem" in (name, from_solver):
        out.pop("linear_solver", None)
    return out
