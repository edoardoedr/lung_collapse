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
