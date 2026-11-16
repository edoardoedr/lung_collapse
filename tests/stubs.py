"""Stand-ins for the heavy libraries the Mac does not have (pyvista, vtk, torch, ...), so that the
numpy parts of the pipeline can be imported and tested there. Import this module first."""

import importlib.abc
import importlib.machinery
import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HEAVY = ("pyvista", "vtk", "vtkmodules", "sklearn", "torch", "warp", "getfem", "gmsh", "itk", "pyacvd")


class _Finder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Last in sys.meta_path: asked only for what the real finders did not find."""

    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in HEAVY:
            return importlib.machinery.ModuleSpec(name, self, is_package=True)
        return None

    def create_module(self, spec):
        m = MagicMock(name=spec.name)
        m.__path__, m.__spec__, m.__name__ = [], spec, spec.name
        return m

    def exec_module(self, module):
        pass


if not any(isinstance(f, _Finder) for f in sys.meta_path):
    sys.meta_path.append(_Finder())


def run_tests(namespace):
    """Run the test_* functions of a module without pytest (not installed on the Mac)."""
    failed = 0
    for name, fn in sorted(namespace.items()):
        if name.startswith("test_") and callable(fn):
            try:
                out = fn()
                print("ok      %s%s" % (name, "" if out is None else "  (%s)" % out))
            except Exception as e:                              # noqa: BLE001
                failed += 1
                print("FAILED  %s: %s: %s" % (name, type(e).__name__, e))
    sys.exit(1 if failed else 0)
