# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""
This module implements a lazy import hook for the ``iris`` package
"""

import importlib
import importlib.abc
import sys
from types import ModuleType


class _IrisFinder(importlib.abc.MetaPathFinder):
    """
    A meta‑path finder that catches the very first import of ``iris``.
    It delegates the actual loading to _IrisLoader, then deletes itself
    so later imports use the standard import machinery.
    """

    def find_spec(self, fullname, path, target=None):
        if fullname != "iris":
            return None  # not our concern

        # Tell the import system to use our loader for this name.
        return importlib.machinery.ModuleSpec(
            name=fullname,
            loader=_IrisLoader(),
            origin="lazy‑iris‑hook",
        )


class _IrisLoader(importlib.abc.Loader):
    """Loads the real ``iris`` package, patches FUTURE, and returns it."""

    def create_module(self, spec):
        # Let the default module creation happen (None means use default).
        return None

    def exec_module(self, module: ModuleType):
        """
        `module` is the empty placeholder that the import system created
        for the name `iris`.  We replace its contents with those of the
        genuine package.
        """

        # Remove our finder so we don’t recurse back into it.
        for i, finder in enumerate(sys.meta_path):
            if isinstance(finder, _IrisFinder):
                del sys.meta_path[i]
                break

        # Remove the placeholder from sys.modules to avoid circular import
        sys.modules.pop("iris", None)

        # Import the *real* iris package now that our hook is gone.
        real_iris = importlib.import_module("iris")

        real_iris.FUTURE.date_microseconds = True

        # Populate the placeholder module with everything from the real one.
        module.__dict__.update(real_iris.__dict__)

        # Put the real module back into sys.modules under the same name.
        sys.modules["iris"] = real_iris

        # --------------------------------------------------------------
        # Re‑install the finder so that future *different* imports
        #     (e.g. reloads) still work, but it will never fire again for
        #     the already‑loaded name.
        # sys.meta_path.insert(0, _IrisFinder())


# Install the finder *once* at program start – before any ``import iris``.
if not any(isinstance(f, _IrisFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _IrisFinder())
