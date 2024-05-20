# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of improver and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Contains a class to trap stdout. Deprecated from PySteps v1.1.0 in favour
of adding "silent_import=True" to the pyconfig file."""

import contextlib
import sys


@contextlib.contextmanager
def redirect_stdout(target=None):
    """Captures stdout and optionally returns it

    Args:
        target:
            Any captured stdout is returned here.
    """
    original = sys.stdout
    sys.stdout = target
    yield
    sys.stdout = original
