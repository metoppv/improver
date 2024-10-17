#!/usr/bin/env python
# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Script to compare netcdf files"""

from improver import cli
from improver.constants import DEFAULT_TOLERANCE


@cli.clizefy
def process(
    actual: cli.inputpath,
    desired: cli.inputpath,
    rtol: float = DEFAULT_TOLERANCE,
    atol: float = DEFAULT_TOLERANCE,
    *,
    ignored_attributes: cli.comma_separated_list = None,
) -> None:
    """
    Compare two netcdf files

    Args:
        actual:
            Path to output data netcdf file
        desired:
            Path to desired/known good data netcdf file
        rtol:
            Relative tolerance for data in variables
        atol:
            Absolute tolerance for data in variables
        ignored_attributes:
            List of attributes to ignore in the comparison. This option allows for
            attributes such as "history" to be ignored, where such attributes often
            vary between files without indicating any differences in the data.

    Returns:
        None
    """
    from improver.utilities import compare

    compare.compare_netcdfs(
        actual,
        desired,
        rtol=rtol,
        atol=atol,
        ignored_attributes=ignored_attributes,
        reporter=print,
    )
