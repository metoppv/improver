#!/usr/bin/env python
# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Script to combine netcdf data."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(
    *cubes: cli.inputcube,
    operation="+",
    new_name=None,
    broadcast=None,
    minimum_realizations=None,
    cell_method_coordinate=None,
    expand_bound=True,
):
    r"""Combine input cubes.

    Combine the input cubes into a single cube using the requested operation.
    The first cube in the input list provides the template for output metadata.
    If coordinates are expanded as a result of this combine operation
    (e.g. expanding time for accumulations / max in period) the upper bound of
    the new coordinate will also be used as the point for the new coordinate.

    Args:
        cubes (iris.cube.CubeList or list of iris.cube.Cube):
            An iris CubeList to be combined.
        operation (str):
            An operation to use in combining input cubes. One of:
            +, -, \*, add, subtract, multiply, min, max, mean, masked_add
        new_name (str):
            New name for the resulting dataset.
        broadcast (str):
            If specified, the input cubes will be broadcast over the coordinate name provided. If
            "threshold" is provided the plugin will try to find a threshold coordinate on the
            probability cube.
        minimum_realizations (int):
            If specified, the input cubes will be filtered to ensure that only realizations that
            include all available lead times are combined. If the number of realizations that
            meet this criteria are fewer than this integer, an error will be raised.
        cell_method_coordinate (str):
            If specified, a cell method is added to the output with the coordinate
            provided. This is only available for max, min and mean operations.
        expand_bound (bool):
            If True then coord bounds will be extended to represent all cubes being combined.
    Returns:
        result (iris.cube.Cube):
            Returns a cube with the combined data.
    """
    from improver.cube_combiner import Combine

    return Combine(
        operation,
        broadcast=broadcast,
        minimum_realizations=minimum_realizations,
        new_name=new_name,
        cell_method_coordinate=cell_method_coordinate,
        expand_bound=expand_bound,
    )(*cubes)
