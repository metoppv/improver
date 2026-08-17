#!/usr/bin/env python
# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Script to apply unresolved-orography wind corrections."""

from improver import cli
from improver.wind_calculations.wind_downscaling import (
    process as wind_downscaling_process,
)


@cli.clizefy
@cli.with_output
def process(
    wind_speed_on_heights: cli.inputcube,
    high_res_orog_cube: cli.inputcube,
    model_orog_cube: cli.inputcube,
    model_orog_stddev_cube: cli.inputcube,
    model_silhouette_roughness_cube: cli.inputcube,
    landmask_cube: cli.inputcube,
    *,
    target_height_levels: cli.comma_separated_list_of_float = None,
    target_wind_speed_cube: cli.inputcube = None,
):
    """Apply unresolved-orography wind corrections."""
    import iris
    from iris.exceptions import CoordinateNotFoundError

    try:
        wind_speed_slices = list(wind_speed_on_heights.slices_over("realization"))
    except CoordinateNotFoundError:
        wind_speed_slices = [wind_speed_on_heights]

    if target_wind_speed_cube is None:
        target_wind_speed_slices = [None] * len(wind_speed_slices)
    else:
        try:
            target_wind_speed_slices = list(
                target_wind_speed_cube.slices_over("realization")
            )
        except CoordinateNotFoundError:
            target_wind_speed_slices = [target_wind_speed_cube] * len(wind_speed_slices)

        if len(target_wind_speed_slices) != len(wind_speed_slices):
            raise ValueError(
                "Mismatch in realization count between wind_speed_on_heights "
                "and target_wind_speed_cube."
            )

    wind_speed_list = iris.cube.CubeList()
    for wind_speed_slice, target_wind_speed_slice in zip(
        wind_speed_slices,
        target_wind_speed_slices,
    ):
        result = wind_downscaling_process(
            wind_speed_slice,
            high_res_orog_cube,
            model_orog_cube,
            model_orog_stddev_cube,
            model_silhouette_roughness_cube,
            landmask_cube,
            target_height_levels=target_height_levels,
            target_wind_speed_cube=target_wind_speed_slice,
        )
        wind_speed_list.append(result)

    wind_speed = wind_speed_list.merge_cube()

    non_dim_coords = [x.name() for x in wind_speed.coords(dim_coords=False)]
    if "realization" in non_dim_coords:
        wind_speed = iris.util.new_axis(wind_speed, "realization")

    return wind_speed
