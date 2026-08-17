#!/usr/bin/env python
# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Script to run wind downscaling."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(
    wind_speed_on_heights: cli.inputcube,
    high_res_orography: cli.inputcube,
    model_orography: cli.inputcube,
    model_orography_stddev: cli.inputcube,
    model_silhouette_roughness: cli.inputcube,
    landmask: cli.inputcube,
    *,
    target_height_levels: cli.comma_separated_list_of_float = None,
    target_wind_speed_cube: cli.inputcube = None,
    output_height_level: float = None,
    output_height_level_units: str = "m",
):
    """Wind downscaling.

    Run unresolved-orography wind-speed downscaling on supplied wind fields.
    All inputs must be on the same horizontal grid.

    Target selection:
        Exactly one of the following must apply:
        1. target_wind_speed_cube is provided:
           The correction is applied to that cube's wind values at its
           single height.
        2. target_height_levels is provided:
           wind_speed_on_heights is interpolated to those heights, then
           corrected.
        3. Neither is provided:
           wind_speed_on_heights is corrected on its native height levels.

    Args:
        wind_speed_on_heights (iris.cube.Cube):
            Cube of wind speed on height levels.
        high_res_orography (iris.cube.Cube):
            High-resolution orography cube.
        model_orography (iris.cube.Cube):
            Model-resolution orography cube.
        model_orography_stddev (iris.cube.Cube):
            Standard deviation of model orography height.
        model_silhouette_roughness (iris.cube.Cube):
            Model silhouette roughness cube.
        landmask (iris.cube.Cube):
            Land-sea mask cube.
        target_height_levels (list[float]):
            Optional comma-separated list of target heights in metres. If not
            provided, the input wind height levels are used unless
            target_wind_speed_cube is supplied. Must not be provided together
            with target_wind_speed_cube.
        target_wind_speed_cube (iris.cube.Cube):
            Optional cube of target wind speed at a single height. If
            provided, this is used directly as the target wind field.
            Must not be provided together with target_height_levels.
        output_height_level (float):
            If only a single height level is desired as output from
            wind-downscaling, this option can be used to select the height
            level. If no units are provided with 'output_height_level_units',
            metres are assumed.
        output_height_level_units (str):
            If a single height level is selected as output using
            'output_height_level', this additional argument may be used to
            specify the units of the value entered to select the level.
            e.g hPa.

    Returns:
        iris.cube.Cube:
            The processed Cube.

    Raises:
        ValueError:
            If the requested height value is not found, if both
            target_height_levels and target_wind_speed_cube are provided, or
            if realization counts differ between wind_speed_on_heights and
            target_wind_speed_cube.
    """
    import warnings

    import iris
    from iris.exceptions import CoordinateNotFoundError

    from improver.utilities.cube_extraction import apply_extraction
    from improver.wind_calculations import wind_downscaling

    if output_height_level_units and output_height_level is None:
        warnings.warn(
            "output_height_level_units has been set but no "
            "associated height level has been provided. These units "
            "will have no effect."
        )

    wind_downscaling_plugin = wind_downscaling.WindDownscaling(
        high_res_orog_cube=high_res_orography,
        model_orog_cube=model_orography,
        model_orog_stddev_cube=model_orography_stddev,
        model_silhouette_roughness_cube=model_silhouette_roughness,
        landmask_cube=landmask,
    )

    # Iterate over realizations and apply corrections member-by-member.
    # If no realization coordinate is present (deterministic data), process
    # the entire cube as a single member.
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
        result = wind_downscaling_plugin(
            wind_speed_slice,
            target_height_levels=target_height_levels,
            target_wind_speed_cube=target_wind_speed_slice,
        )
        wind_speed_list.append(result)
    wind_speed = wind_speed_list.merge_cube()

    # If realization exists as a non-dimension coordinate, reinsert it as
    # a dimension axis so the cube has the expected shape.
    non_dim_coords = [x.name() for x in wind_speed.coords(dim_coords=False)]
    if "realization" in non_dim_coords:
        wind_speed = iris.util.new_axis(wind_speed, "realization")

    # If a specific output height is requested, use apply_extraction to select
    # the corresponding height level from the processed wind cube.
    if output_height_level is not None:
        constraints = {"height": output_height_level}
        units = {"height": output_height_level_units}
        single_level = apply_extraction(
            wind_speed, iris.Constraint(**constraints), units
        )
        if not single_level:
            raise ValueError(
                "Requested height level not found, no cube "
                "returned. Available height levels are:\n"
                "{0:}\nin units of {1:}".format(
                    wind_speed.coord("height").points, wind_speed.coord("height").units
                )
            )
        wind_speed = single_level

    return wind_speed
