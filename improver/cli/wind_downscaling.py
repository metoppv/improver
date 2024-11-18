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
    wind_speed: cli.inputcube,
    sigma: cli.inputcube,
    target_orography: cli.inputcube,
    standard_orography: cli.inputcube,
    silhouette_roughness: cli.inputcube,
    vegetative_roughness: cli.inputcube = None,
    *,
    model_resolution: float,
    output_height_level: float = None,
    output_height_level_units="m",
):
    """Wind downscaling.

    Run wind downscaling to apply roughness correction and height correction
    to wind fields as described in Howard and Clark (2007). All inputs must
    be on the same standard grid.

    Args:
        wind_speed (iris.cube.Cube):
            Cube of wind speed on standard grid.
            Any units can be supplied.
        sigma (iris.cube.Cube):
            Cube of standard deviation of model orography height.
            Units of field: m.
        target_orography (iris.cube.Cube):
            Cube of orography to downscale fields to.
            Units of field: m.
        standard_orography (iris.cube.Cube):
            Cube of orography on standard grid. (interpolated model orography).
            Units of field: m.
        silhouette_roughness (iris.cube.Cube):
            Cube of model silhouette roughness.
            Units of field: dimensionless.
        vegetative_roughness (iris.cube.Cube):
            Cube of vegetative roughness length.
            Units of field: m.
        model_resolution (float):
            Original resolution of model orography (before interpolation to
            standard grid)
            Units of field: m.
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

    Rises:
        ValueError:
            If the requested height value is not found.

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
    try:
        wind_speed_iterator = wind_speed.slices_over("realization")
    except CoordinateNotFoundError:
        wind_speed_iterator = [wind_speed]
    wind_speed_list = iris.cube.CubeList()
    for wind_speed_slice in wind_speed_iterator:
        result = wind_downscaling.RoughnessCorrection(
            silhouette_roughness,
            sigma,
            target_orography,
            standard_orography,
            model_resolution,
            z0_cube=vegetative_roughness,
            height_levels_cube=None,
        )(wind_speed_slice)
        wind_speed_list.append(result)

    wind_speed = wind_speed_list.merge_cube()
    non_dim_coords = [x.name() for x in wind_speed.coords(dim_coords=False)]
    if "realization" in non_dim_coords:
        wind_speed = iris.util.new_axis(wind_speed, "realization")
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
