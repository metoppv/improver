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
    mode: str = "hc_and_rc",
):
    """Wind downscaling.

    Run wind downscaling to apply roughness correction and/or height correction
    to wind fields as described in Howard and Clark (2007).

    Args:
        wind_speed (iris.cube.Cube):
            Cube of wind speed on standard grid. This may be deterministic, or
            contain a realization coordinate. Any units can be supplied.
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
            standard grid).
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
            e.g. hPa.
        mode (str):
            Which correction(s) to apply: "hc_and_rc", "hc", or "rc".

    Returns:
        iris.cube.Cube:
            The processed Cube.

    Raises:
        ValueError:
            If the requested height value is not found.

    """
    import warnings

    from improver.wind_calculations.wind_downscaling import ApplyWindDownscaling

    if output_height_level_units and output_height_level is None:
        warnings.warn(
            "output_height_level_units has been set but no "
            "associated height level has been provided. These units "
            "will have no effect."
        )

    return ApplyWindDownscaling(
        model_resolution=model_resolution,
        output_height_level=output_height_level,
        output_height_level_units=output_height_level_units,
        mode=mode,
    )(
        wind_speed=wind_speed,
        model_orog_stddev=sigma,
        target_orog=target_orography,
        model_orog=standard_orography,
        model_silhouette_roughness=silhouette_roughness,
        model_z0=vegetative_roughness,
    )
