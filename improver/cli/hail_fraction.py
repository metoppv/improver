# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Script to calculate a hail fraction."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(*cubes: cli.inputcubelist, model_id_attr: str = None):
    """
    Calculates the fraction of precipitation that is forecast to fall as hail.

    Args:
        cubes (iris.cube.CubeList or list):
            Contains cubes of the maximum vertical updraught, hail size,
            cloud condensation level temperature, convective cloud top temperature,
            altitude of hail to rain falling level and the altitude of the orography.
        model_id_attr (str):
            Name of the attribute used to identify the source model for
            blending.

    Returns:
        iris.cube.Cube:
            A single cube containing the hail fraction.

    """
    from iris.cube import CubeList

    from improver.precipitation_type.hail_fraction import HailFraction
    from improver.utilities.flatten import flatten

    (
        vertical_updraught,
        hail_size,
        cloud_condensation_level,
        convective_cloud_top,
        hail_melting_level,
        altitude,
    ) = CubeList(flatten(cubes)).extract(
        [
            "maximum_vertical_updraught",
            "diameter_of_hail_stones",
            "air_temperature_at_condensation_level",
            "air_temperature_at_convective_cloud_top",
            "altitude_of_rain_from_hail_falling_level",
            "surface_altitude",
        ]
    )

    return HailFraction(model_id_attr=model_id_attr)(
        vertical_updraught,
        hail_size,
        cloud_condensation_level,
        convective_cloud_top,
        hail_melting_level,
        altitude,
    )
