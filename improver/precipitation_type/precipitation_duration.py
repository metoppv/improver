# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Module containing the PrecipitationDuration class."""

from typing import Optional, Union

import numpy as np
from iris.cube import Cube, CubeList

from improver import PostProcessingPlugin
from improver.metadata.utilities import (
    create_new_diagnostic_cube,
    generate_mandatory_attributes,
)
from improver.utilities.common_input_handle import as_cubelist
from improver.utilities.cube_checker import assert_spatial_coords_match
from improver.utilities.temporal import get_period


class PrecipitationDuration(PostProcessingPlugin):
    """
    Classifies periods of precipitation as wet or dry using specified
    threshold criteria.
    """

    def __init__(
            self,
            min_accumulation_per_hour_light: float,
            min_accumulation_per_hour_heavy: float,
            heavy_rate: float,
        ) -> None:
        """
        Initialise the class.

        Args:
            min_accumulation_per_hour_light:
                The minimum accumulation per hour in the period that defines
                the period as having at least light precipitation (wet).
                Units of mm.
            min_accumulation_per_hour_heavy:
                The minimum accumulation per hour in the period that defines
                the period as having at least heavy precipitation. This
                classification also requires the heavy_rate threshold below
                to be satisfied. Units of mm.
            heavy_rate:
                A rate threshold, which if the maximum rate in the period
                is in excess of contributes to defining the period as having
                heavy precipitation. Units of mm/hr.
        """
        self.min_accumulation_per_hour_light = min_accumulation_per_hour_light
        self.min_accumulation_per_hour_heavy = min_accumulation_per_hour_heavy
        self.heavy_rate = heavy_rate

    @staticmethod
    def _period_in_hours(max_precip_rate: Cube, precip_accumulation: Cube) -> float:
        """Get the periods of the input cubes, check they are equal and
        return the period in hours.

        Args:
            max_precip_rate:
                Maximum preciptation rate in a period.
            precip_accumulation:
                Precipitation accumulation in a period.
        Returns:
            Period in hours.
        Raises:
            ValueError: If periods of two cubes are not equal.
        """
        rate_period = get_period(max_precip_rate) / 3600
        accumulation_period = get_period(precip_accumulation) / 3600
        if not rate_period == accumulation_period:
            raise ValueError(
                "Maximum rate and accumulation periods must match. Periods are:"
                f" max rate {rate_period}, accumulation {accumulation_period}."
            )
        return rate_period


    def process(self, *cubes: Union[Cube, CubeList]) -> Cube:
        """Calculates the hail fraction using the maximum vertical updraught,
        the hail_size, the cloud condensation level temperature, the convective cloud
        top temperature, the altitude of the hail to rain phase change and the
        orography.

        Args:
            cubes:
                max_precip_rate: Maximum preciptation rate in a period.
                precip_accumulation: Precipitation accumulation in a period.

        Returns:
            Something.
        """
        cubes = as_cubelist(*cubes)
        max_precip_rate, precip_accumulation = cubes.extract(
            [
                "lwe_precipitation_rate",
                "lwe_thickness_of_precipitation",
            ]
        )
        period = self._period_in_hours(max_precip_rate, precip_accumulation)

        max_precip_rate.convert_units("mm/hr")
        precip_accumulation.convert_units("mm")

        combined = np.where((acc.data > (min_accumulation_per_hour_light * (period / 3600))) & (rate.data > critical_rate), 1, 0)
        combined_cube = acc.copy(data=combined)
        combined_cube.rename("fraction_of_heavy_precipitation_in_period")
        combined_cube.units = 1
        combined_cube.attributes["critical_rate"] = critical_rate
        combined_cube.attributes["accumulation_min"] = min_accumulation_per_hour
        return combined_cube




        hail_fraction_cube = create_new_diagnostic_cube(
            "hail_fraction",
            "1",
            template_cube=vertical_updraught,
            mandatory_attributes=generate_mandatory_attributes(
                [
                    vertical_updraught,
                    hail_size,
                    cloud_condensation_level,
                    convective_cloud_top,
                    hail_melting_level,
                ],
                model_id_attr=self.model_id_attr,
            ),
            data=hail_fraction,
        )
        return hail_fraction_cube
