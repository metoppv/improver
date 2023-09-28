# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown copyright. The Met Office.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
"""Module containing lightning classes."""
from datetime import timedelta
from typing import Tuple

import iris
import numpy as np
from iris.coords import DimCoord
from iris.cube import Cube, CubeList

from improver import PostProcessingPlugin
from improver.metadata.constants import FLOAT_DTYPE
from improver.metadata.utilities import (
    create_new_diagnostic_cube,
    generate_mandatory_attributes,
)
from improver.threshold import LatitudeDependentThreshold
from improver.utilities.cube_checker import spatial_coords_match
from improver.utilities.rescale import rescale


class LightningFromCapePrecip(PostProcessingPlugin):
    """
    Apply latitude-dependent thresholds to CAPE and precipitation rate to derive a
    presence-of-lightning cube.

    Lightning is based on the presence of both CAPE and precipitation rate with
    thresholds varying linearly between

    +---------------------------+------------+---------------------+
    |                           | CAPE       | Precipitation rate  |
    |                           | (J kg-1)   | (mm h-1)            |
    +===========================+============+=====================+
    | Mid-latitudes             | 350        | 1                   |
    | (above 50 degrees N or S) |            |                     |
    +---------------------------+------------+---------------------+
    | Tropics                   | 500        | 4                   |
    | (below 10 degrees N or S) |            |                     |
    +---------------------------+------------+---------------------+

    """

    @staticmethod
    def _get_inputs(cubes: CubeList) -> Tuple[Cube, Cube]:
        """
        Separates CAPE and precipitation rate cubes and checks that the following
        match: forecast_reference_time, spatial coords, time-bound interval and
        that CAPE time is at the lower bound of precipitation rate time.
        The precipitation rate data must represent a period of 1 or 3 hours.
        """
        cape = cubes.extract(
            iris.Constraint(
                cube_func=lambda cube: "atmosphere_convective_available_potential_energy"
                in cube.name()
            )
        )
        if cape:
            cape = cape.merge_cube()
        else:
            raise ValueError(
                f"No cube named atmosphere_convective_available_potential_energy found "
                f"in {cubes}"
            )
        precip = cubes.extract(
            iris.Constraint(
                cube_func=lambda cube: "precipitation_rate_max" in cube.name()
            )
        )
        if precip:
            precip = precip.merge_cube()
        else:
            raise ValueError(f"No cube named precipitation_rate_max found in {cubes}")
        (cape_time,) = list(cape.coord("time").cells())
        (precip_time,) = list(precip.coord("time").cells())
        if cape_time.point != precip_time.bound[0]:
            raise ValueError(
                f"CAPE cube time ({cape_time.point}) should be valid at the "
                f"precipitation_rate_max cube lower bound ({precip_time.bound[0]})."
            )
        if np.diff(precip_time.bound) not in [timedelta(hours=1), timedelta(hours=3)]:
            raise ValueError(
                f"Precipitation_rate_max cube time window must be one or three hours, "
                f"not {np.diff(precip_time.bound)}."
            )
        if cape.coord("forecast_reference_time") != precip.coord(
            "forecast_reference_time"
        ):
            raise ValueError(
                "Supplied cubes must have the same forecast reference times"
            )
        if not spatial_coords_match([cape, precip]):
            raise ValueError("Supplied cubes do not have the same spatial coordinates")
        return cape, precip

    def process(self, cubes: CubeList, model_id_attr: str = None) -> Cube:
        """
        From the supplied CAPE and precipitation-rate cubes, calculate a probability
        of lightning cube.

        Args:
            cubes:
                Cubes of CAPE and Precipitation rate.
            model_id_attr:
                The name of the dataset attribute to be used to identify the source
                model when blending data from different models.

        Returns:
            Cube of lightning data

        Raises:
            ValueError:
                If one of the cubes is not found or doesn't match the other
        """
        cape, precip = self._get_inputs(cubes)

        cape_true = LatitudeDependentThreshold(
            lambda lat: latitude_to_threshold(lat, midlatitude=350.0, tropics=500.0),
            threshold_units="J kg-1",
            comparison_operator=">",
        )(cape)

        precip_true = LatitudeDependentThreshold(
            lambda lat: latitude_to_threshold(lat, midlatitude=1.0, tropics=4.0),
            threshold_units="mm h-1",
            comparison_operator=">",
        )(precip)

        data = cape_true.data * precip_true.data

        cube = create_new_diagnostic_cube(
            name="probability_of_number_of_lightning_flashes_per_unit_area_above_threshold",
            units="1",
            template_cube=precip,
            data=data.astype(FLOAT_DTYPE),
            mandatory_attributes=generate_mandatory_attributes(
                cubes, model_id_attr=model_id_attr
            ),
        )

        coord = DimCoord(
            np.array([0], dtype=FLOAT_DTYPE),
            units="m-2",
            long_name="number_of_lightning_flashes_per_unit_area",
            var_name="threshold",
            attributes={"spp__relative_to_threshold": "greater_than"},
        )
        cube.add_aux_coord(coord)

        return cube


def latitude_to_threshold(
    latitude: np.ndarray,
    midlatitude: float,
    tropics: float,
) -> np.ndarray:
    """
    Rescale a latitude range into a range of threshold values suitable for
    thresholding a different diagnostic. This is based on the value provided
    for that diagnostic at midlatitude (more than 50 degrees from the equator)
    and in the tropics (closer than 10 degrees from the equator). Varies
    linearly in between.

    Args:
        latitude:
            An array of latitude points (e.g. cube.coord("latitude").points)
        midlatitude:
            The threshold value to return above 50N or below 50S.
        tropics:
            The threshold value to return below 10N or above 10S.

    Returns:
        An array of thresholds, one for each latitude point
    """
    return np.where(
        latitude > 0,
        rescale(latitude, (50.0, 10), (midlatitude, tropics), clip=True),
        rescale(latitude, (-50.0, -10), (midlatitude, tropics), clip=True),
    )
