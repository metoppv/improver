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
    latitude: np.ndarray, midlatitude: float, tropics: float,
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


class LightningMultivariateProbability(PostProcessingPlugin):
    """
    The algorithm outputs the probability of at least one lightning strike within
    20 km of the location in a three hour period. The probabilities are calculated
    for each individual ensemble member, and then they are averaged for the ensemble
    probability forecast.

    Inputs:
    CAPE (cape; calculated with water loading and latent heat of freezing estimated)
    Lifted Index (liftidx)
    Precipitable Water (pwat in mm)
    CIN (cin)
    3-hour Accumulated Precipitation (apcp in inches)
    Outputs:
    20 km lightning probability over the valid time of the accumulated precipitation

    ---------------------------+------------+----------------------------------------

    Regression equation when CAPE and APCP are greater than zero:
    lprob= cape*APCP
    lprob=0.13*alog(lprob+0.7)+0.05

    If APCP is very low, a separate regression equation is used to predict lightning probability.
    The definition of “very low” is raised slightly when PWAT values are high. This is because
    the model often produces showery precipitation that doesn’t access the actual instability in
    very moist environments:
    lprob_noprecip=CAPE/(CIN+100.0)
    lprob_noprecip=0.025*alog(lprob_noprecip+0.31)+0.03
    APCP=APCP-(PWAT/1000)

    IF APCP is less than 0.01 THEN lprob=lprob_noprecip
    If there is no CAPE but the atmosphere is “close” to unstable, lightning does sometimes occur,
    especially when heavy precipitation may have stabilized the atmosphere in the model. Unstable
    values of lifted index are positive here:
    LIFTIDX=LIFTIDX+4.0
    IF LIFTIDX is less than 0 THEN LIFTIDX=0.0
    IF CAPE is less than 0 THEN lprob =0.2*(LIFTIDX*APCP)^0.5

    Finally, the probability of lightning is reduced when there is not much PW, because graupel
    cannot form and start the whole charging process. Therefore we reduce the probability:
    IF PWAT is less than 20 THEN lprob=lprob*(PWAT/20.0)

    Limit final probabilities to 95% as that is as skillful as the regression equations could get:
    IF lprob is greater than 0.95 THEN lprob=0.95

    Make percentage:
    lprob=lprob*100.0
    """

    @staticmethod
    def _get_inputs(cubes: CubeList) -> Tuple[Cube, Cube]:
        """
        Separates CAPE, LI, PW, CIN, and APCP cubes and checks that the following
        match: forecast_reference_time, spatial coords, time-bound interval and
        that CAPE time is at the lower bound of precipitation accumulation time.
        """
        cape = cubes.extract(
            iris.Constraint(
                cube_func=lambda cube: "atmosphere_specific_convective_available_potential_energy"
                in cube.name()
            )
        )
        if cape:
            cape = cape.merge_cube()
        else:
            raise ValueError(
                f"No cube named atmosphere_specific_convective_available_potential_energy found "
                f"in {cubes}"
            )
        lifted_ind_str = (
            "temperature_difference_between_ambient_air_and_air_lifted_adiabatically"
        )
        liftidx = cubes.extract(
            iris.Constraint(cube_func=lambda cube: lifted_ind_str in cube.name())
        )
        if liftidx:
            liftidx = liftidx.merge_cube()
        else:
            raise ValueError(
                f"No cube named: "
                f"temperature_difference_between_ambient_air_and_air_lifted_adiabatically"
                f" found in {cubes}"
            )
        pwat = cubes.extract(
            iris.Constraint(cube_func=lambda cube: "precipitable_water" in cube.name())
        )
        if pwat:
            pwat = pwat.merge_cube()
        else:
            raise ValueError(f"No cube named precipitable_water found " f"in {cubes}")
        cin = cubes.extract(
            iris.Constraint(
                cube_func=lambda cube: "atmosphere_specific_convective_inhibition"
                in cube.name()
            )
        )
        if cin:
            cin = cin.merge_cube()
        else:
            raise ValueError(
                f"No cube named atmosphere_specific_convective_inhibition found "
                f"in {cubes}"
            )
        apcp = cubes.extract(
            iris.Constraint(
                cube_func=lambda cube: "precipitation_amount" in cube.name()
            )
        )
        if apcp:
            apcp = apcp.merge_cube()
        else:
            raise ValueError(f"No cube named precipitation_amount found in {cubes}")

        (cape_time,) = list(cape.coord("time").cells())
        (apcp_time,) = list(apcp.coord("time").cells())
        if cape_time.point != apcp_time.bound[0]:
            raise ValueError(
                f"CAPE cube time ({cape_time.point}) should be valid at the "
                f"precipitation_accumulation cube lower bound ({apcp_time.bound[0]})."
            )
        if np.diff(apcp_time.bound) not in [timedelta(hours=3)]:
            raise ValueError(
                f"Precipitation_accumulation cube time window must be three hours, "
                f"not {np.diff(apcp_time.bound)}."
            )
        # Following time/space checks should be made for all diagnostics
        if cape.coord("forecast_reference_time") != apcp.coord(
            "forecast_reference_time"
        ):
            raise ValueError(
                "Supplied cubes must have the same forecast reference times"
            )
        if not spatial_coords_match([cape, apcp]):
            raise ValueError("Supplied cubes do not have the same spatial coordinates")

        return cape, liftidx, pwat, cin, apcp

    def process(self, cubes: CubeList, model_id_attr: str = None) -> Cube:
        """
        From the supplied CAPE, LIFTIDX, PWAT, CIN, APCP cubes, calculate a probability
        of lightning cube.

        Args:
            cubes:
                Cubes of CAPE, LIFTIDX, PWAT, CIN, APCP.
            model_id_attr:
                The name of the dataset attribute to be used to identify the source
                model when blending data from different models.

        Returns:
            Cube of lightning data

        Raises:
            ValueError:
                If one of the cubes is not found or doesn't match the other
        """
        cape, liftidx, pwat, cin, apcp = self._get_inputs(cubes)

        # Regression equations require math on cubes with incompatible units, so strip data
        templ = apcp.copy()
        cape = cape.data
        liftidx = liftidx.data
        pwat = pwat.data
        cin = cin.data
        apcp = apcp.data / 1000 * 39.3701  # convert (kg m-3) to (inches)

        # Regression equation when CAPE and APCP are greater than zero:
        lprob = cape * apcp
        lprob = 0.13 * np.log(lprob + 0.7) + 0.05

        # If APCP is very low, a separate regression equation is used to predict lightning
        # probability. The definition of “very low” is raised slightly when PWAT values are
        # high. This is because the model often produces showery precipitation that doesn’t
        # access the actual instability in very moist environments:
        lprob_noprecip = cape / (cin + 100.0)
        lprob_noprecip = 0.025 * np.log(lprob_noprecip + 0.31) + 0.03
        apcp = apcp - (pwat / 1000)

        lprob[np.where(apcp < 0.01)] = lprob_noprecip[np.where(apcp < 0.01)]

        # If there is no CAPE but the atmosphere is “close” to unstable, lightning does sometimes
        # occur, especially when heavy precipitation may have stabilized the atmosphere in the
        # model. Unstable values of lifted index are positive here:
        liftidx = liftidx + 4.0

        liftidx[liftidx < 0] = 0

        lprob[np.where(cape <= 0)] = (
            0.2 * (liftidx[np.where(cape <= 0)] * apcp[np.where(cape <= 0)]) ** 0.5
        )

        # Finally, the probability of lightning is reduced when there is not much PWAT, since
        # graupel cannot form required for the charging process. The probability is reduced as:
        lprob[np.where(pwat < 20)] = lprob[np.where(pwat < 20)] * (
            pwat[np.where(pwat < 20)] / 20.0
        )

        # Limit probabilities to 95% as that is as skillful as the regression equations could get:
        lprob[lprob > 0.95] = 0.95

        # Make percentage:
        data = lprob * 100.0

        cube = create_new_diagnostic_cube(
            name="""20_km_lightning_probability_over_the_valid_time_of_the_accumulated_
                precipitation""",
            units="1",
            template_cube=templ,
            data=data.astype(FLOAT_DTYPE),
            mandatory_attributes=generate_mandatory_attributes(
                cubes, model_id_attr=model_id_attr
            ),
        )

        return cube
