# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Module containing lightning classes."""

from datetime import timedelta
from typing import Tuple, Union

import iris
import numpy as np
from iris.coords import DimCoord, CellMethod
from iris.cube import Cube, CubeList

from improver import PostProcessingPlugin
from improver.metadata.constants import FLOAT_DTYPE
from improver.metadata.utilities import (
    create_new_diagnostic_cube,
    generate_mandatory_attributes,
)
from improver.threshold import LatitudeDependentThreshold
from improver.utilities.common_input_handle import as_cubelist
from improver.utilities.cube_checker import spatial_coords_match
from improver.utilities.rescale import rescale
from improver.utilities.spatial import create_vicinity_coord


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

    def __init__(self, model_id_attr: str = None) -> None:
        """
        Initialise the plugin with the model_id_attr.

        Args:
            model_id_attr:
                The name of the dataset attribute to be used to identify the source
                model when blending data from different models.
        """
        self._model_id_attr = model_id_attr

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

    def process(self, *cubes: Union[Cube, CubeList]) -> Cube:
        """
        From the supplied CAPE and precipitation-rate cubes, calculate a probability
        of lightning cube.

        Args:
            cubes:
                Cubes of CAPE and Precipitation rate.

        Returns:
            Cube of lightning data

        Raises:
            ValueError:
                If one of the cubes is not found or doesn't match the other
        """
        cubes = as_cubelist(*cubes)
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
                cubes, model_id_attr=self._model_id_attr
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

        cell_method = CellMethod("Sum", coords="time", comments="of number_of_lightning_flashes_per_unit_area")
        cube.add_cell_method(cell_method)

        return cube


def latitude_to_threshold(
    latitude: np.ndarray, midlatitude: float, tropics: float
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


class LightningMultivariateProbability_USAF2024(PostProcessingPlugin):
    """
    The algorithm outputs the probability of at least one lightning strike within
    20 km of the location in a three hour period. The probabilities are calculated
    for each individual ensemble member, and then they are averaged for the ensemble
    probability forecast.

    Inputs:
    Convective Available Potential Energy (CAPE in J/kg. Most Unstable CAPE is ideal and was used
    to determine the regression statistics, but surface-based CAPE is supported and yields similar
    results but will not forecast elevated convection.)
    Lifted Index (liftind in K)
    Precipitable Water (pwat in kg m-2 or mm. This is used as mm in the regression equations)
    Convective Inhibition (CIN in J/kg)
    3-hour Accumulated Precipitation (apcp in kg m-2 or millimetres)
    Outputs:
    20 km lightning probability over the valid time of the accumulated precipitation


    Description of the algorithm:

    Regression equation when CAPE and APCP are greater than zero:
    lprob= cape*APCP
    lprob=0.13*ln(lprob+0.7)+0.05

    If APCP is very low, a separate regression equation is used to predict lightning probability.
    The definition of “very low” is raised slightly when PWAT values are high. This is because
    the model often produces showery precipitation that doesn’t access the actual instability in
    very moist environments:
    lprob_noprecip=CAPE/(CIN+100.0)
    lprob_noprecip=0.025*ln(lprob_noprecip+0.31)+0.03
    APCP=APCP-(PWAT/1000)

    IF APCP is less than 0.01 inches THEN lprob=lprob_noprecip
    If there is no CAPE but the atmosphere is “close” to unstable, lightning does sometimes occur,
    especially when heavy precipitation may have stabilized the atmosphere in the model. Unstable
    values of lifted index are positive here:
    LIFTIDX=LIFTIDX+4.0
    IF LIFTIDX is less than 0 K THEN LIFTIDX=0.0
    IF CAPE<=0 J/kg THEN lprob=0.2*(LIFTIDX*APCP)^0.5

    Finally, the probability of lightning is reduced when there is not much PW, because graupel
    cannot form and start the whole charging process. Therefore we reduce the probability:
    IF PWAT is less than 20 mm THEN lprob=lprob*(PWAT/20.0)

    Limit final probabilities to 95% as that is as skillful as the regression equations could get:
    IF lprob is greater than 0.95 THEN lprob=0.95
    """

    @staticmethod
    def _extract_input(cubes: CubeList, cube_name: str) -> Cube:
        """Extract the relevant cube based on the cube name.

        Args:
            cubes: Cubes from which to extract required input.
            cube_name: Name of cube to extract.

        Returns:
            The extracted cube.
        """
        try:
            cube = cubes.extract_cube(iris.Constraint(cube_name))
        except iris.exceptions.ConstraintMismatchError:
            raise ValueError(f"No cube named {cube_name} found in {cubes}")
        return cube

    def _get_inputs(self, cubes: CubeList) -> Tuple[Cube, Cube]:
        """
        Separates CAPE, LI, PW, CIN, and APCP cubes and checks that the following
        match: forecast_reference_time, spatial coords, time-bound interval and
        that CAPE time is at the lower bound of precipitation accumulation time.
        """

        output_cubes = iris.cube.CubeList()
        input_names = {
            "atmosphere_specific_convective_available_potential_energy": ["J kg-1"],
            "temperature_difference_between_ambient_air_and_air_lifted_adiabatically": [
                "K"
            ],
            "precipitable_water": ["kg m-2", "mm"],
            "atmosphere_specific_convective_inhibition": ["J kg-1"],
            "precipitation_amount": ["kg m-2", "mm"],
        }

        for input_name, units in input_names.items():
            output_cubes.append(self._extract_input(cubes, input_name))
            if output_cubes[-1].units not in units:
                expected_unit_string = " or ".join(map(str, units))
                received_unit_string = str(output_cubes[-1].units)
                raise ValueError(
                    f"The {output_cubes[-1].name()} units are incorrect, expected "
                    f"units as {expected_unit_string} but received {received_unit_string})."
                )

        cape, liftidx, pwat, cin, apcp = output_cubes

        (apcp_time,) = list(apcp.coord("time").cells())
        for cube in [cape, liftidx, pwat, cin]:
            (cube_time,) = list(cube.coord("time").cells())
            if cube_time.point != apcp_time.bound[0]:
                raise ValueError(
                    f"The {cube.name()} time point ({cube_time.point}) should be valid at the "
                    f"precipitation_accumulation cube lower bound ({apcp_time.bound[0]})."
                )
        if not np.diff(apcp_time.bound) == timedelta(hours=3):
            raise ValueError(
                f"Precipitation_accumulation cube time window must be three hours, "
                f"not {np.diff(apcp_time.bound)}."
            )
        # Following time/space checks should be made for all diagnostics
        for cube in [cape, liftidx, pwat, cin]:
            if cube.coord("forecast_reference_time") != apcp.coord(
                "forecast_reference_time"
            ):
                raise ValueError(
                    f"{cube.name()} and {apcp.name()} do not have the same forecast reference time"
                )
        for cube in [cape, liftidx, pwat, cin]:
            if not spatial_coords_match([cube, apcp]):
                raise ValueError(
                    f"{cube.name()} and {apcp.name()} do not have the same spatial "
                    f"coordinates"
                )

        return cape, liftidx, pwat, cin, apcp

    def process(self, cubes: CubeList, model_id_attr: str = None) -> Cube:
        """
        From the supplied CAPE, Lifted Index, Precipitable Water, CIN, and 3-hr Accumulated
        Precipitation cubes, calculate a probability of lightning cube.

        Args:
            cubes:
                Cubes of CAPE, Lifted Index, Precipitable Water, CIN, and 3-hr Accumulated
                Precipitation cubes.
            model_id_attr:
                The name of the dataset attribute to be used to identify the source
                model when blending data from different models.

        Returns:
            Cube of lightning data

        Raises:
            ValueError:
                If one of the cubes is not found, doesn't match the others, or has incorrect units
        """
        cape, liftidx, pwat, cin, apcp = self._get_inputs(cubes)

        # Regression equations require math on cubes with incompatible units, so strip data
        template = apcp.copy()
        cape = cape.data
        liftidx = liftidx.data
        pwat = pwat.data
        cin = (
            -1 * cin.data
        )  # use inverse convention where CIN is positive instead of negative
        apcp = apcp.data / 1000 * 39.3701  # convert kg m-2 to inches

        # Regression equation when CAPE and APCP are greater than zero:
        lprob = cape * apcp
        lprob = 0.13 * np.log(lprob + 0.7) + 0.05

        # If APCP is very low, a separate regression equation is used to predict lightning
        # probability. The definition of “very low” is raised slightly when PWAT values are
        # high. This is because the model often produces showery precipitation that doesn’t
        # access the actual instability in very moist environments:
        lprob_noprecip = cape / (cin + 100.0)
        lprob_noprecip = 0.025 * np.log(lprob_noprecip + 0.31) + 0.03
        apcp_temp = apcp - (pwat / 1000)

        lprob[np.where(apcp_temp < 0.01)] = lprob_noprecip[np.where(apcp_temp < 0.01)]

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
        low_pwat = np.where(pwat < 20)
        lprob[low_pwat] = lprob[low_pwat] * (pwat[low_pwat] / 20.0)

        # Limit probabilities to 95% as that is as skillful as the regression equations could get:
        lprob[lprob > 0.95] = 0.95

        cube = create_new_diagnostic_cube(
            name=(
                "probability_of_number_of_lightning_flashes_per_unit_area_in_vicinity_above_"
                "threshold"
            ),
            units="1",
            template_cube=template,
            data=lprob.astype(FLOAT_DTYPE),
            mandatory_attributes=generate_mandatory_attributes(
                cubes, model_id_attr=model_id_attr
            ),
        )

        # Add auxiliary coordinate for threshold of any lightning occurring (above 0 flashes)
        coord = DimCoord(
            np.array([0], dtype=FLOAT_DTYPE),
            units="m-2",
            long_name="number_of_lightning_flashes_per_unit_area",
            var_name="threshold",
            attributes={"spp__relative_to_threshold": "greater_than"},
        )
        cube.add_aux_coord(coord)

        # Add auxiliary coordinate for vicinity of 20 km
        vic_coord = create_vicinity_coord(20000)
        cube.add_aux_coord(vic_coord)

        return cube
