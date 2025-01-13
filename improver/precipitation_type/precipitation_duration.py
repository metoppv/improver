# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Module containing the PrecipitationDuration class."""

from typing import Union, Tuple

import numpy as np
from iris import analysis
from iris.cube import Cube, CubeList
from iris.coords import AuxCoord
from iris import Constraint

from improver import PostProcessingPlugin
from improver.metadata.utilities import (
    create_new_diagnostic_cube,
    generate_mandatory_attributes,
)
from improver.metadata.probabilistic import get_threshold_coord_name_from_probability_name
from improver.utilities.common_input_handle import as_cubelist
from improver.utilities.cube_manipulation import MergeCubes


class PrecipitationDuration(PostProcessingPlugin):
    """
    Classifies periods of precipitation intensity using a mix of the maximum
    precipitation rate in the period and the accumulation in the period. These
    classified periods are then used to determine what fraction of a
    constructed longer period would be classified as such. If the target
    period is the same as the native period of the inputs then the returned
    fractions will be binary.
    """

    def __init__(
            self,
            min_accumulation_per_hour: float,
            critical_rate: float,
            target_period: float,
            rate_diagnostic: str = "probability_of_lwe_precipitation_rate_above_threshold",
            accumulation_diagnostic: str = "probability_of_lwe_thickness_of_precipitation_amount_above_threshold",
        ) -> None:
        """
        Initialise the class.

        Args:
            min_accumulation_per_hour:
                The minimum accumulation per hour in the period to define that
                period as light, heavy, etc. To be classified as such the
                critical_rate threshold below must also be satisfied.
                Units of mm.
            critical_rate:
                A rate threshold, which if the maximum rate in the period
                is in excess of contributes to defining the period as being
                light, heavy, etc. Units of mm/hr.
            target_period:
                The period in hours that the final diagnostic represents.
                This should be equivalent to the period covered by the inputs.
                Specifying this explicitly here is entirely for purposes of
                checking that the returned diagnostic represents the period
                that is expected. Without this a missing input file could
                lead to a suddenly different overall period.
            rate_diagnostic:
                The expected diagnostic name for the maximum rate in period
                diagnostic.
            accumulation_diagnostic:
                The expected diagnostic name for the accumulation in period
                diagnostic.
        """
        self.min_accumulation_per_hour = min_accumulation_per_hour
        self.critical_rate = critical_rate
        self.target_period = target_period
        self.rate_diagnostic = rate_diagnostic
        self.accumulation_diagnostic = accumulation_diagnostic
        self.period = None

    def _period_in_hours(self, cubes: CubeList) -> None:
        """Get the periods of the input cubes, check they are equal and
        return the period in hours. Sets the period as a class variable.

        Args:
            cubes:
                A list of cubes.
        Raises:
            ValueError: If periods of all cubes are not equal.
        """
        periods = []
        for cube in cubes:
            cube_periods = [np.diff(ctime.bound)[0].seconds for ctime in cube.coord("time").cells()]
            periods.extend(set(cube_periods))

        try:
            period, = set(periods)
        except ValueError as err:
            raise ValueError(
                "Cube with multiple times with inconsistent periods. Cannot "
                f"return a single time period. Periods are: {period}."
                ) from err

        self.period = period / 3600

    def _construct_thresholds(self):
        """Converts the input threshold units to SI units to match the data.
        The accumulation threshold specified by the user is also mulitplied
        up by the input cube period. So a 0.1 mm accumulation threshold
        will become 0.3 mm for a 3-hour input cube, converted to metres.

        Args:
            period:
                Period of input cubes in hours.

        Returns:
            Tuple of float values, one the accumulation threshold, the other
            the rate threshold.
        """

        min_accumulation_per_hour = AuxCoord([self.min_accumulation_per_hour], units="mm")
        critical_rate = AuxCoord([self.critical_rate], units="mm/hr")
        min_accumulation_per_hour.convert_units("m")
        critical_rate.convert_units("m/s")

        return min_accumulation_per_hour.points[0] * self.period, critical_rate.points[0]

    def construct_constraints(self, accumulation_threshold: float, rate_threshold: float) -> Tuple[Constraint, Constraint]:
        """Construct constraints to use in extracting the accumulation and
        rate data, at relevant thresholds, from the input cubes.

        Args:
            accumulation_threshold:
                The accumulation threshold to be extracted.
            rate_threshold:
                The rate threshold to be extracted.
        Returns:
            Tuple containing an accumulation and rate constraint.
        """
        threshold_name = get_threshold_coord_name_from_probability_name(self.accumulation_diagnostic)
        accumulation_constraint = Constraint(
            self.accumulation_diagnostic,
            coord_values={threshold_name: lambda cell: np.isclose(cell.point, accumulation_threshold).any()}
        )
        threshold_name = get_threshold_coord_name_from_probability_name(self.rate_diagnostic)
        rate_constraint = Constraint(
            self.rate_diagnostic,
            coord_values={threshold_name: lambda cell: np.isclose(cell.point, rate_threshold).any()}
        )
        return accumulation_constraint, rate_constraint

    def _create_output_cube(self, data, cubes):

        classified_precip_cube = create_new_diagnostic_cube(
            "fraction_of_periods_classified_as_wet",
            "1",
            template_cube=cubes[0],
            mandatory_attributes=generate_mandatory_attributes(cubes),
            optional_attributes={
                "minimum_peak_rate (mm/hr)": self.critical_rate,
                "accumulation_minimum (mm)": np.around(self.min_accumulation_per_hour * self.period, decimals=3),
                "input_period_in_hours": self.period,
            },
            data=data,
        )
        return classified_precip_cube

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
        self._period_in_hours(cubes)
        accumulation_threshold, rate_threshold = self._construct_thresholds()
        accumulation_constraint, rate_constraint = self.construct_constraints(accumulation_threshold, rate_threshold)

        precip_accumulation = MergeCubes()(cubes.extract(accumulation_constraint))
        max_precip_rate = MergeCubes()(cubes.extract(rate_constraint))

        if not max_precip_rate.coord("time") == precip_accumulation.coord("time"):
            raise ValueError(
                "Precipitation accumulation and maximum rate in period cubes "
                "have differing time coordinates and cannot be used together."
            )

        total_period = (
            (
                max_precip_rate.coord("time").cell(-1).bound[-1] -
                max_precip_rate.coord("time").cell(0).bound[0]
            ).total_seconds() / 3600
        )
        if not total_period == self.target_period:
            raise ValueError(
                "Input cubes do not combine to create the expected target "
                "period. The period covered by the cubes passed in is: "
                f"{total_period} hours."
            )

        n_periods, = max_precip_rate.coord("time").shape
        short_period_classifications = max_precip_rate.copy(data=max_precip_rate.data * precip_accumulation.data)
        total_period = short_period_classifications.collapsed("time", analysis.SUM)
        total_period_data = total_period.data / n_periods

        classification_fractions = self._create_output_cube(total_period_data, (total_period, precip_accumulation))

        return classification_fractions





