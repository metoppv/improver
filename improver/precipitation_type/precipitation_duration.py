# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Module containing the PrecipitationDuration class."""

from numbers import Number
from typing import Tuple, Union

import numpy as np
from iris import Constraint, analysis
from iris.coords import AuxCoord
from iris.cube import Cube, CubeList
from numpy import ndarray

from improver import PostProcessingPlugin
from improver.metadata.probabilistic import (
    get_threshold_coord_name_from_probability_name,
)
from improver.metadata.utilities import (
    create_new_diagnostic_cube,
    generate_mandatory_attributes,
)
from improver.utilities.common_input_handle import as_cubelist
from improver.utilities.cube_manipulation import MergeCubes, collapse_time


class PrecipitationDuration(PostProcessingPlugin):
    """
    Classifies periods of precipitation intensity using a mix of the maximum
    precipitation rate in the period and the accumulation in the period. These
    classified periods are then used to determine what fraction of a
    constructed longer period would be classified as such.
    """

    def __init__(
        self,
        min_accumulation_per_hour: float,
        critical_rate: float,
        target_period: float,
        accumulation_diagnostic: str = "probability_of_lwe_thickness_of_precipitation_amount_above_threshold",
        rate_diagnostic: str = "probability_of_lwe_precipitation_rate_above_threshold",
    ) -> None:
        """
        Initialise the class.

        Args:
            min_accumulation_per_hour:
                The minimum accumulation per hour in the period, or a list
                of several, used to classify the period. The accumulation is
                used in conjuction wuth the critical rate.
                Units of mm.
            critical_rate:
                A rate threshold, or list of rate thresholds, which if the
                maximum rate in the period is in excess of contributes to
                classifying the period. Units of mm/hr.
            target_period:
                The period in hours that the final diagnostic represents.
                This should be equivalent to the period covered by the inputs.
                Specifying this explicitly here is entirely for purposes of
                checking that the returned diagnostic represents the period
                that is expected. Without this a missing input file could
                lead to a suddenly different overall period.
            accumulation_diagnostic:
                The expected diagnostic name for the accumulation in period
                diagnostic. Used to extract the cubes from the inputs.
            rate_diagnostic:
                The expected diagnostic name for the maximum rate in period
                diagnostic. Used to extract the cubes from the inputs.
        """
        if isinstance(min_accumulation_per_hour, Number):
            min_accumulation_per_hour = [min_accumulation_per_hour]
        min_accumulation_per_hour = [float(x) for x in min_accumulation_per_hour]

        if isinstance(critical_rate, Number):
            critical_rate = [critical_rate]
        critical_rate = [float(x) for x in critical_rate]

        self.min_accumulation_per_hour = min_accumulation_per_hour
        self.critical_rate = critical_rate
        self.target_period = target_period
        self.rate_diagnostic = rate_diagnostic
        self.accumulation_diagnostic = accumulation_diagnostic
        self.period = None
        self.acc_threshold = None
        self.rate_threshold = None

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
            cube_periods = [
                np.diff(ctime.bound)[0].seconds for ctime in cube.coord("time").cells()
            ]
            periods.extend(set(cube_periods))

        try:
            (period,) = set(periods)
        except ValueError as err:
            raise ValueError(
                "Cubes with inconsistent periods. Cannot return a single "
                f"time period. Periods are: {periods}."
            ) from err

        self.period = period / 3600

    def _construct_thresholds(self):
        """Converts the input threshold units to SI units to match the data.
        The accumulation threshold specified by the user is also mulitplied
        up by the input cube period. So a 0.1 mm accumulation threshold
        will become 0.3 mm for a 3-hour input cube, converted to metres.

        Returns:
            Tuple of float values, one the accumulation threshold, the other
            the rate threshold.
        """

        min_accumulation_per_hour = AuxCoord(self.min_accumulation_per_hour, units="mm")
        critical_rate = AuxCoord(self.critical_rate, units="mm/hr")
        min_accumulation_per_hour.convert_units("m")
        critical_rate.convert_units("m/s")

        return min_accumulation_per_hour.points * self.period, critical_rate.points

    def _construct_constraints(
        self, accumulation_threshold: float, rate_threshold: float
    ) -> Tuple[Constraint, Constraint]:
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
        self.acc_threshold = get_threshold_coord_name_from_probability_name(
            self.accumulation_diagnostic
        )
        accumulation_constraint = Constraint(
            self.accumulation_diagnostic,
            coord_values={
                self.acc_threshold: lambda cell: np.isclose(
                    cell.point, accumulation_threshold
                ).any()
            },
        )
        self.rate_threshold = get_threshold_coord_name_from_probability_name(
            self.rate_diagnostic
        )
        rate_constraint = Constraint(
            self.rate_diagnostic,
            coord_values={
                self.rate_threshold: lambda cell: np.isclose(
                    cell.point, rate_threshold
                ).any()
            },
        )
        return accumulation_constraint, rate_constraint

    def _create_output_cube(self, data: ndarray, cubes: Tuple[Cube, Cube]) -> Cube:
        """Create an output cube for the final diagnostic, which is the
        fraction of the constructed period that has been classified as wet
        using the various accumulation and peak rate thresholds.

        Args:
            data:
                The data to be stored in the cube.
            cubes:
                Cubes from which to generate attributes, and the first of
                which is used as a template from which to take coordinates
                etc.
        Returns:
            A Cube with a suitable name and metadata for the diagnostic data
            stored within it.
        """
        classified_precip_cube = create_new_diagnostic_cube(
            "fraction_of_period_classified_as_wet",
            "1",
            template_cube=cubes[0],
            mandatory_attributes=generate_mandatory_attributes(cubes),
            optional_attributes={
                "input_period_in_hours": self.period,
            },
            data=data,
        )
        # Remove any coordinate var names that identify a coordinate as a
        # threshold. We want to generate percentiles by collapsing the
        # realization coordinate, not attempting to use ECC for a probability
        # diagnostic.
        for crd in classified_precip_cube.coords():
            if crd.var_name == "threshold":
                crd.var_name = None

        return classified_precip_cube

    def process(self, *cubes: Union[Cube, CubeList]) -> Cube:
        """Produce a diagnostic that provides the fraction of the day that
        can be classified as exceeding the given thresholds. The fidelity of
        the returned product depends upon the period of the input cubes.

        Args:
            cubes:
                Cubes covering the expected period that include cubes of:
                    max_precip_rate: Maximum preciptation rate in a period.
                    precip_accumulation: Precipitation accumulation in a period.

        Returns:
            A cube of fraction of the target period that is classified as
            exceeding the user specified thresholds. The short period inputs
            are compared against these thresholds to be classified before
            determining how many such periods (what fraction) of the target
            period has been so classified.

        Raises:
            ValueError: If input cubes do not contain the expected diagnostics
                        or diagnostic thresholds.
            ValueError: If the input cubes have differing time coordinates.
            ValueError: If the input cubes do not combine to create the expected
                        target period.
        """
        cubes = as_cubelist(*cubes)
        self._period_in_hours(cubes)
        accumulation_threshold, rate_threshold = self._construct_thresholds()
        accumulation_constraint, rate_constraint = self._construct_constraints(
            accumulation_threshold, rate_threshold
        )

        try:
            precip_accumulation = MergeCubes()(cubes.extract(accumulation_constraint))
            max_precip_rate = MergeCubes()(cubes.extract(rate_constraint))
        except IndexError:
            raise ValueError(
                "Input cubes do not contain the expected diagnostics or thresholds."
            )

        if not max_precip_rate.coord("time") == precip_accumulation.coord("time"):
            raise ValueError(
                "Precipitation accumulation and maximum rate in period cubes "
                "have differing time coordinates and cannot be used together."
            )

        total_period = (
            max_precip_rate.coord("time").cell(-1).bound[-1]
            - max_precip_rate.coord("time").cell(0).bound[0]
        ).total_seconds() / 3600
        if not total_period == self.target_period:
            raise ValueError(
                "Input cubes do not combine to create the expected target "
                "period. The period covered by the cubes passed in is: "
                f"{total_period} hours. Target is {self.target_period} hours."
            )

        (n_periods,) = max_precip_rate.coord("time").shape
        classifications = CubeList()
        # Slice over all thresholds and combine to build all the classification
        # combinations.
        for acc_slice in precip_accumulation.slices_over(self.acc_threshold):
            for rate_slice in max_precip_rate.slices_over(self.rate_threshold):
                # Use the rate slice as a template and add the accumulation
                # threshold coordinate so that an accumulation threshold
                # coordinate can be reconstructed on merging the cube list.
                # The cube will end up with two threshold coordinates.
                acc_coord = acc_slice.coord(self.acc_threshold).copy()
                # All thresholds on the final output are given relative to a
                # 1-hour period.
                acc_coord.points = np.around(acc_coord.points / self.period, decimals=6)

                # We are working with binary probabilities, so make these bool
                # type to reduce memory usage.
                classified = rate_slice.copy(data=rate_slice.data.astype(bool) * acc_slice.data.astype(bool))
                classified.add_aux_coord(acc_coord)
                classifications.append(classified)

        classifications = classifications.merge_cube()
        total_period = collapse_time(classifications, "time", analysis.SUM)
        total_period_data = total_period.data.astype(np.float32) / n_periods

        classification_fractions = self._create_output_cube(
            total_period_data, (total_period, precip_accumulation)
        )

        return classification_fractions
