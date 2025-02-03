# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Module containing the PrecipitationDuration class."""

from numbers import Number
from typing import Optional, Tuple, Union

import numpy as np
from iris import Constraint, analysis
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube, CubeList
from iris.util import new_axis, squeeze
from numpy import ndarray
import itertools

from improver import PostProcessingPlugin
from improver.metadata.probabilistic import (
    get_threshold_coord_name_from_probability_name,
)
from improver.metadata.utilities import (
    create_new_diagnostic_cube,
    generate_mandatory_attributes,
)
from improver.utilities.common_input_handle import as_cubelist
from improver.utilities.cube_manipulation import MergeCubes, collapse_time, enforce_coordinate_ordering


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
        model_id_attr: Optional[str] = None,
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
        self.model_id_attr = model_id_attr
        self.period = None
        self.acc_threshold = None
        self.rate_threshold = None
        self.percentiles = np.array([5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 85, 90, 95])

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

    def _create_output_cube(self, data: ndarray, cubes: Tuple[Cube, Cube], acc_thresh, rate_thresh) -> Cube:
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
        percentile_coord = DimCoord(self.percentiles, long_name="percentile", units="%")

        mandatory_attributes = generate_mandatory_attributes(
            cubes,
            model_id_attr=self.model_id_attr
        )
        mandatory_attributes["input_period_in_hours"] = self.period

        time_coords = [cubes[0].coord("forecast_reference_time").copy()]
        for crd in ["time", "forecast_period"]:
            time_crd = cubes[0].coord(crd).copy()
            time_coords.append(time_crd.collapsed())

        classification_percentiles = Cube(
            data.astype(np.float32),
            dim_coords_and_dims=[
                (percentile_coord, 0),
                (acc_thresh.copy(), 1),
                (rate_thresh.copy(), 2),
                (cubes[0].coord(axis="y"), 3),
                (cubes[0].coord(axis="x"), 4),
            ],
            # aux_coords_and_dims=max_precip_rate.aux_coords,
            units="1",
            attributes=mandatory_attributes,
        )
        for crd in time_coords:
            classification_percentiles.add_aux_coord(crd)
        # Remove any coordinate var names that identify a coordinate as a
        # threshold. We want to generate percentiles by collapsing the
        # realization coordinate, not attempting to use ECC for a probability
        # diagnostic.
        for crd in classification_percentiles.coords():
            if crd.var_name == "threshold":
                crd.var_name = None

        return squeeze(classification_percentiles)

    def process(self, *cubes: Union[Cube, CubeList]) -> Cube:
        """Produce a diagnostic that provides the fraction of the day that
        can be classified as exceeding the given thresholds. The fidelity of
        the returned product depends upon the period of the input cubes.

        This plugin can process an awful lot of data, particularly if
        dealing with a large ensemble. To reduce the memory used, given the
        discrete nature of the period fractions produced, frequency bins are
        used to directly produce percentiles. This means that not all of the
        values need to be held in memory at once, instead we count instances
        of each value and then calculate the percentiles from the counts.
        Unfortunately this method is more convoluted.

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

        # Pick out the threshold coordinates in the inputs.
        acc_thresh = precip_accumulation.coord(var_name="threshold")
        rate_thresh = max_precip_rate.coord(var_name="threshold")

        # Promote any scalar threshold coordinates to dimensions so that the
        # array shapes are as expected.
        if acc_thresh not in precip_accumulation.coords(dim_coords=True):
            precip_accumulation = new_axis(precip_accumulation, acc_thresh)

        if rate_thresh not in max_precip_rate.coords(dim_coords=True):
            max_precip_rate = new_axis(max_precip_rate, rate_thresh)

        # Enforce a coordinate order so we know which dimensions are which
        # when indexing the data arrays.
        enforce_coordinate_ordering(precip_accumulation, ["realization", acc_thresh.name(), "time"])
        enforce_coordinate_ordering(max_precip_rate, ["realization", rate_thresh.name(), "time"])
        # Get the lengths of the threshold arrays, after extraction, for use
        # as looping counts and for creating a target array shape.
        acc_len = acc_thresh.shape[0]
        rate_len = rate_thresh.shape[0]
        # Generate the combinations of the threshold values.
        combinations = list(itertools.product(list(range(acc_len)), list(range(rate_len))))
        realizations = list(range(max_precip_rate.coord("realization").shape[0]))
        # Determine the possible counts that can be achieved which is simply
        # the length of the input time coordinates.
        (n_periods,) = max_precip_rate.coord("time").shape
        possible_values=list(range(0, n_periods + 1))
        # We've ensured that the periods combine to give the target_period.
        # The fractions of that total period that can be returned are
        # therefore simply 0-1 in increments of 1/n_periods.
        fractions = np.arange(0, 1.0001, 1/n_periods)
        # The lookup_percentiles are the target percentiles rescaled to match
        # the number of realizations over which we are counting.
        lookup_percentiles = self.percentiles * (len(realizations) - 1) / 100
        # We create an empty array into which to put our resulting percentiles.
        # We can index this with the accumulation and rate threshold indices
        # to ensure we record the data where we expect.
        generated_percentiles = np.empty((len(self.percentiles), acc_len, rate_len, *max_precip_rate.shape[-2:]))

        for acc_index, rate_index in combinations:
            hit_count = np.zeros((len(possible_values), *max_precip_rate.shape[-2:]), dtype=np.int8)
            for realization in realizations:
                # Mulitply the binary probabilities and then sum over the
                # leading time dimension to count how many of the times have
                # precipitation classified as exceeding both thresholds.
                result = precip_accumulation[realization, acc_index].data.astype(bool) * max_precip_rate[realization, rate_index].data.astype(bool)
                result = np.sum(result, axis=0)
                # Index here corresponds to the possible value, whilst the
                # y,x positions records the counts of the value.
                for index, value in enumerate(possible_values):
                    hit_count[index, result == value] += 1

            # We accumulate the counts over the possible values. The resulting
            # array contains monotonically increasing counts that we can use
            # to determine where each target percentile falls in the possible
            # values.
            cumulated = np.cumsum(hit_count, axis=0)
            resulting_percentiles = []
            for percentile in lookup_percentiles:
                # Find the value below and above the target percentile and
                # apply linear interpolation to determine the percentile value
                percentile_indices_lower = (cumulated <= np.floor(percentile)).sum(axis=0)
                percentile_indices_upper = (cumulated <= np.ceil(percentile)).sum(axis=0)
                interp_fraction = percentile - np.floor(percentile)

                percentile_values = (
                    fractions[percentile_indices_lower] + interp_fraction *
                    (fractions[percentile_indices_upper] - fractions[percentile_indices_lower])
                )
                resulting_percentiles.append(percentile_values)

            resulting_percentiles = np.array(resulting_percentiles)
            # Record the percentiles generated for this accumulation and rate
            # threshold combination.
            generated_percentiles[:, acc_index, rate_index] = resulting_percentiles

        # Store the generated percentiles in a cube with suitable metadata.
        classification_percentiles = self._create_output_cube(
            generated_percentiles, (max_precip_rate, precip_accumulation), acc_thresh, rate_thresh
        )
        return classification_percentiles
