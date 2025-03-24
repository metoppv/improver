# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Module containing the PrecipitationDuration class."""

import itertools
from numbers import Number
from typing import List, Optional, Tuple, Union

import numpy as np
from iris import Constraint
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube, CubeList
from iris.exceptions import CoordinateNotFoundError
from iris.util import new_axis, squeeze
from numpy import ndarray

from improver import PostProcessingPlugin
from improver.metadata.probabilistic import (
    get_threshold_coord_name_from_probability_name,
)
from improver.metadata.utilities import (
    enforce_time_point_standard,
    generate_mandatory_attributes,
)
from improver.utilities.common_input_handle import as_cubelist
from improver.utilities.cube_manipulation import (
    MergeCubes,
    enforce_coordinate_ordering,
)


class PrecipitationDuration(PostProcessingPlugin):
    """
    Classifies periods of precipitation intensity using a mix of the maximum
    precipitation rate in the period and the accumulation in the period. These
    classified periods are then used to determine what fraction of a
    constructed longer period would be classified as such. Percentiles are
    produced directly from these fractions via a frequency table. This is done
    to reduce the memory requirements when potentially combining so much data.
    The discrete nature of the fractions that are possible makes this table
    approach possible.

    Note that we apply linear interpolation in the percentile generation,
    meaning that fractions of the target period that are not multiples of the
    input data period can be returned.

    .. See the documentation for a more detailed discussion of this plugin.
    .. include:: extended_documentation/precipitation/
        precipitation_duration.rst
    """

    def __init__(
        self,
        min_accumulation_per_hour: float,
        critical_rate: float,
        target_period: float,
        percentiles: Union[List[float], ndarray],
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
                used in conjunction with the critical rate.
                Units of mm.
            critical_rate:
                A rate threshold, or list of rate thresholds, which if the
                maximum rate in the period is in excess of contributes to
                classifying the period. Units of mm/hr.
            target_period:
                The period in hours that the final diagnostic represents.
                This should be equivalent to the period covered by the inputs.
                Specifying this explicitly here is entirely for the purpose of
                checking that the returned diagnostic represents the period
                that is expected. Without this a missing input file could
                lead to a suddenly different overall period.
            percentiles:
                A list of percentile values to be returned.
            accumulation_diagnostic:
                The expected diagnostic name for the accumulation in period
                diagnostic. Used to extract the cubes from the inputs.
            rate_diagnostic:
                The expected diagnostic name for the maximum rate in period
                diagnostic. Used to extract the cubes from the inputs.
            model_id_attr:
                The name of the dataset attribute to be used to identify the source
                model when blending data from different models.
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
        self.acc_threshold = get_threshold_coord_name_from_probability_name(
            self.accumulation_diagnostic
        )
        self.rate_threshold = get_threshold_coord_name_from_probability_name(
            self.rate_diagnostic
        )
        self.percentiles = np.sort(np.array(percentiles, dtype=np.float32))

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

    def _construct_thresholds(self) -> Tuple[ndarray, ndarray]:
        """Converts the input threshold units to SI units to match the data.
        The accumulation threshold specified by the user is also multiplied
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

    @staticmethod
    def _construct_constraint(
        diagnostic: str, threshold_name: str, threshold_values: ndarray
    ) -> Constraint:
        """Construct constraint to use in extracting the accumulation and
        rate data, at relevant thresholds, from the input cubes.

        Args:
            diagnostic:
                The name of the diagnostic to be extracted.
            threshold_name:
                The name of the threshold from which to extract data.
            threshold_values:
                The threshold values to extract.
        Returns:
            Iris constraint to extract the target diagnostic and threshold
            values.
        """
        diagnostic_constraint = Constraint(
            diagnostic,
            coord_values={
                threshold_name: lambda cell: np.isclose(
                    cell.point, threshold_values
                ).any()
            },
        )
        return diagnostic_constraint

    def _extract_cubes(
        self,
        cubes: CubeList,
        diagnostic: str,
        threshold_name: str,
        threshold_values: ndarray,
    ) -> Cube:
        """Extract target diagnostics and thresholds from the input cube list.
        Merge these cubes into a single cube with a multi-valued time
        dimension.

        Args:
            cubes:
                The input cubes from which to extract targets.
            diagnostic:
                The diagnostic name.
            threshold_name:
                Name of the threshold coordinate from which to extract specific
                values.
            threshold_values:
                The threshold values to extract.
        Raises:
            ValueError: If the target diagnostic or thresholds are not found.
        Returns:
            cube:
                The merged cube containing the target thresholds.
        """
        diagnostic_constraint = self._construct_constraint(
            diagnostic, threshold_name, threshold_values
        )

        try:
            cube = MergeCubes()(cubes.extract(diagnostic_constraint))
        except IndexError:
            msg = (
                "The requested diagnostic or threshold is not available. "
                f"Requested diagnostic: {diagnostic}, threshold: {threshold_values}"
            )
            raise ValueError(msg)
        return cube

    @staticmethod
    def _structure_inputs(cube: Cube) -> Tuple[Cube, DimCoord]:
        """Ensure threshold and time coordinates are promoted to be dimensions
        of the cube. Ensure the cube is ordered with realization, threshold,
        and time as the three leading dimensions.

        Args:
            cube:
                The cube to be structured.
        Returns:
            Tuple containing:
                The reordered cube and the threshold coordinate.
        """
        # Pick out the threshold coordinates in the inputs.
        thresh = cube.coord(var_name="threshold")
        # Promote any scalar threshold coordinates to dimensions so that the
        # array shapes are as expected.
        if thresh not in cube.coords(dim_coords=True):
            cube = new_axis(cube, thresh)
        # Likewise, ensure the time coordinate is a dimension coordinate in
        # case a period equal to the target period is passed in.
        time_crd = cube.coord("time")
        if time_crd not in cube.coords(dim_coords=True):
            cube = new_axis(cube, time_crd)
        # Enforce a coordinate order so we know which dimensions are which
        # when indexing the data arrays.
        enforce_coordinate_ordering(cube, ["realization", thresh.name(), "time"])
        return cube, thresh

    def _create_output_cube(
        self,
        data: ndarray,
        cubes: Tuple[Cube, Cube],
        acc_thresh: DimCoord,
        rate_thresh: DimCoord,
        spatial_dim_indices,
    ) -> Cube:
        """Create an output cube for the final diagnostic, which is the
        set of percentiles describing the fraction of the constructed
        period that has been classified as wet using the various accumulation
        and peak rate thresholds.

        Args:
            data:
                The percentile data to be stored in the cube.
            cubes:
                Cubes from which to generate attributes, and the first of
                which is used as a template from which to take time and
                spatial coordinates.
            acc_thresh:
                The accumulation threshold to add to the cube.
            rate_thresh:
                The rate threshold to add to the cube.
            spatial_dim_indices:
                Indices of spatial coordinates on the the input cubes.
                This allows us to select the correct dimensions when handling
                either gridded or spot inputs.
        Returns:
            A Cube with a suitable name and metadata for the diagnostic data
            stored within it.
        """
        percentile_coord = DimCoord(self.percentiles, long_name="percentile", units="%")

        mandatory_attributes = generate_mandatory_attributes(
            cubes, model_id_attr=self.model_id_attr
        )
        mandatory_attributes["precipitation_sampling_period_in_hours"] = self.period

        # Add more descriptive long names to accumulation and rate thresholds.
        acc_thresh.long_name = "precipitation_accumulation_threshold_for_wet"
        rate_thresh.long_name = "precipitation_rate_threshold_for_wet"
        # Remove var_names as working with dual threshold coordinates.
        # Iris will otherwise suffix these, e.g. threshold_0.
        acc_thresh.var_name = None
        rate_thresh.var_name = None

        dim_coords = [
            (percentile_coord, 0),
            (acc_thresh.copy(), 1),
            (rate_thresh.copy(), 2),
        ]
        for index in spatial_dim_indices:
            dim_coords.append(
                (cubes[0].coords(dim_coords=True)[index], len(dim_coords))
            )
        # Add any aux coords associated with dim coords.
        aux_coords = [
            (crd, index)
            for index in spatial_dim_indices
            for crd in cubes[0].aux_coords
            if index in crd.cube_dims(cubes[0])
        ]

        classification_percentiles = Cube(
            data.astype(np.float32),
            long_name="fraction_of_time_classified_as_wet",
            dim_coords_and_dims=dim_coords,
            aux_coords_and_dims=aux_coords,
            units="1",
            attributes=mandatory_attributes,
        )
        # The time dimension has been collapsed within the calculation.
        # Here we collapse the coordinate from one of the inputs to apply to
        # the output to match the data manipulation.
        time_coords = [cubes[0].coord("forecast_reference_time").copy()]
        for crd in ["time", "forecast_period"]:
            time_crd = cubes[0].coord(crd).copy()
            time_coords.append(time_crd.collapsed())

        for crd in time_coords:
            classification_percentiles.add_aux_coord(crd)
        enforce_time_point_standard(classification_percentiles)

        return squeeze(classification_percentiles)

    def _calculate_fractions(
        self,
        precip_accumulation,
        max_precip_rate,
        acc_thresh,
        rate_thresh,
        realizations,
        spatial_dims,
    ):
        """Calculate the fractions of the target period that are classified as
        wet using the given accumulation and peak rate thresholds. This is
        done by counting the number of periods in which the thresholds are
        exceeded out of the total number possible. Frequency tables are
        constructed and from these percentiles are calculated.

        Args:
            precip_accumulation:
                Precipitation accumulation in a period.
            max_precip_rate:
                Maximum preciptation rate in a period.
            acc_thresh:
                Accumulation threshold coordinate.
            rate_thresh:
                Max rate threshold coordinate.
            realizations:
                A list of realization indices to loop over.
            spatial_dims:
                A list containing the indices of the spatial coordinates on the
                max_precip_rate cube, which are the same as for the
                precip_accumulation cube.
        Raises:
            ValueError: If the input data is masked.
        Returns:
            An array containing the generated percentiles.
        """
        # Get the lengths of the threshold arrays, after extraction, for use
        # as looping counts and for creating a target array shape.
        acc_len = acc_thresh.shape[0]
        rate_len = rate_thresh.shape[0]

        # Generate the combinations of the threshold values.
        combinations = list(
            itertools.product(list(range(acc_len)), list(range(rate_len)))
        )

        # Determine the possible counts that can be achieved which is simply
        # the length of the input time coordinates.
        (n_periods,) = max_precip_rate.coord("time").shape
        possible_period_counts = list(range(0, n_periods + 1))

        # We've ensured that the periods combine to give the target_period.
        # The fractions of that total period that can be returned are
        # therefore simply 0-1 in increments of 1/n_periods.
        fractions = np.arange(0, 1.0001, 1 / n_periods)

        # The percentile_rank_fractions are the target percentiles rescaled to match
        # the number of realizations over which we are counting.
        percentile_rank_fractions = self.percentiles * (len(realizations) - 1) / 100

        # We create an empty array into which to put our resulting percentiles.
        # We can index this with the accumulation and rate threshold indices
        # to ensure we record the data where we expect.
        generated_percentiles = np.empty(
            (len(self.percentiles), acc_len, rate_len, *spatial_dims)
        )

        hit_count = np.zeros(
            (acc_len, rate_len, n_periods + 1, *spatial_dims),
            dtype=np.int8,
        )
        for realization in realizations:
            # Realize the data to reduce overhead of lazy loading smaller
            # slices which comes to dominate the time with many small slices.
            acc_realized = precip_accumulation[realization].data.astype(bool)
            rate_realized = max_precip_rate[realization].data.astype(bool)
            # Check for masked data and raise exception if present.
            if np.ma.is_masked(acc_realized) or np.ma.is_masked(rate_realized):
                raise ValueError(
                    "Precipitation duration plugin cannot handle masked data."
                )

            for acc_index, rate_index in combinations:
                # Multiply the binary probabilities and then sum over the
                # leading time dimension to count how many of the times
                # precipitation is classified as exceeding both thresholds.
                result = acc_realized[acc_index] * rate_realized[rate_index]
                result = np.sum(result, axis=0)

                for value in possible_period_counts:
                    hit_count[acc_index, rate_index, value, result == value] += 1

        for acc_index, rate_index in combinations:
            # We accumulate the counts over the possible values. The resulting
            # array contains monotonically increasing counts that we can use
            # to determine where each target percentile falls in the possible
            # values.
            cumulated = np.cumsum(hit_count[acc_index, rate_index], axis=0)
            resulting_percentiles = []
            for percentile_rank_fraction in percentile_rank_fractions:
                # Find the value below and above the target percentile and
                # apply linear interpolation to determine the percentile value
                percentile_indices_lower = (
                    cumulated <= np.floor(percentile_rank_fraction)
                ).sum(axis=0)
                percentile_indices_upper = (
                    cumulated <= np.ceil(percentile_rank_fraction)
                ).sum(axis=0)

                interp_fraction = percentile_rank_fraction - np.floor(
                    percentile_rank_fraction
                )

                percentile_values = fractions[
                    percentile_indices_lower
                ] + interp_fraction * (
                    fractions[percentile_indices_upper]
                    - fractions[percentile_indices_lower]
                )
                resulting_percentiles.append(percentile_values)

            resulting_percentiles = np.array(resulting_percentiles)
            # Record the percentiles generated for this accumulation and rate
            # threshold combination.
            generated_percentiles[:, acc_index, rate_index] = resulting_percentiles

        return generated_percentiles

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

        Args:
            cubes:
                Cubes covering the expected period that include cubes of:
                    precip_accumulation: Precipitation accumulation in a period.
                    max_precip_rate: Maximum preciptation rate in a period.
        Returns:
            A cube of percentiles of the fraction of the target period that is
            classified as exceeding the user specified thresholds.
        Raises:
            ValueError: If the input cubes have differing time coordinates.
            ValueError: If the input cubes do not combine to create the expected
                        target period.
            ValueError: If the input cubes lack a realization coordinate.
            ValueError: If the input cubes have differing realization coordinates.
        """
        cubes = as_cubelist(*cubes)
        self._period_in_hours(cubes)
        accumulation_threshold, rate_threshold = self._construct_thresholds()

        precip_accumulation = self._extract_cubes(
            cubes,
            self.accumulation_diagnostic,
            self.acc_threshold,
            accumulation_threshold,
        )
        max_precip_rate = self._extract_cubes(
            cubes, self.rate_diagnostic, self.rate_threshold, rate_threshold
        )

        if max_precip_rate.coord("time") != precip_accumulation.coord("time"):
            raise ValueError(
                "Precipitation accumulation and maximum rate in period cubes "
                "have differing time coordinates and cannot be used together."
            )

        # Check input cubes combine to create the expected target period.
        total_period = (
            max_precip_rate.coord("time").cell(-1).bound[-1]
            - max_precip_rate.coord("time").cell(0).bound[0]
        ).total_seconds() / 3600
        if total_period != self.target_period:
            raise ValueError(
                "Input cubes do not combine to create the expected target "
                "period. The period covered by the cubes passed in is: "
                f"{total_period} hours. Target is {self.target_period} hours."
            )

        # Ensure cubes have the expected dimensions and order.
        precip_accumulation, acc_thresh = self._structure_inputs(precip_accumulation)
        max_precip_rate, rate_thresh = self._structure_inputs(max_precip_rate)

        # Determine size of spatial dimensions so we can work with gridded
        # or spot files.
        spatial_dim_indices = sorted(
            set(
                [
                    max_precip_rate.coord_dims(max_precip_rate.coord(axis=axis))[0]
                    for axis in "xy"
                ]
            )
        )
        spatial_dims = [max_precip_rate.shape[dim] for dim in spatial_dim_indices]

        # Check that a multi-valued realization coordinate is present on the inputs.
        try:
            realizations = list(range(max_precip_rate.coord("realization").shape[0]))
            if len(realizations) == 1:
                raise CoordinateNotFoundError
        except CoordinateNotFoundError as err:
            raise ValueError(
                "This plugin requires input data from multiple realizations."
                "Percentiles are generated by collapsing this coordinate. It "
                "cannot therefore be used with deterministic data or input "
                "with a realization coordinate of length 1."
            ) from err

        # Check realizations are the same as we intend to loop over a counting
        # index and must ensure we are matching the same realizations together.
        if precip_accumulation.coord("realization") != max_precip_rate.coord(
            "realization"
        ):
            raise ValueError(
                "Mismatched realization coordinates between accumulation and "
                "max rate inputs. These must be the same."
            )

        generated_percentiles = self._calculate_fractions(
            precip_accumulation,
            max_precip_rate,
            acc_thresh,
            rate_thresh,
            realizations,
            spatial_dims,
        )

        # Store the generated percentiles in a cube with suitable metadata.
        classification_percentiles = self._create_output_cube(
            generated_percentiles,
            (max_precip_rate, precip_accumulation),
            acc_thresh,
            rate_thresh,
            spatial_dim_indices,
        )
        return classification_percentiles
