# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2020 Met Office.
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
"""Reliability calibration plugins."""

import warnings

import iris
import numpy as np
import scipy

from improver import BasePlugin, PostProcessingPlugin
from improver.calibration.utilities import (
    check_forecast_consistency,
    create_unified_frt_coord,
    filter_non_matching_cubes,
)
from improver.metadata.probabilistic import find_threshold_coordinate
from improver.metadata.utilities import generate_mandatory_attributes
from improver.utilities.cube_manipulation import MergeCubes, collapsed


class ConstructReliabilityCalibrationTables(BasePlugin):

    """A plugin for creating and populating reliability calibration tables."""

    def __init__(
        self,
        n_probability_bins=5,
        single_value_lower_limit=False,
        single_value_upper_limit=False,
    ):
        """
        Initialise class for creating reliability calibration tables. These
        tables include data columns entitled observation_count,
        sum_of_forecast_probabilities, and forecast_count, defined below.

        n_probability_bins (int):
            The total number of probability bins required in the reliability
            tables. If single value limits are turned on, these are included in
            this total.
        single_value_lower_limit (bool):
            Mandates that the lowest bin should be single valued,
            with a small precision tolerance, defined as 1.0E-6.
            The bin is thus 0 to 1.0E-6.
        single_value_upper_limit (bool):
            Mandates that the highest bin should be single valued,
            with a small precision tolerance, defined as 1.0E-6.
            The bin is thus (1 - 1.0E-6) to 1.
        """
        self.single_value_tolerance = 1.0e-6
        self.probability_bins = self._define_probability_bins(
            n_probability_bins, single_value_lower_limit, single_value_upper_limit
        )
        self.table_columns = np.array(
            ["observation_count", "sum_of_forecast_probabilities", "forecast_count"]
        )
        self.expected_table_shape = (len(self.table_columns), n_probability_bins)

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        bin_values = ", ".join(
            ["[{:1.2f} --> {:1.2f}]".format(*item) for item in self.probability_bins]
        )
        result = "<ConstructReliabilityCalibrationTables: " "probability_bins: {}>"
        return result.format(bin_values)

    def _define_probability_bins(
        self, n_probability_bins, single_value_lower_limit, single_value_upper_limit
    ):
        """
        Define equally sized probability bins for use in a reliability table.
        The range 0 to 1 is divided into ranges to give n_probability bins.
        If single_value_lower_limit and / or single_value_upper_limit are True,
        additional bins corresponding to values of 0 and / or 1 will be created,
        each with a width defined by self.single_value_tolerance.

        Args:
            n_probability_bins (int):
                The total number of probability bins desired in the
                reliability tables. This number includes the extrema bins
                (equals 0 and equals 1) if single value limits are turned on,
                in which case the minimum number of bins is 3.
            single_value_lower_limit (bool):
                Mandates that the lowest bin should be single valued,
                with a small precision tolerance, defined as 1.0E-6.
                The bin is thus 0 to 1.0E-6.
            single_value_upper_limit (bool):
                Mandates that the highest bin should be single valued,
                with a small precision tolerance, defined as 1.0E-6.
                The bin is thus (1 - 1.0E-6) to 1.
        Returns:
            numpy.ndarray:
                An array of 2-element arrays that contain the bounds of the
                probability bins. These bounds are non-overlapping, with
                adjacent bin boundaries spaced at the smallest representable
                interval.
        Raises:
            ValueError: If trying to use both single_value_lower_limit and
                        single_value_upper_limit with 2 or fewer probability bins.
        """
        if single_value_lower_limit and single_value_upper_limit:
            if n_probability_bins <= 2:
                msg = (
                    "Cannot use both single_value_lower_limit and "
                    "single_value_upper_limit with 2 or fewer "
                    "probability bins."
                )
                raise ValueError(msg)
            n_probability_bins = n_probability_bins - 2
        elif single_value_lower_limit or single_value_upper_limit:
            n_probability_bins = n_probability_bins - 1

        bin_lower = np.linspace(0, 1, n_probability_bins + 1, dtype=np.float32)
        bin_upper = np.nextafter(bin_lower, 0, dtype=np.float32)
        bin_upper[-1] = 1.0
        bins = np.stack([bin_lower[:-1], bin_upper[1:]], 1).astype(np.float32)

        if single_value_lower_limit:
            bins[0, 0] = np.nextafter(self.single_value_tolerance, 1, dtype=np.float32)
            lowest_bin = np.array([0, self.single_value_tolerance], dtype=np.float32)
            bins = np.vstack([lowest_bin, bins]).astype(np.float32)

        if single_value_upper_limit:
            bins[-1, 1] = np.nextafter(
                1.0 - self.single_value_tolerance, 0, dtype=np.float32
            )
            highest_bin = np.array(
                [1.0 - self.single_value_tolerance, 1], dtype=np.float32
            )
            bins = np.vstack([bins, highest_bin]).astype(np.float32)

        return bins

    def _create_probability_bins_coord(self):
        """
        Construct a dimension coordinate describing the probability bins
        of the reliability table.

        Returns:
            iris.coord.DimCoord:
                A dimension coordinate describing probability bins.
        """
        values = np.mean(self.probability_bins, axis=1, dtype=np.float32)
        probability_bins_coord = iris.coords.DimCoord(
            values, long_name="probability_bin", units=1, bounds=self.probability_bins
        )
        return probability_bins_coord

    def _create_reliability_table_coords(self):
        """
        Construct coordinates that describe the reliability table rows. These
        are observation_count, sum_of_forecast_probabilities, and
        forecast_count. The order used here is the order in which the table
        data is populated, so these must remain consistent with the
        _populate_reliability_bins function.

        Returns:
            (tuple): tuple containing:
                **index_coord** (iris.coord.DimCoord):
                    A numerical index dimension coordinate.
                **name_coord** (iris.coord.AuxCoord):
                    An auxiliary coordinate that assigns names to the index
                    coordinates, where these names correspond to the
                    reliability table rows.
        """
        index_coord = iris.coords.DimCoord(
            np.arange(len(self.table_columns), dtype=np.int32),
            long_name="table_row_index",
            units=1,
        )
        name_coord = iris.coords.AuxCoord(
            self.table_columns, long_name="table_row_name", units=1
        )
        return index_coord, name_coord

    @staticmethod
    def _define_metadata(forecast_slice):
        """
        Define metadata that is specifically required for reliability table
        cubes, whilst ensuring any mandatory attributes are also populated.

        Args:
            forecast_slice (iris.cube.Cube):
                The source cube from which to get pre-existing metadata of use.
        Returns:
            dict:
                A dictionary of attributes that are appropriate for the
                reliability table cube.
        """
        attributes = generate_mandatory_attributes([forecast_slice])
        attributes["title"] = "Reliability calibration data table"
        return attributes

    def _create_reliability_table_cube(self, forecast, threshold_coord):
        """
        Construct a reliability table cube and populate it with the provided
        data. The returned cube will include a cycle hour coordinate, which
        describes the model cycle hour at which the forecast data was produced.
        It will further include the forecast period, threshold coordinate,
        and spatial coordinates from the forecast cube.

        Args:
            forecast (iris.cube.Cube):
                A cube slice across the spatial dimensions of the forecast
                data. This slice provides the time and threshold values that
                relate to the reliability_table_data.
            threshold_coord (iris.coords.DimCoord):
                The threshold coordinate.
        Returns:
            iris.cube.Cube:
                A reliability table cube.
        """

        def _get_coords_and_dims(coord_names):
            """Obtain the requested coordinates and their dimension index from
            the forecast slice cube."""
            coords_and_dims = []
            leading_coords = [probability_bins_coord, reliability_index_coord]
            for coord_name in coord_names:
                crd = forecast_slice.coord(coord_name)
                crd_dim = forecast_slice.coord_dims(crd)
                crd_dim = crd_dim[0] + len(leading_coords) if crd_dim else ()
                coords_and_dims.append((crd, crd_dim))
            return coords_and_dims

        forecast_slice = next(forecast.slices_over(["time", threshold_coord]))
        expected_shape = self.expected_table_shape + forecast_slice.shape
        dummy_data = np.zeros((expected_shape))

        diagnostic = find_threshold_coordinate(forecast).name()
        attributes = self._define_metadata(forecast)

        # Define reliability table specific coordinates
        probability_bins_coord = self._create_probability_bins_coord()
        (
            reliability_index_coord,
            reliability_name_coord,
        ) = self._create_reliability_table_coords()
        frt_coord = create_unified_frt_coord(forecast.coord("forecast_reference_time"))

        # List of required non-spatial coordinates from the forecast
        non_spatial_coords = ["forecast_period", diagnostic]

        # Construct a list of coordinates in the desired order
        dim_coords = [forecast.coord(axis=dim).name() for dim in ["x", "y"]]
        dim_coords_and_dims = _get_coords_and_dims(dim_coords)
        aux_coords_and_dims = _get_coords_and_dims(non_spatial_coords)
        dim_coords_and_dims.append((reliability_index_coord, 0))
        aux_coords_and_dims.append((reliability_name_coord, 0))
        dim_coords_and_dims.append((probability_bins_coord, 1))

        reliability_cube = iris.cube.Cube(
            dummy_data,
            units=1,
            attributes=attributes,
            dim_coords_and_dims=dim_coords_and_dims,
            aux_coords_and_dims=aux_coords_and_dims,
        )
        reliability_cube.add_aux_coord(frt_coord)
        reliability_cube.rename("reliability_calibration_table")

        return reliability_cube

    def _populate_reliability_bins(self, forecast, truth):
        """
        For an x-y slice at a single validity time and threshold, populate
        a reliability table using the provided truth.

        Args:
            forecast (numpy.ndarray):
                An array containing data over an xy slice for a single validity
                time and threshold.
            truth (numpy.ndarray):
                An array containing a thresholded gridded truth at an
                equivalent validity time to the forecast array.
        Returns:
            numpy.ndarray:
                An array containing reliability table data for a single time
                and threshold. The leading dimension corresponds to the rows
                of a calibration table, the second dimension to the number of
                probability bins, and the trailing dimensions are the spatial
                dimensions of the forecast and truth cubes (which are
                equivalent).
        """
        observation_counts = []
        forecast_probabilities = []
        forecast_counts = []

        for bin_min, bin_max in self.probability_bins:

            observation_mask = (
                ((forecast >= bin_min) & (forecast <= bin_max)) & (np.isclose(truth, 1))
            ).astype(int)
            forecast_mask = ((forecast >= bin_min) & (forecast <= bin_max)).astype(int)
            forecasts_probability_values = forecast * forecast_mask

            observation_counts.append(observation_mask)
            forecast_probabilities.append(forecasts_probability_values)
            forecast_counts.append(forecast_mask)

        reliability_table = np.stack(
            [
                np.stack(observation_counts),
                np.stack(forecast_probabilities),
                np.stack(forecast_counts),
            ]
        )
        return reliability_table.astype(np.float32)

    def process(self, historic_forecasts, truths):
        """
        Slice data over threshold and time coordinates to construct reliability
        tables. These are summed over time to give a single table for each
        threshold, constructed from all the provided historic forecasts and
        truths.

        .. See the documentation for an example of the resulting reliability
           table cube.
        .. include:: extended_documentation/calibration/
           reliability_calibration/reliability_calibration_examples.rst

        Note that the forecast and truth data used is probabilistic, i.e. has
        already been thresholded relative to the thresholds of interest, using
        the equality operator required. As such this plugin is agnostic as to
        whether the data is thresholded below or above a given diagnostic
        threshold.

        Args:
            historic_forecasts (iris.cube.Cube):
                A cube containing the historical forecasts used in calibration.
                These are expected to all have a consistent cycle hour, that is
                the hour in the forecast reference time.
            truths (iris.cube.Cube):
                A cube containing the thresholded gridded truths used in
                calibration.
        Returns:
            iris.cube.CubeList:
                A cubelist of reliability table cubes, one for each threshold
                in the historic forecast cubes.
        Raises:
            ValueError: If the forecast and truth cubes have differing
                        threshold coordinates.
        """
        historic_forecasts, truths = filter_non_matching_cubes(
            historic_forecasts, truths
        )

        threshold_coord = find_threshold_coordinate(historic_forecasts)
        truth_threshold_coord = find_threshold_coordinate(truths)
        if not threshold_coord == truth_threshold_coord:
            msg = "Threshold coordinates differ between forecasts and truths."
            raise ValueError(msg)

        time_coord = historic_forecasts.coord("time")

        check_forecast_consistency(historic_forecasts)
        reliability_cube = self._create_reliability_table_cube(
            historic_forecasts, threshold_coord
        )

        reliability_tables = iris.cube.CubeList()
        threshold_slices = zip(
            historic_forecasts.slices_over(threshold_coord),
            truths.slices_over(threshold_coord),
        )
        for forecast_slice, truth_slice in threshold_slices:

            threshold_reliability = []
            time_slices = zip(
                forecast_slice.slices_over(time_coord),
                truth_slice.slices_over(time_coord),
            )
            for forecast, truth in time_slices:

                reliability_table = self._populate_reliability_bins(
                    forecast.data, truth.data
                )

                threshold_reliability.append(reliability_table)

            # Stack and sum reliability tables for all times
            table_values = np.stack(threshold_reliability)
            table_values = np.sum(table_values, axis=0, dtype=np.float32)

            reliability_entry = reliability_cube.copy(data=table_values)
            reliability_entry.replace_coord(forecast_slice.coord(threshold_coord))
            reliability_tables.append(reliability_entry)

        return MergeCubes()(reliability_tables)


class AggregateReliabilityCalibrationTables(BasePlugin):

    """This plugin enables the aggregation of multiple reliability calibration
    tables, and/or the aggregation over coordinates in the tables."""

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        return "<AggregateReliabilityCalibrationTables>"

    @staticmethod
    def _check_frt_coord(cubes):
        """
        Check that the reliability calibration tables do not have overlapping
        forecast reference time bounds. If these coordinates overlap in time it
        indicates that some of the same forecast data has contributed to more
        than one table, thus aggregating them would double count these
        contributions.

        Args:
            cubes (iris.cube.CubeList):
                The list of reliability calibration tables for which the
                forecast reference time coordinates should be checked.
        Raises:
            ValueError: If the bounds overlap.
        """
        bounds = []
        for cube in cubes:
            bounds.extend(cube.coord("forecast_reference_time").bounds)
        bounds = np.concatenate(bounds)
        if not all(x < y for x, y in zip(bounds, bounds[1:])):
            raise ValueError(
                "Reliability calibration tables have overlapping "
                "forecast reference time bounds, indicating that "
                "the same forecast data has contributed to the "
                "construction of both tables. Cannot aggregate."
            )

    def process(self, cubes, coordinates=None):
        """
        Aggregate the input reliability calibration table cubes and return the
        result.

        Args:
            cubes (list or iris.cube.CubeList):
                The cube or cubes containing the reliability calibration tables
                to aggregate.
            coordinates (list or None):
                A list of coordinates over which to aggregate the reliability
                calibration table using summation. If the argument is None and
                a single cube is provided, this cube will be returned
                unchanged.
        """
        coordinates = [] if coordinates is None else coordinates

        try:
            (cube,) = cubes
        except ValueError:
            cubes = iris.cube.CubeList(cubes)
            self._check_frt_coord(cubes)
            cube = cubes.merge_cube()
            coordinates.append("forecast_reference_time")
        else:
            if not coordinates:
                return cube

        result = collapsed(cube, coordinates, iris.analysis.SUM)
        frt = create_unified_frt_coord(cube.coord("forecast_reference_time"))
        result.replace_coord(frt)
        return result


class ManipulateReliabilityTable(BasePlugin):
    """
    A plugin to manipulate the reliability tables before they are used to
    calibrate a forecast. x and y coordinates on the reliability table must be
    collapsed.
    The result is a reliability diagram with monotonic observation frequency.

    Steps taken are:

    1. If non-monotonicity of the observation frequency is detected, try
    combining a pair of bins that appear non-monotonic. Only a single pair of
    bins are combined.

    2. If non-monotonicity of the observation frequency remains after trying
    to combine a single pair of bins, replace non-monotonic bins by assuming a
    constant observation frequency.
    """

    @staticmethod
    def _extract_reliability_table_components(reliability_table):
        """Extract reliability table components from cube

        Args:
            reliability_table (iris.cube.Cube):
                A reliability table to be manipulated.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, iris.coord.DimCoord]:
                Tuple containing the updated observation count,
                forecast probability sum, forecast count and probability bin
                coordinate.
        """
        observation_count = reliability_table.extract(
            iris.Constraint(table_row_name="observation_count")
        ).data
        forecast_probability_sum = reliability_table.extract(
            iris.Constraint(table_row_name="sum_of_forecast_probabilities")
        ).data
        forecast_count = reliability_table.extract(
            iris.Constraint(table_row_name="forecast_count")
        ).data
        probability_bin_coord = reliability_table.coord("probability_bin")
        return (
            observation_count,
            forecast_probability_sum,
            forecast_count,
            probability_bin_coord,
        )

    @staticmethod
    def _sum_pairs(array, upper):
        """
        Returns a new array where a pair of values in the original array have
        been replaced by their sum. Combines the value in the upper index with
        the value in the upper-1 index.

        Args:
            array (numpy.ndarray):
                Array to be modified.
            upper (int):
                Upper index of pair.

        Returns:
            numpy.ndarray:
                Array where a pair of values has been replaced by their sum.
        """
        result = array.copy()
        result[upper - 1] = np.sum(array[upper - 1 : upper + 1])
        return np.delete(result, upper)

    @staticmethod
    def _create_new_bin_coord(probability_bin_coord, upper):
        """
        Create a new probability_bin coordinate by combining two adjacent
        points on the probability_bin coordinate. This matches the combination
        of the data for the two bins.

        Args:
            probability_bin_coord (iris.coord.DimCoord):
                Original probability bin coordinate.
            upper (int):
                Upper index of pair.

        Returns:
            iris.coord.DimCoord:
                Probability bin coordinate with updated points and bounds where
                a pair of bins have been combined to create a single bin.
        """
        old_bounds = probability_bin_coord.bounds
        new_bounds = np.concatenate(
            (
                old_bounds[0 : upper - 1],
                np.array([[old_bounds[upper - 1, 0], old_bounds[upper, 1]]]),
                old_bounds[upper + 1 :],
            )
        )
        new_points = np.mean(new_bounds, axis=1, dtype=np.float32)
        new_bin_coord = iris.coords.DimCoord(
            new_points, long_name="probability_bin", units=1, bounds=new_bounds
        )
        return new_bin_coord

    def _combine_bin_pair(
        self,
        observation_count,
        forecast_probability_sum,
        forecast_count,
        probability_bin_coord,
    ):
        """
        Combine a pair of bins when non-monotonicity of the observation
        frequency is detected. Iterate top-down from the highest forecast
        probability bin to the lowest probability bin when combining the bins.
        Only allow a single pair of bins to be combined.

        Args:
            observation_count (numpy.ndarray):
                Observation count extracted from reliability table.
            forecast_probability_sum (numpy.ndarray):
                Forecast probability sum extracted from reliability table.
            forecast_count (numpy.ndarray):
                Forecast count extracted from reliability table.
            probability_bin_coord (iris.coord.DimCoord):
                Original probability bin coordinate.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, iris.coord.DimCoord]
                Tuple containing the updated observation count,
                forecast probability sum, forecast count and probability bin
                coordinate.
        """
        observation_frequency = np.array(observation_count / forecast_count)
        for upper in np.arange(len(observation_frequency) - 1, 0, -1):
            (diff,) = np.diff(
                [observation_frequency[upper - 1], observation_frequency[upper]]
            )
            if diff < 0:
                forecast_count = self._sum_pairs(forecast_count, upper)
                observation_count = self._sum_pairs(observation_count, upper)
                forecast_probability_sum = self._sum_pairs(
                    forecast_probability_sum, upper
                )
                probability_bin_coord = self._create_new_bin_coord(
                    probability_bin_coord, upper
                )
                break
        return (
            observation_count,
            forecast_probability_sum,
            forecast_count,
            probability_bin_coord,
        )

    @staticmethod
    def _assume_constant_observation_frequency(observation_count, forecast_count):
        """
        Iterate through the observation frequency from the lowest probability
        bin to the highest probability bin. Compare each pair of bins and, if
        a pair is non-monotonic, replace the value of the upper bin with the
        value of the lower bin. Then calculate the new observation count
        required to give a monotonic observation frequency.

        Args:
            observation_count (numpy.ndarray):
                Observation count extracted from reliability table.
            forecast_count (numpy.ndarray):
                Forecast count extracted from reliability table.

        Returns:
            numpy.ndarray:
                Observation count computed from a monotonic observation frequency.

        """
        observation_frequency = np.array(observation_count / forecast_count)
        for index, lower_bin in enumerate(observation_frequency[:-1]):
            (diff,) = np.diff([lower_bin, observation_frequency[index + 1]])
            if diff < 0:
                observation_frequency[index + 1] = lower_bin
        observation_count = observation_frequency * forecast_count
        return observation_count

    @staticmethod
    def _update_reliability_table(
        reliability_table,
        observation_count,
        forecast_probability_sum,
        forecast_count,
        probability_bin_coord,
    ):
        """
        Update the reliability table data and the probability bin coordinate.

        Args:
            reliability_table (iris.cube.Cube):
                A reliability table to be manipulated.
            observation_count (numpy.ndarray):
                Observation count extracted from reliability table.
            forecast_probability_sum (numpy.ndarray):
                Forecast probability sum extracted from reliability table.
            forecast_count (numpy.ndarray):
                Forecast count extracted from reliability table.
            probability_bin_coord (iris.coord.DimCoord):
                Original probability bin coordinate.

        Returns:
            iris.cube.Cube:
                Updated reliability table.
        """
        final_data = np.stack(
            [observation_count, forecast_probability_sum, forecast_count]
        )
        nrows, ncols = final_data.shape
        reliability_table = reliability_table[0:nrows, 0:ncols].copy(data=final_data)
        reliability_table.replace_coord(probability_bin_coord)
        return reliability_table

    def process(self, reliability_table):
        """
        Apply the steps needed to produce a reliability diagram with a
        monotonic observation frequency.

        Args:
            reliability_table (iris.cube.Cube):
                A reliability table to be manipulated. The only coordinates
                expected on this cube are a threshold coordinate,
                a table_row_index coordinate and corresponding table_row_name
                coordinate and a probability_bin coordinate.

        Returns:
            iris.cube.CubeList:
                Containing a reliability table cube for each threshold in the
                input reliablity table. For tables where monotonicity has been
                enforced the probability_bin coordinate will have one less
                bin than the tables that were already monotonic.
        """
        threshold_coord = find_threshold_coordinate(reliability_table)
        reliability_table_cubelist = iris.cube.CubeList()
        for rel_table_slice in reliability_table.slices_over(threshold_coord):
            (
                observation_count,
                forecast_probability_sum,
                forecast_count,
                probability_bin_coord,
            ) = self._extract_reliability_table_components(rel_table_slice)
            # If the observation frequency is non-monotonic adjust the
            # reliability table
            observation_frequency = np.array(observation_count / forecast_count)
            if not np.all(np.diff(observation_frequency) >= 0):
                (
                    observation_count,
                    forecast_probability_sum,
                    forecast_count,
                    probability_bin_coord,
                ) = self._combine_bin_pair(
                    observation_count,
                    forecast_probability_sum,
                    forecast_count,
                    probability_bin_coord,
                )
                observation_count = self._assume_constant_observation_frequency(
                    observation_count, forecast_count
                )
                rel_table_slice = self._update_reliability_table(
                    rel_table_slice,
                    observation_count,
                    forecast_probability_sum,
                    forecast_count,
                    probability_bin_coord,
                )
            reliability_table_cubelist.append(rel_table_slice)
        return reliability_table_cubelist


class ApplyReliabilityCalibration(PostProcessingPlugin):

    """
    A plugin for the application of reliability calibration to probability
    forecasts. This calibration is designed to improve the reliability of
    probability forecasts without significantly degrading their resolution.

    The method implemented here is described in Flowerdew J. 2014.
    """

    def __init__(self, minimum_forecast_count=200, minimum_bin_fraction=0.6):
        """
        Initialise class for applying reliability calibration.

        Args:
            minimum_forecast_count (int):
                The minimum number of forecast counts in a forecast probability
                bin for it to be used in calibration. If the reliability
                table for a forecast threshold includes any bins with
                insufficient counts that threshold will be returned unchanged.
                The default value of 200 is that used in Flowerdew 2014.
            minimum_bin_fraction (float):
                The minimum fraction of forecast count bins associated with a
                probability threshold that must exceed minimum_forecast_count
                for that threshold to be calibrated.
        Raises:
            ValueError: If minimum_forecast_count is less than 1.
            ValueError: If minimum_bin_fraction is not between 0 and 1.
        References:
            Flowerdew J. 2014. Calibrating ensemble reliability whilst
            preserving spatial structure. Tellus, Ser. A Dyn. Meteorol.
            Oceanogr. 66.
        """
        if minimum_forecast_count < 1:
            raise ValueError(
                "The minimum_forecast_count must be at least 1 as empty "
                "bins in the reliability table are not handled."
            )

        if minimum_bin_fraction < 0 or minimum_bin_fraction > 1:
            raise ValueError(
                "The minimum_bin_fraction must be between 0 and 1. Value set "
                f"as {minimum_bin_fraction}"
            )

        self.minimum_forecast_count = minimum_forecast_count
        self.minimum_bin_fraction = minimum_bin_fraction
        self.threshold_coord = None

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        result = "<ApplyReliabilityCalibration: minimum_forecast_count: {}>"
        return result.format(self.minimum_forecast_count)

    @staticmethod
    def _extract_matching_reliability_table(forecast, reliability_table):
        """
        Extract the reliability table with a threshold coordinate
        matching the forecast cube.
        If no matching reliability table is found raise an exception.

        Args:
            forecast (iris.cube.Cube):
                The forecast to be calibrated.
            reliability_table (iris.cube.CubeList):
                The reliability table to use for applying calibration.
        Returns:
            iris.cube.Cube:
                A reliability table who's threshold coordinate matches
                the forecast cube.
        Raises:
            ValueError: If no matching reliability table is found.
        """
        threshold_coord = find_threshold_coordinate(forecast)
        coord_values = {threshold_coord.name(): threshold_coord.points}
        constr = iris.Constraint(coord_values=coord_values)
        if isinstance(reliability_table, iris.cube.Cube):
            extracted = reliability_table.extract(constr)
        else:
            extracted = reliability_table.extract(constr, strict=True)
        if not extracted:
            raise ValueError(
                "No reliability table found to match threshold "
                f"{find_threshold_coordinate(forecast).points[0]}."
            )
        return extracted

    def _ensure_monotonicity_across_thresholds(self, cube):
        """
        Ensures that probabilities change monotonically relative to thresholds
        in the expected order, e.g. exceedance probabilities always remain the
        same or decrease as the threshold values increase, below threshold
        probabilities always remain the same or increase as the threshold
        values increase.

        Args:
            cube (iris.cube.Cube):
                The probability cube for which monotonicity is to be checked
                and enforced. This cube is modified in place.
        Raises:
            ValueError: Threshold coordinate lacks the
                        spp__relative_to_threshold attribute.
        Warns:
            UserWarning: If the probabilities must be sorted to reinstate
                         expected monotonicity following calibration.
        """
        (threshold_dim,) = cube.coord_dims(self.threshold_coord)
        thresholding = self.threshold_coord.attributes.get(
            "spp__relative_to_threshold", None
        )

        if thresholding is None:
            msg = (
                "Cube threshold coordinate does not define whether "
                "thresholding is above or below the defined thresholds."
            )
            raise ValueError(msg)

        if (
            thresholding == "above"
            and not (np.diff(cube.data, axis=threshold_dim) <= 0).all()
        ):
            msg = (
                "Exceedance probabilities are not decreasing monotonically "
                "as the threshold values increase. Forced back into order."
            )
            warnings.warn(msg)
            cube.data = np.sort(cube.data, axis=threshold_dim)[::-1]

        if (
            thresholding == "below"
            and not (np.diff(cube.data, axis=threshold_dim) >= 0).all()
        ):
            msg = (
                "Below threshold probabilities are not increasing "
                "monotonically as the threshold values increase. Forced "
                "back into order."
            )
            warnings.warn(msg)
            cube.data = np.sort(cube.data, axis=threshold_dim)

    def _calculate_reliability_probabilities(self, reliability_table):
        """
        Calculates forecast probabilities and observation frequencies from the
        reliability table. Where the forecast count is zero, Nones are
        returned.

        Args:
            reliability_table (iris.cube.Cube):
                A reliability table for a single threshold from which to
                calculate the forecast probabilities and observation
                frequencies.
        Returns:
            Optional[Tuple[numpy.ndarray, numpy.ndarray]]:
                Tuple containing forecast probabilities calculated by dividing
                the sum of forecast probabilities by the forecast count and
                observation frequency calculated by dividing the observation
                count by the forecast count.
        """
        observation_count = reliability_table.extract(
            iris.Constraint(table_row_name="observation_count")
        ).data
        forecast_count = reliability_table.extract(
            iris.Constraint(table_row_name="forecast_count")
        ).data
        forecast_probability_sum = reliability_table.extract(
            iris.Constraint(table_row_name="sum_of_forecast_probabilities")
        ).data

        forecast_probability = np.array(forecast_probability_sum / forecast_count)
        observation_frequency = np.array(observation_count / forecast_count)

        # If the fraction of bins with forecast counts exceeding minimum_forecast_count
        # if less than minimum_bin_fraction, return None to avoid applying calibration
        # to that probability threshold.
        valid_bins = np.where(forecast_count >= self.minimum_forecast_count)
        if valid_bins[0].size < forecast_count.size * self.minimum_bin_fraction:
            return None, None

        return forecast_probability, observation_frequency

    @staticmethod
    def _interpolate(
        forecast_threshold, reliability_probabilities, observation_frequencies
    ):
        """
        Perform interpolation of the forecast probabilities using the
        reliability table data to produce the calibrated forecast. Where
        necessary linear extrapolation will be applied. Any mask in place on
        the forecast_threshold data is removed and reapplied after calibration.

        Args:
            forecast_threshold (numpy.ndarray):
                The forecast probabilities to be calibrated.
            reliability_probabilities (numpy.ndarray):
                Probabilities taken from the reliability tables.
            observation_frequencies (numpy.ndarray):
                Observation frequencies that relate to the reliability
                probabilities, taken from the reliability tables.

        Returns:
            numpy.ndarray:
                The calibrated forecast probabilities. The final results are
                clipped to ensure any extrapolation has not yielded
                probabilities outside the range 0 to 1.
        """
        shape = forecast_threshold.shape
        mask = forecast_threshold.mask if np.ma.is_masked(forecast_threshold) else None

        forecast_probabilities = np.ma.getdata(forecast_threshold).flatten()

        interpolation_function = scipy.interpolate.interp1d(
            reliability_probabilities, observation_frequencies, fill_value="extrapolate"
        )
        interpolated = interpolation_function(forecast_probabilities.data)

        interpolated = interpolated.reshape(shape).astype(np.float32)

        if mask is not None:
            interpolated = np.ma.masked_array(interpolated, mask=mask)

        return np.clip(interpolated, 0, 1)

    def process(self, forecast, reliability_table):
        """
        Apply reliability calibration to a forecast. The reliability table
        and the forecast cube must share an identical threshold coordinate.

        Args:
            forecast (iris.cube.Cube):
                The forecast to be calibrated.
            reliability_table (iris.cube.Cube or iris.cube.CubeList):
                The reliability table to use for applying calibration.
                x and y dimensions must be collapsed.
        Returns:
            iris.cube.Cube:
                The forecast cube following calibration.
        """
        self.threshold_coord = find_threshold_coordinate(forecast)

        forecast_thresholds = forecast.slices_over(self.threshold_coord)

        uncalibrated_thresholds = []
        calibrated_cubes = iris.cube.CubeList()
        for forecast_threshold in forecast_thresholds:
            reliability_threshold = self._extract_matching_reliability_table(
                forecast_threshold, reliability_table
            )
            (
                reliability_probabilities,
                observation_frequencies,
            ) = self._calculate_reliability_probabilities(reliability_threshold)

            if reliability_probabilities is None:
                calibrated_cubes.append(forecast_threshold)
                uncalibrated_thresholds.append(
                    forecast_threshold.coord(self.threshold_coord).points[0]
                )
                continue

            interpolated = self._interpolate(
                forecast_threshold.data,
                reliability_probabilities,
                observation_frequencies,
            )

            calibrated_cubes.append(forecast_threshold.copy(data=interpolated))

        calibrated_forecast = calibrated_cubes.merge_cube()
        self._ensure_monotonicity_across_thresholds(calibrated_forecast)

        if uncalibrated_thresholds:
            msg = (
                "The following thresholds were not calibrated due to "
                "insufficient forecast counts in reliability table bins: "
                "{}".format(uncalibrated_thresholds)
            )
            warnings.warn(msg)

        return calibrated_forecast
