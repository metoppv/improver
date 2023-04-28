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
"""Reliability calibration plugins."""

import operator
import warnings
from typing import Dict, List, Optional, Tuple, Union

import iris
import numpy as np
import scipy
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube, CubeList
from iris.exceptions import CoordinateNotFoundError
from numpy import ndarray
from numpy.ma.core import MaskedArray

from improver import BasePlugin, PostProcessingPlugin
from improver.calibration.utilities import (
    check_forecast_consistency,
    create_unified_frt_coord,
    filter_non_matching_cubes,
)
from improver.metadata.probabilistic import (
    find_threshold_coordinate,
    probability_is_above_or_below,
)
from improver.metadata.utilities import generate_mandatory_attributes
from improver.utilities.cube_manipulation import (
    MergeCubes,
    collapsed,
    enforce_coordinate_ordering,
    get_dim_coord_names,
)


class ConstructReliabilityCalibrationTables(BasePlugin):

    """A plugin for creating and populating reliability calibration tables."""

    def __init__(
        self,
        n_probability_bins: int = 5,
        single_value_lower_limit: bool = False,
        single_value_upper_limit: bool = False,
    ) -> None:
        """
        Initialise class for creating reliability calibration tables. These
        tables include data columns entitled observation_count,
        sum_of_forecast_probabilities, and forecast_count, defined below.

        n_probability_bins:
            The total number of probability bins required in the reliability
            tables. If single value limits are turned on, these are included in
            this total.
        single_value_lower_limit:
            Mandates that the lowest bin should be single valued,
            with a small precision tolerance, defined as 1.0E-6.
            The bin is thus 0 to 1.0E-6.
        single_value_upper_limit:
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

    def __repr__(self) -> str:
        """Represent the configured plugin instance as a string."""
        bin_values = ", ".join(
            ["[{:1.2f} --> {:1.2f}]".format(*item) for item in self.probability_bins]
        )
        result = "<ConstructReliabilityCalibrationTables: " "probability_bins: {}>"
        return result.format(bin_values)

    def _define_probability_bins(
        self,
        n_probability_bins: int,
        single_value_lower_limit: bool,
        single_value_upper_limit: bool,
    ) -> ndarray:
        """
        Define equally sized probability bins for use in a reliability table.
        The range 0 to 1 is divided into ranges to give n_probability bins.
        If single_value_lower_limit and / or single_value_upper_limit are True,
        additional bins corresponding to values of 0 and / or 1 will be created,
        each with a width defined by self.single_value_tolerance.

        Args:
            n_probability_bins:
                The total number of probability bins desired in the
                reliability tables. This number includes the extrema bins
                (equals 0 and equals 1) if single value limits are turned on,
                in which case the minimum number of bins is 3.
            single_value_lower_limit:
                Mandates that the lowest bin should be single valued,
                with a small precision tolerance, defined as 1.0E-6.
                The bin is thus 0 to 1.0E-6.
            single_value_upper_limit:
                Mandates that the highest bin should be single valued,
                with a small precision tolerance, defined as 1.0E-6.
                The bin is thus (1 - 1.0E-6) to 1.

        Returns:
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

    def _create_probability_bins_coord(self) -> DimCoord:
        """
        Construct a dimension coordinate describing the probability bins
        of the reliability table.

        Returns:
            A dimension coordinate describing probability bins.
        """
        values = np.mean(self.probability_bins, axis=1, dtype=np.float32)
        probability_bins_coord = iris.coords.DimCoord(
            values, long_name="probability_bin", units=1, bounds=self.probability_bins
        )
        return probability_bins_coord

    def _create_reliability_table_coords(self) -> Tuple[DimCoord, AuxCoord]:
        """
        Construct coordinates that describe the reliability table rows. These
        are observation_count, sum_of_forecast_probabilities, and
        forecast_count. The order used here is the order in which the table
        data is populated, so these must remain consistent with the
        _populate_reliability_bins function.

        Returns:
            - A numerical index dimension coordinate.
            - An auxiliary coordinate that assigns names to the index
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
    def _define_metadata(forecast_slice: Cube) -> Dict[str, str]:
        """
        Define metadata that is specifically required for reliability table
        cubes, whilst ensuring any mandatory attributes are also populated.

        Args:
            forecast_slice:
                The source cube from which to get pre-existing metadata of use.

        Returns:
            A dictionary of attributes that are appropriate for the
            reliability table cube.
        """
        attributes = generate_mandatory_attributes([forecast_slice])
        attributes["title"] = "Reliability calibration data table"
        return attributes

    def _create_reliability_table_cube(
        self, forecast: Cube, threshold_coord: DimCoord
    ) -> Cube:
        """
        Construct a reliability table cube and populate it with the provided
        data. The returned cube will include a forecast_reference_time
        coordinate, which will be the maximum range of bounds of the input
        forecast reference times, with the point value set to the latest
        of those in the inputs. It will further include the forecast period,
        threshold coordinate, and spatial coordinates from the forecast cube.

        Args:
            forecast:
                A cube slice across the spatial dimensions of the forecast
                data. This slice provides the time and threshold values that
                relate to the reliability_table_data.
            threshold_coord:
                The threshold coordinate.

        Returns:
            A reliability table cube.
        """

        def _get_coords_and_dims(coord_names: List[str],) -> List[Tuple[DimCoord, int]]:
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
        aux_coords_and_dims = _get_coords_and_dims(non_spatial_coords)
        aux_coords_and_dims.append((reliability_name_coord, 0))
        spatial_coords = [forecast.coord(axis=dim).name() for dim in ["x", "y"]]
        spatial_coords_and_dims = _get_coords_and_dims(spatial_coords)
        try:
            spot_index_coord = _get_coords_and_dims(["spot_index"])
            wmo_id_coord = _get_coords_and_dims(["wmo_id"])
        except CoordinateNotFoundError:
            dim_coords_and_dims = spatial_coords_and_dims
        else:
            dim_coords_and_dims = spot_index_coord
            aux_coords_and_dims.extend(spatial_coords_and_dims + wmo_id_coord)

        dim_coords_and_dims.append((reliability_index_coord, 0))
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

    def _populate_reliability_bins(
        self, forecast: Union[MaskedArray, ndarray], truth: Union[MaskedArray, ndarray]
    ) -> MaskedArray:
        """
        For a spatial slice at a single validity time and threshold, populate
        a reliability table using the provided truth.

        Args:
            forecast:
                An array containing data over a spatial slice for a single validity
                time and threshold.
            truth:
                An array containing a thresholded gridded truth at an
                equivalent validity time to the forecast array.

        Returns:
            An array containing reliability table data for a single time
            and threshold. The leading dimension corresponds to the rows
            of a calibration table, the second dimension to the number of
            probability bins, and the trailing dimension(s) are the spatial
            dimension(s) of the forecast and truth cubes (which are
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
        reliability_table = np.ma.stack(
            [
                np.ma.stack(observation_counts),
                np.ma.stack(forecast_probabilities),
                np.ma.stack(forecast_counts),
            ]
        )

        return reliability_table.astype(np.float32)

    def _populate_masked_reliability_bins(
        self, forecast: ndarray, truth: MaskedArray
    ) -> MaskedArray:
        """
        Support populating the reliability table bins with a masked truth. If a
        masked truth is provided, a masked reliability table is returned.

        Args:
            forecast:
                An array containing data over an xy slice for a single validity
                time and threshold.
            truth:
                An array containing a thresholded gridded truth at an
                equivalent validity time to the forecast array.

        Returns:
            An array containing reliability table data for a single time
            and threshold. The leading dimension corresponds to the rows
            of a calibration table, the second dimension to the number of
            probability bins, and the trailing dimensions are the spatial
            dimensions of the forecast and truth cubes (which are
            equivalent).
        """
        forecast = np.ma.masked_where(np.ma.getmask(truth), forecast)
        table = self._populate_reliability_bins(forecast, truth)
        # Zero data underneath mask to support bitwise addition of masks.
        table.data[table.mask] = 0
        return table

    def _add_reliability_tables(
        self, forecast: Cube, truth: Cube, threshold_reliability: MaskedArray
    ) -> Union[MaskedArray, ndarray]:
        """
        Add reliability tables. The presence of a masked truth is handled
        separately to ensure support for a mask that changes with validity time.

        Args:
            forecast:
                An array containing data over an xy slice for a single validity
                time and threshold.
            truth:
                An array containing a thresholded gridded truth at an
                equivalent validity time to the forecast array.
            threshold_reliability:
                The current reliability table that will be added to.

        Returns:
            An array containing reliability table data for a single time
            and threshold. The leading dimension corresponds to the rows
            of a calibration table, the second dimension to the number of
            probability bins, and the trailing dimensions are the spatial
            dimensions of the forecast and truth cubes (which are
            equivalent).
        """
        if np.ma.is_masked(truth.data):
            table = self._populate_masked_reliability_bins(forecast.data, truth.data)
            # Bitwise addition of masks. This ensures that only points that are
            # masked in both the existing and new reliability tables are kept
            # as being masked within the resulting reliability table.
            mask = threshold_reliability.mask & table.mask
            threshold_reliability = np.ma.array(
                threshold_reliability.data + table.data, mask=mask, dtype=np.float32,
            )
        else:
            np.add(
                threshold_reliability,
                self._populate_reliability_bins(forecast.data, truth.data),
                out=threshold_reliability,
                dtype=np.float32,
            )
        return threshold_reliability

    def process(
        self,
        historic_forecasts: Cube,
        truths: Cube,
        aggregate_coords: Optional[List[str]] = None,
    ) -> Cube:
        """
        Slice data over threshold and time coordinates to construct reliability
        tables. These are summed over time to give a single table for each
        threshold, constructed from all the provided historic forecasts and
        truths. If a masked truth is provided, a masked reliability table is
        returned. If the mask within the truth varies at different timesteps,
        any point that is unmasked for at least one timestep will have
        unmasked values within the reliability table. Therefore historic
        forecast points will only be used if they have a corresponding valid
        truth point for each timestep.

        .. See the documentation for an example of the resulting reliability
           table cube.
        .. include:: extended_documentation/calibration/
           reliability_calibration/reliability_calibration_examples.rst

        Note that the forecast and truth data used is probabilistic, i.e. has
        already been thresholded relative to the thresholds of interest, using
        the equality operator required. As such this plugin is agnostic as to
        whether the data is thresholded below or above a given diagnostic
        threshold.

        `historic_forecasts` and `truths` should have matching validity times.

        Args:
            historic_forecasts:
                A cube containing the historical forecasts used in calibration.
            truths:
                A cube containing the thresholded gridded truths used in
                calibration.
            aggregate_coords:
                Coordinates to aggregate over during construction. This is
                equivalent to constructing then using
                :class:`improver.calibration.reliability_calibration.AggregateReliabilityCalibrationTables`
                but with reduced memory usage due to avoiding large intermediate
                data.

        Returns:
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

        populate_bins_func = self._populate_reliability_bins
        if np.ma.is_masked(truths.data):
            populate_bins_func = self._populate_masked_reliability_bins

        reliability_tables = iris.cube.CubeList()
        threshold_slices = zip(
            historic_forecasts.slices_over(threshold_coord),
            truths.slices_over(threshold_coord),
        )
        for forecast_slice, truth_slice in threshold_slices:

            time_slices = zip(
                forecast_slice.slices_over(time_coord),
                truth_slice.slices_over(time_coord),
            )
            forecast, truth = next(time_slices)
            threshold_reliability = populate_bins_func(forecast.data, truth.data)

            for forecast, truth in time_slices:
                threshold_reliability = self._add_reliability_tables(
                    forecast, truth, threshold_reliability
                )

            reliability_entry = reliability_cube.copy(data=threshold_reliability)
            reliability_entry.replace_coord(forecast_slice.coord(threshold_coord))
            if aggregate_coords:
                reliability_entry = AggregateReliabilityCalibrationTables().process(
                    [reliability_entry], aggregate_coords
                )
            reliability_tables.append(reliability_entry)

        return MergeCubes()(reliability_tables, copy=False)


class AggregateReliabilityCalibrationTables(BasePlugin):

    """This plugin enables the aggregation of multiple reliability calibration
    tables, and/or the aggregation over coordinates in the tables."""

    def __repr__(self) -> str:
        """Represent the configured plugin instance as a string."""
        return "<AggregateReliabilityCalibrationTables>"

    @staticmethod
    def _check_frt_coord(cubes: Union[List[Cube], CubeList]) -> None:
        """
        Check that the reliability calibration tables do not have overlapping
        forecast reference time bounds. If these coordinates overlap in time it
        indicates that some of the same forecast data has contributed to more
        than one table, thus aggregating them would double count these
        contributions.

        Args:
            cubes:
                The list of reliability calibration tables for which the
                forecast reference time coordinates should be checked.

        Raises:
            ValueError: If the bounds overlap.
        """
        lower_bounds = []
        upper_bounds = []
        for cube in cubes:
            lower_bounds.append(cube.coord("forecast_reference_time").bounds[0][0])
            upper_bounds.append(cube.coord("forecast_reference_time").bounds[0][1])
        if not all(x < y for x, y in zip(upper_bounds, lower_bounds[1:])):
            raise ValueError(
                "Reliability calibration tables have overlapping "
                "forecast reference time bounds, indicating that "
                "the same forecast data has contributed to the "
                "construction of both tables. Cannot aggregate."
            )

    def process(
        self,
        cubes: Union[CubeList, List[Cube]],
        coordinates: Optional[List[str]] = None,
    ) -> Cube:
        """
        Aggregate the input reliability calibration table cubes and return the
        result.

        Args:
            cubes:
                The cube or cubes containing the reliability calibration tables
                to aggregate.
            coordinates:
                A list of coordinates over which to aggregate the reliability
                calibration table using summation. If the argument is None and
                a single cube is provided, this cube will be returned
                unchanged.

        Returns:
            Aggregated cube
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

    1. If any bin contains less than the minimum forecast count then try
    combining this bin with whichever neighbour has the lowest sample count.
    This process is repeated for all bins that are below the minimum forecast
    count criterion.

    2. If non-monotonicity of the observation frequency is detected, try
    combining a pair of bins that appear non-monotonic. Only a single pair of
    bins are combined.

    3. If non-monotonicity of the observation frequency remains after trying
    to combine a single pair of bins, replace non-monotonic bins by assuming a
    constant observation frequency.
    """

    def __init__(
        self, minimum_forecast_count: int = 200, point_by_point: bool = False
    ) -> None:
        """
        Initialise class for manipulating a reliability table.

        Args:
            minimum_forecast_count:
                The minimum number of forecast counts in a forecast probability
                bin for it to be used in calibration.
                The default value of 200 is that used in Flowerdew 2014.
            point_by_point:
                Whether to process each point in the input cube independently.
                Please note this option is memory intensive and is unsuitable
                for gridded input

        Raises:
            ValueError: If minimum_forecast_count is less than 1.

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
        self.minimum_forecast_count = minimum_forecast_count
        self.point_by_point = point_by_point

    @staticmethod
    def _extract_reliability_table_components(
        reliability_table: Cube,
    ) -> Tuple[ndarray, ndarray, ndarray, DimCoord]:
        """Extract reliability table components from cube

        Args:
            reliability_table:
                A reliability table to be manipulated.

        Returns:
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
    def _sum_pairs(array: ndarray, upper: int) -> ndarray:
        """
        Returns a new array where a pair of values in the original array have
        been replaced by their sum. Combines the value in the upper index with
        the value in the upper-1 index.

        Args:
            array:
                Array to be modified.
            upper:
                Upper index of pair.

        Returns:
            Array where a pair of values has been replaced by their sum.
        """
        result = array.copy()
        result[upper - 1] = np.sum(array[upper - 1 : upper + 1])
        return np.delete(result, upper)

    @staticmethod
    def _create_new_bin_coord(probability_bin_coord: DimCoord, upper: int) -> DimCoord:
        """
        Create a new probability_bin coordinate by combining two adjacent
        points on the probability_bin coordinate. This matches the combination
        of the data for the two bins.

        Args:
            probability_bin_coord:
                Original probability bin coordinate.
            upper:
                Upper index of pair.

        Returns:
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

    def _combine_undersampled_bins(
        self,
        observation_count: ndarray,
        forecast_probability_sum: ndarray,
        forecast_count: ndarray,
        probability_bin_coord: DimCoord,
    ) -> Tuple[ndarray, ndarray, ndarray, DimCoord]:
        """
        Combine bins that are under-sampled i.e. that have a lower forecast
        count than the minimum_forecast_count, so that information from these
        poorly-sampled bins can contribute to the calibration. If multiple
        bins are below the minimum forecast count, the bin closest to
        meeting the minimum_forecast_count criterion is combined with whichever
        neighbour has the lowest sample count. A new bin is then created by
        summing the neighbouring pair of bins. This process is repeated for all
        bins that are below the minimum forecast count criterion.

        Args:
            observation_count:
                Observation count extracted from reliability table.
            forecast_probability_sum:
                Forecast probability sum extracted from reliability table.
            forecast_count:
                Forecast count extracted from reliability table.
            probability_bin_coord:
                Original probability bin coordinate.

        Returns:
            Tuple containing the updated observation count,
            forecast probability sum, forecast count and probability bin
            coordinate.
        """
        while (
            any(x < self.minimum_forecast_count for x in forecast_count)
            and len(forecast_count) > 1
        ):
            forecast_count_copy = forecast_count.copy()

            # Find index of the bin with the highest forecast count that is
            # below the minimum_forecast_count by setting forecast counts
            # greater than the minimum_forecast_count to NaN.
            forecast_count_copy[forecast_count >= self.minimum_forecast_count] = np.nan
            # Note for multiple occurrences of the maximum,
            # the index of the first occurrence is returned.
            index = np.int32(np.nanargmax(forecast_count_copy))

            # Determine the upper index of the pair of bins to be combined.
            if index == 0:
                # Must use higher bin
                upper = index + 1
            elif index + 1 == len(forecast_count):
                # Index already defines the upper bin
                upper = index
            else:
                # Define upper index to include bin with lowest sample count.
                if forecast_count[index + 1] > forecast_count[index - 1]:
                    upper = index
                else:
                    upper = index + 1

            forecast_count = self._sum_pairs(forecast_count, upper)
            observation_count = self._sum_pairs(observation_count, upper)
            forecast_probability_sum = self._sum_pairs(forecast_probability_sum, upper)
            probability_bin_coord = self._create_new_bin_coord(
                probability_bin_coord, upper
            )

        return (
            observation_count,
            forecast_probability_sum,
            forecast_count,
            probability_bin_coord,
        )

    def _combine_bin_pair(
        self,
        observation_count: ndarray,
        forecast_probability_sum: ndarray,
        forecast_count: ndarray,
        probability_bin_coord: DimCoord,
    ) -> Tuple[ndarray, ndarray, ndarray, DimCoord]:
        """
        Combine a pair of bins when non-monotonicity of the observation
        frequency is detected. Iterate top-down from the highest forecast
        probability bin to the lowest probability bin when combining the bins.
        Only allow a single pair of bins to be combined.

        Args:
            observation_count:
                Observation count extracted from reliability table.
            forecast_probability_sum:
                Forecast probability sum extracted from reliability table.
            forecast_count:
                Forecast count extracted from reliability table.
            probability_bin_coord:
                Original probability bin coordinate.

        Returns:
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
    def _assume_constant_observation_frequency(
        observation_count: ndarray, forecast_count: ndarray
    ) -> ndarray:
        """
        Decide which end bin (highest probability bin or lowest probability
        bin) has the highest sample count. Iterate through the observation
        frequency from the end bin with the highest sample count to the end bin
        with the lowest sample count. Whilst iterating, compare each pair of
        bins and, if a pair is non-monotonic, replace the value of the bin
        closer to the lowest sample count end bin with the value of the
        bin that is closer to the higher sample count end bin. Then calculate
        the new observation count required to give a monotonic observation
        frequency.

        Args:
            observation_count:
                Observation count extracted from reliability table.
            forecast_count:
                Forecast count extracted from reliability table.

        Returns:
            Observation count computed from a monotonic observation frequency.
        """
        observation_frequency = np.array(observation_count / forecast_count)

        iterator = observation_frequency
        operation = operator.lt
        # Top down if forecast count is lower for lowest probability bin,
        # than for highest probability bin.
        if forecast_count[0] < forecast_count[-1]:
            # Reverse array to iterate from top to bottom.
            iterator = observation_frequency[::-1]
            operation = operator.gt

        for index, lower_bin in enumerate(iterator[:-1]):
            (diff,) = np.diff([lower_bin, iterator[index + 1]])
            if operation(diff, 0):
                iterator[index + 1] = lower_bin

        observation_frequency = iterator
        if forecast_count[0] < forecast_count[-1]:
            # Re-reverse array from bottom to top to ensure original ordering.
            observation_frequency = iterator[::-1]

        observation_count = observation_frequency * forecast_count
        return observation_count

    @staticmethod
    def _update_reliability_table(
        reliability_table: Cube,
        observation_count: ndarray,
        forecast_probability_sum: ndarray,
        forecast_count: ndarray,
        probability_bin_coord: DimCoord,
    ) -> Cube:
        """
        Update the reliability table data and the probability bin coordinate.

        Args:
            reliability_table:
                A reliability table to be manipulated.
            observation_count:
                Observation count extracted from reliability table.
            forecast_probability_sum:
                Forecast probability sum extracted from reliability table.
            forecast_count:
                Forecast count extracted from reliability table.
            probability_bin_coord:
                Original probability bin coordinate.

        Returns:
            Updated reliability table.
        """
        final_data = np.stack(
            [observation_count, forecast_probability_sum, forecast_count]
        )
        nrows, ncols = final_data.shape
        reliability_table = reliability_table[0:nrows, 0:ncols].copy(data=final_data)
        reliability_table.replace_coord(probability_bin_coord)
        return reliability_table

    def _enforce_min_count_and_montonicity(self, rel_table_slice: Cube) -> Cube:
        """Apply the steps needed to produce a reliability diagram on a single
        slice of reliability table cube.

        Args:
            reliability_table_slice:
                The reliability table slice to be manipulated. The only
                coordinates expected on this cube are a table_row_index
                coordinate and corresponding table_row_name coordinate and a
                probability_bin coordinate.
        Returns:
            Processed reliability table slice, with reliability steps applied.
        """
        (
            observation_count,
            forecast_probability_sum,
            forecast_count,
            probability_bin_coord,
        ) = self._extract_reliability_table_components(rel_table_slice)

        if np.any(forecast_count < self.minimum_forecast_count):
            (
                observation_count,
                forecast_probability_sum,
                forecast_count,
                probability_bin_coord,
            ) = self._combine_undersampled_bins(
                observation_count,
                forecast_probability_sum,
                forecast_count,
                probability_bin_coord,
            )
            rel_table_slice = self._update_reliability_table(
                rel_table_slice,
                observation_count,
                forecast_probability_sum,
                forecast_count,
                probability_bin_coord,
            )

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
        return rel_table_slice

    def process(self, reliability_table: Cube) -> CubeList:
        """
        Apply the steps needed to produce a reliability diagram with a
        monotonic observation frequency.

        Args:
            reliability_table:
                A reliability table to be manipulated. The only coordinates
                expected on this cube are a threshold coordinate,
                a table_row_index coordinate and corresponding table_row_name
                coordinate and a probability_bin coordinate.

        Returns:
            CubeList containing a reliability table cube for each threshold in
            the input reliablity table. For tables where monotonicity has been
            enforced the probability_bin coordinate will have one less
            bin than the tables that were already monotonic. If
            under-sampled bins have been combined, then the probability_bin
            coordinate will have been reduced until all bins have more than
            the minimum_forecast_count if possible; a single under-sampled
            bin will be returned if combining all bins is still insufficient
            to reach the minimum_forecast_count.
        """
        threshold_coord = find_threshold_coordinate(reliability_table)
        if self.point_by_point:
            y_name = reliability_table.coord(axis="y").name()
            x_name = reliability_table.coord(axis="x").name()

        reliability_table_cubelist = iris.cube.CubeList()
        for rel_table_threshold in reliability_table.slices_over(threshold_coord):
            if self.point_by_point:
                for rel_table_point in rel_table_threshold.slices_over(
                    [y_name, x_name]
                ):
                    rel_table_point_emcam = self._enforce_min_count_and_montonicity(
                        rel_table_point
                    )
                    reliability_table_cubelist.append(rel_table_point_emcam)
            else:
                rel_table_processed = self._enforce_min_count_and_montonicity(
                    rel_table_threshold
                )
                reliability_table_cubelist.append(rel_table_processed)
        return reliability_table_cubelist


class ApplyReliabilityCalibration(PostProcessingPlugin):

    """
    A plugin for the application of reliability calibration to probability
    forecasts. This calibration is designed to improve the reliability of
    probability forecasts without significantly degrading their resolution.

    The method implemented here is described in Flowerdew J. 2014. Calibration
    is always applied as long as there are at least two bins within the input
    reliability table.

    References:
    Flowerdew J. 2014. Calibrating ensemble reliability whilst
    preserving spatial structure. Tellus, Ser. A Dyn. Meteorol.
    Oceanogr. 66.
    """

    def __init__(self, point_by_point: bool = False) -> None:
        """
        Initialise class for applying reliability calibration.

        Args:
            point_by_point:
                Whether to calibrate each point in the input cube independently.
                Utilising this option requires that each spatial point in the
                forecast cube has a corresponding spatial point in the
                reliability table. Please note this option is memory intensive and is
                unsuitable for gridded input.

        """
        self.threshold_coord = None
        self.point_by_point = point_by_point

    @staticmethod
    def _extract_matching_reliability_table(
        forecast: Cube, reliability_table: Union[Cube, CubeList]
    ) -> Cube:
        """
        Extract the reliability table with a threshold coordinate
        matching the forecast cube.
        If no matching reliability table is found raise an exception.

        Args:
            forecast:
                The forecast to be calibrated.
            reliability_table:
                The reliability table to use for applying calibration.

        Returns:
            A reliability table with a threshold coordinate that
            matches the forecast cube.

        Raises:
            ValueError: If no matching reliability table is found.
        """
        threshold_coord = find_threshold_coordinate(forecast)
        coord_values = {threshold_coord.name(): threshold_coord.points}
        constr = iris.Constraint(coord_values=coord_values)
        if isinstance(reliability_table, iris.cube.Cube):
            extracted = reliability_table.extract(constr)
        else:
            extracted = reliability_table.extract_cube(constr)
        if not extracted:
            raise ValueError(
                "No reliability table found to match threshold "
                f"{find_threshold_coordinate(forecast).points[0]}."
            )
        return extracted

    def _ensure_monotonicity_across_thresholds(self, cube: Cube) -> None:
        """
        Ensures that probabilities change monotonically relative to thresholds
        in the expected order, e.g. exceedance probabilities always remain the
        same or decrease as the threshold values increase, below threshold
        probabilities always remain the same or increase as the threshold
        values increase.

        Args:
            cube:
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
        thresholding = probability_is_above_or_below(cube)
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

    def _calculate_reliability_probabilities(
        self, reliability_table: Cube
    ) -> Tuple[Optional[ndarray], Optional[ndarray]]:
        """
        Calculates forecast probabilities and observation frequencies from the
        reliability table. If fewer than two bins are provided, Nones are
        returned as no calibration can be applied. Fewer than two bins can occur
        due to repeated combination of undersampled probability bins,
        please see :class:`.ManipulateReliabilityTable`.

        Args:
            reliability_table:
                A reliability table for a single threshold from which to
                calculate the forecast probabilities and observation
                frequencies.

        Returns:
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

        # If there are fewer than two bins, no calibration can be applied.
        if len(np.atleast_1d(forecast_count)) < 2:
            return None, None

        forecast_probability = np.array(forecast_probability_sum / forecast_count)
        observation_frequency = np.array(observation_count / forecast_count)

        return forecast_probability, observation_frequency

    @staticmethod
    def _interpolate(
        forecast_threshold: Union[MaskedArray, ndarray],
        reliability_probabilities: ndarray,
        observation_frequencies: ndarray,
    ) -> Union[MaskedArray, ndarray]:
        """
        Perform interpolation of the forecast probabilities using the
        reliability table data to produce the calibrated forecast. Where
        necessary linear extrapolation will be applied. Any mask in place on
        the forecast_threshold data is removed and reapplied after calibration.

        Args:
            forecast_threshold:
                The forecast probabilities to be calibrated.
            reliability_probabilities:
                Probabilities taken from the reliability tables.
            observation_frequencies:
                Observation frequencies that relate to the reliability
                probabilities, taken from the reliability tables.

        Returns:
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

    def _apply_calibration(
        self, forecast: Cube, reliability_table: Union[Cube, CubeList],
    ) -> Cube:
        """
        Apply reliability calibration to a forecast.

        Args:
            forecast:
                The forecast to be calibrated.
            reliability_table:
                The reliability table to use for applying calibration.

        Returns:
            The forecast cube following calibration.
        """
        calibrated_cubes = iris.cube.CubeList()
        forecast_thresholds = forecast.slices_over(self.threshold_coord)
        uncalibrated_thresholds = []

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

    def _apply_point_by_point_calibration(
        self, forecast: Cube, reliability_table: CubeList,
    ) -> Cube:
        """
        Apply point by point reliability calibration by iteratively picking a spatial
        coordinate within the forecast cube, extracting the forecast at that point
        and the reliability table corresponding to that point, then passing the
        extracted forecast and reliability table to _get_calibrated_forecast().

        Args:
            forecast:
                The forecast to be calibrated.
            reliability_table:
                The reliability table to use for applying calibration.

        Returns:
            The forecast cube following calibration.
        """

        calibrated_cubes = iris.cube.CubeList()
        y_name = forecast.coord(axis="y").name()
        x_name = forecast.coord(axis="x").name()

        # create list of dimensions
        dim_names = get_dim_coord_names(forecast)
        dim_associated_coords = {}

        # create dictionary with dimension name as keys, containing auxiliary
        # coordinates associated with that dimension.
        for dim_index, dim_name in enumerate(dim_names):
            associated_coords = [
                c for c in forecast.coords(dimensions=dim_index, dim_coords=False)
            ]
            dim_associated_coords[dim_name] = associated_coords

        # slice over the spatial dimension/s of the forecast cube
        # and apply reliability calibration separately to each slice
        # using a slice of the input reliability table at the same
        # spatial point
        for forecast_point in forecast.slices_over([y_name, x_name]):
            y_point = forecast_point.coord(y_name).points[0]
            x_point = forecast_point.coord(x_name).points[0]

            # create reliability table containing only those cubes
            # relating to the currently considered spatial point
            reliability_table_point = reliability_table.extract(
                iris.Constraint(coord_values={y_name: y_point, x_name: x_point})
            )

            calibrated_cube = self._apply_calibration(
                forecast=forecast_point, reliability_table=reliability_table_point
            )
            # remove auxiliary coordinates to ensure cubes can be merged into initial
            # format later
            for coords in dim_associated_coords.values():
                for coord in coords:
                    calibrated_cube.remove_coord(coord.name())
            calibrated_cubes.append(calibrated_cube)

        calibrated_forecast = calibrated_cubes.merge_cube()
        # add auxiliary coordinates back to the calibrated cube
        for dim_coord in dim_associated_coords.keys():
            for coord in dim_associated_coords[dim_coord]:
                dim = [x for x in calibrated_forecast.coord_dims(dim_coord)]
                calibrated_forecast.add_aux_coord(coord, dim)

        # ensure that calibrated forecast dimensions are in the same
        # order as the dimensions in the input forecast
        enforce_coordinate_ordering(calibrated_forecast, dim_names)

        return calibrated_forecast

    def process(
        self, forecast: Cube, reliability_table: Union[Cube, CubeList],
    ) -> Cube:
        """
        Apply reliability calibration to a forecast. The reliability table
        and the forecast cube must share an identical threshold coordinate.

        Args:
            forecast:
                The forecast to be calibrated.
            reliability_table:
                The reliability table to use for applying calibration.

        Returns:
            The forecast cube following calibration.
        """

        self.threshold_coord = find_threshold_coordinate(forecast)

        if self.point_by_point:
            calibrated_forecast = self._apply_point_by_point_calibration(
                forecast=forecast, reliability_table=reliability_table
            )

        else:
            calibrated_forecast = self._apply_calibration(
                forecast=forecast, reliability_table=reliability_table,
            )

        # enforce correct data type
        calibrated_forecast.data = calibrated_forecast.data.astype("float32")

        return calibrated_forecast
