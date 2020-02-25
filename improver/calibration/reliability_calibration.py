# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2019 Met Office.
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

import iris
from iris.exceptions import CoordinateNotFoundError
import numpy as np

from improver import BasePlugin
from improver.utilities.cube_manipulation import MergeCubes
from improver.metadata.utilities import generate_mandatory_attributes
from improver.metadata.probabilistic import find_threshold_coordinate
from improver.calibration.utilities import filter_non_matching_cubes


class ConstructReliabilityCalibrationTables(BasePlugin):

    """A plugin for creating and populating reliability calibration tables."""

    def __init__(self, n_probability_bins=5, single_value_limits=True):
        """
        Initialise class for creating reliability calibration tables. These
        tables include data columns entitled observation_count,
        sum_of_forecast_probabilities, and forecast_count, defined below.

        n_probability_bins (int):
            The total number of probability bins required in the reliability
            tables. If single value limits are turned on, these are included in
            this total.
        single_value_limits (bool):
            Mandates that the extrema bins (0 and 1) should be single valued,
            with a small precision tolerance, defined as 1.0E-6.
            This gives bins of 0 to 1.0E-6 and (1 - 1.0E-6) to 1.
        """
        self.single_value_tolerance = 1.0E-6
        self.probability_bins = self._define_probability_bins(
            n_probability_bins, single_value_limits)
        self.table_columns = np.array(
            ['observation_count', 'sum_of_forecast_probabilities',
             'forecast_count'])
        self.expected_table_shape = (len(self.table_columns),
                                     n_probability_bins)

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        bin_values = ', '.join(
            ['[{:1.2f} --> {:1.2f}]'.format(*item)
             for item in self.probability_bins])
        result = ('<ConstructReliabilityCalibrationTables: '
                  'probability_bins: {}>')
        return result.format(bin_values)

    def _define_probability_bins(self, n_probability_bins,
                                 single_value_limits):
        """
        Define equally sized probability bins for use in a reliability table.
        The range 0 to 1 is divided into ranges to give n_probability bins.
        Note that if single_value_limits is True then two bins will be created
        with values of 0 and 1, each with a width defined by
        self.single_value_tolerance.

        Args:
            n_probability_bins (int):
                The total number of probability bins desired in the
                reliability tables. This number includes the extrema bins
                (equals 0 and equals 1) if single value limits are turned on,
                in which case the minimum number of bins is 3.
            single_value_limits (bool):
                Mandates that the extrema bins (0 and 1) should be single
                valued with a width of single_value_tolerance. This gives
                bins of 0 to single_value_tolerance and
                (1 - single-value_tolerance) to 1.
        Returns:
            numpy.ndarray:
                An array of 2-element arrays that contain the bounds of the
                probability bins. These bounds are non-overlapping, with
                adjacent bin boundaries spaced at the smallest representable
                interval.
        Raises:
            ValueError: If trying to use single_value_limits with 2 or fewer
                        probability bins.
        """
        if single_value_limits:
            if n_probability_bins <= 2:
                msg = ("Cannot use single_value_limits with 2 or fewer "
                       "probability bins.")
                raise ValueError(msg)
            n_probability_bins = n_probability_bins - 2

        bin_lower = np.linspace(0, 1, n_probability_bins + 1, dtype=np.float32)
        bin_upper = np.nextafter(bin_lower, 0, dtype=np.float32)
        bin_upper[-1] = 1.
        bins = np.stack([bin_lower[:-1], bin_upper[1:]], 1).astype(np.float32)

        if single_value_limits:
            bins[0, 0] = np.nextafter(
                self.single_value_tolerance, 1, dtype=np.float32)
            bins[-1, 1] = np.nextafter(
                1. - self.single_value_tolerance, 0, dtype=np.float32)
            lowest_bin = np.array([0, self.single_value_tolerance],
                                  dtype=np.float32)
            highest_bin = np.array(
                [1. - self.single_value_tolerance, 1], dtype=np.float32)
            bins = np.vstack(
                [lowest_bin, bins, highest_bin]).astype(np.float32)
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
            values, long_name='probability_bin', units=1,
            bounds=self.probability_bins)
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
            long_name='table_row_index', units=1)
        name_coord = iris.coords.AuxCoord(
            self.table_columns, long_name='table_row_name', units=1)
        return index_coord, name_coord

    @staticmethod
    def _get_cycle_hours(forecast_reference_time):
        """
        Returns a set of integer representations of the hour of the
        forecast reference time (the cycle hour).

        Args:
            forecast_reference_time (iris.coord.DimCoord):
                The forecast_reference_time coordinate to extract cycle hours
                from.
        Returns:
            set:
                A set of integer representations of the cycle hours.
        """
        cycle_hours = []
        for frt in forecast_reference_time.cells():
            cycle_hours.append(np.int32(frt.point.hour))
        return set(cycle_hours)

    def _check_forecast_consistency(self, forecasts):
        """
        Checks that the forecast cubes are all from a consistent cycle and
        with a consistent forecast period.

        Args:
            forecasts (iris.cube.Cube):
        Raises:
            ValueError: Forecast cubes do not share consistent cycle hour and
                        forecast period.
        """
        n_cycle_hours = len(self._get_cycle_hours(
            forecasts.coord('forecast_reference_time')))
        try:
            n_forecast_periods, = forecasts.coord('forecast_period').shape
        except CoordinateNotFoundError:
            n_forecast_periods = 0
        if n_cycle_hours != 1 or n_forecast_periods != 1:
            msg = ('Forecasts have been provided from differing cycle hours '
                   'or forecast periods, or without these coordinates. These '
                   'coordinates should be present and consistent between '
                   'forecasts. Number of cycle hours found: {}, number of '
                   'forecast periods found: {}.')
            raise ValueError(msg.format(n_cycle_hours, n_forecast_periods))

    @staticmethod
    def _create_unified_frt_coord(forecast_reference_time):
        """
        Constructs a forecast reference time coordinate for the reliability
        calibration cube that records the range of forecast reference times
        of the forecasts used in populating the table.

        Args:
            forecast_reference_time (iris.coord.DimCoord):
                The forecast_reference_time coordinate to be used in the
                coordinate creation.
        Returns:
            iris.coord.DimCoord:
                A dimension coordinate containing the forecast reference time
                coordinate with suitable bounds. The coordinate point is that
                of the latest contributing forecast.
        """
        frt_point = forecast_reference_time.points.max()
        frt_bounds = (forecast_reference_time.points.min(), frt_point)
        return forecast_reference_time[0].copy(points=frt_point,
                                               bounds=frt_bounds)

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

        forecast_slice = next(forecast.slices_over(['time', threshold_coord]))
        expected_shape = self.expected_table_shape + forecast_slice.shape
        dummy_data = np.zeros((expected_shape))

        diagnostic = forecast.coord(var_name='threshold').name()
        attributes = self._define_metadata(forecast)

        # Define reliability table specific coordinates
        probability_bins_coord = self._create_probability_bins_coord()
        reliability_index_coord, reliability_name_coord = (
            self._create_reliability_table_coords())
        frt_coord = self._create_unified_frt_coord(
            forecast.coord('forecast_reference_time'))

        # List of required non-spatial coordinates from the forecast
        non_spatial_coords = ['forecast_period', diagnostic]

        # Construct a list of coordinates in the desired order
        dim_coords = [forecast.coord(axis=dim).name()
                      for dim in ['x', 'y']]
        dim_coords_and_dims = _get_coords_and_dims(dim_coords)
        aux_coords_and_dims = _get_coords_and_dims(non_spatial_coords)
        dim_coords_and_dims.append((reliability_index_coord, 0))
        aux_coords_and_dims.append((reliability_name_coord, 0))
        dim_coords_and_dims.append((probability_bins_coord, 1))

        reliability_cube = iris.cube.Cube(
            dummy_data, units=1, attributes=attributes,
            dim_coords_and_dims=dim_coords_and_dims,
            aux_coords_and_dims=aux_coords_and_dims)
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

            observation_mask = (((forecast >= bin_min) & (forecast <= bin_max))
                                & (np.isclose(truth, 1))).astype(int)
            forecast_mask = ((forecast >= bin_min) &
                             (forecast <= bin_max)).astype(int)
            forecasts_probability_values = forecast * forecast_mask

            observation_counts.append(observation_mask)
            forecast_probabilities.append(forecasts_probability_values)
            forecast_counts.append(forecast_mask)

        reliability_table = np.stack([
                                np.stack(observation_counts),
                                np.stack(forecast_probabilities),
                                np.stack(forecast_counts)])
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
            historic_forecasts, truths)

        threshold_coord = find_threshold_coordinate(historic_forecasts)
        truth_threshold_coord = find_threshold_coordinate(truths)
        if not threshold_coord == truth_threshold_coord:
            msg = "Threshold coordinates differ between forecasts and truths."
            raise ValueError(msg)

        time_coord = historic_forecasts.coord('time')

        self._check_forecast_consistency(historic_forecasts)
        reliability_cube = self._create_reliability_table_cube(
            historic_forecasts, threshold_coord)

        reliability_tables = iris.cube.CubeList()
        threshold_slices = zip(historic_forecasts.slices_over(threshold_coord),
                               truths.slices_over(threshold_coord))
        for forecast_slice, truth_slice in threshold_slices:

            threshold_reliability = []
            time_slices = zip(forecast_slice.slices_over(time_coord),
                              truth_slice.slices_over(time_coord))
            for forecast, truth in time_slices:

                reliability_table = (
                    self._populate_reliability_bins(
                        forecast.data, truth.data))

                threshold_reliability.append(reliability_table)

            # Stack and sum reliability tables for all times
            table_values = np.stack(threshold_reliability)
            table_values = np.sum(table_values, axis=0, dtype=np.float32)

            reliability_entry = reliability_cube.copy(data=table_values)
            reliability_entry.replace_coord(
                forecast_slice.coord(threshold_coord))
            reliability_tables.append(reliability_entry)

        return MergeCubes()(reliability_tables)
