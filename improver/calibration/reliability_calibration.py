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
import numpy as np

from improver import BasePlugin
from improver.utilities.cube_manipulation import merge_cubes
from improver.metadata.utilities import generate_mandatory_attributes


class ConstructRealizationCalibrationTables(BasePlugin):

    """A plugin for creating and populating reliability calibration tables."""

    def __init__(self, n_probability_bins=5, single_value_limits=True):
        """
        Initialise class for creating realization calibration tables.

        n_probability_bins (int):
            The total number of probability bins required in the reliability
            tables. If single value limits are turned on, these are included in
            this total.
        single_value_limits (bool):
            Mandates that the extrema bins (0 and 1) should be single valued,
            with a small precision tolerance of 1.0E-6, e.g. 0 to 1.0E-6 for
            the lowest bin, and 1 - 1.0E-6 to 1 for the highest bin.
        """
        self.probability_bins = self._define_probability_bins(
            n_probability_bins, single_value_limits)
        self.expected_table_shape = (3, n_probability_bins)

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        bin_values = ', '.join(
            ['[{:1.2f} --> {:1.2f}]'.format(*item)
             for item in self.probability_bins])
        result = ('<ConstructRealizationCalibrationTables: '
                  'probability_bins: {}>')
        return result.format(bin_values)

    @staticmethod
    def _define_probability_bins(n_probability_bins, single_value_limits):
        """
        Define equally spaced probability bins for use in a reliability table.
        The range 0 to 1 is divided into ranges to give n_probability bins.
        Note that if single_value_limits is True then two bins will be created
        with values of 0 and 1 and a width of 1.0E-6. The remaining range will
        be split into n_probability_bins - 2 ranges.

        Args:
            n_probability_bins (int):
                The total number of probability bins required in the
                reliability tables. If single value limits are turned on, these
                are included in this total.
            single_value_limits (bool):
                Mandates that the extrema bins (0 and 1) should be single
                valued, with a small precision tolerance of 1.0E-6, e.g. 0 to
                1.0E-6 for the lowest bin, and 1 - 1.0E-6 to 1 for the highest
                bin.
        Returns:
            numpy.ndarray:
                An array of 2-element arrays that contain the bounds of the
                probability bins.
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
            fixed_bin_width = 1.0E-6
            bins[0, 0] = np.nextafter(
                fixed_bin_width, 1, dtype=np.float32)
            bins[-1, 1] = np.nextafter(
                1. - fixed_bin_width, 0, dtype=np.float32)
            lowest_bin = np.array([0, fixed_bin_width], dtype=np.float32)
            highest_bin = np.array(
                [1. - fixed_bin_width, 1], dtype=np.float32)
            bins = np.vstack(
                [lowest_bin, bins, highest_bin]).astype(np.float32)

        return bins

    def _create_probability_bins_coord(self):
        """
        Construct a dimensions coordinate describing the probability bins
        of the reliability table.

        Returns:
            iris.coord.DimCoord:
                A dimcoord describing probability bins.
        """
        values = np.mean(self.probability_bins, axis=1, dtype=np.float32)
        probability_bins_coord = iris.coords.DimCoord(
            values, long_name='probability_bin', units=1,
            bounds=self.probability_bins)
        return probability_bins_coord

    @staticmethod
    def _create_realiability_table_coords():
        """
        Construct coordinates that describe the reliability table rows. These
        are observation_count, sum_of_forecast_probabilities, and
        forecast_count. The order used here is the order in which the table
        data is populated, so these must remain consistent with the
        _populate_reliability_bins function.

        Returns:
            (tuple): tuple containing:
                **index_coord** (iris.coord.DimCoord):
                    A numerical index coordinate.
                **name_coord** (iris.coord.AuxCoord):
                    An auxiliary coordinate that assigns namea to the index
                    coordinates, where these names correspond to the
                    reliability table rows.
        """
        index_coord = iris.coords.DimCoord(
            np.arange((3), dtype=np.int32), long_name='reliability_index',
            units=1)
        name_coord = iris.coords.AuxCoord(
            np.array(
                ['observation_count', 'sum_of_forecast_probabilities',
                 'forecast_count']), long_name='reliability_name', units=1)
        return index_coord, name_coord

    @staticmethod
    def _create_cycle_hour_coord(forecast_reference_time):
        """
        Constructs a coordinate that contains the cycle hour for which the
        reliability table is valid.

        Args:
            forecast_reference_time (iris.coord.DimCoord):
        Returns:
            iris.coord.DimCoord:
                A coord containing an integer unitless representation of the
                cycle hour.
        """
        dt_object = next(forecast_reference_time.cells()).point
        cycle_hour = np.int32(dt_object.hour)
        cycle_coord = iris.coords.DimCoord(cycle_hour, long_name='cycle_hour',
                                           units=1)
        return cycle_coord

    @staticmethod
    def _define_metadata(forecast_slice, diagnostic):
        """
        Define metadata that is specifically required for reliability table
        cubes, whilst ensuring any mandatory attributes are also populated.

        Args:
            forecast_slice (iris.cube.Cube):
                The source cube from which to get pre-existing metadata of use.
            diagnostic (str):
                The name of the diagnostic within the cube.
        Returns:
            dict:
                A dictionary of attributes that are appropriate for the
                reliability table cube.
        """
        attributes = generate_mandatory_attributes([forecast_slice])
        attributes["title"] = "Reliability calibration data table"
        attributes["diagnostic_standard_name"] = diagnostic
        return attributes

    def _create_reliability_table_cube(self, reliability_table_data,
                                       forecast_slice):
        """
        Construct a reliability table cube and populate it with the provided
        data.

        Args:
            reliability_table_data (numpy.ndarray):
                Array of reliability data, with values for each spatial
                coordinate. The data order should correspond to reliability
                table coords, then probability bin coords, followed by spatial
                dimensions.
            forecast_slice (iris.cube.Cube):
                A cube slice across the spatial dimensions of the forecast
                data. This slice provides the time and threshold values that
                relate to the reliability_table_data.
        Returns:
            iris.cube.Cube:
                A reliability table cube.
        Raises:
            ValueError: If the reliability table data does not match the
                        expected dimensions.
        """
        def _get_coords_and_dims(coord_names):
            coords_and_dims = []
            for coord_name in coord_names:
                crd = forecast_slice.coord(coord_name)
                crd_dim = forecast_slice.coord_dims(crd)
                crd_dim = crd_dim[0] + 2 if crd_dim else ()
                coords_and_dims.append((crd, crd_dim))
            return coords_and_dims

        expected_shape = self.expected_table_shape + forecast_slice.shape
        if not reliability_table_data.shape == expected_shape:
            msg = ("The reliability table data does not have the expected "
                   "dimensions.")
            raise ValueError(msg)

        diagnostic = forecast_slice.coord(var_name='threshold').name()
        attributes = self._define_metadata(forecast_slice, diagnostic)

        # Define reliability table specific coordinates
        probability_bins_coord = self._create_probability_bins_coord()
        reliability_index_coord, reliability_name_coord = (
            self._create_realiability_table_coords())
        cycle_coord = self._create_cycle_hour_coord(
            forecast_slice.coord('forecast_reference_time'))

        # List of required non-spatial coordinates from the forecast_slice
        non_spatial_coords = ['forecast_period', diagnostic]

        # Construct a list of coordinates in the desired order
        dim_coords = [forecast_slice.coord(axis=dim).name()
                      for dim in ['x', 'y']]
        dim_coords_and_dims = _get_coords_and_dims(dim_coords)
        aux_coords_and_dims = _get_coords_and_dims(non_spatial_coords)
        dim_coords_and_dims.append((reliability_index_coord, 0))
        aux_coords_and_dims.append((reliability_name_coord, 0))
        dim_coords_and_dims.append((probability_bins_coord, 1))

        reliability_cube = iris.cube.Cube(
            reliability_table_data, units=1, attributes=attributes,
            dim_coords_and_dims=dim_coords_and_dims,
            aux_coords_and_dims=aux_coords_and_dims)
        reliability_cube.add_aux_coord(cycle_coord)
        reliability_cube.rename("reliability_calibration_table")

        return reliability_cube

    def _populate_reliability_bins(self, forecast, truth, threshold):
        """
        For an x-y slice at a single validity time and threshold, populate
        a reliability table using the provided truth.

        Args:
            forecast (numpy.ndarray):
                An array containing data over an xy slice for a single validity
                time and threshold.
            truth (numpy.ndarray):
                An array containing a gridded truth at an equivalent validity
                time to the forecast array.
            threshold (float):
                The threshold value against which the truth should be compared.
        Returns:
            numpy.ndarray:
                An array containing reliability table data for a single time
                and threshold.
        """
        observation_counts = []
        forecast_probabilities = []
        forecast_counts = []

        for bin_min, bin_max in self.probability_bins:

            observation_mask = (((forecast >= bin_min) & (forecast <= bin_max))
                                & (truth > threshold)).astype(int)
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
        threshold constructed from all the provided historic forecasts and
        truths.

        Args:
            historic_forecasts (iris.cube.CubeList):
                A cube list of historic forecast data.
            truths (iris.cube.Cube):
                A cube list of historic analyses that act as truth.
        Returns:
            iris.cube.CubeList:
                A cubelist of reliability table cubes, one for each threshold
                in the historic forecast cubes.
        """
        historic_forecasts = merge_cubes(historic_forecasts)
        truths = merge_cubes(truths)

        threshold_coord = historic_forecasts.coord(var_name='threshold')
        time_coord = historic_forecasts.coord('time')

        reliability_tables = iris.cube.CubeList()
        for threshold_slice in historic_forecasts.slices_over(threshold_coord):

            threshold, = threshold_slice.coord(threshold_coord).points

            threshold_reliability = []
            for time_slice in threshold_slice.slices_over(time_coord):

                time = time_slice.coord('time')
                truth = truths.extract(
                    iris.Constraint(time=next(time.cells()).point))

                reliability_table = (
                    self._populate_reliability_bins(
                        time_slice.data, truth.data, threshold))

                threshold_reliability.append(reliability_table)

            # Stack and sum reliability tables for all times
            table_values = np.stack(threshold_reliability)
            table_values = np.sum(table_values, axis=0, dtype=np.float32)

            reliability_cube = self._create_reliability_table_cube(
                table_values, time_slice)
            reliability_tables.append(reliability_cube)

        return reliability_tables
