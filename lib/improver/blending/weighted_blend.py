# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2018 Met Office.
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
"""Module containing classes for doing weighted blending by collapsing a
   whole dimension."""

import numpy as np
import iris
from iris.analysis import Aggregator
from iris.exceptions import CoordinateNotFoundError

from improver.utilities.cube_manipulation import sort_coord_in_cube
from improver.utilities.cube_checker import find_percentile_coordinate
from improver.utilities.cube_manipulation import enforce_coordinate_ordering
from improver.utilities.temporal import (
    cycletime_to_datetime, cycletime_to_number, forecast_period_coord,
    unify_forecast_reference_time, find_latest_cycletime)


def rationalise_blend_time_coords(
        cubelist, blend_coord, cycletime=None, weighting_coord=None):
    """
    Updates time coordinates on unmerged input cubes before blending depending
    on the coordinate over which the blend will be performed.  Modifies cubes
    in place.

    If blend_coord is forecast_reference_time, ensures the cube does not have
    a forecast_period dimension.  If weighting_coord is forecast_period,
    equalises forecast_reference_time on each cube before blending.

    Args:
        cubelist (iris.cube.CubeList):
            List of cubes containing data to be blended
        blend_coord (str):
            Name of coordinate over which the blend will be performed

    Kwargs:
        cycletime (str or None):
            The cycletime in a YYYYMMDDTHHMMZ format e.g. 20171122T0100Z
        weighting_coord (str or None):
            The coordinate across which weights will be scaled in a
            multi-model blend.

    Raises:
        ValueError: if forecast_reference_time (to be unified) is a
            dimension coordinate
    """
    if "forecast_reference_time" in blend_coord:
        for cube in cubelist:
            coord_names = [x.name() for x in cube.coords()]
            if "forecast_period" in coord_names:
                cube.remove_coord("forecast_period")

    # if blending models using weights by forecast period, set forecast
    # reference times to current cycle time
    if ("model" in blend_coord and weighting_coord is not None
            and "forecast_period" in weighting_coord):
        if cycletime is None:
            cycletime = find_latest_cycletime(cubelist)
        else:
            cycletime = cycletime_to_datetime(cycletime)
        cubelist = unify_forecast_reference_time(cubelist, cycletime)


def conform_metadata(
        cube, cube_orig, coord, cycletime=None):
    """Ensure that the metadata conforms after blending together across
    the chosen coordinate.

    The metadata adjustments are:
        - Forecast reference time: If a cycletime is not present, the
          most recent available forecast_reference_time is used.
          If a cycletime is present, the cycletime is used as the
          forecast_reference_time instead.
        - Forecast period: If a forecast_period coordinate is present,
          and cycletime is not present, the lowest forecast_period is
          used. If a forecast_period coordinate is present, and the
          cycletime is present, forecast_periods are forceably calculated
          from the time and forecast_reference_time coordinate. This is
          because, if the cycletime is present, then the
          forecast_reference_time will also have been just re-calculated, so
          the forecast_period coordinate needs to be reset to match the
          newly calculated forecast_reference_time.
        - Forecast reference time and time: If forecast_reference_time and
          time coordinates are present, then a forecast_period coordinate is
          calculated and added to the cube.
        - Model_id, model_realization and realization coordinates are removed.
        - A title attribute is added to the cube if none is found. Otherwise
          the attributes are unchanged.

    Args:
        cube (iris.cube.Cube):
            Cube containing the metadata to be adjusted.
        cube_orig (iris.cube.Cube):
            Cube containing metadata that may be useful for adjusting
            metadata on the `cube` variable.
        coord (str):
            Coordinate that has been blended. This allows specific metadata
            changes to be limited to whichever coordinate is being blended.

    Keyword Args:
        cycletime (str):
            The cycletime in a YYYYMMDDTHHMMZ format e.g. 20171122T0100Z.

    Returns:
        cube (iris.cube.Cube):
            Cube containing the adjusted metadata.

    """
    # unify time coordinates for cycle and grid (model) blends
    if coord in ["forecast_reference_time", "model"]:
        if cube.coords("forecast_reference_time"):
            if cycletime is None:
                new_cycletime = (
                    np.max(cube_orig.coord("forecast_reference_time").points))
            else:
                cycletime_units = (
                    cube_orig.coord("forecast_reference_time").units.origin)
                cycletime_calendar = (
                    cube.coord("forecast_reference_time").units.calendar)
                new_cycletime = cycletime_to_number(
                    cycletime, time_unit=cycletime_units,
                    calendar=cycletime_calendar)
                # Preserve the data type to avoid converting ints to floats.
                frt_type = cube.coord("forecast_reference_time").dtype
                new_cycletime = np.round(new_cycletime).astype(frt_type)
            cube.coord("forecast_reference_time").points = new_cycletime
            cube.coord("forecast_reference_time").bounds = None

        if cube.coords("forecast_period"):
            forecast_period = (
                forecast_period_coord(cube,
                                      force_lead_time_calculation=True))
            forecast_period.bounds = None
            forecast_period.convert_units(cube.coord("forecast_period").units)
            forecast_period.var_name = cube.coord("forecast_period").var_name
            cube.replace_coord(forecast_period)
        elif cube.coords("forecast_reference_time") and cube.coords("time"):
            forecast_period = (
                forecast_period_coord(cube))
            ndim = cube.coord_dims("time")
            cube.add_aux_coord(forecast_period, data_dims=ndim)

    # update blended cube attributes
    if "title" not in cube.attributes.keys():
        cube.attributes["title"] = "IMPROVER Model Forecast"

    # remove appropriate scalar coordinates
    for crd in ["model_id", "model_realization", "realization"]:
        if cube.coords(crd) and cube.coord(crd).shape == (1,):
            cube.remove_coord(crd)

    return cube


class PercentileBlendingAggregator:
    """Class for the percentile blending aggregator

       This class implements the method described by Combining Probabilities
       by Caroline Jones, 2017. This method implements blending in probability
       space.

       The steps are:
           1. At each geographic point in the cube we take the percentile
              threshold's values across the percentile dimensional coordinate.
              We recalculate, using linear interpolation, their probabilities
              in the pdf of the other points across the coordinate we are
              blending over. Thus at each point we have a set of thresholds
              and their corresponding probability values in each of the
              probability spaces across the blending coordinate.
           2. We do a weighted blend across all the probability spaces,
              combining all the thresholds in all the points in the coordinate
              we are blending over. This gives us an array of thresholds and an
              array of blended probailities for each of the grid points.
           3. We convert back to the original percentile values, again using
              linear interpolation, resulting in blended values at each of the
              original percentiles.

       References:
            Combining Probabilities by Caroline Jones, 2017:
            https://github.com/metoppv/improver/files/1128018/
            Combining_Probabilities.pdf
    """

    def __init__(self):
        """
        Initialise class.
        """
        pass

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        result = ('<PercentileBlendingAggregator>')
        return result

    @staticmethod
    def aggregate(data, axis, arr_percent, arr_weights, perc_dim):
        """ Blend percentile aggregate function to blend percentile data
            along a given axis of a cube.

        Args:
            data (np.array):
                   Array containing the data to blend
            axis (int):
                   The index of the coordinate dimension in the cube. This
                   dimension will be aggregated over.
            arr_percent(np.array):
                     Array of percentile values e.g
                     [0, 20.0, 50.0, 70.0, 100.0],
                     same size as the percentile dimension of data.
            arr_weights(np.array):
                     Array of weights, same size as the axis dimension of data.
            perc_dim (int):
                     The index of the percentile coordinate
            (Note percent and weights have special meaning in Aggregator
             hence the rename.)

        Returns:
            result (np.array):
                     containing the weighted percentile blend data across
                     the chosen coord. The dimension associated with axis
                     has been collapsed, and the rest of the dimensions remain.
        """
        # Iris aggregators support indexing from the end of the array.
        if axis < 0:
            axis += data.ndim
        # Firstly ensure axis coordinate and percentile coordinate
        # are indexed as the first and second values in the data array
        data = np.moveaxis(data, [perc_dim, axis], [1, 0])
        # Determine the rest of the shape
        shape = data.shape[2:]
        input_shape = [data.shape[0],
                       data.shape[1],
                       np.prod(shape, dtype=int)]
        # Flatten the data that is not percentile or coord data
        data = data.reshape(input_shape)
        arr_weights = arr_weights.reshape(input_shape)
        # Create the resulting data array, which is the shape of the original
        # data without dimension we are collapsing over
        result = np.zeros(input_shape[1:], dtype=np.float32)
        # Loop over the flattened data, i.e. across all the data points in
        # each slice of the coordinate we are collapsing over, finding the
        # blended percentile values at each point.
        for i in range(data.shape[-1]):
            result[:, i] = (
                PercentileBlendingAggregator.blend_percentiles(
                    data[:, :, i], arr_percent, arr_weights[:, :, i]))
        # Reshape the data and put the percentile dimension
        # back in the right place
        shape = arr_percent.shape + shape
        result = result.reshape(shape)
        # Percentile is now the leading dimension in the result. This needs
        # to move back to where it was in the input data. The result has
        # one less dimension than the original data as we have collapsed
        # one dimension.
        # If we have collapsed a dimension that was before the percentile
        # dimension in the input data, the percentile dimension moves forwards
        # one place compared to the original percentile dimension.
        if axis < perc_dim:
            result = np.moveaxis(result, 0, perc_dim-1)
        # Else we move the percentile dimension back to where it was in the
        # input data, as we have collapsed along a dimension that came after
        # it in the input cube.
        else:
            result = np.moveaxis(result, 0, perc_dim)
        return result

    @staticmethod
    def blend_percentiles(perc_values, percentiles, weights):
        """ Blend percentiles function, to calculate the weighted blend across
            a given axis of percentile data for a single grid point.

        Args:
            perc_values (np.array):
                Array containing the percentile values to blend, with
                shape: (length of coord to blend, num of percentiles)
            percentiles (np.array):
                Array of percentile values e.g [0, 20.0, 50.0, 70.0, 100.0],
                same size as the percentile dimension of data.
            weights (np.array):
                Array of weights, same size as the axis dimension of data,
                that we will blend over.

        Returns:
            new_combined_perc (np.array):
                Array containing the weighted percentile blend data
                across the chosen coord
        """
        # Find the size of the dimension we want to blend over.
        num = perc_values.shape[0]
        # Create an array to store the weighted blending pdf
        combined_pdf = np.zeros((num, len(percentiles)), dtype=np.float32)
        # Loop over the axis we are blending over finding the values for the
        # probability at each threshold in the pdf, for each of the other
        # points in the axis we are blending over. Use the values from the
        # percentiles if we are at the same point, otherwise use linear
        # interpolation.
        # Then add the probabilities multiplied by the correct weight to the
        # running total.
        for i in range(0, num):
            for j in range(0, num):
                if i == j:
                    recalc_values_in_pdf = percentiles
                else:
                    recalc_values_in_pdf = np.interp(perc_values[i],
                                                     perc_values[j],
                                                     percentiles)
                # Add the resulting probabilities multiplied by the right
                # weight to the running total for the combined pdf.
                combined_pdf[i] += recalc_values_in_pdf*weights[j]

        # Combine and sort the threshold values for all the points
        # we are blending.
        combined_perc_thres_data = np.sort(perc_values.flatten())

        # Combine and sort blended probability values.
        combined_perc_values = np.sort(combined_pdf.flatten())

        # Find the percentile values from this combined data by interpolating
        # back from probability values to the original percentiles.
        new_combined_perc = np.interp(percentiles,
                                      combined_perc_values,
                                      combined_perc_thres_data).astype(
                                          np.float32)
        return new_combined_perc


class MaxProbabilityAggregator:
    """Class for the Aggregator used to calculate the maximum weighted
       probability.

       1. Find the weighted probabilities for each point in the dimension of
          interest by multiplying each probability by the corresponding weight.
       2. Find the maximum weighted probability and return the array with one
          less dimension than the input array.
    """

    def __init__(self):
        """
        Initialise class.
        """
        pass

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        result = ('<MaxProbabilityAggregator>')
        return result

    @staticmethod
    def aggregate(data, axis, arr_weights):
        """ Max probability aggregator method. Used to find the maximum
            weighted probability along a given axis.

        Args:
            data (np.array):
                Array containing the data to blend
            axis (int):
                The index of the coordinate dimension in the cube. This
                dimension will be aggregated over.
            arr_weights (np.array):
                Array of weights, same size as the axis dimension of data.
        Returns:
            result (np.array):
                The data collapsed along the axis dimension, containing the
                maximum weighted probability.
        """
        # Iris aggregators support indexing from the end of the array.
        if axis < 0:
            axis += data.ndim

        # Maintain old functionality, though weights passed in through the
        # weighted blending plugin should always match.
        if arr_weights.shape != data.shape:
            # Reshape the weights to match the shape of the data.
            shape = [len(arr_weights) if i == axis else 1
                     for i in range(data.ndim)]
            arr_weights = arr_weights.reshape(tuple(shape))

        # Calculate the weighted probabilities
        weighted_probs = data*arr_weights
        # Find the maximum along the axis of interest
        result = np.max(weighted_probs, axis=axis)
        return result


class WeightedBlendAcrossWholeDimension:
    """Apply a Weighted blend to a cube, collapsing across the whole
       dimension. Uses one of two methods, either weighted average, or
       the maximum of the weighted probabilities."""

    def __init__(self, coord, weighting_mode, cycletime=None,
                 timeblending=False):
        """Set up for a Weighted Blending plugin

        Args:
            coord (string):
                The name of the coordinate dimension over which the cube will
                be blended.
            weighting_mode (string):
                One of 'weighted_maximum' or 'weighted_mean':
                 - Weighted mean: a normal weighted average over the coordinate
                   of interest.
                 - Weighted_maximum: the points in the coordinate of interest
                   are multiplied by the weights and then the maximum is taken.

        Keyword Args:
            cycletime (str):
                The cycletime in a YYYYMMDDTHHMMZ format e.g. 20171122T0100Z.
            timeblending (bool):
                With the default of False the cube being blended will be
                checked to ensure that slices across the blending coordinate
                all have the same validity time. Setting this to True will
                bypass this test, as is necessary for triangular time
                blending.

        Raises:
            ValueError : If an invalid weighting_mode is given.
        """
        self.coord = coord
        if weighting_mode not in ['weighted_maximum', 'weighted_mean']:
            msg = ("weighting_mode: {} is not recognised, must be either "
                   "weighted_maximum or weighted_mean").format(weighting_mode)
            raise ValueError(msg)
        self.mode = weighting_mode
        self.cycletime = cycletime
        self.timeblending = timeblending

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        description = ('<WeightedBlendAcrossWholeDimension:'
                       ' coord = {0:}, weighting_mode = {1:},'
                       ' cycletime = {2:}, timeblending: {3:}>')
        return description.format(self.coord, self.mode, self.cycletime,
                                  self.timeblending)

    def check_percentile_coord(self, cube):
        """
        Determines if the cube to be blended has a percentile dimension
        coordinate.

        Args:
            cube (iris.cube.Cube):
                The cube to be checked for a percentile coordinate.
        Returns:
            None or perc_coord (iris.coords.DimCoord):
                None if no percentile dimension coordinate is found. If
                such a coordinate is found it is returned.
        Raises:
            ValueError : If there is a percentile coord and it is not a
                dimension coord in the cube.
            ValueError : If there is a percentile dimension with only one
                point, we need at least two points in order to do the blending.
            ValueError : If there is a percentile dimension on the cube and the
                mode for blending is 'weighted_maximum'
        """
        try:
            perc_coord = find_percentile_coordinate(cube)
            perc_dim = cube.coord_dims(perc_coord.name())
            if not perc_dim:
                msg = ('The percentile coord must be a dimension '
                       'of the cube.')
                raise ValueError(msg)
            # Check the percentile coordinate has more than one point,
            # otherwise raise an error as we won't be able to blend.
            if len(perc_coord.points) < 2.0:
                msg = ('Percentile coordinate does not have enough points'
                       ' in order to blend. Must have at least 2 percentiles.')
                raise ValueError(msg)
            # If we have a percentile dimension and the mode is 'max' raise an
            # exception.
            if perc_coord and self.mode == 'weighted_maximum':
                msg = ('The "weighted_maximum" mode is not supported for '
                       'percentile data.')
                raise ValueError(msg)

            return perc_coord
        except CoordinateNotFoundError:
            return None

    def check_compatible_time_points(self, cube):
        """
        Check that the time coordinate only contains a single time. Data
        varying over the blending coordinate should all be for the same
        validity time unless we are triangular time blending. In this case the
        timeblending flag should be true and this function will not raise an
        exception.

        Args:
            cube (iris.cube.Cube):
                The cube upon which the compatibility of the time coords is
                being checked.
        Raises:
            ValueError : If blending over forecast reference time on a cube
                         with multiple times.
        """
        if self.timeblending is True:
            return

        time_points = cube.coord("time").points
        if len(set(time_points)) > 1:
            msg = ("Attempting to blend data for different validity times. The"
                   " time points within the input cube are {}".format(
                    time_points))
            raise ValueError(msg)

    def shape_weights(self, cube, weights):
        """
        The function shapes weights to match the diagnostic cube. A 1D cube of
        weights that vary across the blending coordinate will be broadcast to
        match the complete multidimensional cube shape. A multidimensional cube
        of weights will be checked to ensure that the coordinate names match
        between the two cubes. If they match the order will be enforced and
        then the shape will be checked. If the shapes match the weights will be
        returned as an array.

        Args:
            cube (iris.cube.Cube):
                The data cube on which a coordinate is being blended.
            weights (iris.cube.Cube):
                Cube of blending weights.
        Returns:
            weights_array (np.array):
                An array of weights that matches the cube data shape.
        Raises:
            ValueError: If weights cube coordinates do not match the diagnostic
                        cube in the case of a multidimensional weights cube.
            ValueError: If weights cube shape is not broadcastable to the data
                        cube shape.
        """
        # Check that a multidimensional weights cube has coordinates that match
        # the diagnostic cube. Checking names only to not to be too exacting.
        weight_dims = [crd.name() for crd in weights.coords(dim_coords=True)]
        cube_dims = [crd.name() for crd in cube.coords(dim_coords=True)]
        if set(weight_dims) == set(cube_dims):
            enforce_coordinate_ordering(weights, cube_dims)
        #elif len(weight_dims) > 1:
            #msg = ("Multidimensional weights cube does not contain the same "
                   #"coordinates as the diagnostic cube. Weights: {}, "
                   #"Diagnostic: {}".format(weight_dims, cube_dims))
            #raise ValueError(msg)

        msg = ("Weights cube is not a compatible shape with the data cube. "
               "Weights: {}, Diagnostic: {}".format(weights.shape, cube.shape))

        if weights.shape == cube.shape:
            # Multi-dimensional weights array provided that matches cube.
            weights_array = weights.data.astype(np.float32)
        else:
            # 1D array of weights across the blending coordinate.
            dim_map = []
            dim_coords = [coord.name() for coord in weights.dim_coords]
            print(weights)
            #for i in range(weights.ndim):
                #dim_coords.append(weights.coord(dimensions=[i]).name())
            #if self.coord not in dim_coords:
                #dim_coords.append(self.coord)
            for dim_coord in dim_coords:
                dim_map.append(cube.coord_dims(dim_coord)[0])
            print(dim_coords)
            print(dim_map)
            #coord_dim_thres = cube.coord_dims(self.coord)
            try:
                weights_array = iris.util.broadcast_to_shape(
                    np.array(weights.data, dtype=np.float32),
                    cube.shape, tuple(dim_map))
            except ValueError:
                raise ValueError(msg)

        return weights_array

    @staticmethod
    def check_weights(weights, blend_dim):
        """
        Checks that weights across the blending dimension sum up to 1.

        Args:
            weights (np.array):
                Array of weights shaped to match the data cube.
            blend_dim (integer):
                The dimension in the weights array that is being collapsed.
        Raises:
            ValueError: Raised if the weights do not sum to 1 over the blending
                        dimension.
        """
        sum_of_weights = np.sum(weights, axis=blend_dim)
        msg = ('Weights do not sum to 1 over the blending coordinate. Max sum '
               'of weights: {}'.format(sum_of_weights.max()))
        sum_of_non_zero_weights = sum_of_weights[sum_of_weights > 0]
        if not (np.isclose(sum_of_non_zero_weights, 1)).all():
            raise ValueError(msg)

    def non_percentile_weights(self, cube, weights, custom_aggregator=False):
        """
        Given a 1 or multidimensional cube of weights, reshape and broadcast
        these in such a way as to make them applicable to the data cube. If no
        weights are provided, an array of weights is returned that equally
        weights all slices across the blending coordinate of the cube.

        The output of this function is different depending upon the method
        being used to blend the data.

        weighted_mean:
            reshape and broadcast to match data shape.
        weighted_maximum:
            reshape and broadcast to match data shape, before reordering in
            anticipation of the blending coord being shifted to the -1 position
            by the custom aggregator.

        Args:
            cube (iris.cube.Cube):
                The data cube on which a coordinate is being blended.
            weights (iris.cube.Cube or None):
                Cube of blending weights or None.
        Returns:
            weights_array (np.array):
                An array of weights that matches the cube data shape.
        """
        print("\ntrying to match cube and weights. Cube:\n{}\n weights:\n{}\n\n".format(cube, weights))
        if weights:
            weights_array = self.shape_weights(cube, weights)
        else:
            number_of_fields, = cube.coord(self.coord).shape
            weights_array = (
                np.broadcast_to(1./number_of_fields, cube.shape).astype(
                    np.float32))

        # Our custom aggregator moves the blending coordinate to the -1
        # index, so we need to reshape the weights to match.
        if custom_aggregator:
            coord_dim = cube.coord_dims(self.coord)
            weights_array = np.moveaxis(weights_array, coord_dim, -1)
            self.check_weights(weights_array, -1)
        else:
            blend_dim, = cube.coord_dims(self.coord)
            self.check_weights(weights_array, blend_dim)

        return weights_array.astype(np.float32)

    def percentile_weights(self, cube, weights, perc_coord):
        """
        Given a 1, or multidimensional cube of weights, reshape and broadcast
        these in such a way as to make them applicable to the data cube. If no
        weights are provided, an array of weights is returned that equally
        weights all slices across the blending coordinate of the cube.

        For percentiles the dimensionality of the weights cube is checked
        against the cube without including the percentile coordinate for
        which no weights are likely to ever be provided (e.g. we don't want to
        weight different percentiles differently across the blending
        coordinate). Reshape and broadcast to match the data shape excluding
        the percentile dimension before finally broadcasting to match at the
        end.

        Args:
            cube (iris.cube.Cube):
                The data cube on which a coordinate is being blended.
            weights (iris.cube.Cube or None):
                Cube of blending weights or None.
        Returns:
            weights_array (np.array):
                An array of weights that matches the cube data shape.
        """
        # Percentile blending preserves the percentile dimension, but we will
        # not want to vary weights by percentile. If all the other dimensions
        # match for the cube and weights we can assume that a suitable 3D
        # weights cube has been provided and use it directly. To this end we
        # need to compare the shape of the cube excluding the percentile dim.
        non_perc_crds = [crd.name() for crd in cube.coords(dim_coords=True)
                         if not crd.name() == perc_coord.name()]
        non_perc_slice = next(cube.slices(non_perc_crds))

        # The weights need to be broadcast to match the percentile cube shape,
        # which means broadcasting across the percentile dimension.
        crd_dims = [cube.coord_dims(crd)[0] for crd in non_perc_crds]

        if weights:
            weights_array = self.shape_weights(non_perc_slice, weights)
            weights_array = iris.util.broadcast_to_shape(
                weights_array, cube.shape, tuple(crd_dims))
        else:
            number_of_fields, = cube.coord(self.coord).shape
            weights_array = (
                np.broadcast_to(1./number_of_fields, cube.shape).astype(
                    np.float32))

        blend_dim, = cube.coord_dims(self.coord)
        perc_dim, = cube.coord_dims(perc_coord)

        # The percentile aggregator performs some coordinate reordering on
        # the data. We don't have sufficient information in the aggregator
        # to modify the weight order correctly, so we do it in advance.
        weights_array = np.moveaxis(weights_array,
                                    (blend_dim, perc_dim), (0, 1))

        # Check the weights add up to 1 across the blending dimension.
        self.check_weights(weights_array, 0)

        return weights_array.astype(np.float32)

    def percentile_weighted_mean(self, cube, weights, perc_coord):
        """
        Blend percentile data using the weights provided.

        Args:
            cube (iris.cube.Cube):
                The cube which is being blended over self.coord.
            weights (iris.cube.Cube):
                Cube of blending weights.
            perc_coord (iris.coords.DimCoord):
                The percentile coordinate for this cube.
        Returns:
            cube_new (iris.cube.Cube):
                The cube with percentile values blended over self.coord,
                with suitable weightings applied.
        """
        percentiles = np.array(
            perc_coord.points, dtype=np.float32)
        perc_dim, = cube.coord_dims(perc_coord.name())

        # The iris.analysis.Aggregator moves the coordinate being
        # collapsed to index=-1 in initialisation, before the
        # aggregation method is called. This reduces by 1 the index
        # of all coordinates with an initial index higher than the
        # collapsing coordinate. As we need to know the index of
        # the percentile coordinate at a later step, if it will be
        # changed by this process, we adjust our record (perc_dim)
        # here.
        if cube.coord_dims(self.coord)[0] < perc_dim:
            perc_dim -= 1

        weights_array = self.percentile_weights(cube, weights, perc_coord)

        # Set up aggregator
        PERCENTILE_BLEND = (Aggregator(
            'mean',  # Use CF-compliant cell method.
            PercentileBlendingAggregator.aggregate))
        cube_new = cube.collapsed(self.coord,
                                  PERCENTILE_BLEND,
                                  arr_percent=percentiles,
                                  arr_weights=weights_array,
                                  perc_dim=perc_dim)
        cube_new.data = cube_new.data.astype(np.float32)

        # Ensure collapsed coordinates do not promote themselves
        # to float64.
        for coord in cube_new.coords():
            if coord.points.dtype == np.float64:
                coord.points = coord.points.astype(np.float32)
        return cube_new

    def weighted_mean(self, cube, weights):
        """
        Blend data using a weighted mean using the weights provided.

        Args:
            cube (iris.cube.Cube):
                The cube which is being blended over self.coord.
            weights (iris.cube.Cube or None):
                Cube of blending weights or None.
        Returns:
            cube_new (iris.cube.Cube):
                The cube with values blended over self.coord, with suitable
                weightings applied.
        """
        weights_array = self.non_percentile_weights(cube, weights)

        # Calculate the weighted average.
        cube_new = cube.collapsed(self.coord,
                                  iris.analysis.MEAN,
                                  weights=weights_array)
        cube_new.data = cube_new.data.astype(np.float32)

        return cube_new

    def weighted_maximum(self, cube, weights):
        """
        Blend data using a weighted maximum using the weights provided.
        This entails scaling the data by the weights before then taking
        a maximum across the blending coordinate self.coord.

        Args:
            cube (iris.cube.Cube):
                The cube which is being blended over self.coord.
            weights (iris.cube.Cube):
                Cube of blending weights.
        Returns:
            cube_new (iris.cube.Cube):
                The cube with values blended over self.coord, with suitable
                weightings applied.
        """

        weights_array = self.non_percentile_weights(cube, weights,
                                                    custom_aggregator=True)
        # Set up aggregator
        MAX_PROBABILITY = (Aggregator(
            'maximum',  # Use CF-compliant cell method.
            MaxProbabilityAggregator.aggregate))

        cube_new = cube.collapsed(self.coord, MAX_PROBABILITY,
                                  arr_weights=weights_array)
        cube_new.data = cube_new.data.astype(np.float32)
        return cube_new

    def process(self, cube, weights=None):
        """Calculate weighted blend across the chosen coord, for either
           probabilistic or percentile data. If there is a percentile
           coordinate on the cube, it will blend using the
           PercentileBlendingAggregator but the percentile coordinate must
           have at least two points.

        Args:
            cube (iris.cube.Cube):
                Cube to blend across the coord.
        Keyword Args:
            weights (iris.cube.Cube):
                Cube of blending weights. If None, the diagnostic cube is
                blended with equal weights across the blending dimension.
        Returns:
            result (iris.cube.Cube):
                containing the weighted blend across the chosen coord.
        Raises:
            TypeError : If the first argument not a cube.
            CoordinateNotFoundError : If coordinate to be collapsed not found
                                      in cube.
            CoordinateNotFoundError : If coordinate to be collapsed not found
                                      in provided weights cube.
            ValueError : If coordinate to be collapsed is not a dimension.
        """
        if not isinstance(cube, iris.cube.Cube):
            msg = ('The first argument must be an instance of iris.cube.Cube '
                   'but is {}.'.format(type(cube)))
            raise TypeError(msg)

        if not cube.coords(self.coord):
            msg = ('Coordinate to be collapsed not found in cube.')
            raise CoordinateNotFoundError(msg)

        coord_dim = cube.coord_dims(self.coord)
        if not coord_dim:
            raise ValueError('Blending coordinate {} has no associated '
                             'dimension'.format(self.coord))

        # Ensure input cube and weights cube are ordered equivalently along
        # blending coordinate.
        cube = sort_coord_in_cube(cube, self.coord, order="ascending")
        if weights is not None:
            if not weights.coords(self.coord):
                msg = ('Coordinate to be collapsed not found in weights cube.')
                raise CoordinateNotFoundError(msg)
            weights = sort_coord_in_cube(weights, self.coord,
                                         order="ascending")

        # Check that the time coordinate is single valued if required.
        self.check_compatible_time_points(cube)

        # Check to see if the data is percentile data
        perc_coord = self.check_percentile_coord(cube)

        # Create slices over the threshold coordinate
        #try:
            #cube.coord('threshold')
        #except iris.exceptions.CoordinateNotFoundError:
            #slices_over_threshold = [cube]
            #weights_over_threshold = [weights]
        #else:
            #if self.coord == 'threshold':
                #slices_over_threshold = [cube]
                #weights_over_threshold = [weights]
            #else:
                #slices_over_threshold = cube.slices_over('threshold')
                #try:
                    #weights.coord('threshold')
                #except iris.exceptions.CoordinateNotFoundError:
                    #weights_over_threshold = [weights]
                #else:
                    #weights_over_threshold = [weights] #weights.slices_over('threshold')
        #no_of_cube_slices = 
        #if sum(1 for _ in weights_over_threshold) < no_of_cube_slices:
            #weights_over_threshold = [weights for _ in range(no_of_cube_slices)]
            #print("x")
        #print(slices_over_threshold, weights_over_threshold)
        #cubelist = iris.cube.CubeList([])
        ## For each threshold slice, blend the cube across the blending coord.
        #for cube_thres, weights_thres in zip(slices_over_threshold, weights_over_threshold):
        #print("hello", cube_thres, weights_thres)
        # A selection of blending modes are available:

        # Percentile aggregator
        if perc_coord and self.mode == "weighted_mean":
            cube_new = self.percentile_weighted_mean(cube, weights,
                                                        perc_coord)
        # Weighted mean
        elif self.mode == "weighted_mean":
            cube_new = self.weighted_mean(cube, weights)

        # Maximum probability aggregator.
        elif self.mode == "weighted_maximum":
            cube_new = self.weighted_maximum(cube, weights)

        # Modify the cube metadata and add to the cubelist.
        cube_new = conform_metadata(
            cube_new, cube, coord=self.coord,
            cycletime=self.cycletime)
        #cubelist.append(cube_new)

        # Merge the cubelist to reform a cube that looks like the input but
        # without the coordinate over which blending has occurred.
        #print(cubelist)
        #result = cubelist.merge_cube()
        result = cube_new
        # Add a source realizations attribute if collapsing realizations.
        if self.coord == "realization":
            result.attributes['source_realizations'] = (
                cube.coord(self.coord).points)

        if isinstance(cube.data, np.ma.core.MaskedArray):
            result.data = np.ma.array(result.data)

        return result
