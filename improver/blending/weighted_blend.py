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
"""Module containing classes for doing weighted blending by collapsing a
   whole dimension."""

import warnings

import iris
import numpy as np
from iris.analysis import Aggregator
from iris.coords import AuxCoord
from iris.exceptions import CoordinateNotFoundError

from improver import BasePlugin, PostProcessingPlugin
from improver.metadata.amend import amend_attributes
from improver.metadata.constants.attributes import (
    MANDATORY_ATTRIBUTE_DEFAULTS, MANDATORY_ATTRIBUTES)
from improver.metadata.forecast_times import (
    forecast_period_coord, rebadge_forecasts_as_latest_cycle)
from improver.metadata.probabilistic import find_percentile_coordinate
from improver.utilities.cube_manipulation import (
    MergeCubes, build_coordinate, enforce_coordinate_ordering,
    sort_coord_in_cube, collapsed, get_dim_coord_names)
from improver.utilities.temporal import cycletime_to_number


class MergeCubesForWeightedBlending(BasePlugin):
    """Prepares cubes for cycle and grid blending"""

    def __init__(self, blend_coord, weighting_coord=None, model_id_attr=None):
        """
        Initialise the class

        Args:
            blend_coord (str):
                Name of coordinate over which blending will be performed.  For
                multi-model blending this is flexible to any string containing
                "model".  For all other coordinates this is prescriptive:
                cube.coord(blend_coord) must return an iris.coords.Coord
                instance for all cubes passed into the "process" method.
            weighting_coord (str or None):
                The coordinate across which weights will be scaled in a
                multi-model blend.
            model_id_attr (str or None):
                Name of attribute used to identify model for grid blending.
                None for cycle blending.

        Raises:
            ValueError:
                If trying to blend over model when model_id_attr is not set
        """
        if "model" in blend_coord and model_id_attr is None:
            raise ValueError(
                "model_id_attr required to blend over {}".format(blend_coord))

        # ensure model coordinates are not created for non-model blending
        if "model" not in blend_coord and model_id_attr is not None:
            warnings.warn(
                "model_id_attr not required for blending over {} - "
                "will be ignored".format(blend_coord))
            model_id_attr = None

        self.blend_coord = blend_coord
        self.weighting_coord = weighting_coord
        self.model_id_attr = model_id_attr

    def _create_model_coordinates(self, cubelist):
        """
        Adds numerical model ID and string model configuration scalar
        coordinates to input cubes if self.model_id_attr is specified.
        Sets the original attribute value to "blend", in anticipation.
        Modifies cubes in place.

        Args:
            cubelist (iris.cube.CubeList):
                List of cubes to be merged for blending

        Raises:
            ValueError:
                If self.model_id_attr is not present on all cubes
            ValueError:
                If input cubelist contains cubes from the same model
        """
        model_titles = []
        for i, cube in enumerate(cubelist):
            if self.model_id_attr not in cube.attributes:
                msg = ('Cannot create model ID coordinate for grid blending '
                       'as "model_id_attr={}" was not found within the cube '
                       'attributes'.format(self.model_id_attr))
                raise ValueError(msg)

            model_title = cube.attributes.pop(self.model_id_attr)
            if model_title in model_titles:
                raise ValueError('Cannot create model dimension coordinate '
                                 'with duplicate points')
            model_titles.append(model_title)
            cube.attributes[self.model_id_attr] = "blend"

            new_model_id_coord = build_coordinate([1000 * i],
                                                  long_name='model_id',
                                                  data_type=np.int32)
            new_model_coord = (
                build_coordinate([model_title],
                                 long_name='model_configuration',
                                 coord_type=AuxCoord,
                                 data_type=np.str))

            cube.add_aux_coord(new_model_id_coord)
            cube.add_aux_coord(new_model_coord)

    def process(self, cubes_in, cycletime=None):
        """
        Prepares merged input cube for cycle and grid blending

        Args:
            cubes_in (iris.cube.CubeList or iris.cube.Cube):
                Cubes to be merged.
            cycletime (str or None):
                The cycletime in a YYYYMMDDTHHMMZ format e.g. 20171122T0100Z.
                Can be used in rationalise_blend_time_coordinates.

        Returns:
            iris.cube.Cube:
                Merged cube.

        Raises:
            ValueError:
                If self.blend_coord is not present on all cubes (unless
                blending over models)
        """
        cubes_in = (
            [cubes_in] if isinstance(cubes_in, iris.cube.Cube) else cubes_in)

        if ("model" in self.blend_coord and
                self.weighting_coord is not None and
                "forecast_period" in self.weighting_coord):
            # if blending models using weights by forecast period, unify
            # forecast periods (assuming validity times are identical);
            # method returns a new cubelist with copies of input cubes
            cubelist = rebadge_forecasts_as_latest_cycle(
                cubes_in, cycletime=cycletime)
        else:
            # copy cubes to avoid modifying inputs
            cubelist = [cube.copy() for cube in cubes_in]

        # if input is already a single cube, return here
        if len(cubelist) == 1:
            return cubelist[0]

        # check all input cubes have the blend coordinate
        for cube in cubelist:
            if ("model" not in self.blend_coord and
                    not cube.coords(self.blend_coord)):
                raise ValueError(
                    "{} coordinate is not present on all input "
                    "cubes".format(self.blend_coord))

        # create model ID and model configuration coordinates if blending
        # different models
        if self.model_id_attr is not None:
            self._create_model_coordinates(cubelist)

        # merge resulting cubelist
        result = MergeCubes().process(cubelist, check_time_bounds_ranges=True)

        return result


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
              array of blended probabilities for each of the grid points.
           3. We convert back to the original percentile values, again using
              linear interpolation, resulting in blended values at each of the
              original percentiles.

       References:
            Combining Probabilities by Caroline Jones, 2017:
            https://github.com/metoppv/improver/files/1128018/
            Combining_Probabilities.pdf
    """

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        result = ('<PercentileBlendingAggregator>')
        return result

    @staticmethod
    def aggregate(data, axis, arr_percent, arr_weights, perc_dim):
        """ Blend percentile aggregate function to blend percentile data
            along a given axis of a cube.

        Args:
            data (numpy.ndarray):
                   Array containing the data to blend
            axis (int):
                   The index of the coordinate dimension in the cube. This
                   dimension will be aggregated over.
            arr_percent(numpy.ndarray):
                     Array of percentile values e.g
                     [0, 20.0, 50.0, 70.0, 100.0],
                     same size as the percentile dimension of data.
            arr_weights(numpy.ndarray):
                     Array of weights, same size as the axis dimension of data.
            perc_dim (int):
                     The index of the percentile coordinate
            (Note percent and weights have special meaning in Aggregator
             hence the rename.)

        Returns:
            numpy.ndarray:
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
            perc_values (numpy.ndarray):
                Array containing the percentile values to blend, with
                shape: (length of coord to blend, num of percentiles)
            percentiles (numpy.ndarray):
                Array of percentile values e.g [0, 20.0, 50.0, 70.0, 100.0],
                same size as the percentile dimension of data.
            weights (numpy.ndarray):
                Array of weights, same size as the axis dimension of data,
                that we will blend over.

        Returns:
            numpy.ndarray:
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


class WeightedBlendAcrossWholeDimension(PostProcessingPlugin):
    """Apply a Weighted blend to a cube, collapsing across the whole
       dimension. Uses one of two methods, either weighted average, or
       the maximum of the weighted probabilities."""

    def __init__(self, blend_coord, timeblending=False):
        """Set up for a Weighted Blending plugin

        Args:
            blend_coord (str):
                The name of the coordinate dimension over which the cube will
                be blended.
            timeblending (bool):
                With the default of False the cube being blended will be
                checked to ensure that slices across the blending coordinate
                all have the same validity time. Setting this to True will
                bypass this test, as is necessary for triangular time
                blending.

        Raises:
            ValueError: If the blend coordinate is "threshold".
        """
        if blend_coord == "threshold":
            msg = "Blending over thresholds is not supported"
            raise ValueError(msg)
        self.blend_coord = blend_coord
        self.timeblending = timeblending
        self.cycletime_point = None
        self.crds_to_remove = None

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        description = ('<WeightedBlendAcrossWholeDimension: coord = {}, '
                       'timeblending: {}>')
        return description.format(self.blend_coord, self.timeblending)

    @staticmethod
    def check_percentile_coord(cube):
        """
        Determines if the cube to be blended has a percentile dimension
        coordinate.

        Args:
            cube (iris.cube.Cube):
                The cube to be checked for a percentile coordinate.
        Returns:
            iris.coords.DimCoord or None:
                None if no percentile dimension coordinate is found. If
                such a coordinate is found it is returned.
        Raises:
            ValueError : If there is a percentile coord and it is not a
                dimension coord in the cube.
            ValueError : If there is a percentile dimension with only one
                point, we need at least two points in order to do the blending.
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

    @staticmethod
    def shape_weights(cube, weights):
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
            numpy.ndarray:
                An array of weights that matches the cube data shape.
        Raises:
            ValueError: If weights cube coordinates do not match the diagnostic
                        cube in the case of a multidimensional weights cube.
            ValueError: If weights cube shape is not broadcastable to the data
                        cube shape.
        """
        # Check that a multidimensional weights cube has coordinates that match
        # the diagnostic cube. Checking names only to not to be too exacting.
        weight_dims = get_dim_coord_names(weights)
        cube_dims = get_dim_coord_names(cube)
        if set(weight_dims) == set(cube_dims):
            enforce_coordinate_ordering(weights, cube_dims)
            weights_array = weights.data.astype(np.float32)
        else:
            # Map array of weights to shape of cube to collapse.
            dim_map = []
            dim_coords = [coord.name() for coord in weights.dim_coords]
            # Loop through dim coords in weights cube and find the dim the
            # coord relates to in the cube we are collapsing.
            for dim_coord in dim_coords:
                try:
                    dim_map.append(cube.coord_dims(dim_coord)[0])
                except CoordinateNotFoundError:
                    message = (
                        "{} is a coordinate on the weights cube but it is not "
                        "found on the cube we are trying to collapse.")
                    raise ValueError(message.format(dim_coord))
            try:
                weights_array = iris.util.broadcast_to_shape(
                    np.array(weights.data, dtype=np.float32),
                    cube.shape, tuple(dim_map))
            except ValueError:
                msg = (
                    "Weights cube is not a compatible shape with the"
                    " data cube. Weights: {}, Diagnostic: {}".format(
                        weights.shape, cube.shape))
                raise ValueError(msg)

        return weights_array

    @staticmethod
    def check_weights(weights, blend_dim):
        """
        Checks that weights across the blending dimension sum up to 1.

        Args:
            weights (numpy.ndarray):
                Array of weights shaped to match the data cube.
            blend_dim (int):
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

    def non_percentile_weights(self, cube, weights):
        """
        Given a 1 or multidimensional cube of weights, reshape and broadcast
        these in such a way as to make them applicable to the data cube. If no
        weights are provided, an array of weights is returned that equally
        weights all slices across the blending coordinate of the cube.

        Args:
            cube (iris.cube.Cube):
                The data cube on which a coordinate is being blended.
            weights (iris.cube.Cube or None):
                Cube of blending weights or None.
        Returns:
            numpy.ndarray:
                An array of weights that matches the cube data shape.
        """
        if weights:
            weights_array = self.shape_weights(cube, weights)
        else:
            number_of_fields, = cube.coord(self.blend_coord).shape
            weights_array = (
                np.broadcast_to(1./number_of_fields, cube.shape).astype(
                    np.float32))
        blend_dim, = cube.coord_dims(self.blend_coord)
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
            perc_coord (iris.coords.Coord):
                Percentile coordinate

        Returns:
            numpy.ndarray:
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
            number_of_fields, = cube.coord(self.blend_coord).shape
            weights_array = (
                np.broadcast_to(1./number_of_fields, cube.shape).astype(
                    np.float32))

        blend_dim, = cube.coord_dims(self.blend_coord)
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
                The cube which is being blended over self.blend_coord.
            weights (iris.cube.Cube):
                Cube of blending weights.
            perc_coord (iris.coords.DimCoord):
                The percentile coordinate for this cube.
        Returns:
            iris.cube.Cube:
                The cube with percentile values blended over self.blend_coord,
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
        if cube.coord_dims(self.blend_coord)[0] < perc_dim:
            perc_dim -= 1

        weights_array = self.percentile_weights(cube, weights, perc_coord)

        # Set up aggregator
        PERCENTILE_BLEND = (Aggregator(
            'mean',  # Use CF-compliant cell method.
            PercentileBlendingAggregator.aggregate))

        cube_new = collapsed(cube, self.blend_coord, PERCENTILE_BLEND,
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
                The cube which is being blended over self.blend_coord.
            weights (iris.cube.Cube or None):
                Cube of blending weights or None.

        Returns:
            iris.cube.Cube:
                The cube with values blended over self.blend_coord, with
                suitable weightings applied.
        """
        weights_array = self.non_percentile_weights(cube, weights)

        # Calculate the weighted average.
        cube_new = collapsed(cube, self.blend_coord, iris.analysis.MEAN,
                             weights=weights_array)
        cube_new.data = cube_new.data.astype(np.float32)

        return cube_new

    @staticmethod
    def _get_cycletime_point(input_cube, cycletime):
        """
        For cycle and model blending, establish the single forecast reference
        time to set on the cube after blending.

        Args:
            input_cube (iris.cube.Cube):
                Cube to be blended
            cycletime (str or None):
                The cycletime in a YYYYMMDDTHHMMZ format e.g. 20171122T0100Z.
                If None, the latest forecast reference time is used.

        Returns:
            numpy.int64:
                Forecast reference time point in units of input cube coordinate
        """
        frt_coord = input_cube.coord("forecast_reference_time")
        if cycletime is None:
            return np.max(frt_coord.points)
        frt_units = frt_coord.units.origin
        frt_calendar = frt_coord.units.calendar
        cycletime_point = cycletime_to_number(
            cycletime, time_unit=frt_units, calendar=frt_calendar)
        return np.round(cycletime_point).astype(np.int64)

    def _set_coords_to_remove(self, input_cube):
        """
        Generate a list of coordinate names associated with the blend
        dimension.  Unless these are time-related coordinates, they should be
        removed after blending.

        Args:
            input_cube (iris.cube.Cube):
                Cube to be blended
        """
        time_coords = ["time", "forecast_reference_time", "forecast_period"]
        blend_dim, = input_cube.coord_dims(self.blend_coord)
        self.crds_to_remove = []
        for coord in input_cube.coords():
            if coord.name() in time_coords:
                continue
            if blend_dim in input_cube.coord_dims(coord):
                self.crds_to_remove.append(coord.name())

    def _set_forecast_reference_time_and_period(self, blended_cube):
        """
        For cycle and model blending, update the forecast reference time and
        forecast period coordinate points to the single most recent value,
        rather than the blended average, and remove any bounds from the
        forecast reference time. Modifies cube in place.

        Args:
            blended_cube (iris.cube.Cube)
        """
        blended_cube.coord("forecast_reference_time").points = [
            self.cycletime_point]
        blended_cube.coord("forecast_reference_time").bounds = None
        if blended_cube.coords("forecast_period"):
            blended_cube.remove_coord("forecast_period")
        new_forecast_period = forecast_period_coord(blended_cube)
        time_dim = blended_cube.coord_dims("time")
        blended_cube.add_aux_coord(new_forecast_period, data_dims=time_dim)

    def _update_blended_metadata(self, blended_cube, attributes_dict):
        """
        Update metadata after blending:
        - For cycle and model blending, set a single forecast reference time
        and period using self.cycletime_point or the latest cube contributing
        to the blend
        - Remove scalar coordinates that were previously associated with the
        blend dimension
        - Update attributes as specified via process arguments
        - Set any missing mandatory arguments to their default values
        Modifies cube in place.

        Args:
            blended_cube (iris.cube.Cube)
            attributes_dict (dict or None)
        """
        if self.blend_coord in ["forecast_reference_time", "model_id"]:
            self._set_forecast_reference_time_and_period(blended_cube)
        for coord in self.crds_to_remove:
            blended_cube.remove_coord(coord)
        if attributes_dict is not None:
            amend_attributes(blended_cube, attributes_dict)
        for attr in MANDATORY_ATTRIBUTES:
            if attr not in blended_cube.attributes:
                blended_cube.attributes[attr] = (
                    MANDATORY_ATTRIBUTE_DEFAULTS[attr])

    def process(self, cube, weights=None,
                cycletime=None, attributes_dict=None):
        """Calculate weighted blend across the chosen coord, for either
           probabilistic or percentile data. If there is a percentile
           coordinate on the cube, it will blend using the
           PercentileBlendingAggregator but the percentile coordinate must
           have at least two points.

        Args:
            cube (iris.cube.Cube):
                Cube to blend across the coord.
            weights (iris.cube.Cube):
                Cube of blending weights. If None, the diagnostic cube is
                blended with equal weights across the blending dimension.
            cycletime (str):
                The cycletime in a YYYYMMDDTHHMMZ format e.g. 20171122T0100Z.
                This can be used to manually set the forecast reference time
                on the output blended cube. If not set, the most recent
                forecast reference time from the contributing cubes is used.
            attributes_dict (dict or None):
                Changes to cube attributes to be applied after blending. See
                :func:`~improver.metadata.amend.amend_attributes` for required
                format. If mandatory attributes are not set here, default
                values are used.

        Returns:
            iris.cube.Cube:
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

        if not cube.coords(self.blend_coord):
            msg = 'Coordinate to be collapsed not found in cube.'
            raise CoordinateNotFoundError(msg)

        blend_coord_dims = cube.coord_dims(self.blend_coord)
        if not blend_coord_dims:
            raise ValueError('Blending coordinate {} has no associated '
                             'dimension'.format(self.blend_coord))

        # Ensure input cube and weights cube are ordered equivalently along
        # blending coordinate.
        cube = sort_coord_in_cube(cube, self.blend_coord)
        if weights is not None:
            if not weights.coords(self.blend_coord):
                msg = 'Coordinate to be collapsed not found in weights cube.'
                raise CoordinateNotFoundError(msg)
            weights = sort_coord_in_cube(weights, self.blend_coord)

        # Check that the time coordinate is single valued if required.
        self.check_compatible_time_points(cube)

        # Check to see if the data is percentile data
        perc_coord = self.check_percentile_coord(cube)

        # Establish metadata changes to be made after blending
        self.cycletime_point = (
            self._get_cycletime_point(cube, cycletime) if self.blend_coord in [
                "forecast_reference_time", "model_id"] else None)
        self._set_coords_to_remove(cube)

        # Do blending and update metadata
        if perc_coord:
            result = self.percentile_weighted_mean(cube, weights, perc_coord)
        else:
            result = self.weighted_mean(cube, weights)
        self._update_blended_metadata(result, attributes_dict)

        # Re-mask output
        if isinstance(cube.data, np.ma.core.MaskedArray):
            result.data = np.ma.array(result.data)

        return result
