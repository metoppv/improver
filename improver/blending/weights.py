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
"""Module to create the weights used to blend data."""

import copy

import cf_units
import iris
import numpy as np
from scipy.interpolate import interp1d

from improver import BasePlugin
from improver.utilities.cube_manipulation import (
    check_cube_coordinates, sort_coord_in_cube)


class WeightsUtilities:
    """ Utilities for Weight processing. """

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        result = ('<WeightsUtilities>')
        return result

    @staticmethod
    def normalise_weights(weights, axis=None):
        """Ensures all weights add up to one.

            Args:
                weights (numpy.ndarray):
                    array of weights
                axis (int):
                    The axis that we want to normalise along for a multiple
                    dimensional array. Defaults to None, meaning the whole
                    array is used for the normalisation.

            Returns:
                numpy.ndarray:
                    array of weights where sum = 1.0

            Raises:
                ValueError: any negative weights are found in input.
                ValueError: sum of weights in the input is 0.
        """
        if np.any(weights.min(axis=axis) < 0.0):
            msg = ('Weights must be positive. The weights have at least one '
                   'value < 0.0: {}'.format(weights))
            raise ValueError(msg)

        sumval = np.sum(weights, axis=axis, keepdims=True)
        if np.any(sumval == 0):
            msg = 'Sum of weights must be > 0.0'
            raise ValueError(msg)

        normalised_weights = weights / sumval
        return normalised_weights

    @staticmethod
    def build_weights_cube(cube, weights, blending_coord):
        """Build a cube containing weights for use in blending.

            Args:
                cube (iris.cube.Cube):
                    The cube that is being blended over blending_coord.
                weights (numpy.ndarray):
                    Array of weights
                blending_coord (str):
                    Name of the coordinate over which the weights will be used
                    to blend data, e.g. across model name when grid blending.
            Returns:
                iris.cube.Cube:
                    A cube containing the array of weights.
            Raises:
                ValueError : If weights array is not of the same length as the
                             coordinate being blended over on cube.
        """

        if len(weights) != len(cube.coord(blending_coord).points):
            msg = ("Weights array provided is not the same size as the "
                   "blending coordinate; weights shape: {}, blending "
                   "coordinate shape: {}".format(
                       len(weights), len(cube.coord(blending_coord).points)))
            raise ValueError(msg)

        try:
            weights_cube = next(cube.slices(blending_coord))
        except ValueError:
            weights_cube = iris.util.new_axis(cube, blending_coord)
            weights_cube = next(weights_cube.slices(blending_coord))
        weights_cube.attributes = None
        # Find dim associated with blending_coord and don't remove any coords
        # associated with this dimension.
        blending_dim = cube.coord_dims(blending_coord)
        defunct_coords = [crd.name() for crd in cube.coords(dim_coords=True)
                          if not cube.coord_dims(crd) == blending_dim]
        for crd in defunct_coords:
            weights_cube.remove_coord(crd)
        weights_cube.data = weights
        weights_cube.rename('weights')
        weights_cube.units = 1

        return weights_cube


class ChooseWeightsLinear(BasePlugin):
    """Plugin to interpolate weights linearly to the required points, where
    original weights are provided as a configuration dictionary"""

    def __init__(self, weighting_coord_name, config_dict,
                 config_coord_name="model_configuration"):
        """
        Set up for calculating linear weights from a dictionary or input cube

        Args:
            weighting_coord_name (str):
                Standard name of the coordinate along which the weights will be
                interpolated. For example, if the intention is to provide
                weights varying with forecast period, then this argument would
                be "forecast_period". This coordinate must be included within
                the configuration dictionary.
            config_dict (dict):
                Dictionary containing the configuration information, namely
                an initial set of weights and information regarding the
                points along the specified coordinate at which the weights are
                valid. An example dictionary is shown below.
            config_coord_name (str):
                Name of the coordinate used to select the configuration.
                For example, if the intention is to create weights that scale
                differently with the weighting_coord for different models, then
                "model_configuration" would be the config_coord.

        Dictionary of format::

            {
                "uk_det": {
                    "forecast_period": [7, 12],
                    "weights": [1, 0],
                    "units": "hours"
                }
                "uk_ens": {
                    "forecast_period": [7, 12, 48, 54]
                    "weights": [0, 1, 1, 0]
                    "units": "hours"
                }
            }

        """
        self.weighting_coord_name = weighting_coord_name
        self.config_coord_name = config_coord_name
        self.config_dict = config_dict
        self.weights_key_name = "weights"
        self._check_config_dict()

    def __repr__(self):
        """Represent the plugin instance as a string"""
        msg = ("<ChooseWeightsLinear(): weighting_coord_name = {}, "
               "config_coord_name = {}, config_dict = {}>".format(
                   self.weighting_coord_name, self.config_coord_name,
                   str(self.config_dict)))
        return msg

    def _check_config_dict(self):
        """
        Check whether the items within the configuration dictionary
        are present and of matching lengths.

        Raises:
            ValueError: If items within the configuration dictionary are
                not of matching lengths.
            KeyError: If the required items are not present in the
                configuration dictionary.
        """
        # Check all keys
        for key in self.config_dict.keys():
            weighting_len = (
                len(self.config_dict[key][self.weighting_coord_name]))
            weights_len = (
                len(self.config_dict[key][self.weights_key_name]))
            if weighting_len != weights_len:
                msg = ("{} is {}, {} is {}."
                       "These items in the configuration dictionary "
                       "have different lengths i.e. {} != {}".format(
                           self.weighting_coord_name,
                           self.config_dict[key][self.weighting_coord_name],
                           self.weights_key_name,
                           self.config_dict[key][self.weights_key_name],
                           weighting_len, weights_len))
                raise ValueError(msg)

    def _get_interpolation_inputs_from_dict(self, cube):
        """
        Generate inputs required for linear interpolation.

        Args:
            cube (iris.cube.Cube):
                Cube containing the coordinate information that will be used
                for setting up the interpolation inputs.

        Returns:
            (tuple): tuple containing:
                **source_points** (numpy.ndarray):
                    Points within the configuration dictionary that will
                    be used as the input to the interpolation.

                **target_points** (numpy.ndarray):
                    Points within the cube that will be the target points
                    for the interpolation.

                **source_weights** (numpy.ndarray):
                    Weights from the configuration dictionary that will be
                    used as the input to the interpolation.

                **fill_value** (tuple):
                    Values that be used if extrapolation is required. The
                    fill values will be used as constants that are extrapolated
                    if the target_points are outside the source_points
                    provided. These are equal to the first and last values
                    provided by the source weights.

        """
        config_point, = cube.coord(self.config_coord_name).points
        source_points = (
            self.config_dict[config_point][self.weighting_coord_name])
        source_points = np.array(source_points)
        if "units" in self.config_dict[config_point].keys():
            units = cf_units.Unit(self.config_dict[config_point]["units"])
            source_points = units.convert(
                source_points, cube.coord(self.weighting_coord_name).units)

        target_points = cube.coord(self.weighting_coord_name).points
        source_weights = (
            self.config_dict[config_point][self.weights_key_name])

        fill_value = (source_weights[0], source_weights[-1])
        return source_points, target_points, source_weights, fill_value

    @staticmethod
    def _interpolate_to_find_weights(
            source_points, target_points, source_weights, fill_value, axis=0):
        """
        Use of scipy.interpolate.interp1d to interpolate source_weights
        (valid at source_points) onto target_points grid.  This allows
        the specification of an axis for the interpolation, so that the
        source_weights can be a multi-dimensional numpy array.

        Args:
            source_points (numpy.ndarray):
                Points within the configuration dictionary that will
                be used as the input to the interpolation.
            target_points (numpy.ndarray):
                Points within the cube that will be the target points
                for the interpolation.
            source_weights (numpy.ndarray):
                Weights from the configuration dictionary that will be
                used as the input to the interpolation.
            fill_value (tuple):
                Values to be used if extrapolation is required. The
                fill values are used for target_points that are outside
                the source_points grid.
            axis (int):
                Axis along which the interpolation will occur.

        Returns:
            numpy.ndarray:
                Weights corresponding to target_points following interpolation.
        """
        f_out = interp1d(source_points, source_weights, axis=axis,
                         fill_value=fill_value, bounds_error=False)
        weights = f_out(target_points)
        return weights

    def _create_new_weights_cube(self, cube, weights):
        """Create a cube to contain the output of the interpolation.
        It is currently assumed that the output weights matches the size
        of the input cube.

        Args:
            cube (iris.cube.Cube):
                Cube containing the coordinate information that will be used
                for setting up the new_weights_cube.
            weights (numpy.ndarray):
                Weights calculated following interpolation.

        Returns:
            iris.cube.Cube:
                Cube containing the output from the interpolation. This has
                the same shape as "cube", without the x and y dimensions.
        """
        cubelist = iris.cube.CubeList([])
        for cube_slice, weight in (
                zip(cube.slices_over(self.weighting_coord_name), weights)):
            sub_slice = cube_slice[..., 0, 0]
            sub_slice.data = np.ones(sub_slice.data.shape)*weight
            cubelist.append(sub_slice)

        # re-order dimension coordinates to match input cube
        new_weights_cube = (
            check_cube_coordinates(cube[..., 0, 0], cubelist.merge_cube()))

        # remove all scalar coordinates that are not time-, model- or
        # blend-related
        dim_coords = new_weights_cube.coords(dim_coords=True)
        keep_coords = ["time", "forecast_period", "forecast_reference_time",
                       "model_id", "model_configuration",
                       self.weighting_coord_name, self.config_coord_name]
        for coord in new_weights_cube.coords():
            if coord not in dim_coords and coord.name() not in keep_coords:
                new_weights_cube.remove_coord(coord)

        # remove attributes
        new_weights_cube.attributes = {}

        # rename cube
        new_weights_cube.rename(self.weights_key_name)
        new_weights_cube.units = cf_units.Unit('1')

        return new_weights_cube

    def _calculate_weights(self, cube):
        """Method to wrap the calls to other methods to support calculation
        of the weights by interpolation.

        Args:
            cube (iris.cube.Cube):
                Cube containing the coordinate information that will be used
                for setting up the interpolation and create the new weights
                cube.

        Returns:
            iris.cube.Cube:
                Cube containing the output from the interpolation. This
                has been renamed using the self.weights_key_name but
                otherwise matches the input cube.
        """
        source_points, target_points, source_weights, fill_value = (
            self._get_interpolation_inputs_from_dict(cube))
        axis = 0

        weights = self._interpolate_to_find_weights(
            source_points, target_points, source_weights, fill_value,
            axis=axis)

        new_weights_cube = self._create_new_weights_cube(
            cube, weights)

        return new_weights_cube

    def _define_slice(self, cube):
        """
        Returns a list of coordinates over which to slice the input cube to
        create a list of cubes for blending.

        Args:
            cube (iris.cube.Cube):
                Cube input to plugin

        Returns:
            list of iris.coord.Coord:
                List of coordinates defining the slice to iterate over
        """
        if cube.coord_dims(self.weighting_coord_name):
            slice_list = [
                cube.coord(self.weighting_coord_name),
                cube.coord(axis='y'), cube.coord(axis='x')]
        else:
            slice_list = [cube.coord(axis='y'), cube.coord(axis='x')]
        return slice_list

    def _slice_input_cubes(self, cubes):
        """
        From input iris.cube.Cube or iris.cube.CubeList, create a list of
        cubes with different values of the config coordinate (over which to
        blend), with irrelevant dimensions sliced out.

        Args:
            cubes (iris.cube.Cube or iris.cube.CubeList):
                Cubes passed into the plugin.

        Returns:
            iris.cube.CubeList:
                List of cubes (from which to calculate weights) with
                dimensions (y, x) if weighting_coord is scalar on the input
                cube, or (weighting_coord, y, x) if weighting_coord is
                non-scalar
        """
        if isinstance(cubes, iris.cube.Cube):
            # check how many points there are in the config coordinate
            if len(cubes.coord(self.config_coord_name).points) == 1:
                cubelist = [next(cubes.slices(self._define_slice(cubes)))]
            else:
                # if passed a merged cube, split this up into a cube list
                cubelist = []
                for cube in cubes.slices_over(
                        cubes.coord(self.config_coord_name)):
                    cubelist.append(
                        next(cube.slices(self._define_slice(cube))))
        else:
            cubelist = []
            for cube in cubes:
                cubelist.append(
                    next(cube.slices(self._define_slice(cube))))

        return iris.cube.CubeList(cubelist)

    def process(self, cubes):
        """Calculation of linear weights based on an input dictionary.

        Args:
            cubes (iris.cube.Cube or iris.cube.CubeList):
                Cubes containing the coordinate (source point) information
                that will be used for setting up the interpolation.  Each cube
                should have "self.config_coord_name" as a scalar dimension; if
                a merged cube is passed in, the plugin will split this into a
                list cubes.

        Returns:
            iris.cube.Cube:
                Cube containing the output from the interpolation.
                DimCoords (such as model_id) will be in sorted-ascending order.
        """
        # create 2D cube lists with relevant dimensions only for dict
        # processing
        cubes = self._slice_input_cubes(cubes)

        # calculate weights
        cube_slices = iris.cube.CubeList([])
        for cube in cubes:
            new_weights_cube = self._calculate_weights(cube)
            cube_slices.append(new_weights_cube)

        # normalise weights
        new_weights_cube = cube_slices.merge_cube()
        axis = new_weights_cube.coord_dims(self.config_coord_name)
        new_weights_cube.data = (
            WeightsUtilities.normalise_weights(
                new_weights_cube.data, axis=axis))

        return new_weights_cube


class ChooseDefaultWeightsLinear(BasePlugin):
    """ Calculate Default Weights using Linear Function. """

    def __init__(self, y0val, ynval):
        """
        Set up for calculating default weights using linear function.

        Args:
            y0val (int or float):
                Relative weight of first point.  Must be positive.
            ynval (int or float):
                Relative weight of last point.
        """
        if y0val is None or ynval is None:
            raise ValueError('y0val and ynval are required arguments to the '
                             'ChooseDefaultWeightsLinear plugin')

        if y0val < 0.0:
            msg = ('y0val must be a float >= 0.0, '
                   'y0val = {0:s}'.format(str(y0val)))
            raise ValueError(msg)

        self.y0val = float(y0val)
        self.ynval = float(ynval)

    def linear_weights(self, num_of_weights):
        """Create linear weights

            Args:
                num_of_weights (positive int):
                    Number of weights to create.

            Returns:
                numpy.ndarray:
                    array of weights, sum of all weights = 1.0
        """
        # Special case num_of_weights == 1 i.e. Scalar coordinate.
        if num_of_weights == 1:
            weights = np.array([1.0], dtype=np.float32)
            return weights

        slope = (self.ynval - self.y0val)/(num_of_weights - 1.0)

        weights_list = []
        for tval in range(0, num_of_weights):
            weights_list.append(slope*tval + self.y0val)

        weights = WeightsUtilities.normalise_weights(
            np.array(weights_list, dtype=np.float32))

        return weights

    def process(self, cube, coord_name):
        """
        Calculated weights for a given cube and coord.  Weights scale linearly
        between self.y0val and self.ynval for the cube provided in ascending
        order of blend coordinate.  self.y0val = self.ynval gives equal
        weightings across all input fields.

        Args:
            cube (iris.cube.Cube):
                Cube to blend across the coord.
            coord_name (str):
                Name of coordinate in the cube to be blended.

        Returns:
            iris.cube.Cube:
                1D cube of normalised (sum = 1.0) weights matching length
                of input dimension to be blended

        Raises:
            TypeError : input is not a cube
        """
        if not isinstance(cube, iris.cube.Cube):
            msg = ('The first argument must be an instance of '
                   'iris.cube.Cube but is'
                   ' {0:s}'.format(str(type(cube))))
            raise TypeError(msg)

        weights = self.linear_weights(len(cube.coord(coord_name).points))
        weights_cube = WeightsUtilities.build_weights_cube(cube, weights,
                                                           coord_name)
        return weights_cube

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        desc = ('<ChooseDefaultWeightsLinear y0val='
                '{:4.1f}, ynval={:4.1f}>'.format(self.y0val, self.ynval))
        return desc


class ChooseDefaultWeightsNonLinear(BasePlugin):
    """ Calculate Default Weights using NonLinear Function. """
    def __init__(self, cval):
        """
        Set up for calculating default weights using non-linear function.

        Args:
            cval (float):
                Value greater than 0, less than equal 1.0.  Weights are
                calculated for input cubes in order such that the first has
                weight cval**0, then cval**1, cval**2, etc.  The weights are
                then re-normalised. Thus a value of 1 gives equal weighting
                across all input fields.

        Raises:
            ValueError: an inappropriate value of cval is input.
        """
        if cval is None:
            raise ValueError('cval is a required argument to the '
                             'ChooseDefaultWeightsNonLinear plugin')

        if cval <= 0.0 or cval > 1.0:
            msg = ('cval must be greater than 0.0 and less '
                   'than or equal to 1.0 cval = {}'.format(cval))
            raise ValueError(msg)
        self.cval = cval

    def nonlinear_weights(self, num_of_weights):
        """
        Create nonlinear weights.

        Args:
            num_of_weights (int):
                Number of weights to create

        Returns:
            numpy.ndarray:
                Normalised array of weights
        """
        weights_list = []
        for tval_minus1 in range(0, num_of_weights):
            weights_list.append(self.cval**(tval_minus1))

        weights = WeightsUtilities.normalise_weights(
            np.array(weights_list, dtype=np.float32))

        return weights

    def process(self, cube, coord_name, inverse_ordering=False):
        """
        Calculate nonlinear weights for a given cube and coord.

        Args:
            cube (iris.cube.Cube):
                Cube to be blended across the coord.
            coord_name (str):
                Name of coordinate in the cube to be blended.
            inverse_ordering (bool):
                The input cube blend coordinate will be in ascending order,
                so that calculated blend weights decrease with increasing
                value.  For eg cycle blending by forecast reference time, we
                wish to weight more recent cubes more highly.  This flag gives
                the option to reverse the blend coordinate order so as to have
                higher weights for the higher values.

        Returns:
            iris.cube.Cube:
                1D cube of normalised (sum = 1.0) weights matching input
                dimension to be blended

        Raises:
            TypeError : input is not a cube
        """
        if not isinstance(cube, iris.cube.Cube):
            msg = ('The first argument must be an instance of '
                   'iris.cube.Cube but is'
                   ' {0:s}'.format(str(type(cube))))
            raise TypeError(msg)

        if inverse_ordering:
            # make a copy of the input cube from which to calculate weights
            inverted_cube = cube.copy()
            inverted_cube = sort_coord_in_cube(
                inverted_cube, coord_name, descending=True)
            cube = inverted_cube

        weights = self.nonlinear_weights(len(cube.coord(coord_name).points))
        weights_cube = WeightsUtilities.build_weights_cube(cube, weights,
                                                           coord_name)

        if inverse_ordering:
            # re-sort the weights cube so that it is in ascending order of
            # blend coordinate (and hence matches the input cube)
            weights_cube = sort_coord_in_cube(weights_cube, coord_name)

        return weights_cube

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        desc = ('<ChooseDefaultWeightsNonLinear '
                'cval={0:4.1f}>'.format(self.cval))
        return desc


class ChooseDefaultWeightsTriangular(BasePlugin):
    """ Calculate Default Weights using a Triangular Function. """
    def __init__(self, width, units="no_unit"):
        """Set up for calculating default weights using triangular function.

            Args:
                width (float):
                    The width of the triangular function from the centre point.
                units (cf_units.Unit):
                    The cf units of the width and midpoint.
        """
        self.width = width
        if not isinstance(units, cf_units.Unit):
            units = cf_units.Unit(units)
        self.parameters_units = units

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        msg = ("<ChooseDefaultTriangularWeights "
               "width={}, parameters_units={}>")
        desc = msg.format(self.width, self.parameters_units)
        return desc

    @staticmethod
    def triangular_weights(coord_vals, midpoint, width):
        """Create triangular weights.

            Args:
                coord_vals (numpy.ndarray):
                    An array of coordinate values that we want to calculate
                    weights for.
                midpoint (float):
                    The centre point of the triangular function.
                width (float):
                    The width of the triangular function from the centre point.

            Returns:
                numpy.ndarray:
                    array of weights, sum of all weights should equal 1.0.
        """

        def calculate_weight(point, slope):
            """
            A helper function to calculate the weights for each point using a
            piecewise function to build up the triangular function.
            Args:
                point (float):
                    The point in the coordinate from the cube for
                    which we want to calculate a weight for.
                slope (float):
                    The gradient of the triangle, calculated from
                    1/(width of triangle).

            Returns:
                float:
                    The individual weight calculated by the function.
            """
            if point == midpoint:
                weight = 1
            else:
                weight = 1-abs(point-midpoint)*slope
            return weight

        slope = 1.0/width
        weights = np.zeros(coord_vals.shape, dtype=np.float32)
        # Find the indices of the points where there will be non-zero weights.
        condition = ((coord_vals >= (midpoint-width)) &
                     (coord_vals <= (midpoint+width)))
        points_with_weights = np.where(condition)[0]
        # Calculate for weights for points where we want a non-zero weight.
        for index in points_with_weights:
            weights[index] = calculate_weight(coord_vals[index], slope)
        # Normalise the weights.
        weights = WeightsUtilities.normalise_weights(weights)

        return weights

    def process(self, cube, coord_name, midpoint):
        """Calculate triangular weights for a given cube and coord.

            Args:
                cube (iris.cube.Cube):
                    Cube to blend across the coord.
                coord_name (str):
                    Name of coordinate in the cube to be blended.
                midpoint (float):
                    The centre point of the triangular function.  This is
                    assumed to be provided in the same units as "self.width",
                    ie "self.parameter_units" as initialised.

            Returns:
                iris.cube.Cube:
                    1D cube of normalised (sum = 1.0) weights matching length
                    of input dimension to be blended

            Raises:
                TypeError : input is not a cube
        """
        if not isinstance(cube, iris.cube.Cube):
            msg = ('The first argument must be an instance of '
                   'iris.cube.Cube but is'
                   ' {0:s}'.format(str(type(cube))))
            raise TypeError(msg)

        cube_coord = cube.coord(coord_name)
        coord_vals = cube_coord.points
        coord_units = cube_coord.units

        # Rescale width and midpoint if in different units to the coordinate
        if coord_units != self.parameters_units:
            width_in_coord_units = (
                self.parameters_units.convert(self.width, coord_units))
            midpoint = (
                self.parameters_units.convert(midpoint, coord_units))
        else:
            width_in_coord_units = copy.deepcopy(self.width)

        weights = self.triangular_weights(
            coord_vals, midpoint, width_in_coord_units)

        weights_cube = WeightsUtilities.build_weights_cube(cube, weights,
                                                           coord_name)
        return weights_cube
