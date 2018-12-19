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
"""Module to create the weights used to blend data."""

import copy
import cf_units

import numpy as np
from scipy.interpolate import interp1d

import iris

from improver.utilities.cube_manipulation import check_cube_coordinates


class WeightsUtilities:
    """ Utilities for Weight processing. """

    def __init__(self):
        """Initialise class."""
        pass

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        result = ('<WeightsUtilities>')
        return result

    @staticmethod
    def normalise_weights(weights, axis=None):
        """Ensures all weights add up to one.

            Args:
                weights (numpy.array):
                    array of weights

            Keyword Args:
                axis (int):
                    The axis that we want to normalise along for a multiple
                    dimensional array. Defaults to None, meaning the whole
                    array is used for the normalisation.

            Returns:
                normalised_weights (numpy.array):
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
    def redistribute_weights(weights, forecast_present, method='evenly'):
        """Redistribute weights if any of the forecasts are missing.

            Args:
                weights (numpy.ndarray):
                    Array of weights.
                forecast_present (numpy.ndarray):
                    Size of weights with values set as
                                   1.0 for present
                                   0.0 for missing.
                method (string):
                    Method to redistribute weights, default evenly.

                    Options are:
                        evenly - adding the weights from the
                                 missing forecasts evenly across
                                 the remaining forecasts.
                        proportional - re-weight according to the
                                       proportion of the previous
                                       weights.

            Returns:
                redistributed_weights (numpy.ndarray):
                    Array of weights where sum = 1.0 and missing weights are
                    set to -1.0

            Raises:
                ValueError: the weights input do not add up to 1.
                ValueError: any of the input weights are negative.
                ValueError: an unexpected number of weights are input.
                ValueError: none of the forecasts expected (according to
                            the user input coord_exp_vals) were found on the
                            cube being blended.
                ValueError: an unknown weights redistribution method is
                            entered (only recognised methods are 'evenly'
                            and 'proportional').
        """
        sumval = weights.sum()

        if abs(sumval - 1.0) > 0.0001:
            msg = 'Sum of weights must be 1.0'
            raise ValueError(msg)

        if weights.min() < 0.0:
            msg = 'Weights should be positive or at least one > 0.0'
            raise ValueError(msg)

        if len(weights) != len(forecast_present):
            msg = ('Arrays weights and forecast_present not the same size'
                   ' weights is len {0:}'.format(len(weights)) +
                   ' forecast_present is len {0:}'.format(
                       len(forecast_present)))
            raise ValueError(msg)

        num_forecasts_present = forecast_present.sum()
        if num_forecasts_present == 0:
            msg = 'None of the expected forecasts were found.'
            raise ValueError(msg)
        elif num_forecasts_present < len(forecast_present):
            combined_weights = weights*forecast_present
            if method == 'evenly':
                missing_avg_weight = (
                    1.0 - combined_weights.sum())/num_forecasts_present
                redistributed_weights = combined_weights + missing_avg_weight
            elif method == 'proportional':
                redistributed_weights = (
                    WeightsUtilities.normalise_weights(combined_weights))
            else:
                msg = ('Unknown weights redistribution method'
                       ': {}'.format(method))
                raise ValueError(msg)
            # Set missing values to -1.0
            redistributed_weights = redistributed_weights[np.where(
                forecast_present == 1)]
        elif num_forecasts_present == len(forecast_present):
            redistributed_weights = weights
        return redistributed_weights

    @staticmethod
    def process_coord(cube, coordinate, coord_exp_vals=None,
                      coord_unit='no_unit'):
        """Calculated weights for a given cube and coord.

            Args:
                cube (iris.cube.Cube):
                       Cube to blend across the coord.
                coordinate (string):
                       Name of coordinate in the cube to be blended.
                coord_exp_vals (string):
                       String list of values which are expected on the
                       coordinate to be blended over.
                coord_unit (cf_units.Unit):
                       The unit in which the coord_exp_vals have been passed
                       in.

            Returns:
                (tuple) : tuple containing:
                    **exp_coord_len** (int):
                           The number of forecasts we expect to blend, based on
                           the length of the coordinate we are going to blend
                           over.
                    **exp_forecast_found** (binary mask):
                           Array showing where the input cube coordinate values
                           agree with the input expected coordinate values.

            Raises:
                ValueError: the coordinate to blend over does not exist on
                            the cube being blended.
                ValueError: the length of the expected coordinate input is
                            less than the length of the corresponding cube
                            coordinate.
                ValueError: the input coordinate units cannot be converted
                            to the units of the corresponding cube
                            coordinate.
        """
        if not cube.coords(coordinate):
            msg = ('The coord for this plugin must be '
                   'an existing coordinate in the input cube.')
            raise ValueError(msg)
        cube_coord = cube.coord(coordinate)
        if coord_exp_vals is not None:
            coord_values = [float(x) for x in coord_exp_vals.split(',')]
            if len(coord_values) < len(cube_coord.points):
                msg = ('The cube coordinate has more points '
                       'than requested coord, '
                       'len coord points = {0:d} '.format(len(coord_values)) +
                       'len cube points = {0:d}'.format(
                           len(cube_coord.points)))
                raise ValueError(msg)
            else:
                exp_coord = iris.coords.AuxCoord(coord_values,
                                                 long_name=coordinate,
                                                 units=coord_unit)
                exp_coord_len = len(exp_coord.points)
        else:
            exp_coord_len = len(cube_coord.points)
        # Find which coordinates are present in exp_coord but not in cube_coord
        # ie: find missing forecasts.
        if len(cube_coord.points) < exp_coord_len:
            # Firstly check that coord is in the right units
            # Do not try if coord.units not set
            if (exp_coord.units != cf_units.Unit('1') and
                    str(exp_coord.units) != 'no_unit'):
                if exp_coord.units != cube_coord.units:
                    try:
                        exp_coord.convert_units(cube_coord.units)
                    except ValueError:
                        msg = ('Failed to convert coord units '
                               'requested coord units '
                               '= {0:s} '.format(str(exp_coord.units)) +
                               'cube units '
                               '= {0:s}'.format(str(cube_coord.units)))
                        raise ValueError(msg)
            exp_forecast_found = []
            for exp_point in exp_coord.points:
                if any(abs(exp_point - y) < 1e-5 for y in cube_coord.points):
                    exp_forecast_found.append(1)
                else:
                    exp_forecast_found.append(0)
            exp_forecast_found = np.array(exp_forecast_found)
        else:
            exp_forecast_found = np.ones(exp_coord_len)
        return (exp_coord_len, exp_forecast_found)

    @staticmethod
    def build_weights_cube(cube, weights, blending_coord):
        """Build a cube containing weights for use in blending.

            Args:
                cube (iris.cube.Cube):
                    The cube that is being blended over blending_coord.
                weights (numpy.array):
                    Array of weights
                blending_coord (string):
                    Name of the coordinate over which the weights will be used
                    to blend data, e.g. across model name when grid blending.
            Returns:
                weights_cube (iris.cube.Cube):
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

        print("inside build weights cube", weights_cube)
        print(cube)
        weights_cube.attributes = None
        blending_dim = cube.coord_dims(blending_coord)
        defunct_coords = [crd.name() for crd in cube.coords(dim_coords=True)
                          if not cube.coord_dims(crd) == blending_dim]
        for crd in defunct_coords:
            weights_cube.remove_coord(crd)

        weights_cube.data = weights
        weights_cube.rename('weights')
        weights_cube.units = 1

        return weights_cube


class ChooseWeightsLinear:
    """Plugin to interpolate weights linearly to the required points, where
    original weights are provided as a cube or configuration dictionary"""

    def __init__(self, weighting_coord_name,
                 config_coord_name="model_configuration", config_dict=None):
        """
        Set up for calculating linear weights from a dictionary or input cube

        Args:
            weighting_coord_name (str):
                Standard name of the coordinate along which the weights will be
                interpolated. For example, if the intention is to provide
                weights varying with forecast period, then this argument would
                be "forecast_period".  If a configuration dictionary is
                provided, then this coordinate must be included within the
                configuration dictionary.

        Keyword Args:
            config_coord_name (str):
                Name of the coordinate used to select the configuration.
                For example, if the intention is to create weights that scale
                differently with the weighting_coord for different models, then
                "model_configuration" would be the config_coord.
            config_dict (dict):
                Dictionary containing the configuration information, namely
                an initial set of weights and information regarding the
                points along the specified coordinate at which the weights are
                valid. An example dictionary is shown below.

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
        if self.config_dict:
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

    def _get_interpolation_inputs_from_cube(self, cube, weights_cube):
        """
        Generate inputs required for the linear interpolation.

        Args:
            cube (iris.cube.Cube):
                Cube containing the coordinate information that will be used
                for setting up the interpolation inputs.
            weights_cube (iris.cube.Cube):
                Cube containg the weights that will be interpolated to find
                new weights at the specified points.

        Returns:
            (tuple): tuple containing

                **source_points** (np.ndarray):
                    Points within the configuration dictionary that will
                    be used as the input to the interpolation.

                **target_points** (np.ndarray):
                    Points within the cube that will be the target points
                    for the interpolation.

                **source_weights** (np.ndarray):
                    Weights from the configuration dictionary that will be
                    used as the input to the interpolation.

                **axis** (int):
                    Axis of self.weighting_coord_name within the input cube.
                    This will be used to define the axis of interpolation.

                **fill_value** (tuple):
                    Values that be used if extrapolation is required. The
                    fill values will be used as constants that are extrapolated
                    if the target_points are outside the source_points
                    provided. These are equal to the weight associated with the
                    first and last values along the weighting coord e.g.
                    forecast_period.

        """
        source_points = weights_cube.coord(self.weighting_coord_name).points
        target_points = cube.coord(self.weighting_coord_name).points
        source_weights = weights_cube.core_data()
        axis, = weights_cube.coord_dims(self.weighting_coord_name)

        coord_values = (
            {self.weighting_coord_name: lambda cell: cell == source_points[0]})
        constr = iris.Constraint(coord_values=coord_values)
        lower_fill_value = weights_cube.extract(constr).core_data()

        coord_values = (
            {self.weighting_coord_name:
             lambda cell: cell == source_points[-1]})
        constr = iris.Constraint(coord_values=coord_values)
        upper_fill_value = weights_cube.extract(constr).core_data()

        fill_value = (lower_fill_value, upper_fill_value)
        return source_points, target_points, source_weights, axis, fill_value

    def _get_interpolation_inputs_from_dict(self, cube):
        """
        Generate inputs required for linear interpolation.

        Args:
            cube (iris.cube.Cube):
                Cube containing the coordinate information that will be used
                for setting up the interpolation inputs.

        Returns:
            (tuple): tuple containing

                **source_points** (np.ndarray):
                    Points within the configuration dictionary that will
                    be used as the input to the interpolation.

                **target_points** (np.ndarray):
                    Points within the cube that will be the target points
                    for the interpolation.

                **source_weights** (np.ndarray):
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
            source_points (np.ndarray):
                Points within the configuration dictionary that will
                be used as the input to the interpolation.
            target_points (np.ndarray):
                Points within the cube that will be the target points
                for the interpolation.
            source_weights (np.ndarray):
                Weights from the configuration dictionary that will be
                used as the input to the interpolation.
            fill_value (tuple):
                Values to be used if extrapolation is required. The
                fill values are used for target_points that are outside
                the source_points grid.

        Keyword Args:
            axis (int):
                Axis along which the interpolation will occur.

        Returns:
            weights (np.ndarray):
                Weights corresponding to target_points following interpolation.
        """
        f_out = interp1d(source_points, source_weights, axis=axis,
                         fill_value=fill_value, bounds_error=False)
        weights = f_out(target_points)
        return weights

    @staticmethod
    def _create_coord_and_dims_list(
            base_cube, cube_with_exception_coord, coord_list,
            exception_coord_name):
        """Create a list of coordinates and their dimensions for use
        when constructing an iris.cube.Cube.

        Args:
            base_cube (iris.cube.Cube):
                Cube from which all the coordinates within the coord_list
                will be copied, apart from the coordinate defined by the
                exception_coord_name, which will be taken from the
                cube_with_exception_coord.
            cube_with_exception_coord (iris.cube.Cube):
                Cube from which the exception coordinate will be taken.
            coord_list (iterable):
                List/tuple of coordinates.
            exception_coord_name (str):
                Name of the exception coordinate that will be taken from the
                cube_with_exception_coord.

        Returns:
            new_coord_list (list):
                List of tuples where each tuple is of the form
                (coordinate, index_of_coord). This list, may include the
                exception_coord_name taken from the cube_with_exception_coord.
                If the coordinate does not have an associated dimension e.g.
                the coordinate is a scalar coordinate, then the index of the
                coordinate is set equal to None.
        """
        new_coord_list = []
        for coord in coord_list:
            if (coord.name() == exception_coord_name or
                    base_cube.coord_dims(coord) ==
                    base_cube.coord_dims(exception_coord_name)):
                exception_coord = cube_with_exception_coord.coord(coord.name())
                if cube_with_exception_coord.coord_dims(coord):
                    index, = cube_with_exception_coord.coord_dims(coord)
                    new_coord_list.append((exception_coord, index))
                else:
                    new_coord_list.append((exception_coord, None))
            else:
                if base_cube.coord_dims(coord):
                    index, = base_cube.coord_dims(coord)
                    new_coord_list.append((coord, index))
                else:
                    new_coord_list.append((coord, None))
        return new_coord_list

    def _create_new_weights_cube(self, cube, weights, weights_cube=None):
        """Create a cube to contain the output of the interpolation.
        It is currently assumed that the output weights matches the size
        of the input cube. This will be true if the only difference between
        the cube and the weights cube is along the dimension of the
        self.weighting_coord_name coordinate.

        Args:
            cube (iris.cube.Cube):
                Cube containing the coordinate information that will be used
                for setting up the new_weights_cube.
            weights (np.ndarray):
                Weights calculated following interpolation.

        Kwargs:
            weights_cube (iris.cube.Cube):
                Cube containg the weights that will be interpolated to find
                new weights at the specified points. Ignored if a
                configuration dictionary (self.config_dict) is provided.

        Returns:
            new_weights_cube (iris.cube.Cube):
                Cube containing the output from the interpolation.  If a
                configuration dictionary is not provided
                (self.config_dict is None), this is based on "weights_cube",
                otherwise on "cube".
        """
        if self.config_dict:
            cubelist = iris.cube.CubeList([])
            for cube_slice, weight in (
                    zip(cube.slices_over(self.weighting_coord_name), weights)):
                sub_slice = cube_slice[..., 0, 0]
                sub_slice.remove_coord(sub_slice.coord(axis='x'))
                sub_slice.remove_coord(sub_slice.coord(axis='y'))
                sub_slice.data = np.full_like(sub_slice.data, weight)
                cubelist.append(sub_slice)
            new_weights_cube = (
                check_cube_coordinates(cube[..., 0, 0], cubelist.merge_cube()))
            new_weights_cube.rename(self.weights_key_name)

        else:
            dim_coords_and_dims = (
                self._create_coord_and_dims_list(
                    weights_cube, cube, weights_cube.dim_coords,
                    self.weighting_coord_name))
            aux_coords_and_dims = (
                self._create_coord_and_dims_list(
                    weights_cube, cube, weights_cube.aux_coords,
                    self.weighting_coord_name))

            new_weights_cube = iris.cube.Cube(
                weights, standard_name=weights_cube.standard_name,
                long_name=weights_cube.long_name, units=weights_cube.units,
                attributes=weights_cube.attributes,
                cell_methods=weights_cube.cell_methods,
                dim_coords_and_dims=dim_coords_and_dims,
                aux_coords_and_dims=aux_coords_and_dims)

        return new_weights_cube

    def _calculate_weights(self, cube, weights_cube=None):
        """Method to wrap the calls to other methods to support calculation
        of the weights by interpolation.

        Args:
            cube (iris.cube.Cube):
                Cube containing the coordinate information that will be used
                for setting up the interpolation and create the new weights
                cube.

        Kwargs:
            weights_cube (iris.cube.Cube):
                Cube containing the weights that will be interpolated to find
                new weights at the specified points. Ignored if a
                configuration dictionary (self.config_dict) is provided.

        Returns:
            new_weights_cube (iris.cube.Cube):
                Cube containing the output from the interpolation. This
                has been renamed using the self.weights_key_name but
                otherwise matches the input cube.
        """
        if self.config_dict:
            source_points, target_points, source_weights, fill_value = (
                self._get_interpolation_inputs_from_dict(cube))
            axis = 0
        else:
            source_points, target_points, source_weights, axis, fill_value = (
                self._get_interpolation_inputs_from_cube(
                    cube, weights_cube=weights_cube))

        weights = self._interpolate_to_find_weights(
            source_points, target_points, source_weights, fill_value,
            axis=axis)

        new_weights_cube = self._create_new_weights_cube(
            cube, weights, weights_cube=weights_cube)

        return new_weights_cube

    def _define_slice(self, cube):
        """
        Returns a list of coordinates over which to slice the input cube to
        create a list of cubes for blending.

        Args:
            cube (iris.cube.Cube):
                Cube input to plugin

        Returns:
            slice_list (list):
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
            cubelist (iris.cube.CubeList):
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

    def process(self, cubes, weights_cubes=None):
        """Calculation of linear weights based on an input weights cube
        or dictionary.  If self.config_dict == None, weights are calculated
        individually for each point in self.config_coord_name using the
        input weights_cubes, before being normalised across the
        self.config_coord_name dimension.

        Args:
            cubes (iris.cube.Cube or iris.cube.CubeList):
                Cubes containing the coordinate (source point) information
                that will be used for setting up the interpolation.  Each cube
                should have "self.config_coord_name" as a scalar dimension; if
                a merged cube is passed in, the plugin will split this into a
                list cubes.

        Kwargs:
            weights_cubes (iris.cube.CubeList):
                CubeList where each cube should correspond to a point along
                the self.config_coord_name dimension of the input cube.
                For example, if self.config_coord_name is model_configuration,
                then there should be a separate cube for each model
                configuration.  Ignored a configuration dictionary is provided.

        Returns:
            new_weights_cube (iris.cube.Cube):
                Cube containing the output from the interpolation.
                This cube will only include spatial dimensions if using
                weights_cubes instead of a weights dict.
                DimCoords (such as model_id) will be in sorted-ascending order.
        """
        # check for invalid arguments
        if self.config_dict and weights_cubes is not None:
            raise ValueError(
                'Cannot calculate weights from both dict and cube')

        if self.config_dict:
            # create 2D cube lists with relevant dimensions only for dict
            # processing
            cubes = self._slice_input_cubes(cubes)
        else:
            if isinstance(cubes, iris.cube.Cube):
                cubes = [cubes]
            if isinstance(weights_cubes, iris.cube.Cube):
                weights_cubes = iris.cube.CubeList([weights_cubes])

            # check that the number of weights cubes matches the length of the
            # cubes to be weighted.
            if len(weights_cubes) != len(cubes):
                msg = ("The number of cubes to be weighted needs to be "
                       "the same as the number of weights cubes. "
                       "\nnumber of input cubes is {} != "
                       "number of weights cubes is {}")
                raise ValueError(msg.format(len(cubes), len(weights_cubes)))

        # calculate weights
        cube_slices = iris.cube.CubeList([])
        for cube in cubes:
            if self.config_dict:
                new_weights_cube = self._calculate_weights(cube)
            else:
                coord_point, = cube.coord(self.config_coord_name).points
                coord_values = (
                    {self.config_coord_name: lambda cell: cell == coord_point})
                constr = iris.Constraint(coord_values=coord_values)
                weights_cube, = weights_cubes.extract(constr)
                new_weights_cube = self._calculate_weights(
                    cube, weights_cube=weights_cube)
            cube_slices.append(new_weights_cube)

        # normalise weights
        new_weights_cube = cube_slices.merge_cube()
        axis = new_weights_cube.coord_dims(self.config_coord_name)
        new_weights_cube.data = (
            WeightsUtilities.normalise_weights(
                new_weights_cube.data, axis=axis))

        return new_weights_cube


class ChooseDefaultWeightsLinear:
    """ Calculate Default Weights using Linear Function. """

    def __init__(self, y0val=None, slope=0.0, ynval=None):
        """Set up for calculating default weights using linear function

            Keyword Args:
                y0val (None or positive float):
                    Relative value of starting point.
                slope (float):
                    Slope of the line. Default = 0.0 (equal weights).
                ynval (float or None):
                    Relative weights of last point.
                    Default value is None

            slope OR ynval should be set but NOT BOTH.

            If y0val value is not set or set to None then the code
            uses default values of y0val = 20.0 and ynval = 2.0.

            equal weights when slope = 0.0 or y0val = ynval
        """
        self.slope = slope
        self.ynval = ynval

        if y0val is None:
            self.y0val = 20.0
            self.ynval = 2.0
        elif not isinstance(y0val, float) or y0val < 0.0:
            msg = ('y0val must be a float >= 0.0, '
                   'y0val = {0:s}'.format(str(y0val)))
            raise ValueError(msg)
        else:
            self.y0val = y0val

    def linear_weights(self, num_of_weights):
        """Create linear weights

            Args:
                num_of_weights (Positive Integer):
                                 Number of weights to create.
                y0val (Positive float):
                        relative value of starting point. Default = 1.0
                slope (float):
                        slope of the line. Default = 0.0 (equal weights)
                ynval (Positive float or None):
                        Relative weights of last point.
                        Default value is None

            Returns:
                weights (numpy.array):
                    array of weights, sum of all weights = 1.0

            Raises:
                ValueError: an inappropriate value of y0val is input.
                ValueError: both slope and ynval are set at input.

        """
        # Special case num_of_weights == 1 i.e. Scalar coordinate.
        if num_of_weights == 1:
            weights = np.array([1.0], dtype=np.float32)
            return weights
        if self.ynval is not None:
            if self.slope == 0.0:
                self.slope = (self.ynval - self.y0val)/(num_of_weights - 1.0)
            else:
                msg = ('Relative end point weight or slope must be set'
                       ' but not both.')
                raise ValueError(msg)

        weights_list = []
        for tval in range(0, num_of_weights):
            weights_list.append(self.slope*tval + self.y0val)

        weights = WeightsUtilities.normalise_weights(
            np.array(weights_list, dtype=np.float32))

        return weights

    def process(self, cube, coord_name, coord_vals=None, coord_unit='no_unit',
                weights_distrib_method='evenly'):
        """Calculated weights for a given cube and coord.

            Args:
                cube (iris.cube.Cube):
                       Cube to blend across the coord.
                coord_name (string):
                       Name of coordinate in the cube to be blended.
                coord_vals (string):
                       String list of values which are expected on the
                       coordinate to be blended over.
                coord_unit (cf_units.Unit):
                       The unit in which the coord_exp_vals have been passed
                       in.
                weights_distrib_method (string):
                       The method to use when redistributing weights in cases
                       where there are some forecasts missing. Options:
                       "evenly", "proportional".
            Returns:
                weights (numpy.array):
                    1D array of normalised (sum = 1.0) weights matching length
                    of cube dimension to be blended

            Raises:
                TypeError : input is not a cube
        """
        if not isinstance(cube, iris.cube.Cube):
            msg = ('The first argument must be an instance of '
                   'iris.cube.Cube but is'
                   ' {0:s}'.format(str(type(cube))))
            raise TypeError(msg)

        (num_of_weights,
         exp_coord_found) = WeightsUtilities.process_coord(
             cube, coord_name, coord_vals, coord_unit)

        weights_in = self.linear_weights(num_of_weights)

        weights = WeightsUtilities.redistribute_weights(
            weights_in, exp_coord_found, weights_distrib_method)

        weights_cube = WeightsUtilities.build_weights_cube(cube, weights,
                                                           coord_name)
        return weights_cube

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        desc = '<ChooseDefaultWeightsLinear y0val={0:4.1f}'.format(self.y0val)
        if self.ynval is None:
            desc += ', slope={0:6.2f}>'.format(self.slope)
        else:
            desc += ', ynval={0:4.1f}>'.format(self.ynval)
        return desc


class ChooseDefaultWeightsNonLinear:
    """ Calculate Default Weights using NonLinear Function. """
    def __init__(self, cval=0.85):
        """Set up for calculating default weights using non-linear function.

            Args:
                cval (float):
                       Value greater than 0, less than equal 1.0
                       default = 0.85
                       equal weights when cval = 1.0
        """
        self.cval = cval

    def nonlinear_weights(self, num_of_weights):
        """Create nonlinear weights.

            Args:
                num_of_weights (Positive Integer):
                                 Number of weights to create.

            Returns:
                weights (numpy.array):
                    array of weights, sum of all weights = 1.0

            Raises:
                ValueError: an inappropriate value of cval is input.
        """
        if self.cval <= 0.0 or self.cval > 1.0:
            msg = ('cval must be greater than 0.0 and less '
                   'than or equal to 1.0 '
                   'cval = {0:s}'.format(str(self.cval)))
            raise ValueError(msg)

        weights_list = []
        for tval_minus1 in range(0, num_of_weights):
            weights_list.append(self.cval**(tval_minus1))

        weights = WeightsUtilities.normalise_weights(
            np.array(weights_list, dtype=np.float32))

        return weights

    def process(self, cube, coord_name, coord_vals=None, coord_unit='no_unit',
                weights_distrib_method='evenly'):
        """Calculated weights for a given cube and coord.

            Args:
                cube (iris.cube.Cube):
                       Cube to blend across the coord.
                coord_name (string):
                       Name of coordinate in the cube to be blended.
                coord_vals (string):
                       String list of values which are expected on the
                       coordinate to be blended over.
                coord_unit (cf_units.Unit):
                       The unit in which the coord_exp_vals have been passed
                       in.
                weights_distrib_method (string):
                        The method to use when redistributing weights in cases
                        where there are some forecasts missing. Options:
                        "evenly", "proportional".
            Returns:
                weights (numpy.array):
                    1D array of normalised (sum = 1.0) weights matching length
                    of cube dimension to be blended

            Raises:
                TypeError : input is not a cube
        """
        if not isinstance(cube, iris.cube.Cube):
            msg = ('The first argument must be an instance of '
                   'iris.cube.Cube but is'
                   ' {0:s}'.format(str(type(cube))))
            raise TypeError(msg)

        (num_of_weights,
         exp_coord_found) = WeightsUtilities.process_coord(
             cube, coord_name, coord_vals, coord_unit)

        weights_in = self.nonlinear_weights(num_of_weights)

        weights = WeightsUtilities.redistribute_weights(
            weights_in, exp_coord_found, weights_distrib_method)

        weights_cube = WeightsUtilities.build_weights_cube(cube, weights,
                                                           coord_name)
        return weights_cube

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        desc = ('<ChooseDefaultWeightsNonLinear '
                'cval={0:4.1f}>'.format(self.cval))
        return desc


class ChooseDefaultWeightsTriangular:
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

    @staticmethod
    def triangular_weights(coord_vals, midpoint, width):
        """Create triangular weights.

            Args:
                coord_vals (numpy array):
                    An array of coordinate values that we want to calculate
                    weights for.
                midpoint (float):
                    The centre point of the triangular function.
                width (float):
                    The width of the triangular function from the centre point.

            Returns:
                weights (numpy.array):
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
                weight (float):
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
                coord_name (string):
                    Name of coordinate in the cube to be blended.
                midpoint (float):
                    The centre point of the triangular function.

            Returns:
                weights (numpy.array):
                    1D array of normalised (sum = 1.0) weights matching length
                    of cube dimension to be blended

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
        else:
            width_in_coord_units = copy.deepcopy(self.width)

        weights = self.triangular_weights(
            coord_vals, midpoint, width_in_coord_units)

        weights_cube = WeightsUtilities.build_weights_cube(cube, weights,
                                                           coord_name)
        return weights_cube

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        msg = ("<ChooseDefaultTriangularWeights "
               "width={}, parameters_units={}>")
        desc = msg.format(self.width, self.parameters_units)
        return desc
