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
"""Module containing classes for doing weighted blending by collapsing a
   whole dimension."""

import warnings

import iris
import numpy as np
from iris.analysis import Aggregator
from iris.coords import AuxCoord
from iris.exceptions import CoordinateNotFoundError

from improver import BasePlugin, PostProcessingPlugin
from improver.blending.utilities import find_blend_dim_coord
from improver.metadata.amend import amend_attributes
from improver.metadata.constants import FLOAT_DTYPE, PERC_COORD
from improver.metadata.constants.attributes import (
    MANDATORY_ATTRIBUTE_DEFAULTS,
    MANDATORY_ATTRIBUTES,
)
from improver.metadata.constants.time_types import TIME_COORDS
from improver.metadata.forecast_times import (
    add_blend_time,
    forecast_period_coord,
    rebadge_forecasts_as_latest_cycle,
)
from improver.utilities.cube_manipulation import (
    MergeCubes,
    collapsed,
    enforce_coordinate_ordering,
    get_coord_names,
    get_dim_coord_names,
    sort_coord_in_cube,
)
from improver.utilities.round import round_close
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
                "model_id_attr required to blend over {}".format(blend_coord)
            )

        # ensure model coordinates are not created for non-model blending
        if "model" not in blend_coord and model_id_attr is not None:
            warnings.warn(
                "model_id_attr not required for blending over {} - "
                "will be ignored".format(blend_coord)
            )
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
                msg = (
                    "Cannot create model ID coordinate for grid blending "
                    'as "model_id_attr={}" was not found within the cube '
                    "attributes".format(self.model_id_attr)
                )
                raise ValueError(msg)

            model_title = cube.attributes.pop(self.model_id_attr)
            if model_title in model_titles:
                raise ValueError(
                    "Cannot create model dimension coordinate " "with duplicate points"
                )
            model_titles.append(model_title)

            new_model_id_coord = AuxCoord(
                np.array([1000 * i], dtype=np.int32), units="1", long_name="model_id"
            )
            new_model_coord = AuxCoord(
                [model_title], units="no_unit", long_name="model_configuration"
            )

            cube.add_aux_coord(new_model_id_coord)
            cube.add_aux_coord(new_model_coord)

        model_titles.sort()
        for cube in cubelist:
            cube.attributes[self.model_id_attr] = " ".join(model_titles)

    @staticmethod
    def _remove_blend_time(cube):
        """If present on input, remove existing blend time coordinate (as this will
        be replaced on blending)"""
        if "blend_time" in get_coord_names(cube):
            cube.remove_coord("blend_time")
        return cube

    @staticmethod
    def _remove_deprecation_warnings(cube):
        """Remove deprecation warnings from forecast period and forecast reference
        time coordinates so that these can be merged before blending"""
        for coord in ["forecast_period", "forecast_reference_time"]:
            cube.coord(coord).attributes.pop("deprecation_message", None)
        return cube

    def process(self, cubes_in, cycletime=None):
        """
        Prepares merged input cube for cycle and grid blending. Makes updates to
        metadata (attributes and time-type coordinates) ONLY in so far as these are
        needed to ensure inputs can be merged into a single cube.

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
        cubelist = (
            [cubes_in.copy()]
            if isinstance(cubes_in, iris.cube.Cube)
            else [cube.copy() for cube in cubes_in]
        )

        if "model" in self.blend_coord:
            cubelist = [self._remove_blend_time(cube) for cube in cubelist]
            cubelist = [self._remove_deprecation_warnings(cube) for cube in cubelist]
            if (
                self.weighting_coord is not None
                and "forecast_period" in self.weighting_coord
            ):
                # if blending models using weights by forecast period, unify
                # forecast periods (assuming validity times are identical);
                # method returns a new cubelist with copies of input cubes
                cubelist = rebadge_forecasts_as_latest_cycle(
                    cubelist, cycletime=cycletime
                )

        # if input is already a single cube, return here
        if len(cubelist) == 1:
            return cubelist[0]

        # check all input cubes have the blend coordinate
        for cube in cubelist:
            if "model" not in self.blend_coord and not cube.coords(self.blend_coord):
                raise ValueError(
                    "{} coordinate is not present on all input "
                    "cubes".format(self.blend_coord)
                )

        # create model ID and model configuration coordinates if blending
        # different models
        if self.model_id_attr is not None:
            self._create_model_coordinates(cubelist)

        # merge resulting cubelist
        result = MergeCubes()(cubelist, check_time_bounds_ranges=True)
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

    @staticmethod
    def aggregate(data, axis, percentiles, arr_weights):
        """
        Function to blend percentile data over a given dimension.
        The input percentile data must be provided with the blend coord as the
        first axis and the percentile coord as the second (these are re-ordered
        after aggregator initialisation at the start of this function).  Weights
        data must be provided with the blend coord as the first dimension.

        Args:
            data (numpy.ndarray):
                Array containing the data to blend.
            axis (int):
                The index of the coordinate dimension in the cube. This
                dimension will be aggregated over.
            percentiles (numpy.ndarray):
                Array of percentile values e.g [0, 20.0, 50.0, 70.0, 100.0],
                same size as the percentile (second) dimension of data.
            arr_weights (numpy.ndarray):
                Array of weights, same shape as data, but without the percentile
                dimension (weights do not vary with percentile).

        Note "weights" has special meaning in Aggregator, hence
        using a different name for this variable.

        Returns:
            numpy.ndarray:
                Containing the weighted percentile blend data across
                the chosen coord. The dimension associated with axis
                has been collapsed, and the rest of the dimensions remain.
        """
        # Iris aggregators support indexing from the end of the array.
        if axis < 0:
            axis += data.ndim
        # Blend coordinate is moved to the -1 position in initialisation; move
        # this back to the leading coordinate
        data = np.moveaxis(data, [axis], [0])

        if len(data) != len(arr_weights):
            raise ValueError("Weights shape does not match data")

        # Flatten data and weights over non-blend and percentile dimensions
        grid_shape = data.shape[2:]
        grid_points = np.prod(grid_shape, dtype=int)
        flattened_shape = [data.shape[0], data.shape[1], grid_points]
        data = data.reshape(flattened_shape)
        weights_shape = [data.shape[0], grid_points]
        arr_weights = arr_weights.reshape(weights_shape)

        # Find the blended percentile values at each point in the flattened data
        result = np.zeros(flattened_shape[1:], dtype=FLOAT_DTYPE)
        for i in range(data.shape[-1]):
            result[:, i] = PercentileBlendingAggregator.blend_percentiles(
                data[:, :, i], percentiles, arr_weights[:, i]
            )
        # Reshape the data with a leading percentile dimension
        shape = percentiles.shape + grid_shape
        result = result.reshape(shape)
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
        inputs_to_blend = perc_values.shape[0]
        combined_cdf = np.zeros((inputs_to_blend, len(percentiles)), dtype=FLOAT_DTYPE)
        # Loop over the axis we are blending over finding the values for the
        # probability at each threshold in the cdf, for each of the other
        # points in the axis we are blending over. Use the values from the
        # percentiles if we are at the same point, otherwise use linear
        # interpolation.
        # Then add the probabilities multiplied by the correct weight to the
        # running total.
        for i in range(0, inputs_to_blend):
            for j in range(0, inputs_to_blend):
                if i == j:
                    values_in_cdf = percentiles
                else:
                    values_in_cdf = np.interp(
                        perc_values[i], perc_values[j], percentiles
                    )
                # Add the resulting probabilities multiplied by the right
                # weight to the running total for the combined cdf
                combined_cdf[i] += values_in_cdf * weights[j]

        # Combine and sort the threshold values for all the points
        # we are blending.
        combined_perc_thres_data = np.sort(perc_values.flatten())

        # Combine and sort blended probability values.
        combined_perc_values = np.sort(combined_cdf.flatten())

        # Find the percentile values from this combined data by interpolating
        # back from probability values to the original percentiles.
        new_combined_perc = np.interp(
            percentiles, combined_perc_values, combined_perc_thres_data
        ).astype(FLOAT_DTYPE)
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
        self.cycletime = None
        self.crds_to_remove = None

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        description = (
            "<WeightedBlendAcrossWholeDimension: coord = {}, " "timeblending: {}>"
        )
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
            bool:
                True if there is a multi-valued percentile dimension; False if not
        Raises:
            ValueError : If there is a percentile coord and it is not a
                dimension coord in the cube.
            ValueError : If there is a percentile dimension with only one
                point, we need at least two points in order to do the blending.
        """
        try:
            perc_coord = cube.coord(PERC_COORD)
            perc_dim = cube.coord_dims(PERC_COORD)
            if not perc_dim:
                msg = "The percentile coord must be a dimension of the cube."
                raise ValueError(msg)
            # Check the percentile coordinate has more than one point,
            # otherwise raise an error as we won't be able to blend.
            if len(perc_coord.points) < 2.0:
                msg = (
                    "Percentile coordinate does not have enough points"
                    " in order to blend. Must have at least 2 percentiles."
                )
                raise ValueError(msg)
            return True
        except CoordinateNotFoundError:
            return False

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
        if self.timeblending:
            return

        time_points = cube.coord("time").points
        if len(set(time_points)) > 1:
            msg = (
                "Attempting to blend data for different validity times. The"
                " time points within the input cube are {}".format(time_points)
            )
            raise ValueError(msg)

    @staticmethod
    def shape_weights(cube, weights):
        """
        The function shapes weights to match the diagnostic cube. A cube of
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
            weights_array = weights.data.astype(FLOAT_DTYPE)
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
                        "found on the cube we are trying to collapse."
                    )
                    raise ValueError(message.format(dim_coord))

            try:
                weights_array = iris.util.broadcast_to_shape(
                    np.array(weights.data, dtype=FLOAT_DTYPE),
                    cube.shape,
                    tuple(dim_map),
                )
            except ValueError:
                msg = (
                    "Weights cube is not a compatible shape with the"
                    " data cube. Weights: {}, Diagnostic: {}".format(
                        weights.shape, cube.shape
                    )
                )
                raise ValueError(msg)

        return weights_array

    @staticmethod
    def _normalise_weights(weights):
        """
        Checks that weights across the leading blend dimension sum up to 1.  If
        not, normalise across the blending dimension ignoring any points at which
        the sum of weights is zero.

        Args:
            weights (numpy.ndarray):
                Array of weights shaped to match the data cube.

        Returns:
            numpy.ndarray:
                Weights normalised along the (leading) blend dimension
        """
        sum_of_weights = np.sum(weights, axis=0)
        sum_of_non_zero_weights = sum_of_weights[sum_of_weights > 0]
        if not (np.isclose(sum_of_non_zero_weights, 1)).all():
            weights = np.where(
                sum_of_weights > 0, np.divide(weights, sum_of_weights), 0
            )
        return weights

    def get_weights_array(self, cube, weights):
        """
        Given a 1 or multidimensional cube of weights, reshape and broadcast
        these to the shape of the data cube. If no weights are provided, an
        array of weights is returned that equally weights all slices across
        the blending coordinate.

        Args:
            cube (iris.cube.Cube):
                Template cube to reshape weights, with a leading blend coordinate
            weights (iris.cube.Cube or None):
                Cube of initial blending weights or None
        Returns:
            numpy.ndarray:
                An array of weights that matches the template cube shape.
        """
        if weights:
            weights_array = self.shape_weights(cube, weights)
        else:
            (number_of_fields,) = cube.coord(self.blend_coord).shape
            weight = FLOAT_DTYPE(1.0 / number_of_fields)
            weights_array = np.broadcast_to(weight, cube.shape)

        return weights_array

    def percentile_weighted_mean(self, cube, weights):
        """
        Blend percentile data using the weights provided.

        Args:
            cube (iris.cube.Cube):
                The cube which is being blended over self.blend_coord. Assumes
                self.blend_coord and percentile are leading coordinates (enforced
                in process).
            weights (iris.cube.Cube):
                Cube of blending weights.
        Returns:
            iris.cube.Cube:
                The cube with percentile values blended over self.blend_coord,
                with suitable weightings applied.
        """
        non_perc_slice = next(cube.slices_over(PERC_COORD))
        weights_array = self.get_weights_array(non_perc_slice, weights)
        weights_array = self._normalise_weights(weights_array)

        # Set up aggregator
        PERCENTILE_BLEND = Aggregator(
            "mean",  # Use CF-compliant cell method.
            PercentileBlendingAggregator.aggregate,
        )

        cube_new = collapsed(
            cube,
            self.blend_coord,
            PERCENTILE_BLEND,
            percentiles=cube.coord(PERC_COORD).points,
            arr_weights=weights_array,
        )

        return cube_new

    def weighted_mean(self, cube, weights):
        """
        Blend data using a weighted mean using the weights provided.

        Args:
            cube (iris.cube.Cube):
                The cube which is being blended over self.blend_coord.
                Assumes leading blend dimension (enforced in process)
            weights (iris.cube.Cube or None):
                Cube of blending weights or None.

        Returns:
            iris.cube.Cube:
                The cube with values blended over self.blend_coord, with
                suitable weightings applied.
        """
        weights_array = self.get_weights_array(cube, weights)

        slice_dim = 1
        allow_slicing = cube.ndim > 3

        if allow_slicing:
            cube_slices = cube.slices_over(slice_dim)
        else:
            cube_slices = [cube]

        weights_slices = (
            np.moveaxis(weights_array, slice_dim, 0)
            if allow_slicing
            else [weights_array]
        )

        result_slices = iris.cube.CubeList(
            collapsed(c_slice, self.blend_coord, iris.analysis.MEAN, weights=w_slice)
            for c_slice, w_slice in zip(cube_slices, weights_slices)
        )

        result = result_slices.merge_cube() if allow_slicing else result_slices[0]

        return result

    def _set_coords_to_remove(self, cube):
        """
        Generate a list of coordinate names associated with the blend
        dimension.  Unless these are time-related coordinates, they should be
        removed after blending.

        Args:
            input_cube (iris.cube.Cube):
                Cube to be blended
        """
        (blend_dim,) = cube.coord_dims(self.blend_coord)
        self.crds_to_remove = []
        for coord in cube.coords():
            if coord.name() in TIME_COORDS:
                continue
            if blend_dim in cube.coord_dims(coord):
                self.crds_to_remove.append(coord.name())

    def _get_cycletime_point(self, cube):
        """
        For cycle and model blending, establish the current cycletime to set on
        the cube after blending.

        Returns:
            numpy.int64:
                Cycle time point in units matching the input cube forecast reference
                time coordinate
        """
        frt_coord = cube.coord("forecast_reference_time")
        frt_units = frt_coord.units.origin
        frt_calendar = frt_coord.units.calendar
        # raises TypeError if cycletime is None
        cycletime_point = cycletime_to_number(
            self.cycletime, time_unit=frt_units, calendar=frt_calendar
        )
        return round_close(cycletime_point, dtype=np.int64)

    def _set_blended_time_coords(self, blended_cube):
        """
        For cycle and model blending:
        - Add a "blend_time" coordinate equal to the current cycletime
        - Update the forecast reference time and forecast period coordinate points
        to reflect the current cycle time (behaviour is DEPRECATED)
        - Remove any bounds from the forecast reference time (behaviour is DEPRECATED)
        - Mark the forecast reference time and forecast period as DEPRECATED

        Modifies cube in place.

        Args:
            blended_cube (iris.cube.Cube)
        """
        try:
            cycletime_point = self._get_cycletime_point(blended_cube)
        except TypeError:
            raise ValueError(
                "Current cycle time is required for cycle and model blending"
            )

        add_blend_time(blended_cube, self.cycletime)
        blended_cube.coord("forecast_reference_time").points = [cycletime_point]
        blended_cube.coord("forecast_reference_time").bounds = None
        if blended_cube.coords("forecast_period"):
            blended_cube.remove_coord("forecast_period")
        new_forecast_period = forecast_period_coord(blended_cube)
        time_dim = blended_cube.coord_dims("time")
        blended_cube.add_aux_coord(new_forecast_period, data_dims=time_dim)
        for coord in ["forecast_period", "forecast_reference_time"]:
            msg = f"{coord} will be removed in future and should not be used"
            blended_cube.coord(coord).attributes.update({"deprecation_message": msg})

    def _update_blended_metadata(self, blended_cube, attributes_dict):
        """
        Update metadata after blending:
        - For cycle and model blending, set a single forecast reference time
        and period using current cycletime
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
            self._set_blended_time_coords(blended_cube)
        for coord in self.crds_to_remove:
            blended_cube.remove_coord(coord)
        if attributes_dict is not None:
            amend_attributes(blended_cube, attributes_dict)
        for attr in MANDATORY_ATTRIBUTES:
            if attr not in blended_cube.attributes:
                blended_cube.attributes[attr] = MANDATORY_ATTRIBUTE_DEFAULTS[attr]

    def process(self, cube, weights=None, cycletime=None, attributes_dict=None):
        """Calculate weighted blend across the chosen coord, for either
           probabilistic or percentile data. If there is a percentile
           coordinate on the cube, it will blend using the
           PercentileBlendingAggregator but the percentile coordinate must
           have at least two points.

        Args:
            cube (iris.cube.Cube):
                Cube to blend across the coord.
            weights (iris.cube.Cube):
                Cube of blending weights. This will have 1 or 3 dimensions,
                corresponding either to blend dimension on the input cube with or
                without and additional 2 spatial dimensions. If None, the input cube
                is blended with equal weights across the blending dimension.
            cycletime (str):
                The cycletime in a YYYYMMDDTHHMMZ format e.g. 20171122T0100Z, required
                for cycle or model blending.  This is used to manually set the a "blend
                time" coordinate on a model or cycle blended cube.
            attributes_dict (dict or None):
                Changes to cube attributes to be applied after blending. See
                :func:`~improver.metadata.amend.amend_attributes` for required
                format. If mandatory attributes are not set here, default
                values are used.

        Returns:
            iris.cube.Cube:
                Containing the weighted blend across the chosen coordinate (typically
                forecast reference time or model).
        Raises:
            TypeError : If the first argument not a cube.
            CoordinateNotFoundError : If coordinate to be collapsed not found
                                      in cube.
            CoordinateNotFoundError : If coordinate to be collapsed not found
                                      in provided weights cube.
            ValueError : If coordinate to be collapsed is not a dimension.
        """
        if not isinstance(cube, iris.cube.Cube):
            msg = (
                "The first argument must be an instance of iris.cube.Cube "
                "but is {}.".format(type(cube))
            )
            raise TypeError(msg)

        if not cube.coords(self.blend_coord):
            msg = "Coordinate to be collapsed not found in cube."
            raise CoordinateNotFoundError(msg)

        self.cycletime = cycletime
        output_dims = get_dim_coord_names(next(cube.slices_over(self.blend_coord)))
        self.blend_coord = find_blend_dim_coord(cube, self.blend_coord)

        # Ensure input cube and weights cube are ordered equivalently along
        # blending coordinate.
        cube = sort_coord_in_cube(cube, self.blend_coord)
        if weights is not None:
            if not weights.coords(self.blend_coord):
                msg = "Coordinate to be collapsed not found in weights cube."
                raise CoordinateNotFoundError(msg)
            weights = sort_coord_in_cube(weights, self.blend_coord)

        # Check that the time coordinate is single valued if required.
        self.check_compatible_time_points(cube)

        # Establish coordinates to be removed after blending
        self._set_coords_to_remove(cube)

        # Do blending and update metadata
        perc_coord = self.check_percentile_coord(cube)
        if perc_coord:
            enforce_coordinate_ordering(cube, [self.blend_coord, "percentile"])
            result = self.percentile_weighted_mean(cube, weights)
        else:
            enforce_coordinate_ordering(cube, [self.blend_coord])
            result = self.weighted_mean(cube, weights)
        self._update_blended_metadata(result, attributes_dict)

        # Reorder resulting dimensions to match input
        enforce_coordinate_ordering(result, output_dims)

        return result
