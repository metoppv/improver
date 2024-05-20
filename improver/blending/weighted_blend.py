# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of improver and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Module containing classes for doing weighted blending by collapsing a
   whole dimension."""

import warnings
from typing import List, Optional, Union

import iris
import numpy as np
from iris.analysis import Aggregator
from iris.coords import AuxCoord
from iris.cube import Cube, CubeList
from iris.exceptions import CoordinateNotFoundError
from numpy import ndarray

from improver import BasePlugin, PostProcessingPlugin
from improver.blending import MODEL_BLEND_COORD, MODEL_NAME_COORD
from improver.blending.utilities import find_blend_dim_coord, store_record_run_as_coord
from improver.metadata.constants import FLOAT_DTYPE, PERC_COORD
from improver.metadata.forecast_times import rebadge_forecasts_as_latest_cycle
from improver.utilities.cube_manipulation import (
    MergeCubes,
    collapsed,
    enforce_coordinate_ordering,
    get_coord_names,
    get_dim_coord_names,
    sort_coord_in_cube,
)
from improver.wind_calculations.wind_direction import WindDirection


class MergeCubesForWeightedBlending(BasePlugin):
    """Prepares cubes for cycle and grid blending"""

    def __init__(
        self,
        blend_coord: str,
        weighting_coord: Optional[str] = None,
        model_id_attr: Optional[str] = None,
        record_run_attr: Optional[str] = None,
    ) -> None:
        """
        Initialise the class

        Args:
            blend_coord:
                Name of coordinate over which blending will be performed.  For
                multi-model blending this is flexible to any string containing
                "model".  For all other coordinates this is prescriptive:
                cube.coord(blend_coord) must return an iris.coords.Coord
                instance for all cubes passed into the "process" method.
            weighting_coord:
                The coordinate across which weights will be scaled in a
                multi-model blend.
            model_id_attr:
                Name of attribute used to identify model for grid blending.
            record_run_attr:
                Name of attribute used to record models and cycles blended.
                Ignored if None.

        Raises:
            ValueError:
                If trying to blend over model when model_id_attr is not set
        """
        if "model" in blend_coord and model_id_attr is None:
            raise ValueError(
                "model_id_attr required to blend over {}".format(blend_coord)
            )

        # ensure model coordinates are not created for non-model blending
        if "model" not in blend_coord and (
            model_id_attr is not None and record_run_attr is None
        ):
            warnings.warn(
                "model_id_attr not required for blending over {} - "
                "will be ignored".format(blend_coord)
            )
            model_id_attr = None

        self.blend_coord = blend_coord
        self.weighting_coord = weighting_coord
        self.model_id_attr = model_id_attr
        self.record_run_attr = record_run_attr

    def _create_model_coordinates(self, cubelist: Union[List[Cube], CubeList]) -> None:
        """
        Adds numerical model ID and string model configuration scalar
        coordinates to input cubes if self.model_id_attr is specified.
        Sets the original attribute value to "blend", in anticipation.
        Modifies cubes in place.

        Args:
            cubelist:
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
                    "Cannot create model dimension coordinate with duplicate points"
                )
            model_titles.append(model_title)

            new_model_id_coord = AuxCoord(
                np.array([1000 * i], dtype=np.int32),
                units="1",
                long_name=MODEL_BLEND_COORD,
            )
            new_model_coord = AuxCoord(
                [model_title], units="no_unit", long_name=MODEL_NAME_COORD
            )

            cube.add_aux_coord(new_model_id_coord)
            cube.add_aux_coord(new_model_coord)

    @staticmethod
    def _remove_blend_time(cube: Cube) -> Cube:
        """If present on input, remove existing blend time coordinate (as this will
        be replaced on blending)"""
        if "blend_time" in get_coord_names(cube):
            cube.remove_coord("blend_time")
        return cube

    @staticmethod
    def _remove_deprecation_warnings(cube: Cube) -> Cube:
        """Remove deprecation warnings from forecast period and forecast reference
        time coordinates so that these can be merged before blending"""
        for coord in ["forecast_period", "forecast_reference_time"]:
            try:
                cube.coord(coord).attributes.pop("deprecation_message", None)
            except CoordinateNotFoundError:
                pass
        return cube

    def process(
        self,
        cubes_in: Union[List[Cube], Cube, CubeList],
        cycletime: Optional[str] = None,
    ) -> Cube:
        """
        Prepares merged input cube for cycle and grid blending. Makes updates to
        metadata (attributes and time-type coordinates) ONLY in so far as these are
        needed to ensure inputs can be merged into a single cube.

        Args:
            cubes_in:
                Cubes to be merged.
            cycletime:
                The cycletime in a YYYYMMDDTHHMMZ format e.g. 20171122T0100Z.
                Can be used in rationalise_blend_time_coordinates.

        Returns:
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

        if self.record_run_attr is not None and self.model_id_attr is not None:
            store_record_run_as_coord(
                cubelist, self.record_run_attr, self.model_id_attr
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

        # check all input cubes have the blend coordinate
        for cube in cubelist:
            if "model" not in self.blend_coord and not cube.coords(self.blend_coord):
                raise ValueError(
                    "{} coordinate is not present on all input "
                    "cubes".format(self.blend_coord)
                )

        # create model ID and model configuration coordinates if blending
        # different models
        if "model" in self.blend_coord and self.model_id_attr is not None:
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
        :download:`Combining Probabilities by Caroline Jones, 2017
        <../files/Combining_Probabilities.pdf>`
    """

    @staticmethod
    def aggregate(
        data: ndarray, axis: int, percentiles: ndarray, arr_weights: ndarray
    ) -> ndarray:
        """
        Function to blend percentile data over a given dimension.
        The input percentile data must be provided with the blend coord as the
        first axis and the percentile coord as the second (these are re-ordered
        after aggregator initialisation at the start of this function).  Weights
        data must be provided with the blend coord as the first dimension.

        Args:
            data:
                Array containing the data to blend.
            axis:
                The index of the coordinate dimension in the cube. This
                dimension will be aggregated over.
            percentiles:
                Array of percentile values e.g [0, 20.0, 50.0, 70.0, 100.0],
                same size as the percentile (second) dimension of data.
            arr_weights:
                Array of weights, same shape as data, but without the percentile
                dimension (weights do not vary with percentile).

        Note "weights" has special meaning in Aggregator, hence
        using a different name for this variable.

        Returns:
            Array containing the weighted percentile blend data across
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
    def blend_percentiles(
        perc_values: ndarray, percentiles: ndarray, weights: ndarray
    ) -> ndarray:
        """ Blend percentiles function, to calculate the weighted blend across
            a given axis of percentile data for a single grid point.

        Args:
            perc_values:
                Array containing the percentile values to blend, with
                shape: (length of coord to blend, num of percentiles)
            percentiles:
                Array of percentile values e.g [0, 20.0, 50.0, 70.0, 100.0],
                same size as the percentile dimension of data.
            weights:
                Array of weights, same size as the axis dimension of data,
                that we will blend over.

        Returns:
            Array containing the weighted percentile blend data
            across the chosen coord
        """
        inputs_to_blend = perc_values.shape[0]
        combined_cdf = np.zeros((inputs_to_blend, len(percentiles)), dtype=FLOAT_DTYPE)

        # Loop over the axis we are blending over finding the values for the
        # probability at each threshold in the cdf, for each of the other
        # points in the axis we are blending over.
        # Then add the probabilities multiplied by the correct weight to the
        # running total.
        for i in range(0, inputs_to_blend):
            interp_values = np.reshape(
                np.interp(perc_values, perc_values[i], percentiles),
                (inputs_to_blend, len(percentiles)),
            )
            interp_values[i] = percentiles
            combined_cdf += interp_values * weights[i]

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

    def __init__(self, blend_coord: str, timeblending: bool = False) -> None:
        """Set up for a Weighted Blending plugin

        Args:
            blend_coord:
                The name of the coordinate dimension over which the cube will
                be blended.
            timeblending:
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

    def __repr__(self) -> str:
        """Represent the configured plugin instance as a string."""
        description = (
            "<WeightedBlendAcrossWholeDimension: coord = {}, timeblending: {}>"
        )
        return description.format(self.blend_coord, self.timeblending)

    @staticmethod
    def check_percentile_coord(cube: Cube) -> bool:
        """
        Determines if the cube to be blended has a percentile dimension
        coordinate.

        Args:
            cube:
                The cube to be checked for a percentile coordinate.

        Returns:
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

    def check_compatible_time_points(self, cube: Cube) -> None:
        """
        Check that the time coordinate only contains a single time. Data
        varying over the blending coordinate should all be for the same
        validity time unless we are triangular time blending. In this case the
        timeblending flag should be true and this function will not raise an
        exception.

        Args:
            cube:
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
    def shape_weights(cube: Cube, weights: Cube) -> ndarray:
        """
        The function shapes weights to match the diagnostic cube. A cube of
        weights that vary across the blending coordinate will be broadcast to
        match the complete multidimensional cube shape. A multidimensional cube
        of weights will be checked to ensure that the coordinate names match
        between the two cubes. If they match the order will be enforced and
        then the shape will be checked. If the shapes match the weights will be
        returned as an array.

        Args:
            cube:
                The data cube on which a coordinate is being blended.
            weights:
                Cube of blending weights.

        Returns:
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
    def _normalise_weights(weights: ndarray) -> ndarray:
        """
        Checks that weights across the leading blend dimension sum up to 1.  If
        not, normalise across the blending dimension ignoring any points at which
        the sum of weights is zero.

        Args:
            weights:
                Array of weights shaped to match the data cube.

        Returns:
            Weights normalised along the (leading) blend dimension
        """
        sum_of_weights = np.sum(weights, axis=0)
        sum_of_non_zero_weights = sum_of_weights[sum_of_weights > 0]
        if not (np.isclose(sum_of_non_zero_weights, 1)).all():
            weights = np.where(
                sum_of_weights > 0, np.divide(weights, sum_of_weights), 0
            )
        return weights

    def get_weights_array(self, cube: Cube, weights: Optional[Cube]) -> ndarray:
        """
        Given a 1 or multidimensional cube of weights, reshape and broadcast
        these to the shape of the data cube. If no weights are provided, an
        array of weights is returned that equally weights all slices across
        the blending coordinate.

        Args:
            cube:
                Template cube to reshape weights, with a leading blend coordinate
            weights:
                Cube of initial blending weights or None

        Returns:
            An array of weights that matches the template cube shape.
        """
        if weights:
            weights_array = self.shape_weights(cube, weights)
        else:
            (number_of_fields,) = cube.coord(self.blend_coord).shape
            weight = FLOAT_DTYPE(1.0 / number_of_fields)
            weights_array = np.broadcast_to(weight, cube.shape)

        return weights_array

    def percentile_weighted_mean(self, cube: Cube, weights: Optional[Cube]) -> Cube:
        """
        Blend percentile data using the weights provided.

        Args:
            cube:
                The cube which is being blended over self.blend_coord. Assumes
                self.blend_coord and percentile are leading coordinates (enforced
                in process).
            weights:
                Cube of blending weights.

        Returns:
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

    def weighted_mean(self, cube: Cube, weights: Optional[Cube]) -> Cube:
        """
        Blend data using a weighted mean using the weights provided. Circular
        data identified with a unit of "degrees" are blended appropriately.

        Args:
            cube:
                The cube which is being blended over self.blend_coord.
                Assumes leading blend dimension (enforced in process)
            weights:
                Cube of blending weights or None.

        Returns:
            The cube with values blended over self.blend_coord, with
            suitable weightings applied.
        """

        # If units are degrees, convert degrees to complex numbers.
        if cube.units == "degrees":
            cube.data = WindDirection.deg_to_complex(cube.data)

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

        # If units are degrees, convert complex numbers back to degrees.
        if cube.units == "degrees":
            result.data = WindDirection.complex_to_deg(result.data)

        return result

    def process(self, cube: Cube, weights: Optional[Cube] = None) -> Cube:
        """Calculate weighted blend across the chosen coord, for either
           probabilistic or percentile data. If there is a percentile
           coordinate on the cube, it will blend using the
           PercentileBlendingAggregator but the percentile coordinate must
           have at least two points.

        Args:
            cube:
                Cube to blend across the coord.
            weights:
                Cube of blending weights. This will have 1 or 3 dimensions,
                corresponding either to blend dimension on the input cube with or
                without and additional 2 spatial dimensions. If None, the input cube
                is blended with equal weights across the blending dimension.

        Returns:
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

        # Do blending and update metadata.
        if self.check_percentile_coord(cube):
            enforce_coordinate_ordering(cube, [self.blend_coord, "percentile"])
            result = self.percentile_weighted_mean(cube, weights)
        else:
            enforce_coordinate_ordering(cube, [self.blend_coord])
            result = self.weighted_mean(cube, weights)

        # Reorder resulting dimensions to match input
        enforce_coordinate_ordering(result, output_dims)

        return result
