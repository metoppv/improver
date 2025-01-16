# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Module containing neighbourhood processing utilities."""

from typing import List, Optional, Union

import iris
import numpy as np
from iris.cube import Cube, CubeList
from numpy import ndarray
from scipy.ndimage.filters import correlate

from improver import BasePlugin, PostProcessingPlugin
from improver.constants import DEFAULT_PERCENTILES
from improver.metadata.forecast_times import forecast_period_coord
from improver.nbhood import radius_by_lead_time
from improver.utilities.common_input_handle import as_cube
from improver.utilities.complex_conversion import complex_to_deg, deg_to_complex
from improver.utilities.cube_checker import (
    check_cube_coordinates,
    find_dimension_coordinate_mismatch,
)
from improver.utilities.neighbourhood_tools import boxsum, pad_and_roll
from improver.utilities.spatial import (
    check_if_grid_is_equal_area,
    distance_to_number_of_grid_cells,
)


def check_radius_against_distance(cube: Cube, radius: float) -> None:
    """Check required distance isn't greater than the size of the domain.

    Args:
        cube:
            The cube to check.
        radius:
            The radius, which cannot be more than half of the
            size of the domain.
    """
    axes = []
    for axis in ["x", "y"]:
        coord = cube.coord(axis=axis).copy()
        coord.convert_units("metres")
        axes.append((max(coord.points) - min(coord.points)))

    max_allowed = np.sqrt(axes[0] ** 2 + axes[1] ** 2) * 0.5
    if radius > max_allowed:
        raise ValueError(
            f"Distance of {radius}m exceeds max domain " f"distance of {max_allowed}m"
        )


def circular_kernel(ranges: int, weighted_mode: bool) -> ndarray:
    """
    Method to create a circular kernel.

    Args:
        ranges:
            Number of grid cells in the x and y direction used to create
            the kernel.
        weighted_mode:
            If True, use a circle for neighbourhood kernel with
            weighting decreasing with radius.
            If False, use a circle with constant weighting.

    Returns:
        Array containing the circular smoothing kernel.
        This will have the same number of dimensions as fullranges.
    """
    # The range is square

    area = ranges * ranges
    # Define the size of the kernel based on the number of grid cells
    # contained within the desired radius.
    kernel = np.ones((int(1 + ranges * 2), (int(1 + ranges * 2))))
    # Create an open multi-dimensional meshgrid.
    open_grid = np.array(
        np.ogrid[[slice(-x, x + 1) for x in (ranges, ranges)]], dtype=object
    )
    if weighted_mode:
        # Create a kernel, such that the central grid point has the
        # highest weighting, with the weighting decreasing with distance
        # away from the central grid point.
        open_grid_summed_squared = np.sum(open_grid**2.0).astype(float)
        kernel[:] = (area - open_grid_summed_squared) / area
        mask = kernel < 0.0
    else:
        mask = np.reshape(np.sum(open_grid**2) > area, np.shape(kernel))
    kernel[mask] = 0.0
    return kernel


class BaseNeighbourhoodProcessing(PostProcessingPlugin):
    """
    A base class used to set up neighbourhood radii for a given cube
    based on the forecast period of that cube if required.
    """

    def __init__(
        self, radii: Union[float, List[float]], lead_times: Optional[List] = None
    ) -> None:
        """
        Create a base neighbourhood processing plugin that processes radii
        related arguments.

            radii:
                The radii in metres of the neighbourhood to apply.
                Rounded up to convert into integer number of grid
                points east and north, based on the characteristic spacing
                at the zero indices of the cube projection-x and y coords.
            lead_times:
                List of lead times or forecast periods, at which the radii
                within 'radii' are defined. The lead times are expected
                in hours.
        """
        if isinstance(radii, list):
            self.radii = [float(x) for x in radii]
        else:
            self.radius = float(radii)
        self.lead_times = lead_times
        if self.lead_times is not None:
            if len(radii) != len(lead_times):
                msg = (
                    "There is a mismatch in the number of radii "
                    "and the number of lead times. "
                    "Unable to continue due to mismatch."
                )
                raise ValueError(msg)

    def _find_radii(
        self, cube_lead_times: Optional[ndarray] = None
    ) -> Union[float, ndarray]:
        """Revise radius or radii for found lead times.
        If cube_lead_times is None, no automatic adjustment
        of the radii will take place.
        Otherwise it will interpolate to find the radius at
        each cube lead time as required.

        Args:
            cube_lead_times:
                Array of forecast times found in cube.

        Returns:
            Required neighbourhood sizes.
        """
        radii = np.interp(cube_lead_times, self.lead_times, self.radii)
        return radii

    def process(self, cube: Cube) -> Cube:
        """
        Supply a cube with a forecast period coordinate in order to set the
        correct radius for use in neighbourhood processing.

        Also checks there are no unmasked NaNs in the input cube.

        Args:
            cube:
                Cube to apply a neighbourhood processing method.

        Returns:
            cube:
                The unaltered input cube.
        """

        if np.isnan(cube.data).any():
            raise ValueError("Error: NaN detected in input cube data")

        if self.lead_times:
            # Interpolate to find the radius at each required lead time.
            fp_coord = forecast_period_coord(cube)
            fp_coord.convert_units("hours")
            self.radius = self._find_radii(cube_lead_times=fp_coord.points)
        return cube


class NeighbourhoodProcessing(BaseNeighbourhoodProcessing):
    """Class for applying neighbourhood processing to produce a smoothed field
    within the chosen neighbourhood."""

    def __init__(
        self,
        neighbourhood_method: str,
        radii: Union[float, List[float]],
        lead_times: Optional[List] = None,
        weighted_mode: bool = False,
        sum_only: bool = False,
        re_mask: bool = True,
    ) -> None:
        """
        Initialise class.

        Args:
            neighbourhood_method:
                Name of the neighbourhood method to use. Options: 'circular',
                'square'.
            radii:
                The radii in metres of the neighbourhood to apply.
                Rounded up to convert into integer number of grid
                points east and north, based on the characteristic spacing
                at the zero indices of the cube projection-x and y coords.
            lead_times:
                List of lead times or forecast periods, at which the radii
                within 'radii' are defined. The lead times are expected
                in hours.
            weighted_mode:
                If True, use a circle for neighbourhood kernel with
                weighting decreasing with radius.
                If False, use a circle with constant weighting.
            sum_only:
                If true, return neighbourhood sum instead of mean.
            re_mask:
                If re_mask is True, the original un-neighbourhood processed
                mask is applied to mask out the neighbourhood processed cube.
                If re_mask is False, the original un-neighbourhood processed
                mask is not applied. Therefore, the neighbourhood processing
                may result in values being present in areas that were
                originally masked.

        Raises:
            ValueError: If the neighbourhood_method is not either
                        "square" or "circular".
            ValueError: If the weighted_mode is used with a
                        neighbourhood_method that is not "circular".
        """
        super().__init__(radii, lead_times=lead_times)
        if neighbourhood_method in ["square", "circular"]:
            self.neighbourhood_method = neighbourhood_method
        else:
            msg = "{} is not a valid neighbourhood_method.".format(neighbourhood_method)
            raise ValueError(msg)
        if weighted_mode and neighbourhood_method != "circular":
            msg = (
                "weighted_mode can only be used if neighbourhood_method is circular."
                f" weighted_mode provided: {weighted_mode}, "
                f"neighbourhood_method provided: {neighbourhood_method}."
            )
            raise ValueError(msg)
        self.weighted_mode = weighted_mode
        self.sum_only = sum_only
        self.re_mask = re_mask

    def _calculate_neighbourhood(
        self, data: ndarray, mask: ndarray = None
    ) -> Union[ndarray, np.ma.MaskedArray]:
        """
        Apply neighbourhood processing. Ensures that masked data does not
        contribute to the neighbourhood result. Masked data is either data that
        is masked in the input data array or that corresponds to zeros in the
        input mask.

        Args:
            data:
                Input data array.
            mask:
                Mask of valid input data elements.

        Returns:
            Array containing the smoothed field after the
            neighbourhood method has been applied.
        """

        if not self.sum_only:
            min_val = np.nanmin(data)
            max_val = np.nanmax(data)

        # Data mask to be eventually used for re-masking.
        # (This is OK even if mask is None, it gives a scalar False mask then.)
        # Invalid data where the mask provided == 0.
        data_mask = mask == 0
        if isinstance(data, np.ma.MaskedArray):
            # Include data mask if masked array.
            data_mask = data_mask | data.mask
            data = data.data

        # Define working type and output type.
        if issubclass(data.dtype.type, np.complexfloating):
            loc_data_dtype = np.complex128
            out_data_dtype = np.complex64
        else:
            # Use 64-bit types for enough precision in accumulations.
            loc_data_dtype = np.float64
            out_data_dtype = np.float32
        data = np.array(data, dtype=loc_data_dtype)

        # Replace invalid elements with zeros so they don't count towards
        # neighbourhood sum
        if self.neighbourhood_method == "circular":
            mask_type = np.float32
        else:
            mask_type = np.int64
        valid_data_mask = np.ones(data.shape, dtype=mask_type)
        valid_data_mask[data_mask] = 0
        data[data_mask] = 0

        if self.sum_only:
            max_extreme_data = None
        else:
            area_sum = self._do_nbhood_sum(valid_data_mask)
            max_extreme_data = area_sum.astype(loc_data_dtype)
        # Where data are all ones in nbhood, result will be same as area_sum
        data = self._do_nbhood_sum(data, max_extreme=max_extreme_data)

        if not self.sum_only:
            with np.errstate(divide="ignore", invalid="ignore"):
                # Calculate neighbourhood mean.
                data = data / area_sum
            # For points where all data in the neighbourhood is masked,
            # set result to nan
            data[area_sum == 0] = np.nan
            data = data.clip(min_val, max_val)

        if self.re_mask:
            data = np.ma.masked_array(data, data_mask, copy=False)

        return data.astype(out_data_dtype)

    def _do_nbhood_sum(
        self, data: np.ndarray, max_extreme: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Calculate the sum-in-area from an array.
        As this can be expensive, the method first checks for the extreme cases where the data are:
        All zeros (result will be all zeros too)
        All ones (result will be max_extreme, if supplied)
        Contains outer rows / columns that are completely zero or completely one, these
        rows and columns are trimmed before calculating the area sum and their contents
        will be as for the appropriate all case above.

        Args:
            data:
                Input data array where any masking has already been replaced with zeroes.
            max_extreme:
                Used as the result for any large areas of data that are all ones, allowing an
                optimisation to be used. If not supplied, the optimisation will only be used for
                large areas of zeroes, where a return of zero can be safely predicted.

        Returns:
            Array containing the sum of data within the usable neighbourhood of each point.
        """
        # Determine the smallest box containing all non-zero or all non-one values with a
        # neighbourhood-sized buffer and quit if there are none.
        data_shape = data.shape
        ystart = xstart = 0
        ystop, xstop = data.shape
        size = data.size
        extreme = 0
        fill_value = 0
        half_nb_size = self.nb_size // 2
        # For the two extreme values, 0 and 1, find the size and position of the smallest array
        # that includes all other values with a buffer of the neighbourhood radius.
        # The smallest box from either extreme will be passed to the neighbourhooding method.
        for _extreme, _fill_value in {0: 0, 1: max_extreme}.items():
            if _fill_value is None or issubclass(data.dtype.type, np.complexfloating):
                # We can't take this shortcut if we don't have either a default value/array,
                # or the data values are complex, as comparisons with non-complex values are
                # tricky.
                continue
            nonextreme_indices = np.argwhere(data != _extreme)
            if nonextreme_indices.size == 0:
                # No non-extreme values, so result will be _fill_value if set
                _ystart = _ystop = _xstart = _xstop = 0
            else:
                (_ystart, _xstart), (_ystop, _xstop) = (
                    nonextreme_indices.min(0),
                    nonextreme_indices.max(0) + 1,
                )
                _ystart = max(0, _ystart - half_nb_size)
                _ystop = min(data_shape[0], _ystop + half_nb_size)
                _xstart = max(0, _xstart - half_nb_size)
                _xstop = min(data_shape[1], _xstop + half_nb_size)
            _size = (_ystop - _ystart) * (_xstop - _xstart)
            if _size < size:
                size, extreme, fill_value, ystart, ystop, xstart, xstop = (
                    _size,
                    _extreme,
                    _fill_value,
                    _ystart,
                    _ystop,
                    _xstart,
                    _xstop,
                )
        if size != data.size:
            # If our chosen extreme allows us to process a subset of data, define the default array
            # of neighbourhood sums that we know we will get for regions of extreme data values.
            if isinstance(fill_value, np.ndarray):
                untrimmed = fill_value.astype(data.dtype)
            else:
                untrimmed = np.full(data_shape, fill_value, dtype=data.dtype)
        if size:
            # The subset of data is non-zero in size, so calculate the neighbourhood sums in the
            # subset.
            data = data[ystart:ystop, xstart:xstop]

            # Calculate neighbourhood totals for input data.
            if self.neighbourhood_method == "square":
                data = boxsum(
                    data, self.nb_size, mode="constant", constant_values=extreme
                )
            elif self.neighbourhood_method == "circular":
                data = correlate(data, self.kernel, mode="nearest")
        else:
            data = untrimmed

        # Expand data to the full size again
        if data.shape != data_shape:
            untrimmed[ystart:ystop, xstart:xstop] = data
            data = untrimmed
        return data

    def process(self, cube: Cube, mask_cube: Optional[Cube] = None) -> Cube:
        """
        Call the methods required to apply a neighbourhood processing to a cube.

        Applies neighbourhood processing to each 2D x-y-slice of the input cube.

        If the input cube is masked the neighbourhood sum is calculated from
        the total of the unmasked data in the neighbourhood around each grid
        point. The neighbourhood mean is then calculated by dividing the
        neighbourhood sum at each grid point by the total number of valid grid
        points that contributed to that sum. If a mask_cube is provided then
        this is used to mask each x-y-slice prior to the neighbourhood sum
        or mean being calculated.


        Args:
            cube:
                Cube containing the array to which the neighbourhood processing
                will be applied. Usually thresholded data.
            mask_cube:
                Cube containing the array to be used as a mask. Zero values in
                this array are taken as points to be masked.

        Returns:
            Cube containing the smoothed field after the
            neighbourhood method has been applied.
        """
        super().process(cube)
        check_if_grid_is_equal_area(cube)

        # If the data is masked, the mask will be processed as well as the
        # original_data * mask array.
        check_radius_against_distance(cube, self.radius)

        grid_cells = distance_to_number_of_grid_cells(cube, self.radius)
        if self.neighbourhood_method == "circular":
            self.kernel = circular_kernel(grid_cells, self.weighted_mode)
            self.nb_size = max(self.kernel.shape)
        else:
            self.nb_size = 2 * grid_cells + 1

        try:
            mask_cube_data = mask_cube.data
        except AttributeError:
            mask_cube_data = None

        result_slices = CubeList()
        for cube_slice in cube.slices([cube.coord(axis="y"), cube.coord(axis="x")]):
            cube_slice.data = self._calculate_neighbourhood(
                cube_slice.data, mask_cube_data
            )
            result_slices.append(cube_slice)
        neighbourhood_averaged_cube = result_slices.merge_cube()

        return neighbourhood_averaged_cube


class GeneratePercentilesFromANeighbourhood(BaseNeighbourhoodProcessing):
    """Class for generating percentiles from a circular neighbourhood."""

    def __init__(
        self,
        radii: Union[float, List[float]],
        lead_times: Optional[List] = None,
        percentiles: List = DEFAULT_PERCENTILES,
    ) -> None:
        """
        Create a neighbourhood processing subclass that generates percentiles
        from a 2D circular neighbourhood. A maximum kernel radius of 500
        grid cells is imposed in order to avoid computational inefficiency and
        possible memory errors.

        Args:
            radii:
                The radii in metres of the neighbourhood to apply.
                Rounded up to convert into integer number of grid
                points east and north, based on the characteristic spacing
                at the zero indices of the cube projection-x and y coords.
            lead_times:
                List of lead times or forecast periods, at which the radii
                within 'radii' are defined. The lead times are expected
                in hours.
            percentiles:
                Percentile values at which to calculate; if not provided uses
                DEFAULT_PERCENTILES.
        """
        super().__init__(radii, lead_times=lead_times)
        self.percentiles = tuple(percentiles)

    def pad_and_unpad_cube(self, slice_2d: Cube, kernel: ndarray) -> Cube:
        """
        Method to pad and unpad a two dimensional cube. The input array is
        padded and percentiles are calculated using a neighbourhood around
        each point. The resulting percentile data are unpadded and put into a
        cube.

        Args:
            slice_2d:
                2d cube to be padded with a halo.
            kernel:
                Kernel used to specify the neighbourhood to consider when
                calculating the percentiles within a neighbourhood.

        Returns:
            A cube containing percentiles generated from a
            neighbourhood.

        Examples:

            1. Take the input slice_2d cube with the data, where 1 is an
               occurrence and 0 is an non-occurrence::

                    # fmt: off
                    [
                        [1., 1., 1.],
                        [1., 0., 1.],
                        [1., 1., 1.],
                    ]
                    # fmt: on

            2. Define a kernel. This kernel is effectively placed over each
               point within the input data. Note that the input data is padded
               prior to placing the kernel over each point, so that the kernel
               does not exceed the bounds of the padded data::

                    # fmt: off
                    [
                        [0., 0., 1., 0., 0.],
                        [0., 1., 1., 1., 0.],
                        [1., 1., 1., 1., 1.],
                        [0., 1., 1., 1., 0.],
                        [0., 0., 1., 0., 0.],
                    ]
                    # fmt: on

            3. Pad the input data. The extent of the padding is given by the
               shape of the kernel. The number of values included within the
               calculation of the mean is determined by the size of the
               kernel::

                    # fmt: off
                    [
                        [0.75, 0.75, 1. , 0.5, 1., 0.75, 0.75],
                        [0.75, 0.75, 1. , 0.5, 1., 0.75, 0.75],
                        [1.  , 1.  , 1. , 1. , 1., 1.  , 1.  ],
                        [0.5 , 0.5 , 1. , 0. , 1., 0.5 , 0.5 ],
                        [1.  , 1.  , 1. , 1. , 1., 1.  , 1.  ],
                        [0.75, 0.75, 1. , 0.5, 1., 0.75, 0.75],
                        [0.75, 0.75, 1. , 0.5, 1., 0.75, 0.75],
                    ]
                    # fmt: on

            4. Calculate the values at the percentiles: [10].
               For the point in the upper right corner within the original
               input data e.g. ::

                    # fmt: off
                    [
                        [->1.<-, 1., 1.],
                        [  1.  , 0., 1.],
                        [  1.  , 1., 1.],
                    ]
                    # fmt: on

               When the kernel is placed over this point within the padded
               data, then the following points are included::

                    # fmt: off
                    [
                        [  0.75,   0.75, ->1.<-, 0.5  , 1.  , 0.75, 0.75],
                        [  0.75, ->0.75,   1.  , 0.5<-, 1.  , 0.75, 0.75],
                        [->1.  ,   1.  ,   1.  , 1.   , 1.<-, 1.  , 1.  ],
                        [  0.5 , ->0.5 ,   1.  , 0.<- , 1.  , 0.5 , 0.5 ],
                        [  1.  ,   1.  , ->1.<-, 1.   , 1.  , 1.  , 1.  ],
                        [  0.75,   0.75,   1.  , 0.5  , 1.  , 0.75, 0.75],
                        [  0.75,   0.75,   1.  , 0.5  , 1.  , 0.75, 0.75],
                    ]
                    # fmt: on

               This gives::

                    # fmt: off
                    [0, 0.5, 0.5, 0.75, 1., 1., 1., 1., 1., 1., 1., 1., 1.]
                    # fmt: on

               As there are 13 points within the kernel, this gives the
               following relationship between percentiles and values.

                  ======  ==========
                  Values  Percentile
                  ======  ==========
                  0.      0
                  0.5     8.33
                  0.5     16.67
                  0.75    25.0
                  1.      33.33
                  1.      41.67
                  1.      50.0
                  1.      58.33
                  1.      66.67
                  1.      75.0
                  1.      83.33
                  1.      91.66
                  1.      100.
                  ======  ==========

               Therefore, for the 10th percentile at the value returned for
               the point in the upper right corner of the original input data
               is 0.5.
               When this process is applied to every point within the original
               input data, the result is::

                    # fmt: off
                    [
                        [
                            [0.75, 0.75, 0.5 , 0.5, 0.5 , 0.75, 0.75],
                            [0.75, 0.55, 0.55, 0.5, 0.55, 0.55, 0.55],
                            [0.55, 0.55, 0.5 , 0.5, 0.5 , 0.5 , 0.5 ],
                            [0.5,  0.5 , 0.5 , 0.5, 0.5 , 0.5 , 0.5 ],
                            [0.5,  0.5 , 0.5 , 0.5, 0.5 , 0.55, 0.55],
                            [0.55, 0.55, 0.55, 0.5, 0.55, 0.55, 0.75],
                            [0.75, 0.75, 0.5 , 0.5, 0.5 , 0.75, 0.75],
                        ]
                    ]
                    # fmt: on

            5. The padding is then removed to give::

                   # fmt: off
                   [
                        [
                            [0.5, 0.5, 0.5],
                            [0.5, 0.5, 0.5],
                            [0.5, 0.5, 0.5],
                        ]
                    ]
                    # fmt: on
        """
        kernel_mask = kernel > 0
        nb_slices = pad_and_roll(
            slice_2d.data, kernel.shape, mode="mean", stat_length=max(kernel.shape) // 2
        )
        percentiles = np.array(self.percentiles, dtype=np.float32)

        # Create cube for output percentile data.
        pctcube = self.make_percentile_cube(slice_2d)

        # Collapse neighbourhood windows into percentiles.
        # (Loop over outer dimension to reduce memory footprint.)
        for nb_chunk, perc_chunk in zip(nb_slices, pctcube.data.swapaxes(0, 1)):
            np.percentile(
                nb_chunk[..., kernel_mask],
                percentiles,
                axis=-1,
                out=perc_chunk,
                overwrite_input=True,
            )

        return iris.util.squeeze(pctcube)

    def process(self, cube: Cube) -> Cube:
        """
        Method to apply a circular kernel to the data within the input cube in
        order to derive percentiles over the kernel.

        Args:
            cube:
                Cube containing array to apply processing to. Usually ensemble realizations.

        Returns:
            Cube containing the percentile fields.
            Has percentile as an added dimension.
        """
        super().process(cube)
        if np.ma.is_masked(cube.data):
            msg = (
                "The use of masked input cubes is not yet implemented in"
                " the GeneratePercentilesFromANeighbourhood plugin."
            )
            raise NotImplementedError(msg)

        # Check that the cube has an equal area grid.
        check_if_grid_is_equal_area(cube)
        # Take data array and identify X and Y axes indices
        grid_cell = distance_to_number_of_grid_cells(cube, self.radius)
        check_radius_against_distance(cube, self.radius)
        kernel = circular_kernel(grid_cell, weighted_mode=False)
        # Loop over each 2D slice to reduce memory demand and derive
        # percentiles on the kernel. Will return an extra dimension.
        pctcubelist = iris.cube.CubeList()
        for slice_2d in cube.slices(
            ["projection_y_coordinate", "projection_x_coordinate"]
        ):
            pctcubelist.append(self.pad_and_unpad_cube(slice_2d, kernel))

        result = pctcubelist.merge_cube()
        exception_coordinates = find_dimension_coordinate_mismatch(
            cube, result, two_way_mismatch=False
        )
        result = check_cube_coordinates(
            cube, result, exception_coordinates=exception_coordinates
        )

        # Arrange cube, so that the coordinate order is:
        # realization, percentile, other coordinates.
        required_order = []
        if result.coords("realization", dim_coords=True):
            required_order.append(result.coord_dims("realization")[0])
        if result.coords("percentile", dim_coords=True):
            required_order.append(result.coord_dims("percentile")[0])
        other_coords = []
        for coord in result.dim_coords:
            if coord.name() not in ["realization", "percentile"]:
                other_coords.append(result.coord_dims(coord.name())[0])
        required_order.extend(other_coords)
        result.transpose(required_order)

        return result

    def make_percentile_cube(self, cube: Cube) -> Cube:
        """Returns a cube with the same metadata as the sample cube
        but with an added percentile dimension.

        Args:
            cube:
                Cube to copy meta data from.

        Returns:
            Cube like input but with added percentiles coordinate.
            Each slice along this coordinate is identical.
        """
        pctcubelist = iris.cube.CubeList()
        pct_coord_name = "percentile"
        for pct in self.percentiles:
            pctcube = cube.copy()
            pctcube.add_aux_coord(
                iris.coords.DimCoord(
                    np.float32(pct), long_name=pct_coord_name, units="%"
                )
            )
            pctcubelist.append(pctcube)
        result = pctcubelist.merge_cube()
        # If percentile coord is not already a dimension, promote it.
        # This is required when self.percentiles is length 1.
        if result.coord_dims(pct_coord_name) == ():
            result = iris.util.new_axis(result, scalar_coord=pct_coord_name)
        return result


class MetaNeighbourhood(BasePlugin):
    """
    Meta-processing module which handles probabilities and percentiles
    neighbourhood processing.
    """

    def __init__(
        self,
        neighbourhood_output: str,
        radii: List[float],
        lead_times: Optional[List[int]] = None,
        neighbourhood_shape: str = "square",
        degrees_as_complex: bool = False,
        weighted_mode: bool = False,
        area_sum: bool = False,
        percentiles: List[float] = DEFAULT_PERCENTILES,
        halo_radius: Optional[float] = None,
    ) -> None:
        """
        Initialise the MetaNeighbourhood class.

        Args:
            neighbourhood_output:
                The form of the results generated using neighbourhood processing.
                If "probabilities" is selected, the mean probability with a
                neighbourhood is calculated. If "percentiles" is selected, then
                the percentiles are calculated with a neighbourhood. Calculating
                percentiles from a neighbourhood is only supported for a circular
                neighbourhood, and the input cube should be ensemble realizations.
                The calculation of percentiles from a neighbourhood is notably slower
                than neighbourhood processing using a thresholded probability field.
                Options: "probabilities", "percentiles".
            radii:
                The radius or a list of radii in metres of the neighbourhood to
                apply.
                If it is a list, it must be the same length as lead_times, which
                defines at which lead time to use which nbhood radius. The radius
                will be interpolated for intermediate lead times.
            lead_times:
                The lead times in hours that correspond to the radii to be used.
                If lead_times are set, radii must be a list the same length as
                lead_times.
            neighbourhood_shape:
                Name of the neighbourhood method to use. Only a "circular"
                neighbourhood shape is applicable for calculating "percentiles"
                output.
                Options: "circular", "square".
                Default: "square".
            degrees_as_complex:
                Include this option to process angles as complex numbers.
                Not compatible with circular kernel or percentiles.
            weighted_mode:
                Include this option to set the weighting to decrease with radius.
                Otherwise a constant weighting is assumed.
                weighted_mode is only applicable for calculating "probability"
                neighbourhood output using the circular kernel.
            area_sum:
                Return sum rather than fraction over the neighbourhood area.
            percentiles:
                Calculates value at the specified percentiles from the
                neighbourhood surrounding each grid point. This argument has no
                effect if the output is probabilities.
            halo_radius:
                Set this radius in metres to define the excess halo to clip. Used
                where a larger grid was defined than the standard grid and we want
                to clip the grid back to the standard grid. Otherwise no clipping
                is applied.
        """
        self._neighbourhood_output = neighbourhood_output
        self._neighbourhood_shape = neighbourhood_shape
        self._radius_or_radii, self._lead_times = radius_by_lead_time(radii, lead_times)
        self._degrees_as_complex = degrees_as_complex
        self._weighted_mode = weighted_mode
        self._area_sum = area_sum
        self._percentiles = percentiles
        self._halo_radius = halo_radius

        if neighbourhood_output == "percentiles":
            if weighted_mode:
                raise RuntimeError(
                    "weighted_mode cannot be used with"
                    'neighbourhood_output="percentiles"'
                )
            if degrees_as_complex:
                raise RuntimeError("Cannot generate percentiles from complex numbers")

        if neighbourhood_shape == "circular":
            if degrees_as_complex:
                raise RuntimeError(
                    "Cannot process complex numbers with circular neighbourhoods"
                )

    def process(self, cube: Cube, mask: Cube = None) -> Cube:
        """
        Apply neighbourhood processing to the input cube.

        Args:
            cube: The input cube.
            mask: The mask cube.

        Returns:
            iris.cube.Cube: The processed cube.
        """
        cube = as_cube(cube)
        if mask:
            mask = as_cube(mask)

        if self._degrees_as_complex:
            # convert cube data into complex numbers
            cube.data = deg_to_complex(cube.data)

        if self._neighbourhood_output == "probabilities":
            result = NeighbourhoodProcessing(
                self._neighbourhood_shape,
                self._radius_or_radii,
                lead_times=self._lead_times,
                weighted_mode=self._weighted_mode,
                sum_only=self._area_sum,
                re_mask=True,
            )(cube, mask_cube=mask)
        elif self._neighbourhood_output == "percentiles":
            result = GeneratePercentilesFromANeighbourhood(
                self._radius_or_radii,
                lead_times=self._lead_times,
                percentiles=self._percentiles,
            )(cube)

        if self._degrees_as_complex:
            # convert neighbourhooded cube back to degrees
            result.data = complex_to_deg(result.data)
        if self._halo_radius is not None:
            from improver.utilities.pad_spatial import remove_cube_halo

            result = remove_cube_halo(result, self._halo_radius)
        return result
