# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Module to apply a recursive filter to neighbourhooded data."""

from typing import List, Optional, Tuple

import iris
import numpy as np
from iris.cube import Cube, CubeList
from numpy import ndarray

from improver import PostProcessingPlugin
from improver.generate_ancillaries.generate_orographic_smoothing_coefficients import (
    OrographicSmoothingCoefficients,
)
from improver.utilities.cube_checker import check_cube_coordinates
from improver.utilities.pad_spatial import pad_cube_with_halo, remove_halo_from_cube


class RecursiveFilter(PostProcessingPlugin):
    """
    Apply a recursive filter to the input cube.
    """

    def __init__(
        self,
        iterations: Optional[int] = None,
        edge_width: int = 15,
    ) -> None:
        """
        Initialise the class.

        Args:
            iterations:
                The number of iterations of the recursive filter.
            edge_width:
                Half the width of the padding halo applied before
                recursive filtering.
        Raises:
            ValueError: If number of iterations is not None and is set such
                        that iterations is less than 1.
        """
        if iterations is not None:
            if iterations < 1:
                raise ValueError(
                    "Invalid number of iterations: must be >= 1: {}".format(iterations)
                )
        self.iterations = iterations
        self.edge_width = edge_width
        self.smoothing_coefficient_name_format = "smoothing_coefficient_{}"

    @staticmethod
    def _recurse_forward(
        grid: ndarray, smoothing_coefficients: ndarray, axis: int
    ) -> ndarray:
        """
        Method to run the recursive filter in the forward direction.

        In the forward direction:
            Recursive filtering is calculated as:

        .. math::
            B_i = ((1 - \\rm{smoothing\\_coefficient_{i-1}}) \\times A_i) +
            (\\rm{smoothing\\_coefficient_{i-1}} \\times B_{i-1})

        Progressing from gridpoint i-1 to i:
            :math:`B_i` = new value at gridpoint i

            :math:`A_i` = Old value at gridpoint i

            :math:`B_{i-1}` = New value at gridpoint i - 1

        Args:
            grid:
                2D array containing the input data to which the recursive
                filter will be applied.
            smoothing_coefficients:
                Matching 2D array of smoothing_coefficient values that will be
                used when applying the recursive filter along the specified
                axis.
            axis:
                Index of the spatial axis (0 or 1) over which to recurse.

        Returns:
            2D array containing the smoothed field after the recursive
            filter method has been applied to the input array in the
            forward direction along the specified axis.
        """
        lim = grid.shape[axis]
        for i in range(1, lim):
            if axis == 0:
                grid[i, :] = (1.0 - smoothing_coefficients[i - 1, :]) * grid[
                    i, :
                ] + smoothing_coefficients[i - 1, :] * grid[i - 1, :]
            if axis == 1:
                grid[:, i] = (1.0 - smoothing_coefficients[:, i - 1]) * grid[
                    :, i
                ] + smoothing_coefficients[:, i - 1] * grid[:, i - 1]
        return grid

    @staticmethod
    def _recurse_backward(
        grid: ndarray, smoothing_coefficients: ndarray, axis: int
    ) -> ndarray:
        """
        Method to run the recursive filter in the backwards direction.

        In the backwards direction:
            Recursive filtering is calculated as:

        .. math::
            B_i = ((1 - \\rm{smoothing\\_coefficient}) \\times A_i) +
            (\\rm{smoothing\\_coefficient} \\times B_{i+1})

        Progressing from gridpoint i+1 to i:
            :math:`B_i` = new value at gridpoint i

            :math:`A_i` = Old value at gridpoint i

            :math:`B_{i+1}` = New value at gridpoint i+1

        Args:
            grid:
                2D array containing the input data to which the recursive
                filter will be applied.
            smoothing_coefficients:
                Matching 2D array of smoothing_coefficient values that will be
                used when applying the recursive filter along the specified
                axis.
            axis:
                Index of the spatial axis (0 or 1) over which to recurse.

        Returns:
            2D array containing the smoothed field after the recursive
            filter method has been applied to the input array in the
            backwards direction along the specified axis.
        """
        lim = grid.shape[axis]
        for i in range(lim - 2, -1, -1):
            if axis == 0:
                grid[i, :] = (1.0 - smoothing_coefficients[i, :]) * grid[
                    i, :
                ] + smoothing_coefficients[i, :] * grid[i + 1, :]
            if axis == 1:
                grid[:, i] = (1.0 - smoothing_coefficients[:, i]) * grid[
                    :, i
                ] + smoothing_coefficients[:, i] * grid[:, i + 1]
        return grid

    @staticmethod
    def _run_recursion(
        cube: Cube,
        smoothing_coefficients_x: Cube,
        smoothing_coefficients_y: Cube,
        iterations: int,
    ) -> Cube:
        """
        Method to run the recursive filter.

        Args:
            cube:
                2D cube containing the input data to which the recursive
                filter will be applied.
            smoothing_coefficients_x:
                2D cube containing array of smoothing_coefficient values that
                will be used when applying the recursive filter along the
                x-axis.
            smoothing_coefficients_y:
                2D cube containing array of smoothing_coefficient values that
                will be used when applying the recursive filter along the
                y-axis.
            iterations:
                The number of iterations of the recursive filter

        Returns:
            Cube containing the smoothed field after the recursive filter
            method has been applied to the input cube.
        """
        (x_index,) = cube.coord_dims(cube.coord(axis="x").name())
        (y_index,) = cube.coord_dims(cube.coord(axis="y").name())
        output = cube.data

        for _ in range(iterations):
            output = RecursiveFilter._recurse_forward(
                output, smoothing_coefficients_x.data, x_index
            )
            output = RecursiveFilter._recurse_backward(
                output, smoothing_coefficients_x.data, x_index
            )
            output = RecursiveFilter._recurse_forward(
                output, smoothing_coefficients_y.data, y_index
            )
            output = RecursiveFilter._recurse_backward(
                output, smoothing_coefficients_y.data, y_index
            )
            cube.data = output
        return cube

    def _validate_coefficients(
        self, cube: Cube, smoothing_coefficients: CubeList
    ) -> List[Cube]:
        """Validate the smoothing coefficients cubes.

        Args:
            cube:
                2D cube containing the input data to which the recursive
                filter will be applied.

            smoothing_coefficients:
                A cubelist containing two cubes of smoothing_coefficient values,
                one corresponding to smoothing in the x-direction, and the other
                to smoothing in the y-direction.

        Returns:
            A list of smoothing coefficients cubes ordered: [x-coeffs, y-coeffs].

        Raises:
            ValueError: Smoothing coefficient cubes are not named correctly.
            ValueError: If any smoothing_coefficient cube value is over 0.5
            ValueError: The coordinate to be smoothed within the
                smoothing coefficient cube is not of the expected length.
            ValueError: The coordinate to be smoothed within the
                smoothing coefficient cube does not have the expected points.
        """
        # Ensure cubes are in x, y order.
        smoothing_coefficients.sort(key=lambda cell: cell.name())
        axes = ["x", "y"]

        for axis, smoothing_coefficient in zip(axes, smoothing_coefficients):
            # Check the smoothing coefficient cube name is as expected
            expected_name = self.smoothing_coefficient_name_format.format(axis)
            if smoothing_coefficient.name() != expected_name:
                msg = (
                    "The smoothing coefficient cube name {} does not match the "
                    "expected name {}".format(
                        smoothing_coefficient.name(), expected_name
                    )
                )
                raise ValueError(msg)

            # Check the smoothing coefficients do not exceed an empirically determined
            # maximum value; larger values damage conservation significantly.
            if (smoothing_coefficient.data > 0.5).any():
                raise ValueError(
                    "All smoothing_coefficient values must be less than 0.5. "
                    "A large smoothing_coefficient value leads to poor "
                    "conservation of probabilities"
                )

            for test_axis in axes:
                coefficient_crd = smoothing_coefficient.coord(axis=test_axis)
                if test_axis == axis:
                    expected_points = (
                        cube.coord(axis=test_axis).points[1:]
                        + cube.coord(axis=test_axis).points[:-1]
                    ) / 2
                else:
                    expected_points = cube.coord(axis=test_axis).points

                if len(coefficient_crd.points) != len(
                    expected_points
                ) or not np.allclose(coefficient_crd.points, expected_points):
                    msg = (
                        f"The smoothing coefficients {test_axis} dimension does not "
                        "have the expected length or values compared with the cube "
                        "to which smoothing is being applied.\n\nSmoothing "
                        "coefficient cubes must have coordinates that are:\n"
                        "- one element shorter along the dimension being smoothed "
                        f"({axis}) than in the target cube, with points in that "
                        "dimension equal to the mean of each pair of points along "
                        "the dimension in the target cube\n- equal to the points "
                        "in the target cube along the dimension not being smoothed"
                    )
                    raise ValueError(msg)

        return smoothing_coefficients

    def _pad_coefficients(self, coeff_x, coeff_y):
        """Pad smoothing coefficients"""
        pad_x, pad_y = [
            pad_cube_with_halo(
                coeff,
                2 * self.edge_width,
                2 * self.edge_width,
                pad_method="symmetric",
            )
            for coeff in [coeff_x, coeff_y]
        ]
        return pad_x, pad_y

    @staticmethod
    def _update_coefficients_from_mask(
        coeffs_x: Cube, coeffs_y: Cube, mask: Cube
    ) -> Tuple[Cube, Cube]:
        """
        Zero all smoothing coefficients for data points that are masked

        Args:
            coeffs_x
            coeffs_y
            mask

        Returns:
            Updated smoothing coefficients
        """
        plugin = OrographicSmoothingCoefficients(
            use_mask_boundary=False, invert_mask=False
        )
        plugin.zero_masked(coeffs_x, coeffs_y, mask)
        return coeffs_x, coeffs_y

    def process(
        self, cube: Cube, smoothing_coefficients: CubeList, variable_mask: bool = False,
            mask_zeros: bool = False
    ) -> Cube:
        """
        Set up the smoothing_coefficient parameters and run the recursive
        filter. Smoothing coefficients can be generated using
        :class:`~.OrographicSmoothingCoefficients`
        and :func:`~improver.cli.generate_orographic_smoothing_coefficients`.
        The steps undertaken are:

        1. Split the input cube into slices determined by the co-ordinates in
           the x and y directions.
        2. Construct an array of filter parameters (smoothing_coefficients_x
           and smoothing_coefficients_y) for each cube slice that are used to
           weight the recursive filter in the x- and y-directions.
        3. Pad each cube slice with a square-neighbourhood halo and apply
           the recursive filter for the required number of iterations.
        4. Remove the halo from the cube slice and append the recursed cube
           slice to a 'recursed cube'.
        5. Merge all the cube slices in the 'recursed cube' into a 'new cube'.
        6. Modify the 'new cube' so that its scalar dimension co-ordinates are
           consistent with those in the original input cube.
        7. Return the 'new cube' which now contains the recursively filtered
           values for the original input cube.

        The smoothing_coefficient determines how much "value" of a cell
        undergoing filtering is comprised of the current value at that cell and
        how much comes from the adjacent cell preceding it in the direction in
        which filtering is being applied. A larger smoothing_coefficient
        results in a more significant proportion of a cell's new value coming
        from its neighbouring cell.

        Args:
            cube:
                Cube containing the input data to which the recursive filter
                will be applied.
            smoothing_coefficients:
                A cubelist containing two cubes of smoothing_coefficient values,
                one corresponding to smoothing in the x-direction, and the other
                to smoothing in the y-direction.
            variable_mask
                Determines whether each spatial slice of the input cube can have a
                different mask. If False and cube is masked, a check will be made that
                the same mask is present on each spatial slice. If True, each spatial
                slice of cube may contain a different spatial mask.
            mask_zeros
                If set true all of the values of 0 in the cube will be masked,
                stopping the recursive filter from spreading values into these areas.
                They will then be unmasked later on. If the input cube was masked
                this mask will be reapplied to the output at the end.

        Returns:
            Cube containing the smoothed field after the recursive filter
            method has been applied.

        Raises:
            ValueError:
                If variable_mask is False and the masks on each spatial slice of cube
                are not identical.
        """
        if np.ma.isMaskedArray(cube.data):
            cube_mask = np.ma.getmaskarray(cube.data)
            # If the array is masked this gets the mask so it can be used and
            # reapplied later on.
        else:
            cube_mask = None

        if mask_zeros:
            cube.data = np.ma.masked_where(cube.data == 0.0, cube.data, copy=False)
            # This masks any array element that is zero

        cube_format = next(cube.slices([cube.coord(axis="y"), cube.coord(axis="x")]))
        coeffs_x, coeffs_y = self._validate_coefficients(
            cube_format, smoothing_coefficients
        )

        if not variable_mask and np.ma.is_masked(cube.data):
            # check that all spatial slices have identical masks
            mask_cube = next(
                cube.slices([cube.coord(axis="y"), cube.coord(axis="x")])
            ).data.mask
            for cslice in cube.slices([cube.coord(axis="y"), cube.coord(axis="x")]):
                if not np.array_equal(cslice.data.mask, mask_cube):
                    raise ValueError(
                        "Input cube contains spatial slices with different masks."
                    )

        recursed_cube = iris.cube.CubeList()
        for cslice in cube.slices([cube.coord(axis="y"), cube.coord(axis="x")]):
            padded_cube = pad_cube_with_halo(
                cslice, 2 * self.edge_width, 2 * self.edge_width, pad_method="symmetric"
            )

            slice_coeffs_x = coeffs_x.copy()
            slice_coeffs_y = coeffs_y.copy()

            mask_cube = None
            if np.ma.is_masked(cslice.data):
                mask_cube = cslice.copy(data=cslice.data.mask)
                slice_coeffs_x, slice_coeffs_y = self._update_coefficients_from_mask(
                    slice_coeffs_x,
                    slice_coeffs_y,
                    mask_cube,
                )

            padded_coefficients_x, padded_coefficients_y = self._pad_coefficients(
                slice_coeffs_x, slice_coeffs_y
            )

            new_cube = self._run_recursion(
                padded_cube,
                padded_coefficients_x,
                padded_coefficients_y,
                self.iterations,
            )
            new_cube = remove_halo_from_cube(
                new_cube, 2 * self.edge_width, 2 * self.edge_width
            )

            if mask_cube is not None:
                new_cube.data = np.ma.MaskedArray(new_cube.data, mask=mask_cube.data)

            recursed_cube.append(new_cube)

        new_cube = recursed_cube.merge_cube()
        if mask_zeros:
            new_cube.data = np.ma.getdata(new_cube.data)
            cube.data = np.ma.getdata(cube.data)
            # This unmasks all the data on the cubes
            if cube_mask is not None:
                # Reapplying the original mask so the data doesn't change.
                new_cube.data = np.ma.array(new_cube.data, mask=cube_mask)
                cube.data = np.ma.array(cube.data, mask=cube_mask)

        new_cube = check_cube_coordinates(cube, new_cube)

        return new_cube
