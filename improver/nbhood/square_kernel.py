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
"""This module contains methods for square neighbourhood processing."""

import iris
import numpy as np

from improver.nbhood.circular_kernel import (
    check_radius_against_distance)
from improver.utilities.cube_checker import (
    check_cube_coordinates, check_for_x_and_y_axes)
from improver.utilities.cube_manipulation import clip_cube_data
from improver.utilities.pad_spatial import (
    pad_cube_with_halo, remove_halo_from_cube)
from improver.utilities.spatial import distance_to_number_of_grid_cells


class SquareNeighbourhood:

    """
    Methods for use in application of a square neighbourhood.
    """

    def __init__(self, weighted_mode=True, sum_or_fraction="fraction",
                 re_mask=True):
        """
        Initialise class.

        Args:
            weighted_mode (bool):
                This is included to allow a standard interface for both the
                square and circular neighbourhood plugins.
            sum_or_fraction (str):
                Identifier for whether sum or fraction should be returned from
                neighbourhooding. The sum represents the sum of the
                neighbourhood. The fraction represents the sum of the
                neighbourhood divided by the neighbourhood area.
                Valid options are "sum" or "fraction".
            re_mask (bool):
                If re_mask is True, the original un-neighbourhood processed
                mask is applied to mask out the neighbourhood processed cube.
                If re_mask is False, the original un-neighbourhood processed
                mask is not applied. Therefore, the neighbourhood processing
                may result in values being present in areas that were
                originally masked.
        """
        self.weighted_mode = weighted_mode
        if sum_or_fraction not in ["sum", "fraction"]:
            msg = ("The neighbourhood output can either be in the form of a "
                   "sum of all the points in the neighbourhood or a fraction "
                   "of the sum of the neighbourhood divided by the "
                   "neighbourhood area. The {} option is invalid. "
                   "Valid options are 'sum' or 'fraction'.")
            raise ValueError(msg)
        self.sum_or_fraction = sum_or_fraction
        self.re_mask = re_mask

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        result = ('<SquareNeighbourhood: weighted_mode: {}, '
                  'sum_or_fraction: {}, re_mask: {}>')
        return result.format(self.weighted_mode, self.sum_or_fraction,
                             self.re_mask)

    @staticmethod
    def cumulate_array(cube, iscomplex=False):
        """
        Method to calculate the cumulative sum of an m x n array, by first
        cumulating along the y direction so that the largest values
        are in the nth row, and then cumulating along the x direction,
        so that the largest values are in the mth column. Each grid point
        will contain the cumulative sum from the origin to that grid point.

        Args:
            cube (iris.cube.Cube):
                Cube to which the cumulative summing along the y and x
                direction will be applied. The cube should contain only x and
                y dimensions, so will generally be a slice of a cube ordered
                so that y is first in the cube (i.e. axis=0).
            iscomplex (bool):
                Flag indicating whether cube.data contains complex values.

        Returns:
            iris.cube.Cube:
                Cube to which the cumulative summing
                along the y and x direction has been applied.
        """
        summed_cube = cube.copy()
        if iscomplex:
            data = cube.data.astype(complex)
        elif cube.name().startswith("probability_of"):
            # No need for high precision calculation, just between 0 and 1.
            data = cube.data.astype(np.float32)
        else:
            # Go to high precision for safety.
            data = cube.data.astype(np.longdouble)
        data_summed_along_y = np.cumsum(data, axis=0)
        data_summed_along_x = (
            np.cumsum(data_summed_along_y, axis=1))
        summed_cube.data = data_summed_along_x
        return summed_cube

    @staticmethod
    def calculate_neighbourhood(summed_cube,
                                ymax_xmax_disp, ymin_xmax_disp,
                                ymin_xmin_disp, ymax_xmin_disp,
                                n_rows, n_columns):
        """
        Fast vectorised approach to calculating neighbourhood totals.

        Displacements are calculated as follows for the following input array,
        where the accumulation has occurred from top to
        bottom and left to right::

        | 1 | 2 | 2 | 2 |
        | 1 | 3 | 4 | 4 |
        | 2 | 4 | 5 | 6 |
        | 2 | 4 | 6 | 7 |


        For a 3x3 neighbourhood centred around the point with a value of 5::

        | 1 (C) | 2 | 2                 | 2 (D) |
        | 1     | 3 | 4                 | 4     |
        | 2     | 4 | 5 (Central point) | 6     |
        | 2 (A) | 4 | 6                 | 7 (B) |

        To calculate the value for the neighbourhood sum at the "Central point"
        with a value of 5, calculate::

          Neighbourhood sum = B - A - D + C

        At the central point, this will yield::

          Neighbourhood sum = 7 - 2 - 2 +1 => 4

        Args:
            summed_cube (iris.cube.Cube):
                cube on which to calculate the neighbourhood total.
            ymax_xmax_disp (int):
                Displacement from the point at the centre
                of the neighbourhood.
                Equivalent to point B in the docstring example.
            ymax_xmin_disp (int):
                Displacement from the point at the centre
                of the neighbourhood.
                Equivalent to point A in the docstring example.
            ymin_xmax_disp (int):
                Displacement from the point at the centre
                of the neighbourhood.
                Equivalent to point D in the docstring example.
            ymin_xmin_disp (int):
                Displacement from the point at the centre
                of the neighbourhood.
                Equivalent to point C in the docstring example.
            n_rows (int):
                Number of rows
            n_columns (int):
                Number of columns

        Returns:
            numpy.ndarray:
                Array containing the calculated neighbourhood total.
        """
        flattened = summed_cube.data.flatten()
        ymax_xmax_array = np.roll(flattened, -ymax_xmax_disp)
        ymin_xmax_array = np.roll(flattened, -ymin_xmax_disp)
        ymin_xmin_array = np.roll(flattened, -ymin_xmin_disp)
        ymax_xmin_array = np.roll(flattened, -ymax_xmin_disp)
        neighbourhood_total = (ymax_xmax_array - ymin_xmax_array +
                               ymin_xmin_array - ymax_xmin_array)
        neighbourhood_total.resize(n_rows, n_columns)
        return neighbourhood_total

    def mean_over_neighbourhood(self, summed_cube, summed_mask,
                                cells, iscomplex=False):
        """
        Method to calculate the average value in a square neighbourhood using
        the 4-point algorithm to find the total sum over the neighbourhood.

        The output from the cumulate_array method can be used to
        calculate the sum over a neighbourhood of size
        (2*cells+1)**2. This sum is then divided by the area of
        the neighbourhood to calculate the mean value in the neighbourhood.

        For all points, a fast vectorised approach is taken:

        1. The displacements between the four points used to calculate the
           neighbourhood total sum and the central grid point are calculated.
        2. Within the function calculate_neighbourhood...
           Four copies of the cumulate array output are flattened and rolled
           by these displacements to align the four terms used in the
           neighbourhood total sum calculation.
        3. The neighbourhood total at all points can then be calculated
           simultaneously in a single vector sum.

        Neighbourhood mean = Neighbourhood sum / Neighbourhood area

        Neighbourhood area = (2 * nb_width +1)^2 if there are no missing
        points, nb_width is the neighbourhood width, which is equal to 1 for a
        3x3 neighbourhood.

        Args:
            summed_cube (iris.cube.Cube):
                Summed Cube to which neighbourhood processing is being
                applied. Must be passed through cumulate_array method first.
                The cube should contain only x and y dimensions,
                so will generally be a slice of a cube.
            summed_mask (iris.cube.Cube):
                Summed Mask used to calculate neighbourhood size.
                Must be passed through cumulate_array method first.
                The cube should contain only x and y dimensions,
                so will generally be a slice of a cube.
            cells (int):
                The radius of the neighbourhood in grid points, in the x
                direction (excluding the central grid point).
            iscomplex (bool):
                Flag indicating whether cube.data contains complex values.

        Returns:
            iris.cube.Cube:
                Cube to which square neighbourhood has been applied.
        """
        cube = summed_cube
        check_for_x_and_y_axes(summed_cube)

        # Calculate displacement factors to find 4-points after flattening the
        # array.
        n_rows = len(cube.coord(axis="y").points)
        n_columns = len(cube.coord(axis="x").points)

        # Displacements from the point at the centre of the neighbourhood.
        # Equivalent to point B in the docstring example.
        ymax_xmax_disp = (cells * n_columns) + cells
        # Equivalent to point A in the docstring example.
        ymax_xmin_disp = (cells * n_columns) - cells - 1

        # Equivalent to point D in the docstring example.
        ymin_xmax_disp = (-1 * (cells + 1) * n_columns) + cells
        # Equivalent to point C in the docstring example.
        ymin_xmin_disp = (-1 * (cells + 1) * n_columns) - cells - 1

        # Flatten the cube data and create 4 copies of the flattened
        # array which are rolled to align the 4-points which are needed
        # for the calculation.
        neighbourhood_total = self.calculate_neighbourhood(
            summed_cube, ymax_xmax_disp, ymin_xmax_disp,
            ymin_xmin_disp, ymax_xmin_disp,
            n_rows, n_columns)

        if self.sum_or_fraction == "fraction":
            # Initialise and calculate the neighbourhood area.
            neighbourhood_area = self.calculate_neighbourhood(
                summed_mask, ymax_xmax_disp, ymin_xmax_disp,
                ymin_xmin_disp, ymax_xmin_disp,
                n_rows, n_columns)

            with np.errstate(invalid='ignore', divide='ignore'):
                if iscomplex:
                    cube.data = (neighbourhood_total.astype(complex) /
                                 neighbourhood_area.astype(complex))
                else:
                    cube.data = (neighbourhood_total.astype(float) /
                                 neighbourhood_area.astype(float))
                cube.data[~np.isfinite(cube.data)] = np.nan
        elif self.sum_or_fraction == "sum":
            if iscomplex:
                cube.data = neighbourhood_total.astype(complex)
            else:
                cube.data = neighbourhood_total.astype(float)

        return cube

    @staticmethod
    def set_up_cubes_to_be_neighbourhooded(cube, mask_cube=None):
        """
        Set up a cube ready for neighourhooding the data.

        Args:
            cube (iris.cube.Cube):
                Cube that will be checked for whether the data is masked
                or nan. The cube should contain only x and y dimensions,
                so will generally be a slice of a cube.
            mask_cube (iris.cube.Cube):
                Input Cube containing the array to be used as a mask.

        Returns:
            (tuple): tuple containing:
                **cube** (iris.cube.Cube):
                    Cube with masked or NaN values set to 0.0
                **mask** (iris.cube.Cube):
                    Cube with masked or NaN values set to 0.0
                **nan_array** (numpy.ndarray):
                    numpy array to be used to set the values within
                    the data of the output cube to be NaN.

        """
        # Set up mask_cube
        if not mask_cube:
            mask = cube.copy()
            mask.data = np.real(np.ones_like(mask.data))
        else:
            mask = mask_cube
        # If there is a mask, fill the data array of the mask_cube with a
        # logical array, logically inverted compared to the integer version of
        # the mask within the original data array.

        if isinstance(cube.data, np.ma.MaskedArray):
            index = np.where(cube.data.mask.astype(int) == 1)
            mask.data[index] = 0.0
            cube.data = cube.data.data
        mask.rename('mask_data')
        cube = iris.util.squeeze(cube)
        mask = iris.util.squeeze(mask)
        # Set NaN values to 0 in both the cube data and mask data.
        nan_array = np.isnan(cube.data)
        mask.data[nan_array] = 0.0
        cube.data[nan_array] = 0.0
        #  Set cube.data to 0.0 where mask_cube is 0.0
        cube.data = (cube.data * mask.data).astype(cube.data.dtype)
        return cube, mask, nan_array

    def _pad_and_calculate_neighbourhood(
            self, cube, mask, grid_cells):
        """
        Apply neighbourhood processing consisting of the following steps:

        1. Pad a halo around the input cube to allow vectorised
           neighbourhooding at edgepoints.
        2. Cumulate the array along the x and y axes.
        3. Apply neighbourhood processing to the cumulated array.

        Args:
            cube (iris.cube.Cube):
                Cube with masked or NaN values set to 0.0
            mask (iris.cube.Cube):
                Cube with masked or NaN values set to 0.0
            grid_cells (float or int):
                The number of grid cells along the x axis used to create a
                square neighbourhood.

        Returns:
            iris.cube.Cube:
                Cube containing the smoothed field after the square
                neighbourhood method has been applied with halo added.
        """
        # Pad the iris cube with the neighbourhood radius plus 1 grid point.
        # Since the neighbourhood size is 2*radius + 1, this means that all
        # grid points within the original domain will have data available for
        # the full neighbourhood size.  The halo is removed later.
        padded_cube = pad_cube_with_halo(cube, grid_cells+1, grid_cells+1,
                                         halo_mean_data=False)
        padded_mask = pad_cube_with_halo(mask, grid_cells+1, grid_cells+1,
                                         halo_mean_data=False)

        # Check whether cube contains complex values
        is_complex = np.any(np.iscomplex(cube.data))

        summed_up_cube = self.cumulate_array(padded_cube, is_complex)
        summed_up_mask = self.cumulate_array(padded_mask)
        neighbourhood_averaged_cube = (
            self.mean_over_neighbourhood(summed_up_cube, summed_up_mask,
                                         grid_cells,
                                         is_complex))
        if neighbourhood_averaged_cube.dtype in [np.float64, np.longdouble]:
            neighbourhood_averaged_cube.data = (
                neighbourhood_averaged_cube.data.astype(np.float32))
        return neighbourhood_averaged_cube

    def _remove_padding_and_mask(
            self, neighbourhood_averaged_cube,
            original_cube, mask, grid_cells):
        """
        Remove the halo from the padded array and apply the mask, if required.
        If fraction option set, clip the data so values lie within
        the range of the original cube.

        Args:
            neighbourhood_averaged_cube (iris.cube.Cube):
                Cube containing the smoothed field after the square
                neighbourhood method has been applied.
            original_cube (iris.cube.Cube or None):
                The original cube slice.
            mask (iris.cube.Cube):
                The mask cube created by set_up_cubes_to_be_neighbourhooded.
            grid_cells (float or int):
                The number of grid cells used to create a square
                neighbourhood (assuming an equal area grid).

        Returns:
            iris.cube.Cube:
                Cube containing the smoothed field after the square
                neighbourhood method has been applied and halo removed.
        """
        # Correct neighbourhood averages for masked data, which may have been
        # calculated using larger neighbourhood areas than are present in
        # reality.
        neighbourhood_averaged_cube = remove_halo_from_cube(
            neighbourhood_averaged_cube, grid_cells+1, grid_cells+1)
        if self.re_mask and mask.data.min() < 1.0:
            neighbourhood_averaged_cube.data = np.ma.masked_array(
                neighbourhood_averaged_cube.data,
                mask=np.logical_not(mask.data.squeeze()))
        # Add clipping
        if self.sum_or_fraction == "fraction":
            min_val = np.nanmin(original_cube.data)
            max_val = np.nanmax(original_cube.data)
            neighbourhood_averaged_cube = (
                clip_cube_data(neighbourhood_averaged_cube, min_val, max_val))
        return neighbourhood_averaged_cube

    def run(self, cube, radius, mask_cube=None):
        """
        Call the methods required to apply a square neighbourhood
        method to a cube.

        The steps undertaken are:

        1. Set up cubes by determining, if the arrays are masked.
        2. Pad the input array with a halo and then calculate the neighbourhood
           of the haloed array.
        3. Remove the halo from the neighbourhooded array and deal with a mask,
           if required.

        Args:
            cube (iris.cube.Cube):
                Cube containing the array to which the square neighbourhood
                will be applied.
            radius (float):
                Radius in metres for use in specifying the number of
                grid cells used to create a square neighbourhood.
            mask_cube (iris.cube.Cube):
                Cube containing the array to be used as a mask.

        Returns:
            iris.cube.Cube:
                Cube containing the smoothed field after the square
                neighbourhood method has been applied.
        """
        # If the data is masked, the mask will be processed as well as the
        # original_data * mask array.
        check_radius_against_distance(cube, radius)
        original_attributes = cube.attributes
        original_methods = cube.cell_methods
        grid_cells = distance_to_number_of_grid_cells(cube, radius)

        result_slices = iris.cube.CubeList()
        for cube_slice in cube.slices([cube.coord(axis='y'),
                                       cube.coord(axis='x')]):
            (cube_slice, mask, nan_array) = (
                self.set_up_cubes_to_be_neighbourhooded(cube_slice, mask_cube))
            neighbourhood_averaged_cube = (
                self._pad_and_calculate_neighbourhood(
                    cube_slice, mask, grid_cells))
            neighbourhood_averaged_cube = (
                self._remove_padding_and_mask(
                    neighbourhood_averaged_cube,
                    cube_slice, mask, grid_cells))
            neighbourhood_averaged_cube.data[nan_array.astype(bool)] = np.nan
            result_slices.append(neighbourhood_averaged_cube)

        neighbourhood_averaged_cube = result_slices.merge_cube()

        neighbourhood_averaged_cube.cell_methods = original_methods
        neighbourhood_averaged_cube.attributes = original_attributes

        neighbourhood_averaged_cube = check_cube_coordinates(
            cube, neighbourhood_averaged_cube)
        return neighbourhood_averaged_cube
