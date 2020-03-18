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
"""Module to contain interpolation functions."""

import warnings
import numpy as np
from scipy.interpolate import griddata
from scipy.spatial.qhull import QhullError


def interpolate_missing_data(
        data, method='linear', limit=None, limit_as_maximum=True,
        valid_points=None):
    """
    Args:
        data (numpy.ndarray):
            The field of data to be interpolated across gaps.
        method (str):
            The method to use to fill in the data. This is usually "linear" for
            linear interpolation, and "nearest" for a nearest neighbour
            approach. It can take any method available to the method
            scipy.interpolate.griddata.
        limit (numpy.ndarray):
            The array containing limits for each grid point that are
            imposed on any value in the region that has been interpolated.
        limit_as_maximum (bool):
            If True the test against the limit array is that if the
            interpolated values exceed the limit they should be set to the
            limit value. If False, the test is whether the interpolated values
            fall below the limit value.
        valid_points (numpy.ndarray):
            A boolean array that allows a subset of the unmasked data to be
            chosen as source data for the interpolation process. True values
            in this array mark points that can be used for interpolation if
            they are not otherwise invalid. False values mark points that
            should not be used, even if they are otherwise valid data points.

    Returns:
        numpy.ndarray:
            The original data plus interpolated data in holes where it was
            possible to fill these in.
    """
    if valid_points is None:
        valid_points = np.full_like(data, True, dtype=np.bool)

    # Interpolate linearly across the remaining points
    index = ~np.isnan(data)
    index_valid_data = valid_points[index]
    index[index] = index_valid_data
    data_filled = data

    if np.any(index):
        ynum, xnum = data.shape
        (y_points, x_points) = np.mgrid[0:ynum, 0:xnum]
        values = data[index]
        try:
            data_updated = griddata(
                np.where(index), values, (y_points, x_points), method=method)
        except QhullError:
            data_filled = data
        else:
            data_filled = data_updated

    if limit is not None:
        index = ~np.isfinite(data) & np.isfinite(data_filled)
        limit_data_values(data_filled, index, limit, limit_as_maximum)

    index = ~np.isfinite(data)
    data[index] = data_filled[index]

    return data


def limit_data_values(data, index, limit, limit_as_maximum):
    """
    Impose a limit on data.
    """
    index = index.copy()
    if limit_as_maximum:
        data_violating_limit = (data[index] > limit[index])
    else:
        data_violating_limit = (data[index] < limit[index])
    index[index] = data_violating_limit
    data[index] = limit[index]


class InterpolateUsingDifference:
    """
    Uses interpolation to fill holes in the data contained within the input
    cube. This is achieved by calculating the difference between the input cube
    and a complete (filling the whole domain) reference cube. The difference
    between the data in regions where they overlap is calculated and this
    difference field is then interpolated across the domain. Any holes in the
    input cube data are then filled with data calculated as the reference
    cube data minus the interpolated difference field.
    """

    def __repr__(self):
        """String representation of plugin."""
        return "<InterpolateUsingDifference>"

    @staticmethod
    def _check_inputs(cube, reference_cube, limit):
        """
        Check that the input cubes are compatible and the data is complete or
        masked as expected.
        """
        if np.isnan(reference_cube.data).any():
            raise ValueError(
                'The reference cube contains np.nan data indicating that it '
                'is not complete across the domain.')
        try:
            reference_cube.convert_units(cube.units)
            if limit is not None:
                limit.convert_units(cube.units)
        except ValueError:
            raise ValueError(
                'Reference cube and/or limit do not have units compatible with'
                ' cube.')

    def process(self, cube, reference_cube,
                limit=None, limit_as_maximum=True):
        """
        Apply plugin to input data.

        Args:
            cube (iris.cube.Cube):
                cube for which interpolation is required to fill holes.
            reference_cube (iris.cube.Cube):
                A cube that covers the entire domain that it shares with
                cube.
            limit (iris.cube.Cube or None):
                A cube used to calculate limiting values that the difference
                cube should not violate following interpolation. This can be
                used to ensure that the interpolated field does not get too
                close to or too far away from the reference cube. Any points
                in the interpolated difference field violating the limit are
                set back to the calculated limiting value, ``reference_cube -
                limit``.
            limit_as_maximum (bool):
                If True the test against the values allowed by the limit array
                is that if the interpolated values exceed the limit they should
                be set to the limit value. If False, the test is whether the
                interpolated values fall below the limit value.
        Return:
            iris.cube.Cube:
                A copy of the input cube in which the missing data has been
                populated with values obtained through interpolating the
                difference field and subtracting the result from the reference
                cube.
        Raises:
            ValueError: If the reference cube is not complete across the
                        entire domain.
        """
        if not np.ma.is_masked(cube.data):
            warnings.warn('Input cube unmasked, no data to fill in, returning '
                          'unchanged.')
            return cube

        self._check_inputs(cube, reference_cube, limit)

        invalid_points = cube.data.mask.copy()
        valid_points = ~invalid_points

        difference_field = np.subtract(reference_cube.data, cube.data,
                                       out=np.full(cube.shape, np.nan),
                                       where=valid_points)
        interpolated_difference = interpolate_missing_data(
                difference_field, valid_points=valid_points)

        # If any invalid points remain in the difference field, use nearest
        # neighbour interpolation to fill these with the nearest difference.
        remain_invalid = np.isnan(interpolated_difference)
        if remain_invalid.any():
            interpolated_difference = interpolate_missing_data(
                    difference_field, valid_points=~remain_invalid,
                    method='nearest')

        result = cube.copy()
        result.data[invalid_points] = (
            reference_cube.data[invalid_points] -
            interpolated_difference[invalid_points])

        if limit is not None:
            limit_data_values(result.data, invalid_points, limit.data,
                              limit_as_maximum)

        return result
