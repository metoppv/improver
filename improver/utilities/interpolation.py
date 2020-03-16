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
        if limit_as_maximum:
            data_violating_limit = (data_filled[index] > limit[index])
        else:
            data_violating_limit = (data_filled[index] < limit[index])
        index[index] = data_violating_limit
        data_filled[index] = limit[index]

    index = ~np.isfinite(data)
    data[index] = data_filled[index]

    return data


class InterpolateUsingDifference:
    """
    Calculates the difference between the field that is to be interpolated and
    a complete (filling the whole domain) reference field. The difference
    between the fields in regions where they overlap is calculated and this
    difference is then interpolated across the domain. Any holes in the data
    being interpolated are then filled with data calculated as the reference
    field minus the interpolated difference field.
    """
    # def __init__(self):
    #     """Initialise plugin."""
    #
    # def __repr__(self):
    #     """String representation of plugin."""

    def process(self, field, reference_field,
                limit=None, limit_as_maximum=True):
        """
        Apply plugin to input data.

        Args:
            field (iris.cube.Cube):
                Field for which interpolation is required to fill holes.
            reference_field (iris.cube.Cube):
                A field that covers the entire domain that it shares with
                field.
            limit (iris.cube.Cube or None):
                A field used to calculate limiting values that the difference
                field should not violate following interpolation. This can be
                used to ensure that the interpolated field does not get too
                close to or too far away from the reference field. Any points
                in the interpolated difference field violating the limit are
                set back to the calculated limiting value, ``reference_field -
                limit``.
            limit_as_maximum (bool):
                If True the test against the values allowed by the limit array
                is that if the interpolated values exceed the limit they should
                be set to the limit value. If False, the test is whether the
                interpolated values fall below the limit value.
        Raises:
            ValueError: If the reference field is not complete across the
                        entire domain.
        """
        if np.isnan(reference_field.data).any():
            raise ValueError(
                'The reference field contains np.nan data indicating that it '
                'is not complete across the domain.')

        valid_points = ~field.data.mask
        difference_field = np.subtract(reference_field.data, field.data,
                                       out=np.full(field.shape, np.nan),
                                       where=valid_points)
        if limit is not None:
            limit = reference_field.data - limit.data

        interpolated_difference = interpolate_missing_data(
                difference_field, limit=limit,
                limit_as_maximum=limit_as_maximum,
                valid_points=valid_points)

        return field.copy(
             data=reference_field.data - interpolated_difference)
