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
"""Module to contain indexing operation functions."""

import numpy as np
from scipy.interpolate import griddata
from scipy.spatial.qhull import QhullError


def interpolate_missing_data(
        data, method='linear', limit=None, valid_points=None):
    """
    Args:
        data (numpy.ndarray):
            The field of data to be interpolated across gaps.
        method (str):
            The method to use to fill in the data. This is usually "linear" for
            linear interpolation, and "nearest" for a nearest neighbour
            approach. It can take any method available to the method
            scipy.interpolate.griddata.
        limit(numpy.data):
            The array containing upper limits for each grid point that are
            imposed on any value in the region that has been interpolated.
        valid_points (numpy.ndarray):
            A boolean array that marks valid data that can be used as input
            to the interpolation process. This allows the exclusion of some
            points even though they contain data that could otherwise be used
            in the interpolation.
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
        data_filled_above_limit = (data_filled[index] > limit[index])
        index[index] = data_filled_above_limit
        data_filled[index] = limit[index]

    index = ~np.isfinite(data)
    data[index] = data_filled[index]

    return data
