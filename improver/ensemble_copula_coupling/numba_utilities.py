# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown copyright. The Met Office.
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
"""
This module defines the optional numba utilities for Ensemble Copula Coupling
plugins.
"""

import os

import numpy as np
from numba import config, njit, prange, set_num_threads

config.THREADING_LAYER = "omp"
if "OMP_NUM_THREADS" in os.environ:
    set_num_threads(int(os.environ["OMP_NUM_THREADS"]))


@njit(parallel=True)
def fast_interp_same_x(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    """For each row i of fp, do the equivalent of np.interp(x, xp, fp[i, :]).
    Args:
        x: 1-D array
        xp: 1-D array, sorted in non-decreasing order
        fp: 2-D array with len(xp) columns
    Returns:
        2-D array with shape (len(fp), len(x)), with each row i equal to
            np.interp(x, xp, fp[i, :])
    """
    # check inputs
    if len(x.shape) != 1:
        raise ValueError("x must be 1-dimensional.")
    if len(xp.shape) != 1:
        raise ValueError("xp must be 1-dimensional.")
    if fp.shape[1] != len(xp):
        raise ValueError("Dimension 1 of fp must be equal to length of xp.")
    index = np.searchsorted(xp, x)
    result = np.empty((fp.shape[0], len(x)), dtype=np.float32)
    for row in prange(fp.shape[0]):
        for i, ind in enumerate(index):
            if ind == 0:
                result[row, i] = fp[row, 0]
            elif ind == len(xp):
                result[row, i] = fp[row, -1]
            elif xp[ind] - xp[ind - 1] >= 1e-15:
                result[row, i] = fp[row, ind - 1] + (x[i] - xp[ind - 1]) / (
                    xp[ind] - xp[ind - 1]
                ) * (fp[row, ind] - fp[row, ind - 1])
            else:
                result[row, i] = fp[row, ind - 1]
    return result


@njit(parallel=True)
def fast_interp_same_y(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    """For each row i of xp, do the equivalent of np.interp(x, xp[i], fp).
    Args:
        x: 1-d array
        xp: n * m array, each row must be in non-decreasing order
        fp: 1-d array with length m
    Returns:
        n * len(x) array where each row i is equal to np.interp(x, xp[i], fp)
    """
    # check inputs
    if len(x.shape) != 1:
        raise ValueError("x must be 1-dimensional.")
    if len(fp.shape) != 1:
        raise ValueError("fp must be 1-dimensional.")
    if xp.shape[1] != len(fp):
        raise ValueError("Dimension 1 of xp must be equal to length of fp.")
    # check whether x is non-decreasing
    x_ordered = True
    for i in range(1, len(x)):
        if x[i] < x[i - 1]:
            x_ordered = False
            break
    max_ind = xp.shape[1]
    min_val = fp[0]
    max_val = fp[-1]
    result = np.empty((xp.shape[0], len(x)), dtype=np.float32)
    for i in prange(xp.shape[0]):
        ind = 0
        intercept = 0
        slope = 0
        x_lower = 0
        for j in range(len(x)):
            recalculate = False
            curr_x = x[j]
            # Find the indices of xp[i] to interpolate between. We need the
            # smallest index ind of xp[i] for which xp[i, ind] >= curr_x.
            if x_ordered:
                # Since x and x[i] are non-decreasing, ind for current j must be
                # greater than equal to ind for previous j.
                while (ind < max_ind) and (xp[i, ind] < curr_x):
                    ind = ind + 1
                    recalculate = True
            else:
                ind = np.searchsorted(xp[i], curr_x)
            # linear interpolation
            if ind == 0:
                result[i, j] = min_val
            elif ind == max_ind:
                result[i, j] = max_val
            else:
                if recalculate or not (x_ordered):
                    intercept = fp[ind - 1]
                    x_lower = xp[i, ind - 1]
                    h_diff = xp[i, ind] - x_lower
                    if h_diff < 1e-15:
                        # avoid division by very small values for numerical stability
                        slope = 0
                    else:
                        slope = (fp[ind] - intercept) / h_diff
                result[i, j] = intercept + (curr_x - x_lower) * slope
    return result


@njit(parallel=True)
def fast_interpolate_pointwise(x, xp, fp):
    """
    Given 2 arrays xp and fp with the same dimensions, interpolate along first dimensions
    at values given by x.

    Args:
        x: 1-d array, points at which to evaluate interpolation function
        xp: array with at least 2 dimensions. First dimension is the interpolation dimension.
        fp: array with at least 2 dimensions. First dimension is the interpolation dimension.

    Returns:
        array with dimensions (len(x), ) + xp.shape[1:]
    """

    if xp.shape != fp.shape:
        raise ValueError("xp and fp must have the same shape")
    if len(xp.shape) < 2:
        raise ValueError("xp and fp must have at least 2 dimensions")

    new_shape = (xp.shape[0], xp.size // xp.shape[0])
    result = np.empty((len(x), new_shape[1]), dtype=np.float32)
    xp_reshaped = np.reshape(xp, new_shape)
    fp_reshaped = np.reshape(fp, new_shape)
    for i in prange(xp_reshaped.shape[1]):
        result[:, i] = np.interp(x, xp_reshaped[:, i], fp_reshaped[:, i])
    reshaped_result = np.reshape(result, (len(x),) + tuple(xp.shape[1:]))
    return reshaped_result
