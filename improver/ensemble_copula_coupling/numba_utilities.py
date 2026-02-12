# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
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
def fast_interp_same_y_2d(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    """For each row i of xp, do the equivalent of np.interp(x[i], xp[i], fp).
    This function is distinct from fast_interp_same_y for compatibility with numba.
    The function is essentially the same as fast_interp_same_y but with an
    additional loop over rows to handle a 2-D x array.

    Args:
        x: 2-D array with one row per xp row (shape: n * k)
        xp: n * m array, each row must be in non-decreasing order
        fp: 1-D array with length m
    Returns:
        n * k array where each row i is equal to np.interp(x[i], xp[i], fp)
    """
    # checks
    if len(x.shape) != 2:
        raise ValueError("x must be 2-dimensional for fast_interp_same_y_2d.")
    if len(fp.shape) != 1:
        raise ValueError("fp must be 1-dimensional.")
    if xp.shape[1] != len(fp):
        raise ValueError("Dimension 1 of xp must be equal to length of fp.")
    if x.shape[0] != xp.shape[0]:
        raise ValueError("Rows of x must match rows of xp for 2-D x.")

    n = xp.shape[0]
    m = xp.shape[1]
    k = x.shape[1]
    max_ind = m
    min_val = fp[0]
    max_val = fp[-1]
    result = np.empty((n, k), dtype=np.float32)

    # Loop over each row (sample) in x and xp
    for i in prange(n):
        # Check whether row x[i] is non-decreasing
        x_ordered = True
        # Loop over each element in the row to check ordering
        for t in range(1, k):
            if x[i, t] < x[i, t - 1]:
                x_ordered = False
                break

        ind = 0
        intercept = 0.0
        slope = 0.0
        x_lower = 0.0

        # Loop over each value in the current row of x
        for j in range(k):
            recalculate = False
            curr_x = x[i, j]

            # Find smallest index ind of xp[i] for which xp[i, ind] >= curr_x.
            if x_ordered:
                # Loop forward through xp[i] until the correct interval is found
                while (ind < max_ind) and (xp[i, ind] < curr_x):
                    ind += 1
                    recalculate = True
            else:
                # Use searchsorted to find the interval for unordered x[i]
                ind = np.searchsorted(xp[i], curr_x)

            # linear interpolation
            if ind == 0:
                result[i, j] = min_val
            elif ind == max_ind:
                result[i, j] = max_val
            else:
                if recalculate or not x_ordered:
                    intercept = fp[ind - 1]
                    x_lower = xp[i, ind - 1]
                    h_diff = xp[i, ind] - x_lower
                    if h_diff < 1e-15:
                        # avoid division by very small values for numerical stability
                        slope = 0.0
                    else:
                        slope = (fp[ind] - intercept) / h_diff
                result[i, j] = intercept + (curr_x - x_lower) * slope

    return result


def fast_interp_same_y_nd(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    """Dispatch to 1D or 2D numba kernels.

    Args:
        x: 1-D or 2-D array
        xp: n * m array, each row must be in non-decreasing order
        fp: 1-D array with length m
    """
    if x.ndim == 1:
        return fast_interp_same_y(x, xp, fp)
    if x.ndim == 2:
        return fast_interp_same_y_2d(x, xp, fp)
    raise ValueError("x must be 1D or 2D.")
