import os

import numpy as np
from numba import config, njit, prange, set_num_threads

config.THREADING_LAYER = "omp"
if "OMP_NUM_THREADS" in os.environ:
    set_num_threads(int(os.environ["OMP_NUM_THREADS"]))


@njit(parallel=True)
def fast_interp(x: np.ndarray, xp: np.ndarray, fp: np.ndarray, result: np.ndarray):
    """For each row i of xp, do the equivalent of np.interp(x, xp[i], fp).
    
    Args:
        x: 1-d array
        xp: n * m array, each row must be in non-decreasing order
        fp: 1-d array with length m
        result: n * len(x) array for output
    """
    # check whether x is non-decreasing
    x_ordered = True
    for i in range(1, len(x)):
        if x[i] < x[i - 1]:
            x_ordered = False
            break
    max_ind = xp.shape[1]
    min_val = fp[0]
    max_val = fp[max_ind - 1]
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
                # binary search
                left = 0
                right = max_ind
                while right - left > 1:
                    mid = (left + right) // 2
                    if xp[i, mid] < curr_x:
                        left = mid
                    else:
                        right = mid
                if xp[i, left] >= curr_x:
                    ind = left
                else:
                    ind = right
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
