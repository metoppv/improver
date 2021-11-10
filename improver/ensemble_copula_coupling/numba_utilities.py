# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2021 Met Office.
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
        raise ValueError("Dimension 1 of fp must be equal to lenght of xp.")
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
