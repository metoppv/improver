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
This module defines the optional numba utilities for calibration plugins.
"""

import os

import numpy as np
from numba import config, njit, prange, set_num_threads

# config.THREADING_LAYER = "omp"
# if "OMP_NUM_THREADS" in os.environ:
#     set_num_threads(int(os.environ["OMP_NUM_THREADS"]))

@njit
def forward_fill_nans(x: np.ndarray):
    """Replace nans in place with last non-nan value.
    
    Args:
        x: 1-D array
    """

    last_value = np.nan
    for i in range(len(x)):
        if np.isnan(x[i]):
            x[i] = last_value
        else:
            last_value = x[i]




@njit
def forward_fill(x: np.ndarray, filled_indices: np.ndarray=None):
    """Forward-fill x in place, using the value in the last index of filled_indices.
    
    Args:
        x: 1-D array
        filled_indices: 1-D array of ints, specifying indices of values already filled
    """

    for i in range(len(filled_indices)):
        curr_ind = filled_indices[i]
        next_ind = filled_indices[i + 1] if i < len(filled_indices) - 1 else len(x)
        fill_val = x[curr_ind]
        for j in range(curr_ind + 1, next_ind):
            x[j] = fill_val
