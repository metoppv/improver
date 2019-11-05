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


def choose(index_array, array_set):
    """
    Create a reordered copy of a data array, where an index array of matching
    shape determines how the data is reordered.

    The reordered copy of the array will have the same shape as the inputs.
    If a 3-dimensional array_set is provided, the data coordinates could be
    described as A[i, j, k]. This function does not rearrange the j and k
    coordinates. The reordered array is constructed by substituting into each
    position [i, j, k] in the index_array a value taken from the array_set
    at a matching [j, k] position, but where [i] is determined by the given
    index value. As such the index values must run 0 - N-1 where N is the
    length of the leading dimension of array_set (and equivalently of
    index_array).

    The following figure gives an examples of the expected result of this
    function for a given index_array and array_set.

    .. figure:: extended_documentation/utilities/indexing_operations/
       numpy_choose_test1.png
       :align: center
       :scale: 60 %
       :alt: Diagram to help explain indexing behaviour of choose.

       The input data array is reordered along the leading dimension (i),
       but maintains its sub-array position (j, k), denoted here by the
       different colours.

    Args:
        index_array (numpy.ndarray of int):
            This array must contain integers in the range [0, N-1], where N is
            if the length of the leading dimension of the array_set array.
            These integers determine how array_set will be reordered in the
            returned array.
        array_set (numpy.ndarray):
            A multi-dimensional array, where the leading dimension is in effect
            an indexing dimension. Within this leading dimension are the
            sub-arrays from which values are to be extracted at positions that
            match those given in the index_array.
    Returns:
        numpy.ndarray:
            An array containing the reordered data extracted from array_set.
            The returned array will have the same shape as the index_array and
            array_set arrays.
    Raises:
        ValueError: If index_array and array_set do not have matching shapes.
        IndexError: If an index exceeds the length of the leading dimension
                    of the array_set array (N-1).
    """
    if index_array.shape != array_set.shape:
        msg = ("The choose function only works with an index_array that "
               "matches the shape of array_set.\nindex_array shape: {}\n"
               "array_set shape: {}".format(index_array.shape,
                                            array_set.shape))
        raise ValueError(msg)
    if index_array.max() > array_set.shape[0] - 1:
        msg = ('index_array contains an index that is larger than the '
               'number of sub-arrays from which data can be drawn.\nMax '
               'index: {}\nNumber of sub-arrays: {}'.format(
                   index_array.max(), array_set.shape[0]))
        raise IndexError(msg)

    result = np.array(
        [array_set[index_array[I]][I[1:]]
         for I in np.lib.index_tricks.ndindex(index_array.shape)]
        ).reshape(index_array.shape)

    return result
