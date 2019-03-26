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
    Construct an array from an index array and a set of arrays to choose from.

    The index values specify which array to take the value from. For example,
    if the function is provided with a list of 10 arrays, the index values
    should range from 0-9. Typically this function will be provided with a
    multi-dimensional array rather than a list of arrays, in which case the
    first dimension will be treated as defining the sequence.

    The shape of the index array, and where in that shape the array indices
    are positioned will determine which data is extracted from the arrays.
    The following figure gives some examples to help make this clear.

    .. figure:: extended_documentation/utilities/indexing_operations/
       numpy_choose_example.png
       :scale: 60 %
       :alt: Diagram to help explain indexing behaviour of choose.

       A graphical guide to how the index array is used to extract data from
       the data arrays. The Array column gives the index of the array in the
       list of input arrays (array_set). The Data column shows the contents of
       the 3 arrays passed to the function. The Indices examples are used to
       show which data would be returned (Result) should the index_array be
       shaped in these ways, with these array index values. The colour coding
       is used to help make clear how the shape of the index_array relates to
       the arrays within the array_set.

    Args:
        index_array (np.array of ints):
            This array must contain integers in the range [0, n-1], where n is
            the number of arrays in the array_set (equivalent to the length of
            the leading dimension of array_set if it is a multi-dimensional
            array rather than a list).
        array_set (np.array):
            A multi-dimensional array, where the leading dimension is in effect
            an indexing dimension. Within this leading dimension are the
            sub-arrays from which values are to be extracted at positions that
            match those given in the index_array.
    Returns:
        result (np.array):
            An array containing data extracted from the array_set.
    """
    # Used to correct array shape in output following intermediate broadcast.
    index_array = np.array(index_array)
    index_dims = index_array.ndim
    trim_result = index_array.shape != array_set.shape

    # Broadcast arrays to matching shape. This is included to replicate as much
    # of the numpy choose functionality as possible.
    broadcast_arrays = np.broadcast_arrays(index_array, array_set)
    index_array = broadcast_arrays[0]
    array_set = broadcast_arrays[1]

    # Use indexing to extract the requires values from the sub-arrays.
    result = np.array(
        [array_set[index_array[I]][I[1:]]
         for I in np.lib.index_tricks.ndindex(index_array.shape)]
    ).reshape(index_array.shape)

    # If the index_array shape did not match the array_set shape, it is assumed
    # that the indexing is for the sub-arrays, not the complete array_set. The
    # leading dimension is thus cut away in the result.
    if trim_result:
        result = result[0]

    # Add back any lost empty dimensions to ensure the returned array has the
    # expected dimensionality.
    dim_diff = index_dims - result.ndim
    for _ in range(dim_diff):
        result = np.expand_dims(result, 0)

    return result
