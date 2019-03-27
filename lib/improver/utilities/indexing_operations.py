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
    should range from 0-9. This function should be provided with a
    multi-dimensional array and is not designed to accept a list of arrays,
    which is the preferred input to numpy choose. The first dimension of the
    multi-dimensional array will be treated as defining the sequence of
    sub-arrays.

    The shape of the index array, and where in that shape the array indices
    are positioned will determine which data is extracted from the arrays. The
    two arrays will be broadcast to a common shape if they do not match and
    such a broadcasting operation is possible. The returned array will have the
    same shape as the sub-arrays in most instances. If the index_array has the
    same number of dimensions as the array_set, the returned array will have an
    equivalent number of dimensions (i.e. one more than the sub-arrays).

    The following figure gives some examples of the expected result of this
    function when a series of index_arrays of differing shape and
    dimensionality are used. The array_set is the same for each example,
    shown down the left hand side; this array has the shape (3, 2, 2).

    .. figure:: extended_documentation/utilities/indexing_operations/
       numpy_choose_example.png
       :scale: 60 %
       :alt: Diagram to help explain indexing behaviour of choose.

       The Array column gives the index in the outer most dimension of the
       multi-dimensional input array (array_set). The Data column shows the
       contents of the 3 sub-arrays that correspond to the given Array index.
       The Indices examples are used to show which data would be returned
       (Result) should the index_array be shaped in these various ways, with
       these array index values. The colour coding is used to help make clear
       how the shape of the index_array relates to the arrays within the
       array_set.

    The examples above demonstrate the application of choose with a series of
    index_arrays of different dimensionality. In all but the last of these
    cases the choose function will use broadcasting to construct an index array
    that matches the shape of the sub-arrays from which data is being taken.
    The figure below illustrates the results of this intermediate broadcasting
    step.

    .. figure:: extended_documentation/utilities/indexing_operations/
       numpy_choose_broadcasting.png
       :scale: 60 %
       :alt: Diagram to help explain broadcasting behaviour of choose.

       The form of the index_arrays after they have been broadcast to match the
       sub-arrays from which values are being taken.

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
    Raises:
        ValueError: If input arrays cannot be broadcast to a common shape.
        IndexError: If an index exceeds the number of available sub-arrays.
    """
    # Used to correct array shape in output following intermediate broadcast.
    index_array = np.array(index_array)
    index_dims = index_array.ndim
    trim_result = index_array.shape != array_set.shape

    # Broadcast arrays to matching shape. This is included to replicate as much
    # of the numpy choose functionality as possible.
    try:
        broadcast_arrays = np.broadcast_arrays(index_array, array_set)
    except ValueError as err:
        msg = '{}\nindex_array shape: {}\narray_set shape: {}'.format(
            err, index_array.shape, array_set.shape)
        raise ValueError(msg)

    # numpy choose can handle the condition below, but we currently have no
    # need to accommodate this behaviour, so we trap it and raise an error.
    if broadcast_arrays[1].ndim > array_set.ndim :
        msg = ('Dimensionality of array_set has increased which will prevent '
               'indexing from working as expected.')
        raise IndexError(msg)

    index_array = broadcast_arrays[0]
    array_set = broadcast_arrays[1]

    # Use indexing to extract the requires values from the sub-arrays.
    try:
        result = np.array(
            [array_set[index_array[I]][I[1:]]
             for I in np.lib.index_tricks.ndindex(index_array.shape)]
        ).reshape(index_array.shape)
    except IndexError as err:
        msg = ('{}\nindex_array contains an index that is larger than the '
               'number of sub-arrays from which data can be drawn.\nMax '
               'index: {}\nNumber of sub-arrays: {}'.format(
                   err, index_array.max(), array_set.shape[0]))
        raise IndexError(msg)

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
