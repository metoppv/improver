# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017 Met Office.
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
Plugins written for the Improver site specific process chain.

"""

import numpy as np
from iris import Constraint
from iris.time import PartialDateTime


class ConditionalListExtract(object):
    '''
    Performs a numerical comparison, the type selected with method, of data
    in an array and returns an array of indices in that data array that
    fulfill the comparison.

    Args:
    -----
    method          : which comparison to make, e.g. not_equal_to.
    data            : array of values to be filtered.
    indices_list    : list of indices in the data array that should be
                      considered.
    comparison_value: the value against which numbers in data are to be
                      compared.

    Returns:
    --------
    array_of_indices.tolist():
                  a list of the the indices of data values that fulfill the
                  comparison condition.
    '''

    def __init__(self, method):
        self.method = method

    def process(self, data, indices_list, comparison_value):
        ''' Call the data comparison method passed in'''
        array_of_indices = np.array(indices_list)
        function = getattr(self, self.method)
        subset = function(data, array_of_indices, comparison_value)

        return array_of_indices[0:2, subset[0]].tolist()

    @staticmethod
    def less_than(data, array_of_indices, comparison_value):
        ''' Return indices of array for which value < comparison_value '''
        return np.where(
            data[[array_of_indices[0],
                  array_of_indices[1]]] < comparison_value
            )

    @staticmethod
    def greater_than(data, array_of_indices, comparison_value):
        ''' Return indices of array for which value > comparison_value '''
        return np.where(
            data[[array_of_indices[0],
                  array_of_indices[1]]] > comparison_value
            )

    @staticmethod
    def equal_to(data, array_of_indices, comparison_value):
        ''' Return indices of array for which value == comparison_value '''
        return np.where(
            data[[array_of_indices[0],
                  array_of_indices[1]]] == comparison_value
            )

    @staticmethod
    def not_equal_to(data, array_of_indices, comparison_value):
        ''' Return indices of array for which value != comparison_value '''
        return np.where(
            data[[array_of_indices[0],
                  array_of_indices[1]]] != comparison_value
            )


# Common shared functions

def nearest_n_neighbours(i, j, no_neighbours, exclude_self=False):
    '''
    Returns a coordinate list of n points comprising the original
    coordinate (i,j) plus the n-1 neighbouring points on a cartesian grid.
    e.g. n = 9

    (i-1, j-1) | (i-1, j) | (i-1, j+1)
    ----------------------------------
      (i, j-1) |  (i, j)  | (i, j+1)
    ----------------------------------
    (i+1, j-1) | (i+1, j) | (i+1, j+1)

    n must be in the sequence (2*d(ij) + 1)**2 where d(ij) is the +- in the
    index (1,2,3, etc.); equivalently sqrt(n) is an odd integer and n >= 9.

    exclude_self = True will return the list without the i,j point about which
    the list was constructed.

    Args:
    -----
    i, j            : central coordinate about which to find neighbours.
    no_neighbours   : no. of neighbours to return (9, 25, 49, etc).
    exclude_self    : boolean, if True, (i,j) excluded from returned list.

    Returns:
    --------
    list of array indices that neighbour the central (i,j) point.

    '''
    # Check n is a valid no. for neighbour finding.
    root_no_neighbours = np.sqrt(no_neighbours)
    delta_neighbours = (root_no_neighbours - 1)/2
    if not np.mod(delta_neighbours, 1) == 0 or delta_neighbours < 1:
        raise ValueError(
            'Invalid neareat no. of neighbours request. N={} is not a valid '
            'square no. (sqrt N must be odd)'.format(no_neighbours))

    delta_neighbours = int(delta_neighbours)
    n_indices = [(i+a, j+b)
                 for a in range(-delta_neighbours, delta_neighbours+1)
                 for b in range(-delta_neighbours, delta_neighbours+1)]
    if exclude_self is True:
        n_indices.pop(no_neighbours/2)
    return np.array(
        [np.array(n_indices)[:, 0], np.array(n_indices)[:, 1]]
        ).astype(int).tolist()


def node_edge_test(node_list, cube):
    '''
    Node lists produced using the nearest_n_neighbours function may overspill
    the domain of the array from which data is to be extracted. This function
    checks whether the cube of interest is a global domain with a wrapped
    boundary using the iris.cube.Cube.coord().circular property. In cases of
    wrapped boundaries, the neighbouring points addresses are appropriately
    modified. Otherwise the points are discarded.

    Args
    ----
    node_list : list[[i],[j]] of indices.
    cube      : the cube for which data will be extracted using the
                indices (e.g. cube.data[node_list]).

    Returns
    -------
    node_list : modified node_list with points beyond the cube boundary
                either changed or discarded as appropriate.

    '''

    node_list = np.array(node_list)

    for k, coord in enumerate(['y', 'x']):
        coord_max = cube.coord(axis=coord).shape[0]
        circular = cube.coord(axis=coord).circular
        max_list = np.where(node_list[k] >= coord_max)[0].tolist()
        min_list = np.where(node_list[k] < 0)[0].tolist()
        if circular:
            node_list[k, min_list] = node_list[k, min_list] + coord_max
            node_list[k, max_list] = node_list[k, max_list] - coord_max
        else:
            node_list = np.delete(node_list,
                                  np.hstack((min_list, max_list)), 1)

    return node_list.tolist()


def get_nearest_coords(cube, latitude, longitude):
    '''
    Uses the iris cube method nearest_neighbour_index to find the nearest grid
    points to a given latitude-longitude position.
    '''

    i_latitude = (cube.coord(axis='y').nearest_neighbour_index(latitude))
    j_longitude = (cube.coord(axis='x').nearest_neighbour_index(longitude))
    return i_latitude, j_longitude


def index_of_minimum_difference(whole_list, subset_list=None):
    '''
    Returns the index of the minimum value in a list.
    '''
    if subset_list is None:
        subset_list = np.arange(len(whole_list))
    return subset_list[np.argmin(abs(whole_list[subset_list]))]


def list_entry_from_index(list_in, index_in):
    '''
    Extracts index_in element from each list in a list of lists, and returns
    as a list.
    e.g.
         list_in = [[0,1,2],[5,6,7],[8,9,10]]
         index_in = 1
         Returns [1,6,9]

    '''
    n_columns = len(list_in)
    return [list_in[n_col][index_in] for n_col in range(n_columns)]


def datetime_constraint(time_in):
    '''
    Constructs an iris equivalence constraint from a python datetime object.

    '''
    return Constraint(
        time=PartialDateTime(*[int(time_in.strftime("%{}".format(x)))
                               for x in
                               ['Y', 'm', 'd', 'H']])
        )
