# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2018 Met Office.
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

import warnings
import numpy as np

import iris

from improver.utilities.temporal import extract_cube_at_time


class ConditionalListExtract(object):
    '''
    Performs a numerical comparison, the type selected with method, of data
    in a 2D array and returns an array of indices in that data array that
    fulfill the comparison.

    '''

    def __init__(self, method):
        """
        Get selected method of comparison.

        Args:
            method (string):
                Which comparison to make, e.g. not_equal_to.

        """
        self.method = method

    def process(self, data, indices_list, comparison_value):
        """
        Call the data comparison method passed in.

        Args:
            data (numpy.array):
                Array of values to be filtered.

            indices_list (list):
                Indices in the data array that should be considered.

            comparison_value(float):
                Value against which numbers in data are to be compared.

        Returns:
            array_of_indices (list):
                A list of the the indices of data values that fulfill the
                comparison condition.

        """

        array_of_indices = np.array(indices_list)
        try:
            function = getattr(self, self.method)
        except AttributeError:
            raise AttributeError('Unknown method "{}" passed to {}.'.format(
                self.method, self.__class__.__name__))

        subset = function(data, array_of_indices, comparison_value)

        return array_of_indices[0:2, subset[0]].tolist()

    @staticmethod
    def less_than(data, array_of_indices, comparison_value):
        """ Return indices of array for which value < comparison_value """
        return np.where(
            data[[array_of_indices[0],
                  array_of_indices[1]]] < comparison_value
            )

    @staticmethod
    def greater_than(data, array_of_indices, comparison_value):
        """ Return indices of array for which value > comparison_value """
        return np.where(
            data[[array_of_indices[0],
                  array_of_indices[1]]] > comparison_value
            )

    @staticmethod
    def equal_to(data, array_of_indices, comparison_value):
        """ Return indices of array for which value == comparison_value """
        return np.where(
            data[[array_of_indices[0],
                  array_of_indices[1]]] == comparison_value
            )

    @staticmethod
    def not_equal_to(data, array_of_indices, comparison_value):
        """ Return indices of array for which value != comparison_value """
        return np.where(
            data[[array_of_indices[0],
                  array_of_indices[1]]] != comparison_value
            )


def nearest_n_neighbours(i, j, no_neighbours, exclude_self=False):
    """
    Returns a coordinate list of n points comprising the original
    coordinate (i,j) plus the n-1 neighbouring points on a cartesian grid.
    e.g.::

      n = 9

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
        i, j (ints):
            Central coordinate about which to find neighbours.

        no_neighbours (int):
            No. of neighbours to return (9, 25, 49, etc).

        exclude_self (boolean):
            If True, the central coordinate (i,j) is excluded from returned
            list.

    Returns:
        numpy.array:
            Array of neighbouring indices:

    """
    # Check n is a valid no. for neighbour finding.
    root_no_neighbours = np.sqrt(no_neighbours)
    delta_neighbours = (root_no_neighbours - 1)/2
    if not np.mod(delta_neighbours, 1) == 0 or delta_neighbours < 1:
        raise ValueError(
            'Invalid nearest no. of neighbours request. N={} is not a valid '
            'square no. (sqrt N must be odd)'.format(no_neighbours))

    delta_neighbours = int(delta_neighbours)
    n_indices = [(i+a, j+b)
                 for a in range(-delta_neighbours, delta_neighbours+1)
                 for b in range(-delta_neighbours, delta_neighbours+1)]
    if exclude_self is True:
        n_indices.pop(no_neighbours//2)
    return np.array(
        [np.array(n_indices)[:, 0], np.array(n_indices)[:, 1]]
        ).astype(int).tolist()


def node_edge_check(node_list, cube):
    """
    Node lists produced using the nearest_n_neighbours function may overspill
    the domain of the array from which data is to be extracted. This function
    checks whether the cube of interest is a global domain with a wrapped
    boundary using the iris.cube.Cube.coord().circular property. In cases of
    wrapped boundaries, the neighbouring points addresses are appropriately
    modified. Otherwise the points are discarded.

    Args:
        node_list (list):
            List of indices with a structure [[i],[j]].

        cube (iris.cube.Cube):
            A cube containing the grid from which the i,j coordinates have been
            selected, and which will be used to determine if these points fall
            on the edge of the domain.

    Returns:
        node_list (list):
            Modified node_list with points beyond the cube boundary either
            changed or discarded as appropriate.

    """

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
            indices_for_removal = np.hstack((min_list, max_list)).astype(int)
            node_list = np.delete(node_list, indices_for_removal, 1)

    return node_list.tolist()


def index_of_minimum_difference(whole_list, subset_list=None):
    """
    Returns the index of the minimum value in a list.

    Args:
        whole_list (numpy.array):
            Array to be searched for a minimum value.

        subset_list (numpy.array/None):
            Array of indices to include in the search. If None the entirity of
            whole_list is searched.

    Returns:
        int:
            Index of the minimum value in whole_list.

    """
    whole_list = np.array(whole_list)

    if subset_list is None:
        subset_list = np.arange(len(whole_list))
    return subset_list[np.argmin(abs(whole_list[subset_list]))]


def list_entry_from_index(list_in, index_in):
    """
    Extracts index_in element from each list in a list of lists, and returns
    as a list, e.g.::

         list_in = [[0,1,2],[5,6,7],[8,9,10]]
         index_in = 1
         Returns [1,6,9]

    Args:
        list_in (list):
            The input list.
        index_in (int):
            Chosen index.
    Returns:
        list:
            The extracted value returned as a list.
    """
    return list(list(zip(*list_in))[index_in])


def construct_neighbour_hash(neighbour_finding):
    """
    Constructs a hash from the various neighbour finding options. This is used
    to avoid repeating the same neighbour search more than once.

    Args:
        neighbour_finding (dict):
            A dictionary containing the method, vertical_bias, and
            land_constraint options for neighbour finding.

    Returns:
        string:
            A concatenated string of the options
            e.g. 'fast_nearest_neighbour-None-False'

    """
    return '{}-{}-{}'.format(neighbour_finding['method'],
                             neighbour_finding['vertical_bias'],
                             neighbour_finding['land_constraint'])


def apply_bias(vertical_bias, dzs):
    """
    Bias neighbour selection to look for grid points with an
    altitude that is above or below the site if vertical_bias is
    not None.

    Args:
        vertical_bias (string/None):
            Sets the preferred vertical displacement of the grid point
            relative to the site; above/below/None.

        dzs (numpy.array):
            1D array of vertical displacements calculated as the subtraction of
            grid orography altitudes from spot site altitudes.

    Returns:
        dz_subset (numpy.array):
            Indices of grid points that satisfy bias condition if any are
            available, otherwise it returns the whole set.

    """
    if vertical_bias == 'above':
        dz_subset, = np.where(dzs <= 0)
    elif vertical_bias == 'below':
        dz_subset, = np.where(dzs >= 0)

    if vertical_bias is None or dz_subset.size == 0:
        dz_subset = np.arange(len(dzs))

    return dz_subset


def extract_ad_at_time(additional_diagnostics, time, time_extract):
    """
    Extracts additional diagnostics at the required time.

    Args:
        additional_diagnostics (dict):
            Dictionary of additional time varying diagnostics needed
            for the extraction method in use.

        time (datetime.datetime object):
            Time at which forecast data is needed.

        time_extract (iris.Constraint):
            Iris constraint for the desired time.

    Returns:
        ad_extracted (dict):
            Dictionary of the additional diagnostics but only data
            at the desired time.

    """
    ad_extracted = {}
    for key in additional_diagnostics.keys():
        cubes = additional_diagnostics[key]
        ad_extracted[key] = extract_cube_at_time(cubes, time, time_extract)
    return ad_extracted
