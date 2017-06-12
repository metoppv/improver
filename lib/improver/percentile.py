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
"""Module containing percentiling classes."""


import iris
from iris.exceptions import CoordinateNotFoundError
from iris import FUTURE

FUTURE.netcdf_promote = True


class PercentileConverter(object):

    """Plugin for converting from a set of values to a PDF.

    Generate percentiles together with min, max, mean, stdev.

    """

    # Default percentile boundaries to calculate at.
    DEFAULT_PERCENTILES = [0, 5, 10, 20, 25, 30, 40, 50,
                           60, 70, 75, 80, 90, 95, 100]

    def __init__(self, collapse_coord, percentiles=None):
        """Create a PDF plugin with a given source plugin.

        Parameters
        ----------
        collapse_coord : str or iris.coord.DimCoord (or list of either)
            The name or coordinate definition of the coordinate(s) to collapse
            over.

        percentiles : list (optional)
            Percentile values at which to calculate; if not provided uses
            DEFAULT_PERCENTILES.

        """
        if not isinstance(collapse_coord, list):
            collapse_coord = [collapse_coord]
        for coord_item in collapse_coord:
            _type_test(coord_item)

        if percentiles is not None:
            self.percentiles = map(int, percentiles)
        else:
            self.percentiles = self.DEFAULT_PERCENTILES

        # Collapsing multiple coordinates results in a new percentile
        # coordinate, its name suffixed by the original coordinate names. Such
        # a collapse is cummutative (i.e. coordinate order doesn't matter).
        # However the coordinates are sorted such that the resulting
        # percentile coordinate has a consistent name regardless of the order
        # in which the user provides the original coordinate names.
        self.collapse_coord = sorted(collapse_coord, _sort_coord)

    def process(self, cube):
        """Create a cube containing the percentiles as a new dimension.

        What's generated is:
            * 13 percentiles - (5%, 10%, 20%, 25%, 30%, 40%, 50%, 60%,
              70%, 75%, 80%, 90%, 95%)

        Parameters
        ----------
        cube : iris.cube.Cube instance
            Given the collapse coordinate, convert the set of values
            along that coordinate into a PDF and extract percentiles
            and min, max, mean, stdev.

        Returns
        -------
        cube : iris.cube.Cube instance
            A single merged cube of all the cubes produced by each percentile
            collapse.

        """
        try:
            return cube.collapsed(self.collapse_coord,
                                  iris.analysis.PERCENTILE,
                                  percent=self.percentiles)
        except:
            raise CoordinateNotFoundError(
                "Coordinate '{}' not found in cube passed to {}.".format(
                    self.collapse_coord, self.__class__.__name__))


def _type_test(collapse_coord):
    """Test for input type."""
    if not isinstance(collapse_coord, (basestring, iris.coords.Coord)):
        raise ValueError('collapse_coord is {!r}, which does not specify '
                         'a single coordinate.'.format(collapse_coord))


def _sort_coord(item_a, item_b):
    """Sorting function to alphabetically sort either strings representing
    coordinates or iris.coords.Coord objects by their names.

    """
    name_a, name_b = item_a, item_b
    if isinstance(item_a, iris.coords.Coord):
        name_a = item_a.name()
    if isinstance(item_b, iris.coords.Coord):
        name_b = item_b.name()
    if name_a > name_b:
        return 1
    if name_a == name_b:
        return 0
    else:
        return -1
