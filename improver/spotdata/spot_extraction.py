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

"""Spot data extraction from diagnostic fields using neighbour cubes."""

import iris
import numpy as np

from improver import BasePlugin
from improver.metadata.constants.attributes import MANDATORY_ATTRIBUTE_DEFAULTS
from improver.metadata.constants.mo_attributes import MOSG_GRID_ATTRIBUTES
from improver.metadata.utilities import create_coordinate_hash
from improver.spotdata.build_spotdata_cube import build_spotdata_cube
from improver.utilities.cube_manipulation import enforce_coordinate_ordering


class SpotExtraction(BasePlugin):
    """
    For the extraction of diagnostic data using neighbour cubes that contain
    spot-site information and the appropriate grid point from which to source
    data.
    """

    def __init__(self, neighbour_selection_method='nearest'):
        """
        Args:
            neighbour_selection_method (str):
                The neighbour cube may contain one or several sets of grid
                coordinates that match a spot site. These are determined by
                the neighbour finding method employed. This keyword is used to
                extract the desired set of coordinates from the neighbour cube.
        """
        self.neighbour_selection_method = neighbour_selection_method

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        return ('<SpotExtraction: neighbour_selection_method: {}>'.format(
                    self.neighbour_selection_method))

    def extract_coordinates(self, neighbour_cube):
        """
        Extract the desired set of grid coordinates that correspond to spot
        sites from the neighbour cube.

        Args:
            neighbour_cube (iris.cube.Cube):
                A cube containing information about the spot data sites and
                their grid point neighbours.
        Returns:
            iris.cube.Cube:
                A cube containing only the x and y grid coordinates for the
                grid point neighbours given the chosen neighbour selection
                method. The neighbour cube contains the indices stored as
                floating point values, so they are converted to integers
                in this cube.
        Raises:
            ValueError if the neighbour_selection_method expected is not found
            in the neighbour cube.
        """
        method = iris.Constraint(
            neighbour_selection_method_name=self.neighbour_selection_method)
        index_constraint = iris.Constraint(
            grid_attributes_key=['x_index', 'y_index'])
        coordinate_cube = neighbour_cube.extract(method & index_constraint)
        if coordinate_cube:
            coordinate_cube.data = np.rint(coordinate_cube.data).astype(int)
            return coordinate_cube

        available_methods = (
            neighbour_cube.coord('neighbour_selection_method_name').points)
        raise ValueError(
            'The requested neighbour_selection_method "{}" is not available in'
            ' this neighbour_cube. Available methods are: {}.'.format(
                self.neighbour_selection_method, available_methods))

    @staticmethod
    def extract_diagnostic_data(coordinate_cube, diagnostic_cube):
        """
        Extracts diagnostic data from the desired grid points in the diagnostic
        cube. The neighbour finding routine that produces the coordinate cube
        works in x-y order. As such, the diagnostic cube is changed to match
        before the indices are used to extract data.

        Args:
            coordinate_cube (iris.cube.Cube):
                A cube containing the x and y grid coordinates for the grid
                point neighbours.
            diagnostic_cube (iris.cube.Cube):
                A cube of diagnostic data from which spot data is being taken.
        Returns:
            numpy.ndarray:
                An array of diagnostic values at the grid coordinates found
                within the coordinate cube.
        """
        enforce_coordinate_ordering(
            diagnostic_cube, [diagnostic_cube.coord(axis='x').name(),
                              diagnostic_cube.coord(axis='y').name()])
        spot_values = diagnostic_cube.data[tuple(coordinate_cube.data.T)]
        return spot_values

    @staticmethod
    def build_diagnostic_cube(neighbour_cube, diagnostic_cube,
                              spot_values):
        """
        Builds a spot data cube containing the extracted diagnostic values.

        Args:
            neighbour_cube (iris.cube.Cube):
                This cube is needed as a source for information about the spot
                sites which needs to be included in the spot diagnostic cube.
            diagnostic_cube (iris.cube.Cube):
                The cube is needed to provide the name and units of the
                diagnostic that is being processed.
            spot_values (numpy.ndarray):
                An array containing the diagnostic values extracted for the
                required spot sites.
        Returns:
            iris.cube.Cube:
                A spot data cube containing the extracted diagnostic data.
        """

        neighbour_cube = build_spotdata_cube(
            spot_values, diagnostic_cube.name(), diagnostic_cube.units,
            neighbour_cube.coord('altitude').points,
            neighbour_cube.coord(axis='y').points,
            neighbour_cube.coord(axis='x').points,
            neighbour_cube.coord('wmo_id').points)
        return neighbour_cube

    def process(self, neighbour_cube, diagnostic_cube, new_title=None):
        """
        Create a spot data cube containing diagnostic data extracted at the
        coordinates provided by the neighbour cube.

        .. See the documentation for more details about the inputs and output.
        .. include:: /extended_documentation/spotdata/spot_extraction/
           spot_extraction_examples.rst

        Args:
            neighbour_cube (iris.cube.Cube):
                A cube containing information about the spot data sites and
                their grid point neighbours.
            diagnostic_cube (iris.cube.Cube):
                A cube of diagnostic data from which spot data is being taken.
            new_title (str or None):
                New title for spot-extracted data.  If None, this attribute is
                reset to a default value, since it has no prescribed standard
                and may therefore contain grid information that is no longer
                correct after spot-extraction.
        Returns:
            iris.cube.Cube:
                A cube containing diagnostic data for each spot site, as well
                as information about the sites themselves.
        """
        # Check we are using a matched neighbour/diagnostic cube pair
        check_grid_match([neighbour_cube, diagnostic_cube])

        coordinate_cube = self.extract_coordinates(neighbour_cube)

        # Deal with leading dimensions such as thresholds, realizations, etc.
        data_cubes = iris.cube.CubeList()
        for cube in diagnostic_cube.slices(
                [diagnostic_cube.coord(axis='x').name(),
                 diagnostic_cube.coord(axis='y').name()]):

            spot_values = self.extract_diagnostic_data(coordinate_cube, cube)
            spotdata_cube = self.build_diagnostic_cube(neighbour_cube, cube,
                                                       spot_values)

            # Add scalar coordinates onto the spot cube which can be promoted
            # to reform and leading dimensions.
            for coord in cube.coords(dim_coords=False):
                spotdata_cube.add_aux_coord(coord)
            data_cubes.append(spotdata_cube)

        spotdata_cube = data_cubes.merge_cube()

        # Copy attributes from the diagnostic cube that describe the data's
        # provenance
        spotdata_cube.attributes = diagnostic_cube.attributes
        spotdata_cube.attributes['model_grid_hash'] = (
            neighbour_cube.attributes['model_grid_hash'])

        # Remove grid attributes and update title
        for attr in MOSG_GRID_ATTRIBUTES:
            spotdata_cube.attributes.pop(attr, None)
        spotdata_cube.attributes["title"] = (
            MANDATORY_ATTRIBUTE_DEFAULTS["title"]
            if new_title is None else new_title)

        return spotdata_cube


def check_grid_match(cubes):
    """
    Checks that cubes are on, or originate from, compatible coordinate grids.
    Each cube is first checked for an existing 'model_grid_hash' which can be
    used to encode coordinate information on cubes that do not themselves
    contain a coordinate grid (e.g. spotdata cubes). If this is not found a new
    hash is generated to enable comparison. If the cubes are not compatible, an
    exception is raised to prevent the use of unmatched cubes.

    Args:
        cubes (list of iris.cube.Cube):
            A list of cubes to check for grid compatibility.
    Raises:
        ValueError: Raised if the cubes are not on matching grids as
                    identified by the model_grid_hash.
    """
    def _get_grid_hash(cube):
        try:
            cube_hash = cube.attributes['model_grid_hash']
        except KeyError:
            cube_hash = create_coordinate_hash(cube)
        return cube_hash

    cubes = iter(cubes)
    reference_hash = _get_grid_hash(next(cubes))

    for cube in cubes:
        cube_hash = _get_grid_hash(cube)
        if cube_hash != reference_hash:
            raise ValueError('Cubes do not share or originate from the same '
                             'grid, so cannot be used together.')
