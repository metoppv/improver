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

"""Spot data extraction from diagnostic fields using neighbour cubes."""

import numpy as np

import iris
from improver.utilities.cube_manipulation import (enforce_coordinate_ordering,
                                                  compare_attributes)
from improver.spotdata_new.build_spotdata_cube import build_spotdata_cube


class SpotExtraction():
    """
    For the extraction of diagnostic data using neighbour cubes that contain
    spot-site information and the appropriate grid point from which to source
    data.
    """

    def __init__(self, neighbour_selection_method='nearest',
                 grid_metadata_identifier='mosg'):
        """
        Args:
            neighbour_selection_method (str):
                The neighbour cube may contain one or several sets of grid
                coordinates that match a spot site. These are determined by
                the neighbour finding method employed. This keyword is used to
                extract the desired set of coordinates from the neighbour cube.
            grid_metadata_identifier (str):
                A string to search for in the input cube attributes that
                can be used to ensure that the neighbour cube being used has
                been created for the model/grid of the diagnostic cube.
        """
        self.neighbour_selection_method = neighbour_selection_method
        self.grid_metadata_identifier = grid_metadata_identifier

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        return ('<SpotExtraction: neighbour_selection_method: {}, '
                'grid_metadata_identifier: {}>'.format(
                    self.neighbour_selection_method,
                    self.grid_metadata_identifier))

    def extract_coordinates(self, neighbour_cube):
        """
        Extract the desired set of grid coordinates that correspond to spot
        sites from the neighbour cube.

        Args:
            neighbour_cube (iris.cube.Cube):
                A cube containing information about the spot data sites and
                their grid point neighbours.
        Returns:
            coordinate_cube (iris.cube.Cube):
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
            spot_values (np.array):
                An array of diagnostic values at the grid coordinates found
                within the coordinate cube.
        """
        diagnostic_cube = enforce_coordinate_ordering(
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
            spot_values (np.array):
                An array containing the diagnostic values extracted for the
                required spot sites.
        Returns:
            neighbour_cube (iris.cube.Cube):
                A spot data cube containing the extracted diagnostic data.
        """

        neighbour_cube = build_spotdata_cube(
            spot_values, diagnostic_cube.name(), diagnostic_cube.units,
            neighbour_cube.coord('altitude').points,
            neighbour_cube.coord(axis='y').points,
            neighbour_cube.coord(axis='x').points,
            neighbour_cube.coord('wmo_id').points)
        return neighbour_cube

    def process(self, neighbour_cube, diagnostic_cube):
        """
        Create a spot data cube containing diagnostic data extracted at the
        coordinates provided by the neighbour cube.

        .. See the documentation for more details about the inputs and output.
        .. include:: extended_documentation/spot_extraction_examples.rst

        Args:
            neighbour_cube (iris.cube.Cube):
                A cube containing information about the spot data sites and
                their grid point neighbours.
            diagnostic_cube (iris.cube.Cube):
                A cube of diagnostic data from which spot data is being taken.
        Returns:
            spotdata_cube (iris.cube.Cube):
                A cube containing diagnostic data for each spot site, as well
                as information about the sites themselves.
        """
        # Check we are using a matched neighbour/diagnostic cube pair
        check_grid_match(self.grid_metadata_identifier,
                         [neighbour_cube, diagnostic_cube])

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
        # provenance.
        spotdata_cube.attributes = diagnostic_cube.attributes

        return spotdata_cube


def check_grid_match(grid_metadata_identifier, cubes):
    """
    Uses the provided grid_metadata_identifier to extract and compare
    attributes on the input cubes. The expectation is that all the metadata
    identified should match for the cubes to be deemed compatible.

    Args:
        grid_metadata_identifier (str):
            A partial or complete attribute name. Attributes matching this are
            compared between the two cubes.
        cubes (list of iris.cube.Cube items):
            List of cubes for which the attributes should be tested.
    Raises:
        ValueError: Raised if the metadata extracted is not identical on
                    all cubes.
    """
    # Allow user to bypass cube comparison by setting identifier to None.
    if grid_metadata_identifier is None:
        return

    comparison_result = compare_attributes(
        cubes, attribute_filter=grid_metadata_identifier)

    print(comparison_result)

    # Check that all dictionaries returned are empty, indicating matches.
    if not all(not item for item in comparison_result):
        raise ValueError('Cubes do not share the metadata identified '
                         'by the grid_metadata_identifier ({})'.format(
                             grid_metadata_identifier))
