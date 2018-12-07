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

"""Apply temperature lapse rate adjustments to a spot data cube."""

import numpy as np

import iris
from improver.spotdata_new.spot_extraction import SpotExtraction


class SpotLapseRateAdjust:
    """
    Adjusts spot data temperatures by a lapse rate to better represent the
    conditions at their altitude that may not be captured by the model
    orography.
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
            grid_metadata_identifier (str or None):
                A string to search for in the input cube's attributes that
                can be used to ensure that the cubes being used are for the
                same model/grid. This test can be bypassed by setting this to
                None.
        """
        self.neighbour_selection_method = neighbour_selection_method
        self.grid_metadata_identifier = grid_metadata_identifier

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        return ('<SpotLapseRateAdjust: neighbour_selection_method: {}, '
                'grid_metadata_identifier: {}>'.format(
                    self.neighbour_selection_method,
                    self.grid_metadata_identifier))

    def check_grid_match(self, cubes):
        """
        Uses the provided grid_metadata_identifier to extract and compare
        attributes on the input cubes. The expectation is that all the metadata
        identified should match for the cubes to be deemed compatible.

        Args:
            cubes (list of cubes):
                List of cubes for which the attributes should be tested.
        Raises:
            ValueError: Raised if the metadata extracted is not identical on
                        all cubes.
        """
        def _get_attributes(cube):
            """Build dictionary of attributes that match the
            self.grid_metadata_identifier."""

            attributes = cube.attributes
            attributes = {k:v for (k, v) in attributes.items()
                          if self.grid_metadata_identifier in k}
            return attributes
        def _compare_attributes(attributes, reference_attributes):
            """Compare keys and values of attributes with reference_attributes.
            Raise an exception if they do not match exactly."""

            match = ((attributes.items() & reference_attributes.items()) ==
                     reference_attributes.items())
            if match is not True:
                raise ValueError('Cubes do not share the metadata identified '
                                 'by the grid_metadata_identifier ({})'.format(
                                  self.grid_metadata_identifier))

        # Allow user to bypass cube comparison by setting identifier to None.
        if self.grid_metadata_identifier is None:
            return

        reference_attributes = _get_attributes(cubes[0])
        for cube in cubes:
            attributes = _get_attributes(cube)
            _compare_attributes(attributes, reference_attributes)

    def process(self, spot_data_cube, neighbour_cube, gridded_lapse_rate_cube):
        """
        Extract lapse rates from the appropriate grid points and apply them to
        the spot extracted temperatures.

        Args:
            spot_data_cube (iris.cube.Cube):
                The spot data cube of temperatures extracted from the gridded
                fields. These temperatures will have been extracted using the
                same neighbour_cube and neighbour_selection_method that are
                being used here.
            neighbour_cube (iris.cube.Cube):
                The neighbour_cube that contains the grid coordinates at which
                lapse rates should be extracted and the vertical displacement
                between those grid points on the model orography and the spot
                data sites actual altitudes.
            gridded_lapse_rate_cube (iris.cube.Cube):
                A cube of temperature lapse rates on the same grid as that from
                which the spot data temperatures were extracted.
        Returns:
            new_spot_cube (iris.cube.Cube):
                A copy of the input spot_data_cube with the data modified by
                the lapse rates to give a better representation of the site's
                temperatures.
        """
        # Check the cubes are compatible.
        self.check_grid_match([spot_data_cube, neighbour_cube,
                               gridded_lapse_rate_cube])

        # Extract the lapse rates that correspond to the spot sites.
        extraction_plugin = SpotExtraction(
            neighbour_selection_method=self.neighbour_selection_method)
        spot_lapse_rate = extraction_plugin.process(neighbour_cube,
                                                    gridded_lapse_rate_cube)

        # Extract vertical displacements between the model orography and sites.
        method_constraint = iris.Constraint(
            neighbour_selection_method_name=self.neighbour_selection_method)
        data_constraint = iris.Constraint(
            grid_attributes_key='vertical_displacement')
        vertical_displacement = neighbour_cube.extract(method_constraint &
                                                       data_constraint)

        # Create a copy of the input cube with modified temperatures.
        new_temperatures = (
            spot_data_cube.data + (
                spot_lapse_rate.data * vertical_displacement.data)
            ).astype(np.float32)
        new_spot_cube = spot_data_cube.copy(data=new_temperatures)

        return new_spot_cube
