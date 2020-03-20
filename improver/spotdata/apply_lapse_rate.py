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

"""Apply temperature lapse rate adjustments to a spot data cube."""

import iris
import numpy as np

from improver import PostProcessingPlugin
from improver.spotdata.spot_extraction import SpotExtraction, check_grid_match


class SpotLapseRateAdjust(PostProcessingPlugin):
    """
    Adjusts spot data temperatures by a lapse rate to better represent the
    conditions at their altitude that may not be captured by the model
    orography.
    """

    def __init__(self, neighbour_selection_method='nearest'):
        """
        Args:
            neighbour_selection_method (str):
                The neighbour cube may contain one or several sets of grid
                coordinates that match a spot site. These are determined by
                the neighbour finding method employed. This keyword is used to
                extract the desired set of coordinates from the neighbour cube.
                The methods available will be all, or a subset, of the
                following::

                  - nearest
                  - nearest_land
                  - nearest_land_minimum_dz
                  - nearest_minimum_dz

                The method available in a neighbour cube will depend on the
                options that were specified when it was created.
        """
        self.neighbour_selection_method = neighbour_selection_method

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        return ('<SpotLapseRateAdjust: neighbour_selection_method: {}'
                '>'.format(
                    self.neighbour_selection_method))

    def process(self, spot_data_cube, neighbour_cube, gridded_lapse_rate_cube):
        """
        Extract lapse rates from the appropriate grid points and apply them to
        the spot extracted temperatures.

        The calculation is::

         lapse_rate_adjusted_temperatures = temperatures + lapse_rate *
         vertical_displacement

        Args:
            spot_data_cube (iris.cube.Cube):
                A spot data cube of temperatures for the spot data sites,
                extracted from the gridded temperature field. These
                temperatures will have been extracted using the same
                neighbour_cube and neighbour_selection_method that are being
                used here.
            neighbour_cube (iris.cube.Cube):
                The neighbour_cube that contains the grid coordinates at which
                lapse rates should be extracted and the vertical displacement
                between those grid points on the model orography and the spot
                data sites actual altitudes. This cube is only updated when
                a new site is added.
            gridded_lapse_rate_cube (iris.cube.Cube):
                A cube of temperature lapse rates on the same grid as that from
                which the spot data temperatures were extracted.
        Returns:
            iris.cube.Cube:
                A copy of the input spot_data_cube with the data modified by
                the lapse rates to give a better representation of the site's
                temperatures.
        """
        # Check the cubes are compatible.
        check_grid_match([neighbour_cube, spot_data_cube,
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

        # Apply lapse rate adjustment to the temperature at each site.
        new_temperatures = (
            spot_data_cube.data + (
                spot_lapse_rate.data * vertical_displacement.data)
            ).astype(np.float32)
        new_spot_cube = spot_data_cube.copy(data=new_temperatures)
        return new_spot_cube
