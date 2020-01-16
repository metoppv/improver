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
"""Module for calculating the probability of specific precipitation phases."""

import operator
import iris
import numpy as np
from cf_units import Unit

from improver import BasePlugin
from improver.nbhood.nbhood import GeneratePercentilesFromANeighbourhood
from improver.metadata.utilities import (
    create_new_diagnostic_cube, generate_mandatory_attributes)
from improver.utilities.cube_checker import spatial_coords_match


class PrecipPhaseProbability(BasePlugin):
    """
    This plugin converts a falling-phase-change-level cube into the
    probability of a specific precipitation phase being found at the surface.

    For snow; the 80th percentile is taken from a neighbourhood around
    each point and is compared with the orography. Where the orography is
    higher, the returned probability-of-snow is 1, else 0.
    For rain, the above method is modified to get the 20th percentile
    and where the orography is lower than the percentile value, the returned
    probability-of-rain is 1, else 0.
    """
    def __init__(self, radius=10000.):
        """
        Initialise plugin
        Args:
            radius (float):
                Neighbourhood radius from which 80th percentile is found (m)
        """
        self.percentile_plugin = GeneratePercentilesFromANeighbourhood
        self._nbhood_shape = 'circular'
        self.radius = radius

    def _extract_input_cubes(self, cubes):
        """
        Separates the input list into the required cubes for this plugin,
        detects whether snow or rain are required from the input phase-level
        cube name and appropriately initialises the percentile_plugin, sets
        the appropriate comparator operator for comparing with orography and
        the unique part of the output cube name.

        Converts units of falling_level_cube to that of orography_cube if
        necessary. Sets flag for snow or rain depending on name of
        falling_level_cube.

        Args:
            cubes (iris.cube.CubeList or list):
                Contains cubes of the altitude of the phase-change level (this
                can be snow->sleet, or sleet->rain) and the altitude of the
                orography. The name of the phase-change level cube must be
                either "altitude_of_snow_falling_level" or
                "altitude_of_rain_falling_level". The name of the orography
                cube must be "surface_altitude".

        Raises:
            ValueError: If cubes with the expected names cannot be extracted.
            ValueError: If cubes does not have the expected length of 2.
            ValueError: If the extracted cubes do not have matching spatial
                        coordinates.

        """
        if isinstance(cubes, list):
            cubes = iris.cube.CubeList(cubes)
        if len(cubes) != 2:
            raise ValueError(f'Expected 2 cubes, found {len(cubes)}')

        if not spatial_coords_match(cubes[0], cubes[1]):
            raise ValueError('Spatial coords mismatch between '
                             f'{cubes[0]} and '
                             f'{cubes[1]}')

        extracted_cube = cubes.extract('altitude_of_snow_falling_level')
        if extracted_cube:
            self.falling_level_cube, = extracted_cube
            self.param = 'snow'
            self.comparator = operator.gt
            self.get_discriminating_percentile = self.percentile_plugin(
                self._nbhood_shape, self.radius, percentiles=[80.])
        else:
            extracted_cube = cubes.extract('altitude_of_rain_falling_level')
            if not extracted_cube:
                raise ValueError(
                    'Could not extract a rain or snow falling-level '
                    f'cube from {cubes}')
            self.falling_level_cube, = extracted_cube
            self.param = 'rain'
            self.comparator = operator.lt
            # We want rain at or above the surface, so inverse of 80th
            # centile is the 20th centile.
            self.get_discriminating_percentile = self.percentile_plugin(
                self._nbhood_shape, self.radius, percentiles=[20.])

        orography_name = 'surface_altitude'
        extracted_cube = cubes.extract(orography_name)
        if extracted_cube:
            self.orography_cube, = extracted_cube
        else:
            raise ValueError(f'Could not extract {orography_name} cube from '
                             f'{cubes}')

        if self.falling_level_cube.units != self.orography_cube.units:
            self.falling_level_cube = self.falling_level_cube.copy()
            self.falling_level_cube.convert_units(self.orography_cube.units)

    def process(self, cubes):
        """
        Derives the probability of a precipitation phase at the surface. If
        the snow-sleet falling-level is supplied, this is the probability of
        snow at (or below) the surface. If the sleet-rain falling-level is
        supplied, this is the probability of rain at (or above) the surface.

        Args:
            cubes (iris.cube.CubeList or list):
                Contains cubes of the altitude of the phase-change level (this
                can be snow->sleet, or sleet->rain) and the altitude of the
                orography.

        Returns:
            iris.cube.Cube:
                Contains the probability of a specific precipitation phase
                reaching the surface orography. If the falling_level_cube was
                snow->sleet, then this will be the probability of snow at the
                surface. If the falling_level_cube was sleet->rain, then this
                will be the probability of rain at the surface.
                The probabilities are categorical (1 or 0) allowing
                precipitation to be divided uniquely between snow, sleet and
                rain phases.
        """
        self._extract_input_cubes(cubes)
        processed_falling_level = iris.util.squeeze(
            self.get_discriminating_percentile(
                self.falling_level_cube))

        result_data = np.where(
            self.comparator(
                self.orography_cube.data,
                processed_falling_level.data),
            1, 0).astype('float32')
        mandatory_attributes = generate_mandatory_attributes(
            [self.falling_level_cube])

        cube = create_new_diagnostic_cube(
            f'probability_of_{self.param}_at_surface',
            Unit('1'),
            self.falling_level_cube,
            mandatory_attributes,
            data=result_data)
        return cube
