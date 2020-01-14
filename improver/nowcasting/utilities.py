#!/usr/bin/env python
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
"""Module with utilities required for nowcasting."""

import iris
import numpy as np
from cf_units import Unit

from improver import BasePlugin
from improver.utilities.temporal import (
    extract_nearest_time_point, iris_time_to_datetime)


class ExtendRadarMask(BasePlugin):
    """
    Extend the mask on radar rainrate data based on the radar coverage
    composite
    """

    def __init__(self):
        """
        Initialise with known values of the coverage composite for which radar
        data is valid.  All other areas will be masked.
        """
        self.coverage_valid = [1, 2]

    def process(self, radar_data, coverage):
        """
        Update the mask on the input rainrate cube to reflect where coverage
        is valid

        Args:
            radar_data (iris.cube.Cube):
                Radar data with mask corresponding to radar domains
            coverage (iris.cube.Cube):
                Radar coverage data containing values:
                    0: outside composite
                    1: precip detected
                    2: precip not detected & 1/32 mm/h detectable at this range
                    3: precip not detected & 1/32 mm/h NOT detectable

        Returns:
            iris.cube.Cube:
                Radar data with mask extended to mask out regions where
                1/32 mm/h are not detectable
        """
        # check cube coordinates match
        for crd in radar_data.coords():
            if coverage.coord(crd.name()) != crd:
                raise ValueError('Rain rate and coverage composites unmatched '
                                 '- coord {}'.format(crd.name()))

        # accommodate data from multiple times
        radar_data_slices = radar_data.slices([radar_data.coord(axis='y'),
                                               radar_data.coord(axis='x')])
        coverage_slices = coverage.slices([coverage.coord(axis='y'),
                                           coverage.coord(axis='x')])

        cube_list = iris.cube.CubeList()
        for rad, cov in zip(radar_data_slices, coverage_slices):
            # create a new mask that is False wherever coverage is valid
            new_mask = ~np.isin(cov.data, self.coverage_valid)

            # remask rainrate data
            remasked_data = np.ma.MaskedArray(rad.data.data, mask=new_mask)
            cube_list.append(rad.copy(remasked_data))

        return cube_list.merge_cube()


class ApplyOrographicEnhancement(BasePlugin):

    """Apply orographic enhancement to precipitation rate input, either to
     add or subtract an orographic enhancement component."""

    def __init__(self, operation):
        """Initialise class.

        Args:
            operation (str):
                Operation ("add" or "subtract") to apply to the incoming cubes.

        Raises:
            ValueError: Operation not supported.

        """
        # A minimum precipitation rate in mm/h that will be used as a lower
        # precipitation rate threshold.
        self.min_precip_rate_mmh = 1/32.
        self.operation = operation

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        result = ('<ApplyOrographicEnhancement: operation: {}>')
        return result.format(self.operation)

    @staticmethod
    def _select_orographic_enhancement_cube(precip_cube, oe_cubes,
                                            allowed_time_diff=1800):
        """Select the orographic enhancement cube with the required time
        coordinate.

        Args:
            precip_cube (iris.cube.Cube):
                Cube containing the input precipitation fields.
            oe_cubes (iris.cube.Cube or iris.cube.CubeList):
                Cube or CubeList containing the orographic enhancement fields.
            allowed_time_diff (int):
                An int in seconds to define a limit to the maximum difference
                between the datetime of the precipitation cube and the time
                points available within the orographic enhancement cube.
                If this limit is exceeded, then an error is raised.


        Returns:
            iris.cube.Cube:
                Cube containing the orographic enhancement fields at the
                required time.

        """
        time_point, = iris_time_to_datetime(precip_cube.coord("time").copy())
        oe_cube = extract_nearest_time_point(
            oe_cubes, time_point, allowed_dt_difference=allowed_time_diff)
        return oe_cube

    def _apply_orographic_enhancement(self, precip_cube, oe_cube):
        """Combine the precipitation rate cube and the orographic enhancement
        cube.

        Args:
            precip_cube (iris.cube.Cube):
                Cube containing the input precipitation field.
            oe_cube (iris.cube.Cube):
                Cube containing the orographic enhancement field matching
                the validity time of the precipitation cube.

        Returns:
            iris.cube.Cube:
                Cube containing the precipitation rate field modified by the
                orographic enhancement cube.

        """
        # Convert orographic enhancement into the units of the precipitation
        # rate cube.
        oe_cube.convert_units(precip_cube.units)

        # Set orographic enhancement to be zero for points with a
        # precipitation rate of < 1/32 mm/hr.
        original_units = Unit("mm/hr")
        threshold_in_cube_units = (
            original_units.convert(self.min_precip_rate_mmh,
                                   precip_cube.units))

        # Ignore invalid warnings generated if e.g. a NaN is encountered
        # within the less than (<) comparison.
        with np.errstate(invalid='ignore'):
            oe_cube.data[precip_cube.data < threshold_in_cube_units] = 0.

        # Add / subtract orographic enhancement where data is not masked
        cube = precip_cube.copy()
        if self.operation == "add":
            cube.data = cube.data + oe_cube.data
        elif self.operation == "subtract":
            cube.data = cube.data - oe_cube.data
        else:
            msg = ("Operation '{}' not supported for combining "
                   "precipitation rate and "
                   "orographic enhancement.".format(self.operation))
            raise ValueError(msg)

        return cube

    def _apply_minimum_precip_rate(self, precip_cube, cube):
        """Ensure that negative precipitation rates are capped at the defined
        minimum precipitation rate.

        Args:
            precip_cube (iris.cube.Cube):
                Cube containing a precipitation rate input field.
            cube (iris.cube.Cube):
                Cube containing the precipitation rate field after combining
                with orographic enhancement.

        Returns:
            iris.cube.Cube:
                Cube containing the precipitation rate field where any
                negative precipitation rates have been capped at the defined
                minimum precipitation rate.

        """
        if self.operation == "subtract":
            original_units = Unit("mm/hr")
            threshold_in_cube_units = (
                original_units.convert(self.min_precip_rate_mmh,
                                       cube.units))
            threshold_in_precip_cube_units = (
                original_units.convert(self.min_precip_rate_mmh,
                                       precip_cube.units))

            # Ignore invalid warnings generated if e.g. a NaN is encountered
            # within the less than (<) comparison.
            with np.errstate(invalid='ignore'):
                # Create a mask computed from where the input precipitation
                # cube is greater or equal to the threshold and the result
                # of combining the precipitation rate input cube with the
                # orographic enhancement has generated a cube with
                # precipitation rates less than the threshold.
                mask = ((precip_cube.data >= threshold_in_precip_cube_units) &
                        (cube.data <= threshold_in_cube_units))

                # Set any values lower than the threshold to be equal to
                # the minimum precipitation rate.
                cube.data[mask] = threshold_in_cube_units
        return cube

    def process(self, precip_cubes, orographic_enhancement_cube):
        """Apply orographic enhancement by modifying the input fields. This can
        include either adding or deleting the orographic enhancement component
        from the input precipitation fields.

        Args:
            precip_cubes (iris.cube.Cube or iris.cube.CubeList):
                Cube or CubeList containing the input precipitation fields.
            orographic_enhancement_cube (iris.cube.Cube):
                Cube containing the orographic enhancement fields.

        Returns:
            iris.cube.CubeList:
                CubeList of precipitation rate cubes that have been updated
                using orographic enhancement.
        """
        if isinstance(precip_cubes, iris.cube.Cube):
            precip_cubes = iris.cube.CubeList([precip_cubes])

        updated_cubes = iris.cube.CubeList([])
        for precip_cube in precip_cubes:
            oe_cube = self._select_orographic_enhancement_cube(
                precip_cube, orographic_enhancement_cube.copy())
            cube = self._apply_orographic_enhancement(precip_cube, oe_cube)
            cube = self._apply_minimum_precip_rate(precip_cube, cube)
            updated_cubes.append(cube)
        return updated_cubes
