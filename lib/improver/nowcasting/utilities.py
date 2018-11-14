#!/usr/bin/env python
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
"""Module with utilities required for nowcasting."""

import numpy as np

from cf_units import Unit
import iris

from improver.cube_combiner import CubeCombiner
from improver.utilities.cube_checker import check_cube_coordinates
from improver.utilities.cube_manipulation import merge_cubes
from improver.utilities.temporal import (
    extract_nearest_time_point, iris_time_to_datetime)


class ApplyOrographicEnhancement(object):

    """Apply orographic enhancement to precipitation rate input, either to
     add or subtract an orographic enhancement component."""

    def __init__(self, operation):
        """Initialise class.

        Args:
            operation (str):
                Operation (+, add, -, subtract) to apply to the incoming cubes.

        Raises:
            ValueError: Operation not supported.

        """
        # A minimum precipitation rate in mm/h that will be used as a lower
        # precipitation rate threshold.
        self.min_precip_rate_mmh = 1/32.

        possible_operations = ['+', 'add', '-', 'subtract']

        if operation in possible_operations:
            self.operation = operation
        else:
            msg = ("Operation '{}' not supported for combining "
                   "precipitation rate and "
                   "orographic enhancement.".format(operation))
            raise ValueError(msg)

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        result = ('<ApplyOrographicEnhancement: operation: {}>')
        return result.format(self.operation)

    @staticmethod
    def _select_orographic_enhancement_cube(precip_cube, oe_cubes):
        """Select the orographic enhancement cube with the required time
        coordinate.

        Args:
            precip_cube (iris.cube.Cube):
                Cube containing the input precipitation fields.
            oe_cubes (iris.cube.Cube or iris.cube.CubeList):
                Cube or CubeList containing the orographic enhancement fields.

        Returns:
            oe_cube (iris.cube.Cube):
                Cube containing the orographic enhancement fields at the
                required time.

        """
        time_point, = iris_time_to_datetime(precip_cube.coord("time").copy())
        oe_cube = extract_nearest_time_point(oe_cubes, time_point)
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
            cube (iris.cube.Cube):
                Cube containing the precipitation rate field modified by the
                orographic enhancement cube.

        """
        # Ensure the orographic enhancement cube matches the
        # dimensions of the precip_cube.
        oe_cube = check_cube_coordinates(precip_cube, oe_cube)

        # Ensure that orographic enhancement is in the units of the
        # precipitation rate cube.
        oe_cube.convert_units(precip_cube.units)

        # Set orographic enhancement to be zero for points with a
        # precipitation rate of < 1/32 mm/hr.
        original_units = Unit("mm/hr")
        threshold_in_cube_units = (
            original_units.convert(self.min_precip_rate_mmh,
                                   precip_cube.units))
        oe_cube.data[precip_cube.data < threshold_in_cube_units] = 0.

        # Use CubeCombiner to combine the cubes.
        temp_cubelist = iris.cube.CubeList([precip_cube, oe_cube])
        cube = CubeCombiner(self.operation).process(
            temp_cubelist, precip_cube.name())
        return cube

    def _apply_minimum_precip_rate(self, precip_cube, cube):
        """Ensure that negative precipitation rates are capped at +1/32 mm/hr.

        Args:
            precip_cube (iris.cube.Cube):
                Cube containing a precipitation rate input field.
            cube (iris.cube.Cube):
                Cube containing the precipitation rate field after combining
                with orographic enhancement.

        Returns:
            cube (iris.cube.Cube):
                Cube containing the precipitation rate field where any
                negative precipitation rates have been capped at +1/32 mm/hr.

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

                # Set any values lower than the tolerance to be 1/32 mm/hr.
                cube.data[mask] = threshold_in_cube_units
        return cube

    def process(self, precip_cubes, orographic_enhancement_cubes):
        """Apply orographic enhancement by modifying the input fields. This can
        include either adding or deleting the orographic enhancement component
        from the input precipitation fields.

        Args:
            precip_cubes (iris.cube.Cube or iris.cube.CubeList):
                Cube or CubeList containing the input precipitation fields.
            orographic_enhancement_cubes (iris.cube.Cube or
                                          iris.cube.CubeList):
                Cube or CubeList containing the orographic enhancement fields.

        Returns:
            updated_cubes (iris.cube.CubeList):
                CubeList of precipitation rate cubes that have been updated
                using orographic enhancement.
        """
        if isinstance(precip_cubes, iris.cube.Cube):
            precip_cubes = iris.cube.CubeList([precip_cubes])

        if isinstance(orographic_enhancement_cubes, iris.cube.CubeList):
            orographic_enhancement_cube = (
                merge_cubes(orographic_enhancement_cubes))
        else:
            orographic_enhancement_cube = orographic_enhancement_cubes

        updated_cubes = iris.cube.CubeList([])
        for precip_cube in precip_cubes:
            oe_cube = self._select_orographic_enhancement_cube(
                precip_cube, orographic_enhancement_cube.copy())
            cube = self._apply_orographic_enhancement(precip_cube, oe_cube)
            cube = self._apply_minimum_precip_rate(precip_cube, cube)
            updated_cubes.append(cube)
        return updated_cubes
