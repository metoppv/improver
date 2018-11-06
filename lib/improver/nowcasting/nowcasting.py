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

import datetime

from cf_units import Unit
import iris
from iris.time import PartialDateTime

from improver.cube_combiner import CubeCombiner
from improver.utilities.cube_checker import check_cube_coordinates
from improver.utilities.temporal import iris_time_to_datetime, extract_nearest_time_point


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

    def extract_using_datetime(cube, dt, dt_increment=datetime.timedelta(hours=1)):
        #dt_half_hour = dt.replace(minute=30, second=0)

        if dt >= dt_half_hour:
            lower_dt = dt_half_hour
            upper_dt = dt_half_hour + dt_increment
        else:
            lower_dt = dt_half_hour - dt_increment
            upper_dt = dt_half_hour

        constr = (
            iris.Constraint(time=lambda cell: lower_dt <= cell < upper_dt))
        cube = cube.extract(constr)
        return cube


    @staticmethod
    def _extract_orographic_enhancement_cube(precip_cube, oe_cubes):
        """Extract the orographic enhancement cube with the required time
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
        time_point = iris_time_to_datetime(precip_cube.coord("time"))
        oe_cube = extract_nearest_time_point(oe_cubes, time_point)

        print(oe_cube)
        print(oe_cube[0].coord("time"))
        if not oe_cube:
            msg = ("There is no orographic enhancement available for "
                   "a time of {}".format(
                       iris_time_to_datetime(precip_cube.coord("time"))))
            raise ValueError(msg)
        if len(oe_cube)>1:
            msg = ("There are multiple orographic enhancements available for "
                   "a time of {}. Only one should be available.".format(
                       iris_time_to_datetime(precip_cube.coord("time"))))
            raise ValueError(msg)
        return oe_cube[0]

    def _apply_cube_combiner(self, precip_cube, oe_cube):
        """Combine the precipitation rate cube and the orographic enhancement
        cube.

        Args:
            precip_cube (iris.cube.Cube):
                Cube containing the input precipitation field.
            oe_cube (iris.cube.Cube):
                Cube containing the orographic enhancement field.

        Returns:
            cube (iris.cube.Cube):
                Cube containing the precipitation rate field modified by the
                orographic enhancement cube.

        """
        # Ensure the orographic enhancement cube matches the
        # dimensions of the precip_cube.
        oe_cube = check_cube_coordinates(precip_cube, oe_cube)
        temp_cubelist = iris.cube.CubeList([precip_cube, oe_cube])
        cube = CubeCombiner(self.operation).process(
            temp_cubelist, precip_cube.name())
        return cube

    @staticmethod
    def apply_minimum_precip_rate(cube):
        """Ensure that negative precipitation rates are capped at +1/32 mm/hr.

        Args:
            cube (iris.cube.Cube):
                Cube containing a precipitation rate field.

        Returns:
            cube (iris.cube.Cube):
                Cube containing the precipitation rate field where any
                negative precipitation rates have been capped at +1/32 mm/hr.

        """
        original_units = Unit("mm/hr")
        cube.data[cube.data<0] = original_units.convert(1/32., cube.units)
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

        if isinstance(orographic_enhancement_cubes, iris.cube.Cube):
            orographic_enhancement_cubes = (
                iris.cube.CubeList([orographic_enhancement_cubes]))

        updated_cubes = iris.cube.CubeList([])
        for precip_cube in precip_cubes:
            oe_cube = self._extract_orographic_enhancement_cube(
                precip_cube, orographic_enhancement_cubes)
            cube = self._apply_cube_combiner(precip_cube, oe_cube)
            cube = self.apply_minimum_precip_rate(cube)
            updated_cubes.append(cube)
        return updated_cubes
