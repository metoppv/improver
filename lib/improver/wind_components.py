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
"""Module containing plugin to resolve wind components."""

import numpy as np
import iris
from iris.analysis.cartography import rotate_winds

from improver.utilities.cube_manipulation import compare_coords

DEG_TO_RAD = np.pi/180.


class ResolveWindComponents(object):
    """Plugin to resolve wind components along specified projection axes"""

    def __init__(self, target_cs):
        """
        Initialise plugin

        Args:
            target_cs (iris.coord_systems.CoordSystem):
                Coordinate system onto which to transform wind components
        """
        self.target_cs = target_cs

    def __repr__(self):
        """Represent the plugin instance as a string"""
        return ('<ResolveWindComponents: target_cs {}>'.format(self.target_cs))

    @staticmethod
    def resolve_wind_components(speed, angle):
        """
        Perform trigonometric reprojection onto x and y axes

        Args:
            speed (iris.cube.Cube):
                Cube containing 2D array of wind speeds
            angle_deg (iris.cube.Cube):
                Cube containing 2D array of wind directions as angles from N
        Returns:
            iris.cube.Cube:
                Cube containing wind speed component in the
                positive x-direction
            iris.cube.Cube:
                Cube containing wind speed component in the
                positive y-direction
        """
        angle.convert_units('degrees')
        sin_angle = np.sin(DEG_TO_RAD*angle.data)
        cos_angle = np.cos(DEG_TO_RAD*angle.data)
        uspeed = np.multiply(speed.data, sin_angle)
        vspeed = np.multiply(speed.data, cos_angle)
        return [speed.copy(data=uspeed), speed.copy(data=vspeed)]

    def process(self, wind_speed, wind_dir):

        """
        Convert wind speed and direction into u,v components along
        specified projection axes.

        Args:
            wind_speed (iris.cube.Cube):
                Cube containing wind speed values
            wind_dir (iris.cube.Cube):
                Cube containing wind direction values (in degrees)

        Returns:
            ucube (iris.cube.Cube):
                Cube containing wind speeds in the positive projection x-axis
                direction, with units and projection matching wind_speed cube.
            vcube (iris.cube.Cube):
                Cube containing wind speeds in the positive projection y-axis
                direction, with units and projection matching wind_speed cube.
        """

        # check input cube coordinates match
        unmatched_coords = compare_coords([wind_speed, wind_dir])
        if unmatched_coords != [{}, {}]:
            msg = 'Wind speed and direction cubes have unmatched coordinates'
            raise ValueError('{} {}'.format(msg, unmatched_coords))

        # slice over x and y
        speed_slices = wind_speed.slices([wind_speed.coord(axis='y').name(),
                                          wind_speed.coord(axis='x').name()])
        dir_slices = wind_dir.slices([wind_dir.coord(axis='y').name(),
                                      wind_dir.coord(axis='x').name()])

        # resolve components from true North
        uvcubelist = [self.resolve_wind_components(speed, angle)
                      for speed, angle in zip(speed_slices, dir_slices)]
        uvcubelist = np.array(uvcubelist).T.tolist()

        # merge cubelists
        ucube = iris.cube.CubeList(uvcubelist[0]).merge_cube()
        vcube = iris.cube.CubeList(uvcubelist[1]).merge_cube()

        # rotate winds onto target coordinate system
        ucube, vcube = rotate_winds(ucube, vcube, self.target_cs())

        # relabel final cubes with CF compliant data names
        ucube.rename("grid_eastward_wind")
        vcube.rename("grid_northward_wind")

        return ucube, vcube
