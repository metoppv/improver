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

from improver.utilities.cube_manipulation import compare_coords

DEG_TO_RAD = np.pi/180.


class ResolveWindComponents(object):

    """Plugin to resolve wind components along projection axes"""

    def __init__(self):
        """Initialise plugin"""
        pass

    def __repr__(self):
        """Represent the plugin instance as a string"""
        return ('<ResolveWindComponents>')

    @staticmethod
    def reproject_angles(wind_dir):
        """
        Reprojects wind directions from true north to grid north.

        Args:
            wind_dir (iris.cube.Cube):
                Cube containing wind direction angles with respect to
                true north

        Returns:
            reprojected (iris.cube.Cube):
                Cube containing wind direction angles with respect to
                grid north
        """

        # TODO write this function
        reprojected = wind_dir.copy()

        return reprojected

    @staticmethod
    def resolve_wind_components(speed, angle_deg):
        """
        Performs trigonometric conversion onto x and y axes

        Args:
            speed (np.ndarray):
                2D array of wind speeds
            angle_deg (np.ndarray):
                2D array of wind directions in degrees with respect
                to local grid north
        Returns:
            uspeed (np.ndarray):
                Wind speed component in the positive x-direction
            vspeed (np.ndarray):
                Wind speed component in the positive y-direction
        """
        sin_angle = np.sin(DEG_TO_RAD*angle_deg)
        cos_angle = np.cos(DEG_TO_RAD*angle_deg)
        uspeed = np.multiply(speed, sin_angle)
        vspeed = np.multiply(speed, cos_angle)
        return uspeed, vspeed

    def process(self, wind_speed, wind_dir):

        """
        Converts wind speed and direction into u,v components along
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

        # check input cube projections & dimensions match
        unmatched_coords = compare_coords([wind_speed, wind_dir])
        if unmatched_coords != [{}, {}]:
            msg = 'Wind speed and direction cubes have unmatched coordinates'
            raise ValueError('{} {}'.format(msg, unmatched_coords))

        # slice over x and y
        speed_slices = wind_speed.slices([wind_speed.coord(axis='y').name(),
                                          wind_speed.coord(axis='x').name()])
        dir_slices = wind_dir.slices([wind_dir.coord(axis='y').name(),
                                      wind_dir.coord(axis='x').name()])

        ucubelist = iris.cube.CubeList([])
        vcubelist = iris.cube.CubeList([])

        for speed, angle in zip(speed_slices, dir_slices):

            # convert wind directions to be with respect to projection y-axis
            # "north", rather than true north
            reproj_angle = self.reproject_angles(angle)

            # resolve wind speeds onto projection x and y axes
            reproj_angle.convert_units('degrees')
            uspeed, vspeed = self.resolve_wind_components(
                speed.data, reproj_angle.data)

            # create cubes to append
            ucubelist.append(speed.copy(data=uspeed))
            vcubelist.append(speed.copy(data=vspeed))

        # merge cubelists and rename
        ucube = ucubelist.merge_cube()
        ucube.rename("grid_eastward_wind")
        vcube = vcubelist.merge_cube()
        vcube.rename("grid_northward_wind")

        return ucube, vcube
