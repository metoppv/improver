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
"""Module containing plugin to resolve wind components."""

import numpy as np
from iris.analysis import Linear
from iris.analysis.cartography import rotate_winds
from iris.coord_systems import GeogCS
from iris.coords import DimCoord
from iris.cube import Cube

from improver import BasePlugin
from improver.utilities.cube_manipulation import compare_coords

# Global coordinate reference system used in StaGE (GRS80)
GLOBAL_CRS = GeogCS(semi_major_axis=6378137.0,
                    inverse_flattening=298.257222101)


class ResolveWindComponents(BasePlugin):
    """
    Plugin to resolve wind components along an input cube's projection axes,
    given directions with respect to true North
    """

    def __repr__(self):
        """Represent the plugin instance as a string"""
        return '<ResolveWindComponents>'

    @staticmethod
    def calc_true_north_offset(reference_cube):
        """
        Calculate the angles between grid North and true North, as a
        matrix of values on the grid of the input reference cube.

        Args:
            reference_cube (iris.cube.Cube):
                2D cube on grid for which "north" is required.  Provides both
                coordinate system (reference_cube.coord_system()) and template
                spatial grid on which the angle adjustments should be provided.

        Returns:
            numpy.ndarray:
                Angle in radians by which wind direction wrt true North at
                each point must be rotated to be relative to grid North.
        """
        reference_x_coord = reference_cube.coord(axis='x')
        reference_y_coord = reference_cube.coord(axis='y')

        # find corners of reference_cube grid in lat / lon coordinates
        latlon = [GLOBAL_CRS.as_cartopy_crs().transform_point(
            reference_x_coord.points[i], reference_y_coord.points[j],
            reference_cube.coord_system().as_cartopy_crs()) for i in [0, -1]
                                                            for j in [0, -1]]
        latlon = np.array(latlon).T.tolist()

        # define lat / lon coordinates to cover the reference_cube grid at an
        # equivalent resolution
        lat_points = np.linspace(np.floor(min(latlon[1])),
                                 np.ceil(max(latlon[1])),
                                 len(reference_y_coord.points))
        lon_points = np.linspace(np.floor(min(latlon[0])),
                                 np.ceil(max(latlon[0])),
                                 len(reference_x_coord.points))

        lat_coord = DimCoord(lat_points, 'latitude', units='degrees',
                             coord_system=GLOBAL_CRS)
        lon_coord = DimCoord(lon_points, 'longitude', units='degrees',
                             coord_system=GLOBAL_CRS)

        # define a unit vector wind towards true North over the lat / lon grid
        udata = np.zeros(reference_cube.shape, dtype=np.float32)
        vdata = np.ones(reference_cube.shape, dtype=np.float32)

        ucube_truenorth = Cube(udata, "grid_eastward_wind",
                               dim_coords_and_dims=[(lat_coord, 0),
                                                    (lon_coord, 1)])
        vcube_truenorth = Cube(vdata, "grid_northward_wind",
                               dim_coords_and_dims=[(lat_coord, 0),
                                                    (lon_coord, 1)])

        # rotate unit vector onto reference_cube coordinate system
        ucube, vcube = rotate_winds(
            ucube_truenorth, vcube_truenorth, reference_cube.coord_system())

        # unmask and regrid rotated winds onto reference_cube grid
        ucube.data = ucube.data.data
        ucube = ucube.regrid(reference_cube, Linear())
        vcube.data = vcube.data.data
        vcube = vcube.regrid(reference_cube, Linear())

        # ratio of u to v winds is the tangent of the angle which is the
        # true North to grid North rotation
        angle_adjustment = np.arctan2(ucube.data, vcube.data)

        return angle_adjustment

    @staticmethod
    def resolve_wind_components(speed, angle, adj):
        """
        Perform trigonometric reprojection onto x and y axes

        Args:
            speed (iris.cube.Cube):
                Cube containing wind speed data
            angle (iris.cube.Cube):
                Cube containing wind directions as angles from true North
            adj (numpy.ndarray):
                2D array of wind direction angle adjustments in radians, to
                convert zero reference from true North to grid North.
                Broadcast automatically if speed and angle cubes have extra
                dimensions.

        Returns:
            (tuple): tuple containing:
                **u_speed** (iris.cube.Cube):
                    Cube containing wind vector component in the positive
                    x-direction

                **v_speed** (iris.cube.Cube):
                    Cube containing wind vector component in the positive
                    y-direction
        """
        angle.convert_units('radians')
        angle.data += adj

        # output vectors should be pointing "to" not "from"
        if "wind_from_direction" in angle.name():
            angle.data += np.pi
        sin_angle = np.sin(angle.data)
        cos_angle = np.cos(angle.data)
        uspeed = np.multiply(speed.data, sin_angle)
        vspeed = np.multiply(speed.data, cos_angle)
        return [speed.copy(data=uspeed), speed.copy(data=vspeed)]

    def process(self, wind_speed, wind_dir):

        """
        Convert wind speed and direction into u,v components along input cube
        projection axes.

        Args:
            wind_speed (iris.cube.Cube):
                Cube containing wind speed values
            wind_dir (iris.cube.Cube):
                Cube containing wind direction values relative to true North

        Returns:
            (tuple): tuple containing:
                **ucube** (iris.cube.Cube):
                    Cube containing wind speeds in the positive projection
                    x-axis direction, with units and projection matching
                    wind_speed cube.

                **vcube** (iris.cube.Cube):
                    Cube containing wind speeds in the positive projection
                    y-axis direction, with units and projection matching
                    wind_speed cube.
        """
        # check cubes contain the correct data (assuming CF standard names)
        if "wind_speed" not in wind_speed.name():
            msg = '{} cube does not contain wind speeds'
            raise ValueError('{} {}'.format(wind_speed.name(), msg))

        if "wind" not in wind_dir.name() or "direction" not in wind_dir.name():
            msg = '{} cube does not contain wind directions'
            raise ValueError('{} {}'.format(wind_dir.name(), msg))

        # check input cube coordinates match
        unmatched_coords = compare_coords([wind_speed, wind_dir])
        if unmatched_coords != [{}, {}]:
            msg = 'Wind speed and direction cubes have unmatched coordinates'
            raise ValueError('{} {}'.format(msg, unmatched_coords))

        # calculate angle adjustments for wind direction
        wind_dir_slice = next(
            wind_dir.slices([wind_dir.coord(axis='y').name(),
                             wind_dir.coord(axis='x').name()]))
        adj = self.calc_true_north_offset(wind_dir_slice)

        # calculate grid eastward and northward speeds
        ucube, vcube = self.resolve_wind_components(wind_speed, wind_dir, adj)

        # relabel final cubes with CF compliant data names corresponding to
        # positive wind speeds along the x and y axes
        ucube.rename("grid_eastward_wind")
        vcube.rename("grid_northward_wind")

        return ucube, vcube
