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
from cartopy.crs import Geodetic

from improver.utilities.cube_manipulation import compare_coords

DEG_TO_RAD = np.pi/180.


class ResolveWindComponents(object):
    """Plugin to resolve wind components along specified projection axes"""

    def __init__(self):
        """Initialise plugin"""
        pass

    def __repr__(self):
        """Represent the plugin instance as a string"""
        return ('<ResolveWindComponents>')

    @staticmethod
    def calc_true_north_offset(reference_cube):
        """
        Calculate the angles between grid north and true north, as a
        matrix of values on the grid of the input reference cube.

        Args:
            reference_cube (iris.cube.Cube):
                2D cube on grid for which "north" is required.  Provides both
                coordinate system (reference_cube.coord_system()) and template
                spatial grid on which the angle adjustments should be provided.

        Returns:
            angle_adjustment (numpy.ndarray):
                Angle in radians to be added to "wind_to_direction" at each
                point on the x-y input grid, so that the new direction is with
                respect to grid north.  Equivalent to the clockwise angular
                rotation at each point from true north to grid north.
        """
        # extrapolate coordinates half a point out in the y-direction, so that
        # so that diffs will be centred
        ypoints = list(reference_cube.coord(axis='y').points)
        grid_length = ypoints[1] - ypoints[0]
        ypoints.append(ypoints[-1] + grid_length)
        ypoints = np.array(ypoints) - 0.5*grid_length

        # get longitudes of extrapolated coordinates
        xv, yv = np.meshgrid(reference_cube.coord(axis='x').points, ypoints)
        newshape = xv.shape

        reference_cs = reference_cube.coord_system().as_cartopy_crs()
        lat_lon_cs = Geodetic()

        lon_points = []
        lat_points = []
        for x, y in zip(xv.flatten(), yv.flatten()):
            lon, lat = lat_lon_cs.transform_point(x, y, reference_cs)
            lon_points.append(lon)
            lat_points.append(lat)
        lon_points = np.array(lon_points).reshape(newshape)
        lat_points = np.array(lat_points).reshape(newshape)

        # calculate longitude differences in degrees along y-axis
        lon_diffs = np.diff(lon_points, axis=0)

        # interpolate latitude to original x/y grid (approx)
        lat_points_interp = [np.interp(np.arange(len(ypoints)-1)+0.5,
                                       np.arange(len(ypoints)),
                                       points) for points in lat_points.T]
        lat_points = np.array(lat_points_interp).T

        # convert longitude differences in degrees to distance at the latitude
        # of each point
        def longitude_difference_to_km(lon_diff, lat):
            """Converts longitude differences into km at a given latitude"""
            lat_adj = np.cos(DEG_TO_RAD*lat)
            earth_radius = 6371.
            distance_km = lat_adj*lon_diff*DEG_TO_RAD*earth_radius
            return distance_km

        lon_diffs_km = longitude_difference_to_km(lon_diffs, lat_points)

        # calculate d(lon)/dy ratio (defines quadratic in sin theta to solve)
        ycoord = reference_cube.coord(axis='y').copy()
        ycoord.convert_units('km')
        grid_length_km = ycoord.points[1] - ycoord.points[0]
        grid_length_array = np.full(
            reference_cube.shape, grid_length_km, dtype=np.float32)

        distance_ratio = np.divide(lon_diffs_km, grid_length_array)

        # solve quadratic in sin theta
        num = 1. + 4.*np.multiply(distance_ratio, distance_ratio)
        sin_theta = np.divide(np.sqrt(num) - 1., 2.*distance_ratio)

        angle_adjustment = np.arcsin(sin_theta)
        return angle_adjustment

    @staticmethod
    def resolve_wind_components(speed, angle, adj):
        """
        Perform trigonometric reprojection onto x and y axes

        Args:
            speed (iris.cube.Cube):
                Cube containing 2D array of wind speeds
            angle (iris.cube.Cube):
                Cube containing 2D array of wind directions as angles from
                true north
            adj (numpy.ndarray):
                2D array of wind direction angle adjustments in radians, to
                convert zero reference from true north to grid north
        Returns:
            iris.cube.Cube:
                Cube containing wind vector component in the positive
                x-direction
            iris.cube.Cube:
                Cube containing wind vector component in the positive
                y-direction
        """
        angle.convert_units('radians')
        angle.data += adj

        # vector should be pointing "to" not "from"
        if angle.name() == "wind_from_direction":
            angle.data += np.pi
            angle.data = np.where(angle.data < 2.*np.pi, angle.data,
                                  angle.data - 2.*np.pi)
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
                Cube containing wind direction values

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

        x_coord = wind_speed.coord(axis='x').name()
        y_coord = wind_speed.coord(axis='y').name()

        # calculate angle adjustments for wind direction
        adj = self.calc_true_north_offset(
            next(wind_dir.slices([y_coord, x_coord])))

        # slice over x and y
        speed_slices = wind_speed.slices([y_coord, x_coord])
        dir_slices = wind_dir.slices([y_coord, x_coord])

        # adjust directions and resolve components onto grid axes
        uvcubelist = [self.resolve_wind_components(speed, angle, adj)
                      for speed, angle in zip(speed_slices, dir_slices)]
        uvcubelist = np.array(uvcubelist).T.tolist()

        # merge cubelists
        ucube = iris.cube.CubeList(uvcubelist[0]).merge_cube()
        vcube = iris.cube.CubeList(uvcubelist[1]).merge_cube()

        # relabel final cubes with CF compliant data names corresponding to
        # positive wind speeds along the x and y axes
        ucube.rename("grid_eastward_wind")
        vcube.rename("grid_northward_wind")

        return ucube, vcube
