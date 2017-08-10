# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017 Met Office.
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
"""Module containing neighbourhood processing utilities."""

import copy
import math

import iris
from iris.exceptions import CoordinateNotFoundError
import numpy as np


# Maximum radius of the neighbourhood width in grid cells.
MAX_RADIUS_IN_GRID_CELLS = 500


class Utilities(object):

    """
    Utilities for neighbourhood processing.
    """

    def __init__(self):
        """
        Initialise class.
        """
        pass

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        result = ('<Utilities>')
        return result

    @staticmethod
    def find_required_lead_times(cube):
        """
        Determine the lead times within a cube, either by reading the
        forecast_period coordinate, or by calculating the difference between
        the time and the forecast_reference_time. If the forecast_period
        coordinate is present, the points are assumed to represent the
        desired lead times with the bounds not being considered. The units of
        the forecast_period, time and forecast_reference_time coordinates are
        converted, if required.

        Parameters
        ----------
        cube : Iris.cube.Cube
            Cube from which the lead times will be determined.

        Returns
        -------
        required_lead_times : Numpy array
            Array containing the lead times, at which the radii need to be
            calculated.

        """
        if cube.coords("forecast_period"):
            try:
                cube.coord("forecast_period").convert_units("hours")
            except ValueError as err:
                msg = "For forecast_period: {}".format(err)
                raise ValueError(msg)
            required_lead_times = cube.coord("forecast_period").points
        else:
            if cube.coords("time") and cube.coords("forecast_reference_time"):
                try:
                    cube.coord("time").convert_units(
                        "hours since 1970-01-01 00:00:00")
                    cube.coord("forecast_reference_time").convert_units(
                        "hours since 1970-01-01 00:00:00")
                except ValueError as err:
                    msg = "For time/forecast_reference_time: {}".format(err)
                    raise ValueError(msg)
                required_lead_times = (
                    cube.coord("time").points -
                    cube.coord("forecast_reference_time").points)
            else:
                msg = ("The forecast period coordinate is not available "
                       "within {}."
                       "The time coordinate and forecast_reference_time "
                       "coordinate were also not available for calculating "
                       "the forecast_period.".format(cube))
                raise CoordinateNotFoundError(msg)
        return required_lead_times

    @staticmethod
    def adjust_nsize_for_ens(ens_factor, num_ens, width):
        """
        Adjust neighbourhood size according to ensemble size.

        Parameters
        ----------
        ens_factor : float
            The factor with which to adjust the neighbourhood size
            for more than one ensemble member.
            If ens_factor = 1.0 this essentially conserves ensemble
            members if every grid square is considered to be the
            equivalent of an ensemble member.
        num_ens : float
            Number of realizations or ensemble members.
        width : float
            radius or width appropriate for a single forecast in m.

        Returns
        -------
        new_width : float
            new neighbourhood radius (m).

        """
        if num_ens <= 1.0:
            new_width = width
        else:
            new_width = (ens_factor *
                         math.sqrt((width**2.0)/num_ens))
        return new_width

    @staticmethod
    def check_cube_coordinates(cube, new_cube):
        """
        Find and promote to dimension coordinates any scalar coordinates in
        new_cube that were originally dimension coordinates in the progenitor
        cube. If coordinate is in new_cube that is not in the old cube, keep
        coordinate in its current position.

        Parameters
        ----------
        cube : iris.cube.Cube
            The input cube provided to nbhood.
        new_cube : iris.cube.Cube
            The cube produced by the neighbourhooding process that must be
            checked for demoted dimensional coordinates.

        Returns
        -------
        new_cube : iris.cube.Cube
            Modified neighbourhooded cube with relevant scalar coordinates
            promoted to dimension coordinates.

        Raises
        ------
        iris.exceptions.CoordinateNotFoundError raised if the final dimension
        coordinates of the returned cube do not match the input cube.

        """
        # Promote available and relevant scalar coordinates
        for coord in new_cube.aux_coords[::-1]:
            if coord in cube.dim_coords:
                new_cube = iris.util.new_axis(new_cube, coord)

        # Ensure dimension order matches; if lengths are unequal a coordinate
        # is missing, so raise an appropriate error.
        cube_dimension_order = {coord.name(): cube.coord_dims(coord.name())[0]
                                for coord in cube.dim_coords}
        print "cube = ", cube
        print "cube_dimension_order = ", cube_dimension_order
        print "new_cube = ", new_cube
        correct_order = []
        for coord in new_cube.dim_coords:
            if coord in cube.dim_coords:
                correct_order.append(cube_dimension_order[coord.name()])
            else:
                new_coord_dim = new_cube.coord_dims(coord.name())[0]
                print "new_coord_dim = ", new_coord_dim
                if len(correct_order)>0:
                    correct_order[correct_order>new_coord_dim] += 1
                else:
                    new_coord_dim = -1
                correct_order.append(new_coord_dim)
            print "correct_order = ", correct_order

        #correct_order = [cube_dimension_order[coord.name()]
                         #for coord in new_cube.dim_coords]
        print "correct_order = ", correct_order
        if len(cube_dimension_order) == len(correct_order):
            new_cube.transpose(correct_order)
        else:
            msg = ('Returned cube dimension coordinates do not match input '
                   'cube dimension coordinates. \n input cube shape {} '
                   ' returned cube shape {}'.format(
                    cube.shape, new_cube.shape))
            raise iris.exceptions.CoordinateNotFoundError(msg)

        return new_cube

    @staticmethod
    def check_if_grid_is_equal_area(cube):
        """
        Identify whether the grid is an equal area grid.
        If not, raise an error.

        Parameters
        ----------
        cube : Iris.cube.Cube
            Cube with coordinates that will be checked.

        Raises
        ------
        ValueError : Invalid grid: projection_x/y coords required
        ValueError : Intervals between points along the x and y axis vary.
                     Therefore the grid is not an equal area grid.

        """

        try:
            for coord_name in ['projection_x_coordinate',
                               'projection_y_coordinate']:
                cube.coord(coord_name)
        except CoordinateNotFoundError:
            raise ValueError("Invalid grid: projection_x/y coords required")
        for coord_name in ['projection_x_coordinate',
                           'projection_y_coordinate']:
            if np.sum(np.diff(np.diff(cube.coord(coord_name).points))) > 0:
                msg = ("Intervals between points along the x and y axis vary."
                       "Therefore the grid is not an equal area grid.")
                raise ValueError(msg)
