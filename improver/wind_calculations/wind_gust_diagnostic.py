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
"""Module containing plugin for WindGustDiagnostic."""

import warnings

import iris
import numpy as np

from improver import PostProcessingPlugin
from improver.metadata.probabilistic import find_percentile_coordinate


class WindGustDiagnostic(PostProcessingPlugin):

    """Plugin for calculating wind-gust diagnostic.

    In the model a shear-driven turbulence parameterization is used to
    estimate wind gusts but in convective situations this can over-estimate the
    convective gust.
    This diagnostic takes the Maximum of the values at each grid point of
    * a chosen percentile of the wind-gust forecast and
    * a chosen percentile of the wind-speed forecast
    to produce a better estimate of wind-gust.
    For example a typical wind-gust could be MAX(gust(50%),windspeed(95%))
    an extreme wind-gust forecast could be MAX(gust(95%), windspeed(100%))

    Scientific Reference: *Roberts N., Mylne K.*
    Poster - European Meteorological Society Conference 2017.

    See
    https://github.com/metoppv/improver/files/1244828/WindGustChallenge_v2.pdf
    for a discussion of the problem and proposed solutions.

    """

    def __init__(self, percentile_gust, percentile_windspeed):
        """
        Create a WindGustDiagnostic plugin for a given set of percentiles.

        Args:
            percentile_gust (float):
                Percentile value required from wind-gust cube.
            percentile_windspeed (float):
                Percentile value required from wind-speed cube.

        """
        self.percentile_gust = percentile_gust
        self.percentile_windspeed = percentile_windspeed

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        desc = ('<WindGustDiagnostic: wind-gust perc='
                '{0:3.1f}, wind-speed perc={1:3.1f}>'
                .format(self.percentile_gust, self.percentile_windspeed))
        return desc

    def add_metadata(self, cube):
        """Add metadata to cube for windgust diagnostic.

        Args:
            cube (iris.cube.Cube):
                Cube containing the wind-gust diagnostic data.
        Returns:
            iris.cube.Cube:
                Cube containing the wind-gust diagnostic data with
                corrected Metadata.

        """
        result = cube
        result.rename("wind_speed_of_gust")
        if self.percentile_gust == 50.0 and self.percentile_windspeed == 95.0:
            diagnostic_txt = 'Typical gusts'
        elif (self.percentile_gust == 95.0 and
              self.percentile_windspeed == 100.0):
            diagnostic_txt = 'Extreme gusts'
        else:
            diagnostic_txt = str(self)
        result.attributes.update({'wind_gust_diagnostic': diagnostic_txt})

        return result

    @staticmethod
    def extract_percentile_data(cube, req_percentile, standard_name):
        """Extract percentile data from cube.

        Args:
            cube (iris.cube.Cube):
                Cube contain one or more percentiles.
            req_percentile (float):
                Required percentile value
            standard_name (str):
                Standard name of the data.

        Returns:
            (tuple): tuple containing:
                **result** (iris.cube.Cube):
                    Cube containing the required percentile data
                **perc_coord** (iris.coords.Coord):
                    Percentile coordinate.

        """
        if not isinstance(cube, iris.cube.Cube):
            msg = ('Expecting {0:s} data to be an instance of '
                   'iris.cube.Cube but is'
                   ' {1}.'.format(standard_name, type(cube)))
            raise TypeError(msg)
        perc_coord = find_percentile_coordinate(cube)
        if cube.standard_name != standard_name:
            msg = ('Warning mismatching name for data expecting'
                   ' {0:s} but found {1:s}'.format(standard_name,
                                                   cube.standard_name))
            warnings.warn(msg)
        constraint = (
            iris.Constraint(coord_values={perc_coord.name(): req_percentile}))
        result = cube.extract(constraint)
        if result is None:
            msg = ('Could not find required percentile '
                   '{0:3.1f} in cube'.format(req_percentile))
            raise ValueError(msg)
        return result, perc_coord

    def process(self, cube_gust, cube_ws):
        """
        Create a cube containing the wind_gust diagnostic.

        Args:
            cube_gust (iris.cube.Cube):
                Cube contain one or more percentiles of wind_gust data.
            cube_ws (iris.cube.Cube):
                Cube contain one or more percentiles of wind_speed data.

        Returns:
            iris.cube.Cube:
                Cube containing the wind-gust diagnostic data.

        """

        # Extract wind-gust data
        (req_cube_gust,
         perc_coord_gust) = self.extract_percentile_data(cube_gust,
                                                         self.percentile_gust,
                                                         "wind_speed_of_gust")
        # Extract wind-speed data
        (req_cube_ws,
         perc_coord_ws) = (
             self.extract_percentile_data(cube_ws,
                                          self.percentile_windspeed,
                                          "wind_speed"))
        if perc_coord_gust.name() != perc_coord_ws.name():
            msg = ('Percentile coord of wind-gust data'
                   'does not match coord of wind-speed data'
                   ' {0:s} {1:s}.'.format(perc_coord_gust.name(),
                                          perc_coord_ws.name()))
            raise ValueError(msg)

        # Check times are compatible.
        msg = ('Could not match time coordinate')
        wg_time = req_cube_gust.coords('time')
        ws_time = req_cube_ws.coords('time')
        if len(wg_time) == 0 or len(ws_time) == 0:
            raise ValueError(msg)

        if not all(wg_point == ws_point for wg_point, ws_point
                   in zip(wg_time[0].points, ws_time[0].points)):
            if wg_time[0].bounds is None:
                raise ValueError(msg)
            if not all((point >= bounds[0] and point <= bounds[1])
                       for point, bounds in zip(ws_time[0].points,
                                                wg_time[0].bounds)):
                raise ValueError(msg)

        # Add metadata to gust cube
        req_cube_gust = self.add_metadata(req_cube_gust)

        # Calculate wind-gust diagnostic
        result = req_cube_gust.copy(
            data=np.maximum(req_cube_gust.data, req_cube_ws.data))

        # Update metadata
        result.remove_coord(perc_coord_gust.name())
        return result
