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
"""Module containing plugin for WindGustDiagnositic."""

import iris
from iris.exceptions import CoordinateNotFoundError
from iris import FUTURE

FUTURE.netcdf_promote = True


class WindGustDiagnostic(object):

    """Plugin for calculating wind-gust diagnostic.

    In the model a shear-driven turbulence parameterization is used to
    estimate wind gusts but in convective situations this can over-estimate the
    convective gust.
    This diagnostic takes the Maximum of the values at eachgrid point of
        a chosen percentile of the wind-gust forecast and
        a chosen percentile of the wind-speed forecast
    to produce a better estimate of wind-gust.
    For example a typical wind-gust could be MAX(gust(50%),windspeed(95%))
    an extreme wind-gust forecast could be MAX(gust(95%), windspeed(100%))

    See
    https://github.com/metoppv/improver/files/1244828/WindGustChallenge_v2.pdf
    for a discussion of the problem and proposed solutions.
    """

    def __init__(self, percentile_gust, percentile_windspeed):
        """
        Create a WindGustDiagnostic plugin for a given set of percentiles.

        Args:
            percentile_gust: float
                Percentile value required from wind-gust cube.
            percentile_windspeed: float
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
            cube: iris.cube.Cube instance
                Cube containing the wind-gust diagnostic data.
        Returns:
            result : iris.cube.Cube instance
                Cube containing the wind-gust diagnostic data with
                corrected Metadata.

        """
        result = cube
        result.standard_name = "wind_speed_of_gust"
        result.long_name = "wind_gust_diagnostic"
        if self.percentile_gust == 50.0 and self.percentile_windspeed == 95.0:
            diagnostic_txt = 'Typical gusts'
        elif (self.percentile_gust == 95.0 and
              self.percentile_windspeed == 100.0):
            diagnostic_txt = 'Extreme gusts'
        else:
            diagnostic_txt = str(self)
        result.attributes.update({'wind_gust_diagnostic': diagnostic_txt})

        return result

    def process(self, cube_gust, cube_ws):
        """
        Create a cube containing the wind_gust diagnostic.

        Args:
            cube_gust : iris.cube.Cube instance
                Cube contain one or more percentiles of wind_gust data.
            cube_ws : iris.cube.Cube instance
                Cube contain one or more percentiles of wind_speed data.

        Returns:
            result : iris.cube.Cube instance
                Cube containing the wind-gust diagnostic data.

        """

        # Extract wind-gust data
        # raise CoordinateNotFoundError(
        #    "Coordinate '{}' not found in cube passed to {}.".format(
        #        self.collapse_coord, self.__class__.__name__))

        # Extract wind-speed data

        # Calculate wind-gust diagnostic
        # result = cube.collapsed(perc_coord, iris.analysis.MAX)
        # Update metadata
        result = self.add_metadata(cube_gust)
        return result
