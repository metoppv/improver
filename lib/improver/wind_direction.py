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
"""Module containing wind direction averaging plugins."""

import iris
import numpy as np
from improver.utilities.cube_manipulation import enforce_float32_precision


class WindDirection(object):
    """Plugin to calculate average wind direction from ensemble members.

    This holds the function to calculate the average wind direction
    from ensemble members using a complex numbers approach.


    """

    def __init__(self):
        """
        Initialise class.
        """
        pass

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        desc = ('<WindDirection>')
        return desc

    @staticmethod
    def deg_to_complex(angle_deg, r=1):
        """Takes an array or float containing an angle in degrees
           and returns a complex equivalent.
           The r value can be used to weigh values - but it is set
           to 1 for now. """

        # Convert from degrees to radians.
        angle_rad = np.deg2rad(angle_deg)

        # Derive real and imaginary components.
        a = r*np.cos(angle_rad)
        b = r*np.sin(angle_rad)

        # Combine components into a complex number and return.
        return a + 1j*b

    @staticmethod
    def complex_to_deg(complex_in):
        """Take complex number and find angle in degrees from 360.
           the "np.angle" function returns negative numbers when the input
           is greater than 180. Therefore additional processing is needed
           to ensure that the angle is between 0-359."""

        angle = np.angle(complex_in, deg=True)

        # If the input is an array then need to use native np.functions
        # for speed rather than applying vector functions to the array.
        if type(angle) is np.ndarray:

            # If angle negative value - add 360 degrees.
            angle = np.where(angle < 0, angle+360, angle)

            # If angle == 360 - set to zero degrees.
            # Due to floating point - need to round value before using
            # equal operator.
            round_angle = np.around(angle, 2)
            angle = np.where(round_angle == 360, 0.0, angle)
        else:
            if angle < 0:
                angle = angle + 360.0
            if round(angle, 2) == 360.00:
                angle = 0.0
        return angle

    @staticmethod
    def find_r_values(complex_in):
        """Takes input wind direction in complex values and returns array
           containing r values using Pythagoras theorem."""

        return np.sqrt(np.square(complex_in.real)+np.square(complex_in.imag))

    @staticmethod
    def calc_polar_mean_std_dev(wind_dir_complex, wind_dir_deg_mean):
        """The averaged wind direction complex values have r values
           between 0-1. So we need to recalculate the complex values so they
           retain their angle but r is fixed as r=1. They can then be used to
           find the distance between the average angle and the other
           wind direction data points. From this information - we can calculate
           a confidence value."""

        wind_dir_complex_mean_fixr = WindDirection.deg_to_complex(
                                                             wind_dir_deg_mean)

        # Find difference in the distance between all the observed points and
        # mean point with fixed r=1.
        # For maths to work - the "wind_dir_complex_mean_fixr array" needs to
        # be "tiled" so that it is the same dimension as "wind_dir_complex".
        wind_dir_complex_mean_tile = np.tile(wind_dir_complex_mean_fixr,
                                             (wind_dir_complex.shape[0], 1, 1))

        # Calculate distance from each wind direction data point to the
        # average point.
        difference = wind_dir_complex - wind_dir_complex_mean_tile
        dist_from_mean = np.sqrt(np.square(difference.real) +
                                 np.square(difference.imag))

        # Find average distance.
        dist_from_mean_avg = np.mean(dist_from_mean, axis=0)

        # If we have two points at opposite ends of the compass
        # (eg. 270 and 90), then their separation distance is 2.
        # Normalise the array using 2 as the maximum possible value.
        dist_from_mean_norm = 1 - dist_from_mean_avg*0.5

        return dist_from_mean_norm

    @staticmethod
    def wind_dir_decider(wind_dir_deg, wind_dir_deg_mean, r_vals):
        """If the wind direction is so widely scattered that the r value
           is nearly zero then this indicates that the average wind direction
           is essentially meaningless.
           We therefore subistuite this meaningless average wind
           direction value for the wind direction taken from the first
           ensemble member."""

        # Threshold r value.
        r_thresh = 0.01

        # Mask True if r values below threshold.
        where_low_r = np.where(r_vals < r_thresh, True, False)

        # If the whole array contains good r-values, return origonal array.
        if not where_low_r.any():
            return wind_dir_deg_mean

        # Takes first ensemble member.
        first_member = wind_dir_deg[0]

        # If the r-value is low - subistite average wind direction value for
        # the wind direction taken from the first ensemble member.
        reprocessed_wind_dir_mean = np.where(where_low_r, first_member,
                                             wind_dir_deg_mean)

        return reprocessed_wind_dir_mean

    def process(self, cube_ens_wdir):
        """
        Create a cube containing the wind direction averaged over the ensemble
        members.

        Args:
            cube_wdir (iris.cube.Cube):
                Cube containing wind direction from multiple ensemble members.

        Returns:
            result (iris.cube.Cube):
                Cube containing the wind direction averaged from the
                ensemble members.

        Raises
        ------
        TypeError: If cube_wdir is not a cube.

        """

        if not isinstance(cube_ens_wdir, iris.cube.Cube):
            msg = "Wind direction input is not a cube, but {}"
            raise TypeError(msg.format(type(cube_ens_wdir)))

        # Force input cube to float32.
        enforce_float32_precision(cube_ens_wdir)

        # Creates result cube - copies input and removes realization dimension.
        for cube_mean_wdir in cube_ens_wdir.slices_over("realization"):
            cube_mean_wdir.remove_coord('realization')
            break

        y_coord_name = cube_ens_wdir.coord(axis="y").name()
        x_coord_name = cube_ens_wdir.coord(axis="x").name()
        wind_dir_deg = next(cube_ens_wdir.slices(["realization",
                                                  y_coord_name,
                                                  x_coord_name])).data

        # Convert wind direction from degrees to complex numbers.
        wind_dir_complex = WindDirection.deg_to_complex(wind_dir_deg)

        # Find the complex average - which actually signifies a point between
        # all of the data points in POLAR co-cordinates (not degrees).
        wind_dir_complex_mean = np.mean(wind_dir_complex, axis=0)

        # Convert complex average to degrees to produce average wind direction.
        wind_dir_deg_mean = WindDirection.complex_to_deg(wind_dir_complex_mean)

        # Find r values for wind direction average.
        r_vals = WindDirection.find_r_values(wind_dir_complex_mean)

        # Calculate standard deviation from polar mean.
        polar_mean_std_dev = WindDirection.calc_polar_mean_std_dev(
                                                             wind_dir_complex,
                                                             wind_dir_deg_mean)

        # Finds any meaningless averages and subistuite with
        # the wind direction taken from the first ensemble member.
        wind_dir_deg_mean = WindDirection.wind_dir_decider(wind_dir_deg,
                                                           wind_dir_deg_mean,
                                                           r_vals)

        cube_mean_wdir.data[0] = wind_dir_deg_mean
        return cube_mean_wdir, r_vals, polar_mean_std_dev
