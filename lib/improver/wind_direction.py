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
from iris.coords import CellMethod
import numpy as np
from improver.utilities.cube_checker import check_cube_coordinates
from improver.utilities.cube_manipulation import enforce_float32_precision


class WindDirection(object):
    """Plugin to calculate average wind direction from ensemble realizations.

    Science background:
    Taking an average wind direction is tricky since an average of two wind
    directions at 10 and 350 degrees is 180 when it should be 0 degrees.
    Converting the wind direction angles to complex numbers allows us to
    find a useful numerical average. ::

        z = a + bi
        a = r*Cos(theta)
        b = r*Sin(theta)
        r = radius

    The average of two complex numbers is NOT the ANGLE between two points
    it is the MIDPOINT in cartesian space.
    Therefore if there are two data points with radius=1 at 90 and 270 degrees
    then the midpoint is at (0,0) with radius=0 and therefore its average angle
    is meaningless. ::

                   N
                   |
        W---x------o------x---E
                   |
                   S

    In the rare case that a meaningless complex average is calculated, the
    code rejects the calculated complex average and simply uses the wind
    direction taken from the first ensemble realization.

    The steps are:

    1) Take data from all ensemble realizations.
    2) Convert the wind direction angles to complex numbers.
    3) Find complex average and their radius values.
    4) Convert the complex average back into degrees.
    5) If any point has an radius of nearly zero - replace the
       calculated average with the wind direction from the first ensemble.
    6) Calculate the confidence measure of the wind direction.

    Step 6 still needs more development so it is only included in the code
    as a placeholder.

    """

    def __init__(self):
        """Initialise class."""
        pass

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        desc = ('<WindDirection>')
        return desc

    @staticmethod
    def deg_to_complex(angle_deg, radius=1):
        """Converts degrees to complex values.

        The radius value can be used to weigh values - but it is set
        to 1 for now.

        Args:
            angle_deg (np.ndarray or float):
                3D array or float - wind direction angles in degrees.

        Keyword Args:
            radius (np.ndarray):
                3D array or float - radius value for each point, default=1.

        Returns:
            (np.ndarray or float):
                3D array or float - wind direction translated to
                complex numbers.

        """

        # Convert from degrees to radians.
        angle_rad = np.deg2rad(angle_deg)

        # Derive real and imaginary components (also known as a and b)
        real = radius*np.cos(angle_rad)
        imag = radius*np.sin(angle_rad)

        # Combine components into a complex number and return.
        return real + 1j*imag

    @staticmethod
    def complex_to_deg(complex_in):
        """Converts complex to degrees.

        The "np.angle" function returns negative numbers when the input
        is greater than 180. Therefore additional processing is needed
        to ensure that the angle is between 0-359.

        Args:
            complex_in (np.ndarray):
                3D array - wind direction angles in
                complex number form.

        Returns:
            angle (np.ndarray):
                3D array - wind direction in angle form

        Raises
        ------
        TypeError: If complex_in is not an array.

        """

        if not isinstance(complex_in, np.ndarray):
            msg = "Input data is not a numpy array, but {}"
            raise TypeError(msg.format(type(complex_in)))

        angle = np.angle(complex_in, deg=True)

        # If angle negative value - add 360 degrees.
        angle = np.where(angle < 0, angle+360, angle)

        # If angle == 360 - set to zero degrees.
        # Due to floating point - need to round value before using
        # equal operator.
        round_angle = np.around(angle, 2)
        angle = np.where(round_angle == 360, 0.0, angle)

        return angle

    @staticmethod
    def find_r_values(complex_in):
        """Find radius values from complex numbers.

        Takes input wind direction in complex values and returns array
        containing r values using Pythagoras theorem.

        Args:
            complex_in (np.ndarray or float):
                3D array or float - wind direction angles in complex numbers.

        Returns:
            (np.ndarray or float):
                3D array or float - Radius values extracts from
                complex numbers.
        """

        return np.sqrt(np.square(complex_in.real)+np.square(complex_in.imag))

    @staticmethod
    def calc_confidence_measure(wind_dir_complex, wind_dir_deg_mean,
                                r_vals, r_thresh, realization_axis):
        """Find confidence measure of polar numbers.

        The average wind direction complex values represent the midpoint
        between the different values and so have r values between 0-1.

        1) From wind_dir_deg_mean - create a new set of complex values.
           Therefore they will have the same angle but r is fixed as r=1.
        2) Find the distance between the mean point and all the ensemble
           realization wind direction complex values.
        3) Find the average distance between the mean point and the wind
           direction values. Large average distance == low confidence.
        4) A confidence value that is between 1 for confident (small spread in
           ensemble realizations) and 0 for no-confidence. Set to 0 if r value
           is below threshold as any r value is regarded as meaningless.

        Args:
            wind_dir_complex (np.ndarray):
                3D array - wind direction angles in complex numbers.
            wind_dir_deg_mean (np.ndarray):
                3D array - average wind direction in angles.
            r_vals (np.ndarray):
                3D array - Radius taken from average complex wind direction
                angle.
            r_thresh (float):
                Any r value below threshold is regarded as meaningless.
            realization_axis (int):
                Axis over which to average the arrays over.

        Returns:
            dist_from_mean_norm (np.ndarray):
                3D array - The average distance from mean normalised - used
                as a confidence value.
        """

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
        dist_from_mean_avg = np.mean(dist_from_mean, axis=realization_axis)

        # If we have two points at opposite ends of the compass
        # (eg. 270 and 90), then their separation distance is 2.
        # Normalise the array using 2 as the maximum possible value.
        dist_from_mean_norm = 1 - dist_from_mean_avg*0.5

        # With two points directly opposite (270 and 90) it returns a
        # confidence value of 0.29289322 instead of zero due to precision
        # error.
        #
        # angles | confidence
        # 270/90 | 0.29289322
        # 270/89 | 0.295985
        # 270/88 | 0.299091
        # 270/87 | 0.30221
        # Therefore any confidence value where the r is less than the threshold
        # should be set to zero.
        dist_from_mean_norm = np.where(r_vals < r_thresh,
                                       0.0, dist_from_mean_norm)
        return dist_from_mean_norm

    @staticmethod
    def wind_dir_decider(wind_dir_deg, wind_dir_deg_mean, r_vals, r_thresh):
        """If the wind direction is so widely scattered that the r value
           is nearly zero then this indicates that the average wind direction
           is essentially meaningless.
           We therefore substitute this meaningless average wind
           direction value for the wind direction taken from the first
           ensemble realization.

        Args:
            wind_dir_deg (np.ndarray):
                3D array - wind direction angles from ensembles (in degrees).
            wind_dir_deg_mean (np.ndarray):
                2D array - average wind direction angle (in degrees).
            r_vals (np.ndarray):
                2D array - Radius taken from average complex wind direction
                angle.
            r_thresh (float):
                Any r value below threshold is regarded as meaningless.

        Returns:
            reprocessed_wind_dir_mean (np.ndarray):
                3D array - Wind direction degrees where ambigious values have
                been replaced with data from first ensemble realization.
        """

        # Mask True if r values below threshold.
        where_low_r = np.where(r_vals < r_thresh, True, False)

        # If the whole array contains good r-values, return origonal array.
        if not where_low_r.any():
            return wind_dir_deg_mean

        # Takes first ensemble realization.
        first_realization = wind_dir_deg[0]

        # If the r-value is low - subistite average wind direction value for
        # the wind direction taken from the first ensemble realization.
        reprocessed_wind_dir_mean = np.where(where_low_r, first_realization,
                                             wind_dir_deg_mean)

        return reprocessed_wind_dir_mean

    @staticmethod
    def process(cube_ens_wdir):
        """Create a cube containing the wind direction averaged over the
        ensemble realizations.

        Args:
            cube_ens_wdir (iris.cube.Cube):
                Cube containing wind direction from multiple ensemble
                realizations.

        Returns:
            cube_mean_wdir (iris.cube.Cube):
                Cube containing the wind direction averaged from the
                ensemble realizations.
            cube_r_vals (np.ndarray):
                3D array - Radius taken from average complex wind direction
                angle.
            cube_confidence_measure (np.ndarray):
                3D array - The average distance from mean normalised - used
                as a confidence value.

        Raises
        ------
        TypeError: If cube_wdir is not a cube.

        """

        # Any points where the r-values are below the threshold is regarded as
        # containing ambigous data.
        r_thresh = 0.01

        if not isinstance(cube_ens_wdir, iris.cube.Cube):
            msg = "Wind direction input is not a cube, but {}"
            raise TypeError(msg.format(type(cube_ens_wdir)))

        try:
            cube_ens_wdir.convert_units("degrees")
        except ValueError as err:
            msg = "Input cube cannot be converted to degrees: {}".format(err)
            raise ValueError(msg)

        # Force input cube to float32.
        enforce_float32_precision(cube_ens_wdir)

        # Creates cubelists to hold data.
        wdir_cube_list = iris.cube.CubeList()
        r_vals_cube_list = iris.cube.CubeList()
        confidence_measure_cube_list = iris.cube.CubeList()

        y_coord_name = cube_ens_wdir.coord(axis="y").name()
        x_coord_name = cube_ens_wdir.coord(axis="x").name()
        for slice_ens_wdir in cube_ens_wdir.slices(["realization",
                                                    y_coord_name,
                                                    x_coord_name]):
            # Extract wind direction data.
            wind_dir_deg = slice_ens_wdir.data
            realization_axis = slice_ens_wdir.coord_dims("realization")[0]

            # Copies input cube and remove realization dimension to create
            # cubes for storing results.
            slice_mean_wdir = next(slice_ens_wdir.slices_over("realization"))
            slice_mean_wdir.remove_coord("realization")

            # Convert wind direction from degrees to complex numbers.
            wind_dir_complex = WindDirection.deg_to_complex(wind_dir_deg)

            # Find the complex average -  which actually signifies a point
            # between all of the data points in POLAR coordinates.
            # NOT the average DEGREE ANGLE.
            wind_dir_complex_mean = np.mean(wind_dir_complex,
                                            axis=realization_axis)

            # Convert complex average values to degrees to produce average
            # wind direction.
            wind_dir_deg_mean = WindDirection.complex_to_deg(
                wind_dir_complex_mean)

            # Find radius values for wind direction average.
            r_vals = WindDirection.find_r_values(wind_dir_complex_mean)

            # Calculate the confidence measure based on the difference
            # between the complex average and the individual ensemble
            # realizations.
            # TODO: This will still need some further investigation.
            #        This is will be the subject of another ticket.
            confidence_measure = WindDirection.calc_confidence_measure(
                wind_dir_complex, wind_dir_deg_mean, r_vals, r_thresh,
                realization_axis)

            # Finds any meaningless averages and substitute with
            # the wind direction taken from the first ensemble realization.
            wind_dir_deg_mean = WindDirection.wind_dir_decider(
                wind_dir_deg, wind_dir_deg_mean, r_vals, r_thresh)

            # Save data into cubes (create new cubes for r and
            # confidence measure data).
            slice_mean_wdir.data = wind_dir_deg_mean
            slice_r_vals = slice_mean_wdir.copy(data=r_vals)
            slice_confidence_measure = (
                slice_mean_wdir.copy(data=confidence_measure))
            # Append to cubelists.
            wdir_cube_list.append(slice_mean_wdir)
            r_vals_cube_list.append(slice_r_vals)
            confidence_measure_cube_list.append(slice_confidence_measure)

        # Combine cubelists into cube.
        cube_mean_wdir = wdir_cube_list.merge_cube()
        cube_r_vals = r_vals_cube_list.merge_cube()
        cube_confidence_measure = confidence_measure_cube_list.merge_cube()

        # Check that the dimensionality of coordinates of the output cube
        # matches the input cube.
        first_slice = next(cube_ens_wdir.slices_over(["realization"]))
        cube_mean_wdir = check_cube_coordinates(first_slice, cube_mean_wdir)

        # Change cube identifiers.
        cube_mean_wdir.add_cell_method(CellMethod("mean",
                                                  coords="realization"))
        cube_r_vals.long_name = "radius_of_complex_average_wind_from_direction"
        cube_r_vals.units = None
        cube_confidence_measure.long_name = (
            "confidence_measure_of_wind_from_direction")
        cube_confidence_measure.units = None

        return cube_mean_wdir, cube_r_vals, cube_confidence_measure
