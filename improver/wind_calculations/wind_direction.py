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
"""Module containing wind direction averaging plugins."""

import iris
import numpy as np
from iris.coords import CellMethod

from improver import BasePlugin
from improver.nbhood.nbhood import NeighbourhoodProcessing
from improver.utilities.cube_checker import check_cube_coordinates


class WindDirection(BasePlugin):
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

    Args:
        backup_method (str):
            Backup method to use if the complex numbers approach has low
            confidence.
            "first_realization" uses the value of realization zero.
            "neighbourhood" (default) recalculates using the complex numbers
            approach with additional realizations extracted from neighbouring
            grid points from all available realizations.

    """

    def __init__(self, backup_method='neighbourhood'):
        """Initialise class."""
        self.backup_methods = ['first_realization', 'neighbourhood']
        self.backup_method = backup_method
        if self.backup_method not in self.backup_methods:
            msg = ('Invalid option for keyword backup_method '
                   '({})'.format(self.backup_method))
            raise ValueError(msg)

        # Any points where the r-values are below the threshold is regarded as
        # containing ambigous data.
        self.r_thresh = 0.01

        # Creates cubelists to hold data.
        self.wdir_cube_list = iris.cube.CubeList()
        self.r_vals_cube_list = iris.cube.CubeList()
        self.confidence_measure_cube_list = iris.cube.CubeList()
        # Radius used in neighbourhood plugin as determined in IMPRO-491
        self.nb_radius = 6000.  # metres
        # Initialise neighbourhood plugin ready for use
        self.nbhood = NeighbourhoodProcessing('square',
                                              self.nb_radius,
                                              weighted_mode=False)

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        return (
            '<WindDirection: backup_method "{}"; neighbourhood radius "{}"m>'
        ).format(self.backup_method, self.nb_radius)

    def _reset(self):
        """Empties working data objects"""
        self.realization_axis = None
        self.wdir_complex = None
        self.wdir_slice_mean = None
        self.wdir_mean_complex = None
        self.r_vals_slice = None
        self.confidence_slice = None

    @staticmethod
    def deg_to_complex(angle_deg, radius=1):
        """Converts degrees to complex values.

        The radius value can be used to weigh values - but it is set
        to 1 for now.

        Args:
            angle_deg (numpy.ndarray or float):
                3D array or float - wind direction angles in degrees.
            radius (numpy.ndarray):
                3D array or float - radius value for each point, default=1.

        Returns:
            numpy.ndarray or float:
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
            complex_in (numpy.ndarray):
                3D array - wind direction angles in
                complex number form.

        Returns:
            numpy.ndarray:
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

        # We don't need 64 bit precision.
        angle = angle.astype(np.float32)

        return angle

    def calc_wind_dir_mean(self):
        """Find the mean wind direction using complex average which actually
           signifies a point between all of the data points in POLAR
           coordinates - NOT the average DEGREE ANGLE.

        Uses:
            self.wdir_complex (numpy.ndarray or float):
                3D array or float - wind direction angles in degrees.
            self.realization_axis (int):
                Axis to collapse over.

        Defines:
            self.wdir_mean_complex (numpy.ndarray or float):
                3D array or float - wind direction angles as complex numbers
                collapsed along an axis using np.mean().
            self.wdir_slice_mean (numpy.ndarray or float):
                3D array or float - wind direction angles in degrees collapsed
                along an axis using np.mean().
        """
        self.wdir_mean_complex = np.mean(self.wdir_complex,
                                         axis=self.realization_axis)
        self.wdir_slice_mean.data = self.complex_to_deg(self.wdir_mean_complex)

    def find_r_values(self):
        """Find radius values from complex numbers.

        Takes input wind direction in complex values and returns array
        containing r values using Pythagoras theorem.

        Uses:
            self.wdir_mean_complex (numpy.ndarray or float):
                3D array or float - wind direction angles in complex numbers.
            self.wdir_slice_mean (iris.cube.Cube):
                3D array or float - mean wind direction angles in complex
                numbers.

        Defines:
            self.r_vals_slice (iris.cube.Cube):
                Contains r values and inherits meta-data from
                self.wdir_slice_mean.
        """

        r_vals = (np.sqrt(np.square(self.wdir_mean_complex.real) +
                          np.square(self.wdir_mean_complex.imag)))
        self.r_vals_slice = self.wdir_slice_mean.copy(data=r_vals)

    def calc_confidence_measure(self):
        """Find confidence measure of polar numbers.

        The average wind direction complex values represent the midpoint
        between the different values and so have r values between 0-1.

        1) From self.wdir_slice_mean - create a new set of complex values.
           Therefore they will have the same angle but r is fixed as r=1.
        2) Find the distance between the mean point and all the ensemble
           realization wind direction complex values.
        3) Find the average distance between the mean point and the wind
           direction values. Large average distance == low confidence.
        4) A confidence value that is between 1 for confident (small spread in
           ensemble realizations) and 0 for no-confidence. Set to 0 if r value
           is below threshold as any r value is regarded as meaningless.

        Uses:
            self.wdir_complex (numpy.ndarray):
                3D array - wind direction angles in complex numbers.
            self.wdir_slice_mean (iris.cube.Cube):
                Contains average wind direction in angles.
            self.realization_axis (int):
                Axis to collapse over.
            self.r_vals_slice.data (numpy.ndarray):
                3D array - Radius taken from average complex wind direction
                angle.
            self.r_thresh (float):
                Any r value below threshold is regarded as meaningless.

        Defines:
            self.confidence_slice (iris.cube.Cube):
                Contains the average distance from mean normalised - used
                as a confidence value. Inherits meta-data from
                self.wdir_slice_mean
        """

        # Recalculate complex mean with radius=1.
        wdir_mean_complex_r1 = self.deg_to_complex(self.wdir_slice_mean.data)

        # Find difference in the distance between all the observed points and
        # mean point with fixed r=1.
        # For maths to work - the "wdir_mean_complex_r1 array" needs to
        # be "tiled" so that it is the same dimension as "self.wdir_complex".
        wind_dir_complex_mean_tile = np.tile(wdir_mean_complex_r1,
                                             (self.wdir_complex.shape[0],
                                              1, 1))

        # Calculate distance from each wind direction data point to the
        # average point.
        difference = self.wdir_complex - wind_dir_complex_mean_tile
        dist_from_mean = np.sqrt(np.square(difference.real) +
                                 np.square(difference.imag))

        # Find average distance.
        dist_from_mean_avg = np.mean(dist_from_mean,
                                     axis=self.realization_axis)

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
        dist_from_mean_norm = np.where(self.r_vals_slice.data < self.r_thresh,
                                       0.0, dist_from_mean_norm)
        self.confidence_slice = self.wdir_slice_mean.copy(
            data=dist_from_mean_norm)

    def wind_dir_decider(self, where_low_r, wdir_cube):
        """If the wind direction is so widely scattered that the r value
           is nearly zero then this indicates that the average wind direction
           is essentially meaningless.
           We therefore substitute this meaningless average wind
           direction value for the wind direction calculated from a larger
           sample by smoothing across a neighbourhood of points before
           rerunning the main technique.
           This is invoked rarely (1 : 100 000)

        Arguments:
            where_low_r (numpy.ndarray):
                Array of boolean values. True where original wind direction
                estimate has low confidence. These points are replaced
                according to self.backup_method
            wdir_cube (iris.cube.Cube):
                Contains array of wind direction data (realization, y, x)

        Uses:
            self.wdir_slice_mean (iris.cube.Cube):
                Containing average wind direction angle (in degrees).
            self.wdir_complex (numpy.ndarray):
                3D array - wind direction angles from ensembles (in complex).
            self.r_vals_slice.data (numpy.ndarray):
                2D array - Radius taken from average complex wind direction
                angle.
            self.r_thresh (float):
                Any r value below threshold is regarded as meaningless.
            self.realization_axis (int):
                Axis to collapse over.
            self.n_realizations (int):
                Number of realizations available in the plugin. Used to set the
                neighbourhood radius as this is used to adjust the radius again
                in the neighbourhooding plugin.

        Defines:
            self.wdir_slice_mean.data (numpy.ndarray):
                2D array - Wind direction degrees where ambigious values have
                been replaced with data from first ensemble realization.
        """
        if self.backup_method == 'neighbourhood':
            # Performs smoothing over a 6km square neighbourhood.
            # Then calculates the mean wind direction.
            child_class = WindDirection(backup_method="first_realization")
            child_class.wdir_complex = self.nbhood.process(
                wdir_cube.copy(data=self.wdir_complex)).data
            child_class.realization_axis = self.realization_axis
            child_class.wdir_slice_mean = self.wdir_slice_mean.copy()
            child_class.calc_wind_dir_mean()
            improved_values = child_class.wdir_slice_mean.data
        else:
            # Takes realization zero (control member).
            improved_values = wdir_cube.extract(
                iris.Constraint(realization=0)).data

        # If the r-value is low - substitute average wind direction value for
        # the wind direction taken from the first ensemble realization.
        self.wdir_slice_mean.data = np.where(where_low_r, improved_values,
                                             self.wdir_slice_mean.data)

    def process(self, cube_ens_wdir):
        """Create a cube containing the wind direction averaged over the
        ensemble realizations.

        Args:
            cube_ens_wdir (iris.cube.Cube):
                Cube containing wind direction from multiple ensemble
                realizations.

        Returns:
            iris.cube.Cube:
                Cube containing the wind direction averaged from the
                ensemble realizations.
            cube_r_vals (numpy.ndarray):
                3D array - Radius taken from average complex wind direction
                angle.
            cube_confidence_measure (numpy.ndarray):
                3D array - The average distance from mean normalised - used
                as a confidence value.

        Raises
        ------
        TypeError: If cube_wdir is not a cube.

        """

        if not isinstance(cube_ens_wdir, iris.cube.Cube):
            msg = "Wind direction input is not a cube, but {}"
            raise TypeError(msg.format(type(cube_ens_wdir)))

        try:
            cube_ens_wdir.convert_units("degrees")
        except ValueError as err:
            msg = "Input cube cannot be converted to degrees: {}".format(err)
            raise ValueError(msg)

        self.n_realizations = len(cube_ens_wdir.coord('realization').points)
        y_coord_name = cube_ens_wdir.coord(axis="y").name()
        x_coord_name = cube_ens_wdir.coord(axis="x").name()
        for wdir_slice in cube_ens_wdir.slices(["realization",
                                                y_coord_name,
                                                x_coord_name]):
            self._reset()
            # Extract wind direction data.
            self.wdir_complex = self.deg_to_complex(wdir_slice.data)
            self.realization_axis, = wdir_slice.coord_dims("realization")

            # Copies input cube and remove realization dimension to create
            # cubes for storing results.
            self.wdir_slice_mean = next(
                wdir_slice.slices_over("realization"))
            self.wdir_slice_mean.remove_coord("realization")

            # Derive average wind direction.
            self.calc_wind_dir_mean()

            # Find radius values for wind direction average.
            self.find_r_values()

            # Calculate the confidence measure based on the difference
            # between the complex average and the individual ensemble
            # realizations.
            self.calc_confidence_measure()

            # Finds any meaningless averages and substitute with
            # the wind direction taken from the first ensemble realization.
            # Mask True if r values below threshold.
            where_low_r = np.where(self.r_vals_slice.data < self.r_thresh,
                                   True, False)
            # If the any point in the array contains poor r-values,
            # trigger decider function.
            if where_low_r.any():
                self.wind_dir_decider(where_low_r, wdir_slice)

            # Append to cubelists.
            self.wdir_cube_list.append(self.wdir_slice_mean)
            self.r_vals_cube_list.append(self.r_vals_slice)
            self.confidence_measure_cube_list.append(self.confidence_slice)

        # Combine cubelists into cube.
        cube_mean_wdir = self.wdir_cube_list.merge_cube()
        cube_r_vals = self.r_vals_cube_list.merge_cube()
        cube_confidence_measure = (
            self.confidence_measure_cube_list.merge_cube())

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
