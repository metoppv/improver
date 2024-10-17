# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Module containing wind direction averaging plugins."""
import iris
import numpy as np
from iris.coords import CellMethod
from iris.cube import Cube
from numpy import ndarray

from improver import PostProcessingPlugin
from improver.nbhood.nbhood import NeighbourhoodProcessing
from improver.utilities.complex_conversion import complex_to_deg, deg_to_complex
from improver.utilities.cube_checker import check_cube_coordinates


class WindDirection(PostProcessingPlugin):
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

    Args:
        backup_method:
            Backup method to use if the complex numbers approach has low
            confidence.
            "first_realization" uses the value of realization zero.
            "neighbourhood" (default) recalculates using the complex numbers
            approach with additional realizations extracted from neighbouring
            grid points from all available realizations.
    """

    def __init__(self, backup_method: str = "neighbourhood") -> None:
        """Initialise class."""
        self.backup_methods = ["first_realization", "neighbourhood"]
        self.backup_method = backup_method
        if self.backup_method not in self.backup_methods:
            msg = "Invalid option for keyword backup_method ({})".format(
                self.backup_method
            )
            raise ValueError(msg)

        # Any points where the r-values are below the threshold is regarded as
        # containing ambigous data.
        self.r_thresh = 0.01

        # Creates cubelists to hold data.
        self.wdir_cube_list = iris.cube.CubeList()
        self.r_vals_cube_list = iris.cube.CubeList()
        # Radius used in neighbourhood plugin as determined in IMPRO-491
        self.nb_radius = 6000.0  # metres
        # Initialise neighbourhood plugin ready for use
        self.nbhood = NeighbourhoodProcessing(
            "square", self.nb_radius, weighted_mode=False
        )

    def __repr__(self) -> str:
        """Represent the configured plugin instance as a string."""
        return (
            '<WindDirection: backup_method "{}"; neighbourhood radius "{}"m>'
        ).format(self.backup_method, self.nb_radius)

    def _reset(self) -> None:
        """Empties working data objects"""
        self.realization_axis = None
        self.wdir_complex = None
        self.wdir_slice_mean = None
        self.wdir_mean_complex = None
        self.r_vals_slice = None

    def calc_wind_dir_mean(self) -> None:
        """Find the mean wind direction using complex average which actually
           signifies a point between all of the data points in POLAR
           coordinates - NOT the average DEGREE ANGLE.

        Uses:
            self.wdir_complex:
                3D array or float - wind direction angles in degrees.
            self.realization_axis:
                Axis to collapse over.

        Defines:
            self.wdir_mean_complex:
                3D array or float - wind direction angles as complex numbers
                collapsed along an axis using np.mean().
            self.wdir_slice_mean:
                3D array or float - wind direction angles in degrees collapsed
                along an axis using np.mean().
        """
        self.wdir_mean_complex = np.mean(self.wdir_complex, axis=self.realization_axis)
        self.wdir_slice_mean.data = complex_to_deg(self.wdir_mean_complex)

    def find_r_values(self) -> None:
        """Find radius values from complex numbers.

        Takes input wind direction in complex values and returns array
        containing r values using Pythagoras theorem.

        Uses:
            self.wdir_mean_complex:
                3D array or float - wind direction angles in complex numbers.
            self.wdir_slice_mean:
                3D array or float - mean wind direction angles in complex
                numbers.

        Defines:
            self.r_vals_slice:
                Contains r values and inherits meta-data from
                self.wdir_slice_mean.
        """

        r_vals = np.sqrt(
            np.square(self.wdir_mean_complex.real)
            + np.square(self.wdir_mean_complex.imag)
        )
        self.r_vals_slice = self.wdir_slice_mean.copy(data=r_vals)

    def wind_dir_decider(self, where_low_r: ndarray, wdir_cube: Cube) -> None:
        """If the wind direction is so widely scattered that the r value
           is nearly zero then this indicates that the average wind direction
           is essentially meaningless.
           We therefore substitute this meaningless average wind
           direction value for the wind direction calculated from a larger
           sample by smoothing across a neighbourhood of points before
           rerunning the main technique.
           This is invoked rarely (1 : 100 000)

        Args:
            where_low_r:
                Array of boolean values. True where original wind direction
                estimate has low confidence. These points are replaced
                according to self.backup_method
            wdir_cube:
                Contains array of wind direction data (realization, y, x)

        Uses:
            self.wdir_slice_mean:
                Containing average wind direction angle (in degrees).
            self.wdir_complex:
                3D array - wind direction angles from ensembles (in complex).
            self.r_vals_slice.data:
                2D array - Radius taken from average complex wind direction
                angle.
            self.r_thresh:
                Any r value below threshold is regarded as meaningless.
            self.realization_axis:
                Axis to collapse over.
            self.n_realizations:
                Number of realizations available in the plugin. Used to set the
                neighbourhood radius as this is used to adjust the radius again
                in the neighbourhooding plugin.

        Defines:
            self.wdir_slice_mean.data:
                2D array - Wind direction degrees where ambigious values have
                been replaced with data from first ensemble realization.
        """
        if self.backup_method == "neighbourhood":
            # Performs smoothing over a 6km square neighbourhood.
            # Then calculates the mean wind direction.
            child_class = WindDirection(backup_method="first_realization")
            child_class.wdir_complex = self.nbhood(
                wdir_cube.copy(data=self.wdir_complex)
            ).data
            child_class.realization_axis = self.realization_axis
            child_class.wdir_slice_mean = self.wdir_slice_mean.copy()
            child_class.calc_wind_dir_mean()
            improved_values = child_class.wdir_slice_mean.data
        else:
            # Takes realization zero (control member).
            improved_values = wdir_cube.extract(iris.Constraint(realization=0)).data

        # If the r-value is low - substitute average wind direction value for
        # the wind direction taken from the first ensemble realization.
        self.wdir_slice_mean.data = np.where(
            where_low_r, improved_values, self.wdir_slice_mean.data
        )

    def process(self, cube_ens_wdir: Cube) -> Cube:
        """Create a cube containing the wind direction averaged over the
        ensemble realizations.

        Args:
            cube_ens_wdir:
                Cube containing wind direction from multiple ensemble
                realizations.

        Returns:
            - Cube containing the wind direction averaged from the
              ensemble realizations.

        Raises:
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

        self.n_realizations = len(cube_ens_wdir.coord("realization").points)
        y_coord_name = cube_ens_wdir.coord(axis="y").name()
        x_coord_name = cube_ens_wdir.coord(axis="x").name()
        for wdir_slice in cube_ens_wdir.slices(
            ["realization", y_coord_name, x_coord_name]
        ):
            self._reset()
            # Extract wind direction data.
            self.wdir_complex = deg_to_complex(wdir_slice.data)
            (self.realization_axis,) = wdir_slice.coord_dims("realization")
            # Copies input cube and remove realization dimension to create
            # cubes for storing results.
            self.wdir_slice_mean = next(wdir_slice.slices_over("realization"))
            self.wdir_slice_mean.remove_coord("realization")

            # Derive average wind direction.
            self.calc_wind_dir_mean()

            # Find radius values for wind direction average.
            self.find_r_values()

            # Finds any meaningless averages and substitute with
            # the wind direction taken from the first ensemble realization.
            # Mask True if r values below threshold.
            where_low_r = np.where(self.r_vals_slice.data < self.r_thresh, True, False)
            # If the any point in the array contains poor r-values,
            # trigger decider function.
            if where_low_r.any():
                self.wind_dir_decider(where_low_r, wdir_slice)

            # Append to cubelists.
            self.wdir_cube_list.append(self.wdir_slice_mean)

        # Combine cubelists into cube.
        cube_mean_wdir = self.wdir_cube_list.merge_cube()

        # Check that the dimensionality of coordinates of the output cube
        # matches the input cube.
        first_slice = next(cube_ens_wdir.slices_over(["realization"]))
        cube_mean_wdir = check_cube_coordinates(first_slice, cube_mean_wdir)

        # Change cube identifiers.
        cube_mean_wdir.add_cell_method(CellMethod("mean", coords="realization"))
        return cube_mean_wdir
