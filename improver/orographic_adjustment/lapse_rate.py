# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Module containing generic lapse rate calculation plugins."""

import numpy as np
from iris.cube import Cube, CubeList
from numpy import ndarray
from scipy.interpolate import make_splrep

from improver import BasePlugin
from improver.utilities import neighbourhood_tools


class LapseRate(BasePlugin):
    """Class containing methods to calculate the modelled lapse rate from a specified neighbourhood
    and to apply the lapse rate to a new orography to calculate the expected value of the variable."""

    _local_functions = None
    orography_windows = None
    diagnostic_windows = None

    def __init__(
        self,
        nbhood_radius: int = 7,
        lapse_rate_function: callable = make_splrep,
        **lapse_rate_function_kwargs,
    ):
        """
        Initialise the plugin.

        Args:
            nbhood_radius:
                Radius of neighbourhood around each point. The neighbourhood
                will be a square array with side length 2*nbhood_radius + 1.
                The default value of 7 is from the referenced paper.
        """
        self.nbhood_radius = nbhood_radius

        if self.nbhood_radius < 0:
            msg = "Neighbourhood radius is less than zero"
            raise ValueError(msg)

        # nbhood_size=3 corresponds to a 3x3 array centred on the
        # central point.
        self.nbhood_size = int((2 * nbhood_radius) + 1)

        # Used in the neighbourhood checks, ensures that the center
        # of the array is non NaN.
        self.ind_central_point = self.nbhood_size // 2

        self._calc_function = lapse_rate_function
        if not callable(self._calc_function):
            msg = "The provided lapse rate function is not callable."
            raise ValueError(msg)
        self._calc_function_kwargs = lapse_rate_function_kwargs

    def _create_windows(self, data: ndarray) -> ndarray:
        """Uses neighbourhood tools to pad and generate rolling windows
        of the given dataset.

        Args:
            data:
                2D array of the source diagnostic data

        Returns:
            - Rolling windows of the padded dataset.
        """
        window_shape = (self.nbhood_size, self.nbhood_size)
        return neighbourhood_tools.pad_and_roll(
            data, window_shape, mode="constant", constant_values=np.nan
        )

    def _create_local_functions(self):
        """Calculates the local lapse rate function for each point in the dataset.
        The resulting array of functions is stored as an attribute of the class.
        """

        def _fit_function(diag_window, orog_window):
            if np.isnan(diag_window[self.ind_central_point]) or np.isnan(
                orog_window[self.ind_central_point]
            ):
                return lambda x: np.nan
            sort_idx = np.argsort(orog_window)
            return self._calc_function(
                orog_window[sort_idx],
                diag_window[sort_idx],
                **self._calc_function_kwargs,
            )

        vectorized_fit = np.vectorize(_fit_function, signature="(n),(n)->()")
        self._local_functions = vectorized_fit(
            np.moveaxis(self.diagnostic_windows, 0, -1),
            np.moveaxis(self.orography_windows, 0, -1),
        )

    def _apply_new_orography_to_functions(self, new_orography: ndarray) -> ndarray:
        """Applies the local lapse rate functions to a new orography to calculate the expected value of the variable.

        Args:
            new_orography:
                2D array of orographies to apply the local functions to, in metres

        Returns:
            2D array of the same shape as new_orography, containing the expected value of the variable at each point.
        """
        vectorized_apply = np.vectorize(lambda func, orog: func(orog))
        return vectorized_apply(self._local_functions, new_orography)

    def _clip_to_local_range(self, adjusted_data: ndarray) -> ndarray:
        """Clips the adjusted data to the local range of the diagnostic data in the neighbourhood.

        Args:
            adjusted_data:
                2D array of the same shape as new_orography, containing the expected value of the variable at each point.

        Returns:
            2D array of the same shape as adjusted_data, containing the clipped expected value of the variable at each point.
        """
        max_in_area = np.nanmax(self.diagnostic_windows, axis=(0, 1))
        min_in_area = np.nanmin(self.diagnostic_windows, axis=(0, 1))
        return np.clip(adjusted_data, min_in_area, max_in_area)

    def process(self, diagnostic: Cube, orography: Cube, new_orography: Cube) -> Cube:
        """Main processing method for the plugin. Takes in the diagnostic and orography datasets, calculates the local lapse rate functions, and applies them to a new orography.

        Args:
            diagnostic:
                The source diagnostic data
            orography:
                Source orography data
            new_orography:
                Target orography data

        Returns:
            Cube of the diagnostic data adjusted to the new orography, with the same metadata as the input diagnostic cube.
        """
        orography.convert_units(new_orography.units)
        self.orography_windows = self._create_windows(orography.data)
        xy_coords = [orography.coord(axis="y"), orography.coord(axis="x")]
        adjusted_slices = CubeList()
        for diagnostic_slice in diagnostic.slices(xy_coords):
            self.diagnostic_windows = self._create_windows(diagnostic.data)
            self._create_local_functions()
            adjusted_data = self._apply_new_orography_to_functions(new_orography.data)
            adjusted_data = self._clip_to_local_range(adjusted_data)
            adjusted_cube = diagnostic_slice.copy(data=adjusted_data)
            adjusted_slices.append(adjusted_cube)
        adjusted_cube = adjusted_slices.merge_cube()
        return adjusted_cube
