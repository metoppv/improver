# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Module containing generic lapse rate calculation plugins."""

import warnings

import numpy as np
from iris.cube import Cube, CubeList
from numpy import ndarray
from scipy.interpolate import make_smoothing_spline

from improver import BasePlugin
from improver.nbhood.nbhood import circular_kernel
from improver.utilities import neighbourhood_tools


class OrogLapseRate(BasePlugin):
    """Class containing methods to calculate the modelled lapse rate from a specified neighbourhood
    and to apply the lapse rate to a new orography to calculate the expected value of the variable."""

    _local_functions = None
    orography_windows = None
    diagnostic_windows = None
    weighted = False

    def __init__(
        self,
        nbhood_radius: int = 3,
        lapse_rate_function: callable = make_smoothing_spline,
        lam: float = 100000.0,
        weighted: bool = False,
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
        self.weighted = weighted

        if self.nbhood_radius < 0:
            msg = "Neighbourhood radius is less than zero"
            raise ValueError(msg)

        # nbhood_size=3 corresponds to a 3x3 array centred on the
        # central point.
        self.nbhood_size = int((2 * nbhood_radius) + 1)

        # Used in the neighbourhood checks, ensures that the centre
        # of the array is non NaN and allows us to find the centre
        # of the flattened window arrays.
        self.ind_central_point = self.nbhood_size // 2
        self.central_idx = 2 * self.nbhood_radius**2 + 2 * self.nbhood_radius

        self._calc_function = lapse_rate_function
        if not callable(self._calc_function):
            msg = "The provided lapse rate function is not callable."
            raise ValueError(msg)
        self._calc_function_kwargs = lapse_rate_function_kwargs
        if lam is not None:
            self._calc_function_kwargs["lam"] = lam

        if self.weighted:
            self.weights = circular_kernel(self.nbhood_radius, True).flatten()
        else:
            self.weights = np.ones(self.nbhood_size * self.nbhood_size)

    def _create_windows(self, data: ndarray) -> ndarray:
        """Uses neighbourhood tools to pad and generate rolling windows
        of the given dataset. Data from outside the bounds of the original
        dataset is padded with NaNs, and the resulting windows are reshaped
        so that the innermost dimension contains each point in the neighbourhood.

        Args:
            data:
                2D array of the source diagnostic data

        Returns:
            Rolling windows of the padded dataset where innermost dimension.
            contains each point in the neighbourhood.
        """
        window_shape = (self.nbhood_size, self.nbhood_size)
        resulting_shape = [*data.shape, self.nbhood_size * self.nbhood_size]
        return neighbourhood_tools.pad_and_roll(
            data, window_shape, mode="constant", constant_values=np.nan
        ).reshape(resulting_shape)

    def _adjust_duplicate_orography_points(self):
        """If there are any duplicate orography points in the window, adjust them within the max_adjustment limit
        to satisfy the requirement of unique orography points for the spline fit."""
        vectorized_function = np.vectorize(
            self._adjust_duplicate_points, signature="(n)->(n)"
        )
        self.orography_windows = np.apply_along_axis(
            vectorized_function, -1, self.orography_windows
        )

    def _adjust_duplicate_points(
        self, orog_window: ndarray, max_adjustment: float = 0.0625
    ) -> ndarray:
        """If there are any duplicate orography points in the window, adjust them within the max_adjustment limit
        to satisfy the requirement of unique orography points for the spline fit."""
        unique_orog_points = np.unique(orog_window)
        if len(unique_orog_points) == len(orog_window):
            return orog_window
        orog_window = orog_window.copy()
        for orog_point in unique_orog_points:
            duplicate_indices = np.where(orog_window == orog_point)[0]
            for i, idx in enumerate(duplicate_indices):
                adjustment = (i - (len(duplicate_indices) - 1) / 2) * max_adjustment
                orog_window[idx] += adjustment
        return orog_window

    def _create_local_functions(self):
        """Calculates the local lapse rate function for each point in the dataset.
        The resulting array of functions is stored as an attribute of the class.
        """

        def _fit_function(diag_window, orog_window):
            if len(orog_window.shape) != 1:
                raise ValueError(
                    "The innermost dimension of the input windows must be 1-dimensional."
                )
            if np.allclose(diag_window, 0.0, atol=1e-5):
                return lambda x: 0.0
            sort_idx = np.argsort(orog_window)
            sort_idx = sort_idx[np.isfinite(diag_window[sort_idx])]
            sort_idx = sort_idx[self.weights[sort_idx] > 0]
            try:
                result = self._calc_function(
                    orog_window[sort_idx],
                    diag_window[sort_idx],
                    w=self.weights[sort_idx],
                    **self._calc_function_kwargs,
                )
            except Exception as e:
                warnings.warn(f"{e}")
                # Central point of the flattened window array is n + n(2n+1) where n is the neighbourhood radius.
                result = lambda x: diag_window[self.central_idx]
            return result

        vectorized_fit = np.vectorize(_fit_function, signature="(n),(n)->()")
        self._local_functions = vectorized_fit(
            self.diagnostic_windows, self.orography_windows
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
        max_in_area = np.nanmax(self.diagnostic_windows, axis=-1)
        min_in_area = np.nanmin(self.diagnostic_windows, axis=-1)
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
        self._adjust_duplicate_orography_points()
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
