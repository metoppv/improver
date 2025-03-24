# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Module containing class for recalibrating blended probabilities.

.. See the documentation for a more detailed discussion of this plugin.
.. include:: extended_documentation/calibration/beta_recalibration/
   beta_recalibration.rst
"""

from typing import Any, Dict

import cf_units
import iris
import numpy as np
from iris.exceptions import CoordinateNotFoundError
from scipy.stats import beta

from improver import PostProcessingPlugin
from improver.metadata.probabilistic import is_probability


class BetaRecalibrate(PostProcessingPlugin):
    """Recalibrate probabilities using the cumulative distribution function
    of the beta distribution.
    """

    def __init__(self, recalibration_dict: Dict[str, Any]):
        """
        Args:
            recalibration_dict:
                Dictionary from which to calculate alpha and beta parameters for
                recalibrating blended output using the beta distribution. Dictionary
                format is as specified below. Weights will be interpolated over the
                forecast period from the values specified in the dictionary.

        Recalibration dictionary format::

            {
                "forecast_period": [7, 12],
                "alpha": [1, 1.5],
                "beta": [1.3, 2],
                "units": "hours",
            }

        The "units" key is optional. If it is omitted, it is assumed that the units
        are the same as those used in forecast_period coordinate of the input cube.
        """
        self.recalibration_dict = recalibration_dict

    def process(self, cube):
        """Recalibrate cube using the beta distribution with the alpha
        and beta parameters specified in self.recalibration_dict.

        Args:
            cube:
                A cube containing a forecast_period coordinate.

        Returns:
            A cube having the same dimensions as the input, with data
            transformed by the beta distribution cdf.

        Raises:
            CoordinateNotFoundError: if cube does not contain probability data
            CoordinateNotFoundError: if cube does not contain forecast_period coordinate
            RuntimeError: if any interpolated values of alpha or beta are <= 0
        """

        if not (is_probability(cube)):
            raise CoordinateNotFoundError(
                "Input cube must be a probability forecast "
                "and contain a threshold coordinate."
            )
        if len(cube.coords("forecast_period")) == 0:
            raise CoordinateNotFoundError(
                "Recalibration input must contain forecast_period coordinate."
            )
        forecast_period = self.recalibration_dict["forecast_period"]
        cube_forecast_period = cube.coord("forecast_period").points
        if "units" in self.recalibration_dict.keys():
            # convert interpolation points to cube units
            units = cf_units.Unit(self.recalibration_dict["units"])
            forecast_period = [
                units.convert(x, cube.coord("forecast_period").units)
                for x in forecast_period
            ]
        a = np.interp(
            cube_forecast_period, forecast_period, self.recalibration_dict["alpha"]
        )
        b = np.interp(
            cube_forecast_period, forecast_period, self.recalibration_dict["beta"]
        )
        if np.any(a <= 0) or np.any(b <= 0):
            raise RuntimeError("interpolated alpha and beta parameters must be > 0")

        cubelist = iris.cube.CubeList([])
        for i, slice in enumerate(cube.slices_over("forecast_period")):
            slice.data = beta(a[i], b[i]).cdf(slice.data).astype(np.float32)
            cubelist.append(slice)
        return cubelist.merge_cube()
