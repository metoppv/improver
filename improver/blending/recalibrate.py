# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Module containing class for recalibrating blended probabilities."""

import cf_units
import iris
import numpy as np
from scipy.stats import beta
from improver import PostProcessingPlugin

from typing import Optional, Dict, Any


class Recalibrate(PostProcessingPlugin):

    def __init__(self, recalibration_dict: Optional[Dict[str, Any]] = None):
        """
        Args:

            recalibration_dict:
                Dictionary from which to calculate alpha and beta parameters for
                recalibrating blended output using the beta distribution. Dictionary
                format is as specified below. Weights will be interpolated over the 
                forecast period from the values specified in the dictionary.
        
        
        Recalibration dictionary format::
            {   "forecast_period": [7, 12],
                "alpha": [1, 1.5],
                "beta": [1.3, 2],
                "units": "hours"
            }
        """
        self.recalibration_dict = recalibration_dict


    def process(self, cube):
        """Recalibrate cube using the beta distribution with the alpha
        and beta parameters specified in recalibration_dict.
        
        Args:
            cube:
                A cube containing a forecast_period coordinate.
            recalibration_dict:
                A dictionary giving the weights at some subset of
                forecast_period values. Weights at other values are
                linearly interpolated.

        Returns:
            A cube having the same dimensions as the input, with data
            transformed by the beta distribution cdf.    
        """

        if len(cube.coords("threshold")) == 0:
            raise ValueError("Recalibration input must contain threshold coordinate.")
        if len(cube.coords("forecast_period")) == 0:
            raise ValueError("Recalibration input must contain forecast_period coordinate.")
        x = self.recalibration_dict["forecast_period"].values
        xp = cube.coord("forecast_period").points
        if "units" in self.recalibration_dict.keys():
            units = cf_units.Unit(self.recalibration_dict["units"])
            x = units.convert(x, cube.coord("forecast_period").units)
        a = np.interp(x, xp, self.recalibration_dict["alpha"].values)
        b = np.interp(x, xp, self.recalibration_dict["beta"].values)
        cubelist = iris.cube.CubeList([])
        for i, slice in enumerate(cube.slices_over("forecast_period")):
            slice.data = beta(a[i], b[i]).cdf(slice.data)
            cubelist.append(slice)
        return cubelist.merge_cube()
