# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2021 Met Office.
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
"""Module containing neighbourhood processing utilities."""

from typing import List, Optional, Union

import numpy as np
from iris.cube import Cube
from numpy import ndarray

from improver import BasePlugin
from improver.metadata.forecast_times import forecast_period_coord


class BaseNeighbourhoodProcessing(BasePlugin):
    """
    A base class used to set up neighbourhood radii for a given cube
    based on the forecast period of that cube if required.
    """

    def __init__(
        self, radii: Union[float, List[float]], lead_times: Optional[List] = None,
    ) -> None:
        """
        Create a base neighbourhood processing plugin that processes radii
        related arguments.

            radii:
                The radii in metres of the neighbourhood to apply.
                Rounded up to convert into integer number of grid
                points east and north, based on the characteristic spacing
                at the zero indices of the cube projection-x and y coords.
            lead_times:
                List of lead times or forecast periods, at which the radii
                within 'radii' are defined. The lead times are expected
                in hours.
        """
        if isinstance(radii, list):
            self.radii = [float(x) for x in radii]
        else:
            self.radius = float(radii)
        self.lead_times = lead_times
        if self.lead_times is not None:
            if len(radii) != len(lead_times):
                msg = (
                    "There is a mismatch in the number of radii "
                    "and the number of lead times. "
                    "Unable to continue due to mismatch."
                )
                raise ValueError(msg)

    def _find_radii(
        self, cube_lead_times: Optional[ndarray] = None
    ) -> Union[float, ndarray]:
        """Revise radius or radii for found lead times.
        If cube_lead_times is None, no automatic adjustment
        of the radii will take place.
        Otherwise it will interpolate to find the radius at
        each cube lead time as required.

        Args:
            cube_lead_times:
                Array of forecast times found in cube.

        Returns:
            Required neighbourhood sizes.
        """
        radii = np.interp(cube_lead_times, self.lead_times, self.radii)
        return radii

    def process(self, cube: Cube) -> Cube:
        """
        Supply a cube with a forecast period coordinate in order to set the
        correct radius for use in neighbourhood processing.

        Also checkes there are no unmasked NaNs in the input cube.

        Args:
            cube:
                Cube to apply a neighbourhood processing method.

        Returns:
            cube:
                The unaltered input cube.
        """

        if np.isnan(cube.data).any():
            raise ValueError("Error: NaN detected in input cube data")

        if self.lead_times:
            # Interpolate to find the radius at each required lead time.
            fp_coord = forecast_period_coord(cube)
            fp_coord.convert_units("hours")
            self.radius = self._find_radii(cube_lead_times=fp_coord.points)
        return cube
