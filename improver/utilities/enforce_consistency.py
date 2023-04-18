# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown copyright. The Met Office.
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
"""Provides utilities for enforcing consistency between forecasts."""
from iris.cube import Cube

from improver import PostProcessingPlugin


class EnforceConsistentPercentiles(PostProcessingPlugin):
    """Enforce that the percentiles of the forecast provided are equal to, or exceed,
    the reference forecast by a minimum percentage exceedance. For example, wind speed
    forecasts may be provided as the reference forecast for wind gust forecasts with
    a minimum percentage exceedance of 10%. Wind speed forecasts of 10 m/s will mean
    that the resulting wind gust forecast will be at least 11 m/s.
    """

    def __init__(self, min_percentage_exceedance: float = 0.0) -> None:
        """Initialise class.

        Args:
            min_percentage_exceedance: The minimum percentage by which the reference
                forecast must be exceeded by the accompanying forecast. The reference
                forecast therefore provides a lower bound for the accompanying forecast
                with this forecast limited to a minimum of the reference forecast plus
                the min_percentage_exceedance multiplied by the reference forecast.
                The percentage is expected as a value between 0 and 100.

        """
        if min_percentage_exceedance < 0 or min_percentage_exceedance > 100:
            msg = (
                "The percentage representing the minimum that the reference "
                "forecast must be exceeded by is outside the range 0 to 100. "
                f"The value provided is {min_percentage_exceedance}."
            )
            raise ValueError(msg)

        self.min_percentage_exceedance = min_percentage_exceedance

    def process(self, ref_forecast_cube: Cube, forecast_cube: Cube) -> Cube:
        """Enforce a lower bound for the forecast cube using the reference forecast
        cube plus a minimum percentage exceedance.

        Args:
            ref_forecast (iris.cube.Cube)
                Cube with percentiles that are equal to, or provide a lower bound,
                for the accompanying forecast. The reference forecast is expected
                to be the same shape as forecast_cube but have a different name.
            forecast_cube (iris.cube.Cube):
                Cube with percentiles that will have consistency enforced.

        Returns:
            Cube with percentiles that has had consistency enforced.
        """
        updated_forecast_cube = forecast_cube.copy()
        lower_bound = (
            ref_forecast_cube.data
            + ref_forecast_cube.data * self.min_percentage_exceedance / 100
        )
        condition1 = updated_forecast_cube.data < lower_bound
        updated_forecast_cube.data[condition1] = lower_bound[condition1]
        return updated_forecast_cube
