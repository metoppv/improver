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
import warnings

import numpy as np
from iris.cube import Cube

from improver import PostProcessingPlugin


class EnforceConsistentForecasts(PostProcessingPlugin):
    """Enforce that the forecasts provided are no less than or no greater than some
    linear function of a reference forecast. For example, wind speed forecasts may be
    provided as the reference forecast for wind gust forecasts with a requirement that
    wind gusts are not less than 110% of the corresponding wind speed forecast. Wind
    speed forecasts of 10 m/s will mean that the resulting wind gust forecast will be
    at least 11 m/s.
    """

    def __init__(
        self,
        additive_amount: float = 0.0,
        multiplicative_amount: float = 1.0,
        comparison_operator: str = ">=",
        diff_for_warning: float = None,
    ) -> None:
        """
        Initialise class for enforcing consistency between a forecast and a linear
        function of a reference forecast.

        Args:
            additive_amount: The amount to be added to the reference forecast prior to
                enforcing consistency between the forecast and reference forecast. If
                both an additive_amount and multiplicative_amount are specified then
                addition occurs after multiplication.
            multiplicative_amount: The amount to multiply the reference forecast by
                prior to enforcing consistency between the forecast and reference
                forecast. If both an additive_amount and multiplicative_amount are
                specified then addition occurs after multiplication.
            comparison_operator: Determines whether the forecast is enforced to be not
                less than or not greater than the reference forecast. Valid choices are
                ">=", for not less than, and "<=" for not greater than.
            diff_for_warning: If assigned, the plugin will raise a warning if any
                absolute change in forecast value is greater than this value.
        """

        self.additive_amount = additive_amount
        self.multiplicative_amount = multiplicative_amount
        self.comparison_operator = comparison_operator
        self.diff_for_warning = diff_for_warning

    def process(self, forecast: Cube, reference_forecast: Cube) -> Cube:
        """
        Function to enforce that the values in forecast_cube are not less than or not
        greater than a linear function of the corresponding values in
        reference_forecast.

        Args:
            forecast: A forecast cube of probabilities or percentiles
            reference_forecast: A reference forecast cube used to determine the bound
                of the forecast cube.

        Returns:
            A forecast cube with identical metadata to forecast but the forecasts are
            enforced to be not less than or not greater than a linear function of
            reference_forecast.
        """
        # calculate forecast_bound by applying specified linear transformation to
        # reference_forecast
        forecast_bound = reference_forecast.copy()
        forecast_bound.data = self.multiplicative_amount * forecast_bound.data
        forecast_bound.data = self.additive_amount + forecast_bound.data

        # assign forecast_bound to be either an upper or lower bound depending on input
        # comparison_operator
        lower_bound = None
        upper_bound = None
        if self.comparison_operator == ">=":
            lower_bound = forecast_bound.data
        elif self.comparison_operator == "<=":
            upper_bound = forecast_bound.data
        else:
            msg = (
                f"comparison_operator must be either '>=' or '<=', not "
                f"{self.comparison_operator}."
            )
            raise ValueError(msg)

        new_forecast = forecast.copy()
        new_forecast.data = np.clip(new_forecast.data, lower_bound, upper_bound)

        diff = new_forecast.data - forecast.data
        max_abs_diff = np.max(np.abs(diff))
        if self.diff_for_warning is not None and max_abs_diff > self.diff_for_warning:
            warnings.warn(
                f"Inconsistency between forecast {forecast.name} and "
                f"{reference_forecast.name} is greater than {self.diff_for_warning}. "
                f"Maximum absolute difference reported was {max_abs_diff}"
            )

        return new_forecast