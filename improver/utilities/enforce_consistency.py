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
from typing import List, Optional, Union

import numpy as np
from iris.cube import Cube

from improver import PostProcessingPlugin
from improver.metadata.probabilistic import is_probability


class EnforceConsistentForecasts(PostProcessingPlugin):
    """Enforce that the forecasts provided are no less than, no greater than, or between
    some linear function(s) of a reference forecast. For example, wind speed forecasts
    may be provided as the reference forecast for wind gust forecasts with a requirement
    that wind gusts are not less than 110% of the corresponding wind speed forecast.
    Wind speed forecasts of 10 m/s will mean that the resulting wind gust forecast will
    be at least 11 m/s.
    """

    def __init__(
        self,
        additive_amount: Union[float, List[float]] = 0.0,
        multiplicative_amount: Union[float, List[float]] = 1.0,
        comparison_operator: Union[str, List[str]] = ">=",
        diff_for_warning: Optional[float] = None,
    ) -> None:
        """
        Initialise class for enforcing a forecast to be either greater than or equal to,
        or less than or equal to a linear function of a reference forecast. Can also
        enforce that a forecast is between two bounds created from the reference
        forecast, if this is the case then a list of two elements must be provided for
        additive_amount, multiplicative_amount, and comparison_operator.

        Args:
            additive_amount: The amount to be added to the reference forecast prior to
                enforcing consistency between the forecast and reference forecast. If
                both an additive_amount and multiplicative_amount are specified then
                addition occurs after multiplication. This option cannot be used for
                probability forecasts, if it is then an error will be raised.
            multiplicative_amount: The amount to multiply the reference forecast by
                prior to enforcing consistency between the forecast and reference
                forecast. If both an additive_amount and multiplicative_amount are
                specified then addition occurs after multiplication. This option cannot
                be used for probability forecasts, if it is then an error will be raised.
            comparison_operator: Determines whether the forecast is enforced to be not
                less than or not greater than the reference forecast. Valid choices are
                ">=", for not less than, and "<=" for not greater than. If provided as
                a list then each of ">=" and "<=" must be in the list exactly once.
            diff_for_warning: If assigned, the plugin will raise a warning if any
                absolute change in forecast value is greater than this value.
        """

        self.additive_amount = additive_amount
        self.multiplicative_amount = multiplicative_amount
        self.comparison_operator = comparison_operator
        self.diff_for_warning = diff_for_warning

    @staticmethod
    def calculate_bound(
        cube: Cube, additive_amount: float, multiplicative_amount: float
    ) -> Cube:
        """
        Function to calculate a linear transformation of the reference forecast.

        Args:
            cube: An iris cube.
            additive_amount: The amount to be added to the cube. If both an
                additive_amount and multiplicative_amount are specified then addition
                occurs after multiplication.
            multiplicative_amount: The amount to multiply the cube by. If both an
                additive_amount and multiplicative_amount are specified then addition
                occurs after multiplication.

        Returns:
            A cube with identical metadata to input cube but with transformed data.
        """

        output = cube.copy()
        output.data = multiplicative_amount * output.data
        output.data = additive_amount + output.data

        return output

    def process(self, forecast: Cube, reference_forecast: Cube) -> Cube:
        """
        Function to enforce that the values in the forecast cube are not less than or
        not greater than a linear function of the corresponding values in
        reference_forecast, or between two bounds generated from two different linear
        functions of the reference_forecast.

        Args:
            forecast: A forecast cube
            reference_forecast: A reference forecast cube used to determine the bound/s
                of the forecast cube.

        Returns:
            A forecast cube with identical metadata to forecast but the forecasts are
            enforced to be within the calculated bounds.

        Raises:
            ValueError: If units of forecast and reference cubes are different and
                cannot be converted to match.
            ValueError: If additive_amount and multiplicative_amount are not 0.0 and 1.0,
                respectively, when a probability forecast is input.
            ValueError: If incorrect comparison_operator is input.
            ValueError: If contradictory bounds are generated.
            ValueError: If any of additive_amount, multiplicative_amount, or
                comparison_operator are lists when they are not all lists.

        Warns:
            Warning: If difference between generated bounds and forecast is greater than
                diff_for_warning.
        """

        # check forecast and reference units match
        try:
            reference_forecast.convert_units(forecast.units)
        except ValueError:
            if forecast.units != reference_forecast.units:
                msg = (
                    "The units in the forecast and reference cubes do not match and "
                    "cannot be converted to match. The units of forecast were "
                    f"{forecast.units}, the units of reference_forecast were "
                    f"{reference_forecast.units}."
                )
                raise ValueError(msg)

        # linear transformation cannot be applied to probability forecasts
        if self.additive_amount != 0.0 or self.multiplicative_amount != 1.0:
            if is_probability(forecast):
                msg = (
                    "For probability data, additive_amount must be 0.0 and "
                    "multiplicative_amount must be 1.0. The input additive_amount was "
                    f"{self.additive_amount}, the input multiplicative_amount was "
                    f"{self.multiplicative_amount}."
                )
                raise ValueError(msg)

        # calculate forecast_bound by applying specified linear transformation to
        # reference_forecast
        check_if_list = [
            isinstance(item, list)
            for item in [
                self.additive_amount,
                self.multiplicative_amount,
                self.comparison_operator,
            ]
        ]
        if all(check_if_list):
            lower_bound = self.calculate_bound(
                reference_forecast,
                self.additive_amount[0],
                self.multiplicative_amount[0],
            ).data
            upper_bound = self.calculate_bound(
                reference_forecast,
                self.additive_amount[1],
                self.multiplicative_amount[1],
            ).data
            if self.comparison_operator == ["<=", ">="]:
                upper_bound, lower_bound = lower_bound, upper_bound
            elif self.comparison_operator == [">=", "<="]:
                pass
            else:
                msg = (
                    "When comparison operators are provided as a list, the list must be "
                    f"either ['>=', '<='] or ['<=', '>='], not {self.comparison_operator}."
                )
                raise ValueError(msg)
            if np.any(lower_bound > upper_bound):
                msg = (
                    "The provided reference_cube, additive_amount and "
                    "multiplicative_amount have created contradictory bounds. Some of"
                    "the values in the lower bound are greater than the upper bound."
                )
                raise ValueError(msg)
        elif any(check_if_list):
            msg = (
                "If any of additive_amount, multiplicative_amount, or comparison_operator "
                "are input as a list, then they must all be input as a list of 2 elements. "
            )
            raise ValueError(msg)
        else:
            bound = self.calculate_bound(
                reference_forecast, self.additive_amount, self.multiplicative_amount
            )
            lower_bound = None
            upper_bound = None
            if self.comparison_operator == ">=":
                lower_bound = bound.data
            elif self.comparison_operator == "<=":
                upper_bound = bound.data
            else:
                msg = (
                    "When enforcing consistency with one bound, comparison_operator "
                    f"must be either '>=' or '<=', not {self.comparison_operator}."
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
