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
"""Simple bias correction plugins."""

from typing import Dict

import iris
from iris.cube import Cube

from improver import BasePlugin
from improver.calibration.utilities import (
    check_forecast_consistency,
    create_unified_frt_coord,
    filter_non_matching_cubes,
)
from improver.metadata.utilities import (
    create_new_diagnostic_cube,
    generate_mandatory_attributes,
)
from improver.utilities.cube_manipulation import collapsed, get_dim_coord_names


def mean_additive_error(forecasts, truths):
    """Evaluate the mean error between the forecast and truth dataset."""
    forecast_errors = truths - forecasts
    return forecast_errors.data


def apply_additive_bias(forecast, bias):
    forecast = forecast + bias
    return forecast.data


class CalculateForecastBias(BasePlugin):
    """
    A plugin to evaluate the forecast bias from the historical forecast and truth
    value(s).
    """

    def __init__(self):
        """
        Initialise class for applying simple bias correction.
        """
        self.error_method = mean_additive_error

    def _define_metadata(self, forecast_slice: Cube) -> Dict[str, str]:
        """
        Define metadata that is specifically required for reliability table
        cubes, whilst ensuring any mandatory attributes are also populated.
        Args:
            forecast_slice:
                The source cube from which to get pre-existing metadata of use.
        Returns:
            A dictionary of attributes that are appropriate for the
            reliability table cube.
        """
        attributes = generate_mandatory_attributes([forecast_slice])
        attributes["title"] = "Forecast error data"
        return attributes

    def _create_forecast_bias_cube(self, forecasts):
        """Create a cube to store the forecast bias data."""

        attributes = self._define_metadata(forecasts)
        forecast_bias_cube = create_new_diagnostic_cube(
            name=f"{forecasts.name()} forecast error",
            units=forecasts.units,
            template_cube=forecasts,
            mandatory_attributes=attributes,
        )

        return forecast_bias_cube

    def _collapse_time(self, bias):
        """Collapse the time dimension coordinate if present."""
        if "time" in get_dim_coord_names(bias):
            frt_coord = create_unified_frt_coord(bias.coord("forecast_reference_time"))
            mean_bias = collapsed(bias, "forecast_reference_time", iris.analysis.MEAN)
            mean_bias.data = mean_bias.data.astype(bias.dtype)
            mean_bias.replace_coord(frt_coord)
        else:
            mean_bias = bias
        # Remove valid time in favour of frt coordinate
        mean_bias.remove_coord("time")

        return mean_bias

    def process(self, historic_forecasts, truths):

        # Ensure that valid times match over forecasts/truth
        historic_forecasts, truths = filter_non_matching_cubes(
            historic_forecasts, truths
        )
        # Ensure that input forecasts are for consitent period/valid-hour
        check_forecast_consistency(historic_forecasts)
        # Remove truth frt to enable cube maths
        truths.remove_coord("forecast_reference_time")

        # Create template cube to store the forecast bias
        bias = self._create_forecast_bias_cube(historic_forecasts)
        bias.data = self.error_method(historic_forecasts, truths)
        bias = self._collapse_time(bias)
        return bias


class ApplySimpleBiasCorrection(BasePlugin):
    """
    A Plugin to apply a simple bias correction on a per member basis using
    the specified bias terms.
    """

    def __init__(self):
        """
        Initialise class for applying simple bias correction.
        """
        self.correction_method = apply_additive_bias

    def process(self, forecast, bias_terms):
        corrected_forecast = forecast.copy()
        corrected_forecast.data = self.correction_method(forecast, bias_terms)
        return corrected_forecast
