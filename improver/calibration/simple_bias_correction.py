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

from typing import Dict, Optional

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
        Define metadata for forecast error cube, whilst ensuring any mandatory
        attributes are also populated.

        Args:
            forecast_slice:
                The source cube from which to get pre-existing metadata of use.

        Returns:
            A dictionary of attributes that are appropriate for the forecast error
            (bias) cube.
        """
        # NEED TO ESTABLISH WHAT ATTRIBUTES SHOULD BE USED FOR BIAS CUBES.
        attributes = generate_mandatory_attributes([forecast_slice])
        attributes["title"] = "Forecast error data"
        return attributes

    def _create_forecast_bias_cube(self, forecasts: Cube):
        """
        Create a cube to store the forecast bias data.

        Args:
            forecasts:
                Cube containing the reference forecasts to use in calculation
                of forecast error.

        Returns:
            A copy of the forecast cube with the attributes updated to reflect
            the cube is the forecast error of the associated diagnostic.
        """
        attributes = self._define_metadata(forecasts)
        # NEED TO ESTABLISH WHAT THE APPROPRIATE DIAGNOSTIC NAME SHOULD BE HERE.
        forecast_bias_cube = create_new_diagnostic_cube(
            name=f"{forecasts.name()} forecast error",
            units=forecasts.units,
            template_cube=forecasts,
            mandatory_attributes=attributes,
        )

        return forecast_bias_cube

    def _collapse_time(self, bias: Cube):
        """
        Collapse the time dimension coordinate if present by taking the
        mean value along this dimension.

        Args:
            bias:
                Cube containing the forecast error values.

        Returns:
            Cube containing the mean forecast error over the range of
            specified forecasts. A single forecast reference time is assigned
            with the bounds reflecting the range of forecast times over which
            the forecast error is evaluated.
        """
        # NEED TO ESTABLISH WHETHER FRT IS THE SUITABLE COORDINATE TO USE HERE.
        # - this coord is flagged as deprecated
        # - check what is used in reliability calibration
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

    def process(self, historic_forecasts: Cube, truths: Cube):
        """
        Evaluate forecast error over the set of historic forecasts and associated
        truth values.

        Where mulitple forecasts are provided, forecasts must have
        consistent forecast period and valid-hour. The resultant value returned
        is the mean value over the set of forecast/truth pairs.

        Args:
            historic_forecasts:
                Cube containing one or more historic forecasts over which to evaluate
                the forecast error.
            truths:
                Cube containing one or more truth values from which to evaluate the forecast
                error.

        Returns:
            A cube containing the mean forecast error value evaluated over the set of
            historic forecasts and truth values.
        """
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

    def process(self, forecast: Cube, bias: Cube, lower_bound: Optional[float]) -> Cube:
        """
        Apply bias correction using the specified bias values.

        Where a lower bound is specified, all values that fall below this
        lower bound (after bias correction) will be remapped to this value
        to ensure physical realistic values.

        Args:
            forecast:
                The cube to which bias correction is to be applied.
            bias:
                The cube containing the bias values for which to use in
                the bias correction.
            lower_bound:
                A lower bound below which all values will be remapped to
                after applying the bias correction.

        Returns:
            Bias corrected forecast cube.
        """
        corrected_forecast = forecast.copy()
        corrected_forecast.data = self.correction_method(forecast, bias)

        if lower_bound:
            below_lower_bound = corrected_forecast.data < lower_bound
            corrected_forecast.data[below_lower_bound] = lower_bound

        return corrected_forecast
