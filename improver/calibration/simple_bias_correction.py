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
from iris.cube import Cube, CubeList
from numpy import ndarray

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


def evaluate_additive_error(
    forecasts: Cube, truths: Cube, collapse_dim: str
) -> ndarray:
    """
    Evaluate the mean additive error (error = forecast - truth) between the
    forecast and truth dataset.

    Args:
        forecasts:
            Cube containing set of historic forecasts.
        truths:
            Cube containing set of corresponding truth values.
        collapse_dim:
            Dim coordinate over which to evaluate the mean value.

    Returns:
        An array containing the mean additive forecast error values.
    """
    forecast_errors = forecasts - truths
    if collapse_dim in get_dim_coord_names(forecast_errors):
        mean_forecast_error = collapsed(
            forecast_errors, collapse_dim, iris.analysis.MEAN
        )
        return mean_forecast_error.data
    return forecast_errors.data


def apply_additive_correction(forecast: Cube, bias: Cube) -> ndarray:
    """
    Apply additive correction to forecast using the specified bias values,
    where the bias is expected to be defined as forecast - truth.

    Args:
        forecast:
            Cube containing the forecast to which bias correction is to be applied.
        bias:
            Cube containing the bias values to apply to the forecast.

    Returns:
        An array containing the corrected forecast values.
    """
    corrected_forecast = forecast - bias
    return corrected_forecast.data


class CalculateForecastBias(BasePlugin):
    """
    A plugin to evaluate the forecast bias from the historical forecast and truth
    value(s).
    """

    def __init__(self):
        """
        Initialise class for calculating forecast bias.
        """
        self.error_method = evaluate_additive_error

    def _define_metadata(self, forecasts: Cube) -> Dict[str, str]:
        """
        Define metadata for forecast bias cube, whilst ensuring any mandatory
        attributes are also populated.

        Args:
            forecasts:
                The source cube from which to get pre-existing metadata.

        Returns:
            A dictionary of attributes that are appropriate for the bias cube.
        """
        attributes = generate_mandatory_attributes([forecasts])
        attributes["title"] = "Forecast bias data"
        return attributes

    def _create_bias_cube(self, forecasts: Cube) -> Cube:
        """
        Create a cube to store the forecast bias data.

        Where multiple reference forecasts values are provided via forecasts,
        the time dimension will be collapsed to a single value represented by
        a single forecast_reference_time with bounds set using the range of
        forecast_reference_time values present in forecasts.

        Args:
            forecasts:
                Cube containing the reference forecasts to use in calculation
                of forecast bias.

        Returns:
            A copy of the forecasts cube with the attributes updated to reflect
            the cube is the forecast error of the associated diagnostic. If a time
            dimension is present in the forecasts, this will be collapsed to a single
            value.
        """
        attributes = self._define_metadata(forecasts)
        forecast_bias_cube = create_new_diagnostic_cube(
            name=f"{forecasts.name()}_forecast_error",
            units=forecasts.units,
            template_cube=forecasts,
            mandatory_attributes=attributes,
        )
        # Collapse the time values down to a single value as mean value
        # will be stored where multiple forecast_reference_times are passed in.
        if "time" in get_dim_coord_names(forecast_bias_cube):
            frt_coord = create_unified_frt_coord(
                forecast_bias_cube.coord("forecast_reference_time")
            )
            forecast_bias_cube = collapsed(
                forecast_bias_cube, "forecast_reference_time", iris.analysis.MEAN
            )
            forecast_bias_cube.data = forecast_bias_cube.data.astype(
                forecast_bias_cube.dtype
            )
            forecast_bias_cube.replace_coord(frt_coord)
        # Remove valid time in favour of frt coordinate
        forecast_bias_cube.remove_coord("time")

        return forecast_bias_cube

    def process(self, historic_forecasts: Cube, truths: Cube) -> Cube:
        """
        Evaluate forecast bias over the set of historic forecasts and associated
        truth values.

        Where multiple forecasts values are provided, forecasts must have consistent
        forecast period and valid-hour. The resultant value returned is the mean value
        over the set of forecast/truth pairs.

        Args:
            historic_forecasts:
                Cube containing one or more historic forecasts over which to evaluate
                the forecast bias.
            truths:
                Cube containing one or more truth values from which to evaluate the forecast
                bias.

        Returns:
            A cube containing the forecast bias values evaluated over the set of historic
            forecasts and truth values.
        """
        # Ensure that valid times match over forecasts/truth
        historic_forecasts, truths = filter_non_matching_cubes(
            historic_forecasts, truths
        )
        # Ensure that input forecasts are for consistent period/valid-hour
        check_forecast_consistency(historic_forecasts)
        # Remove truth frt to enable cube maths
        truths.remove_coord("forecast_reference_time")

        # Create template cube to store the forecast bias
        bias = self._create_bias_cube(historic_forecasts)
        bias.data = self.error_method(historic_forecasts, truths, collapse_dim="time")
        return bias


class ApplyBiasCorrection(BasePlugin):
    """
    A Plugin to apply a simple bias correction on a per member basis using
    the specified bias values.
    """

    def __init__(self):
        """
        Initialise class for applying simple bias correction.
        """
        self.correction_method = apply_additive_correction

    def _get_mean_bias(self, bias_values: CubeList) -> Cube:
        """
        Evaluate the mean bias from the input cube(s) in bias_values.

        Where multiple cubes are provided, each bias value must represent
        a single forecast_reference_time to ensure that the resultant value
        is the true mean over the set of reference forecasts. This is done
        by checking the forecast_reference_time bounds; if a bias_value is
        defined over a range of frt values (ie. bounds exist) an error will
        be raised.

        Args:
            bias_values:
                Cubelist containing the input bias cube(s).

        Returns:
            Cube containing the mean bias evaulated from set of bias_values.
        """
        # Currently only support for cases where the input bias_values are defined
        # over a single forecast_reference_time.
        if len(bias_values) == 1:
            return bias_values[0]
        else:
            # Loop over bias_values and check bounds on each cube.
            for bias_cube in bias_values:
                if bias_cube.coord("forecast_reference_time").bounds is not None:
                    raise ValueError(
                        "Collapsing multiple bias values to a mean value is unsupported for "
                        "bias values defined over multiple reference forecast values."
                    )
            bias_values = bias_values.merge_cube()
            mean_bias = collapsed(
                bias_values, "forecast_reference_time", iris.analysis.MEAN
            )
            return mean_bias

    def process(
        self, forecast: Cube, bias: CubeList, lower_bound: Optional[float]
    ) -> Cube:
        """
        Apply bias correction using the specified bias values.

        Where a lower bound is specified, all values that fall below this
        lower bound (after bias correction) will be remapped to this value
        to ensure physically realistic values.

        Args:
            forecast:
                The cube to which bias correction is to be applied.
            bias:
                The cubelist containing the bias values for which to use in
                the bias correction.
            lower_bound:
                A lower bound below which all values will be remapped to
                after the bias correction step.

        Returns:
            Bias corrected forecast cube.
        """
        bias = self._get_mean_bias(bias)

        corrected_forecast = forecast.copy()
        corrected_forecast.data = self.correction_method(forecast, bias)

        if lower_bound is not None:
            below_lower_bound = corrected_forecast.data < lower_bound
            corrected_forecast.data[below_lower_bound] = lower_bound

        return corrected_forecast
