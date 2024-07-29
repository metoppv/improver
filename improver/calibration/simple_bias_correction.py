# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Simple bias correction plugins."""

import warnings
from typing import Dict, Optional, Union

import iris
import numpy.ma as ma
from iris.cube import Cube, CubeList
from numpy import ndarray

from improver import BasePlugin
from improver.calibration import add_warning_comment, split_forecasts_and_bias_files
from improver.calibration.utilities import (
    check_forecast_consistency,
    create_unified_frt_coord,
    filter_non_matching_cubes,
    get_frt_hours,
)
from improver.metadata.probabilistic import is_probability
from improver.metadata.utilities import (
    create_new_diagnostic_cube,
    generate_mandatory_attributes,
)
from improver.utilities.common_input_handle import as_cubelist
from improver.utilities.cube_manipulation import (
    clip_cube_data,
    collapsed,
    get_dim_coord_names,
)


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
    # Set the masks explicitly to inherit the masks from both cubes
    if isinstance(forecasts.data, ma.MaskedArray) or isinstance(
        truths.data, ma.MaskedArray
    ):
        forecast_errors.data.mask = ma.mask_or(
            ma.asarray(forecasts.data).mask, ma.asarray(truths.data).mask
        )
    if collapse_dim in get_dim_coord_names(forecast_errors):
        mean_forecast_error = collapsed(
            forecast_errors, collapse_dim, iris.analysis.MEAN
        )
        return mean_forecast_error.data
    return forecast_errors.data


def apply_additive_correction(
    forecast: Cube, bias: Cube, fill_masked_bias_values: bool = True
) -> ndarray:
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
    if fill_masked_bias_values and isinstance(bias.data, ma.masked_array):
        bias.data = ma.MaskedArray.filled(bias.data, 0.0)
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

    def _ensure_single_valued_forecast(self, forecasts: Cube) -> Cube:
        """
        Check to see if an ensemble based dimension (realization, percentile, threshold)
        is present. If threshold dim present, or realization/percentile dimensions have
        multiple values a ValueError will be raised. Otherwise the percentile/realization
        dim coord is demoted to an aux coord.

        Args:
            forecast:
                Cube containing historical forecast values used in bias correction.

        Returns:
            Cube with unit realization/percentile dim coords demoted to aux coordinate.
        """
        forecast_dim_coords = get_dim_coord_names(forecasts)
        if is_probability(forecasts):
            raise ValueError(
                "Forecasts provided as probability data. Historical forecasts must be single"
                "valued realisable forecast (realization, percentile or ensemble mean)."
            )
        elif "percentile" in forecast_dim_coords:
            if forecasts.coord("percentile").points.size > 1:
                raise ValueError(
                    "Multiple percentile values detected. Expect historical forecasts"
                    "to be single valued forecasts."
                )
        elif "realization" in forecast_dim_coords:
            if forecasts.coord("realization").points.size > 1:
                raise ValueError(
                    "Multiple realization values detected. Expect historical forecasts"
                    "to be single valued forecasts."
                )
        forecasts = iris.util.squeeze(forecasts)
        return forecasts

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
            name=f"forecast_error_of_{forecasts.name()}",
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

        The historical forecasts are expected to be representative single-valued forecasts
        (eg. control or ensemble mean forecast). If a non-unit ensemble based dimension
        (realization, threshold or percentile) is present then a ValueError will be raised.

        The bias here is evaluated point-by-point and the associated bias cube
        will retain the same spatial dimensions as the input cubes. By using a
        point-by-point approach, the bias-correction enables a form of statistical
        downscaling where coherent biases exist between a coarse forecast dataset and
        finer truth dataset.

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
        self._ensure_single_valued_forecast(historic_forecasts)

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

    def __init__(
        self,
        lower_bound: Optional[float] = None,
        upper_bound: Optional[float] = None,
        fill_masked_bias_values: Optional[bool] = False,
    ):
        """
        Initialise class for applying simple bias correction.

        Args:
            lower_bound:
                A lower bound below which all values will be remapped to
                after the bias correction step.
            upper_bound:
                An upper bound above which all values will be remapped to
                after the bias correction step.
            fill_masked_bias_values:
                Flag to specify whether masked areas in the bias data
                should be filled to an appropriate fill value.
        """
        self._correction_method = apply_additive_correction
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound
        self._fill_masked_bias_values = fill_masked_bias_values

    def _split_forecasts_and_bias(self, cubes: CubeList):
        """
        Wrapper for the split_forecasts_and_bias_files function.

        Args:
            cubes:
                Cubelist containing the input forecast and bias cubes.

        Return:
            - Cube containing the forecast data to be bias-corrected.
            - Cubelist containing the bias data to use in bias-correction.
              Or None if no bias data is provided.
        """
        forecast_cube, bias_cubes = split_forecasts_and_bias_files(cubes)

        # Check whether bias data supplied, if not then return unadjusted input cube.
        # This behaviour is to allow spin-up of the bias-correction terms.
        if not bias_cubes:
            msg = (
                "There are no forecast_error (bias) cubes provided for calibration. "
                "The uncalibrated forecast will be returned."
            )
            warnings.warn(msg)
            forecast_cube = add_warning_comment(forecast_cube)
            return forecast_cube, None
        else:
            bias_cubes = as_cubelist(bias_cubes)
            return forecast_cube, bias_cubes

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
            Cube containing the mean bias evaluated from set of bias_values.
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
                        "bias values defined over multiple reference forecast values. Bias cube"
                        f"for frt: {bias_cube.coord('forecast_reference_time').points} has bounds"
                        f"{bias_cube.coord('forecast_reference_time').bounds}, expected {None}."
                    )
            bias_values = bias_values.merge_cube()
            frt_coord = create_unified_frt_coord(
                bias_values.coord("forecast_reference_time")
            )
            mean_bias = collapsed(
                bias_values, "forecast_reference_time", iris.analysis.MEAN
            )
            mean_bias.replace_coord(frt_coord)
            return mean_bias

    def _check_forecast_bias_consistent(
        self, forecast: Cube, bias_data: CubeList
    ) -> None:
        """Check that forecast and bias values are defined over the same
        valid-hour and forecast-period.

        Checks that between the bias_data Cubes there is a single coordinate forecast_reference_time
        and single coordinate forecast_period. Then check forecast Cube contains the same single
        coordinate forecast_reference_time and single coordinate forecast_period.

        Args:
            forecast:
                Cube containing forecast data to be bias-corrected.
            bias:
                CubeList containing bias data to use in bias-correction.
        """
        bias_frt_hours = []
        for cube in bias_data:
            bias_frt_hours.extend(get_frt_hours(cube.coord("forecast_reference_time")))
        bias_frt_hours = set(bias_frt_hours)
        fcst_frt_hours = set(get_frt_hours(forecast.coord("forecast_reference_time")))
        combined_frt_hours = fcst_frt_hours | bias_frt_hours
        if len(bias_frt_hours) != 1:
            raise ValueError(
                "Multiple forecast_reference_time valid-hour values detected across bias datasets."
            )
        elif len(combined_frt_hours) != 1:
            raise ValueError(
                "forecast_reference_time valid-hour differ between forecast and bias datasets."
            )

        bias_period = []
        for cube in bias_data:
            bias_period.extend(cube.coord("forecast_period").points)
        bias_period = set(bias_period)
        fcst_period = set(forecast.coord("forecast_period").points)
        combined_period_values = fcst_period | bias_period
        if len(bias_period) != 1:
            raise ValueError(
                "Multiple forecast period values detected across bias datasets."
            )
        elif len(combined_period_values) != 1:
            print(combined_period_values)
            raise ValueError(
                "Forecast period differ between forecast and bias datasets."
            )

    def process(self, *cubes: Union[Cube, CubeList],) -> Cube:
        """        Split then apply bias correction using the specified bias values.

        Where the bias data is defined point-by-point, the bias-correction will also
        be applied in this way enabling a form of statistical downscaling where coherent
        biases exist between a coarse forecast dataset and finer truth dataset.

        Where a lower bound is specified, all values that fall below this
        lower bound (after bias correction) will be remapped to this value
        to ensure physically realistic values.

        If fill_masked_bias_values is True, the masked areas in bias data will be
        filled using an appropriate fill value to leave the forecast data unchanged
        in the masked areas.

        Args:
            cubes:
                A list of cubes containing:
                - A Cube containing the forecast to be calibrated. The input format is expected
                to be realizations.
                - A cube or cubelist containing forecast bias data over a specified
                set of forecast reference times. If a list of cubes is passed in, each cube
                should represent the forecast error for a single forecast reference time; the
                mean value will then be evaluated over the forecast_reference_time coordinate.

        Returns:
            Bias corrected forecast cube.
        """
        cubes = as_cubelist(*cubes)
        forecast, bias_cubes = self._split_forecasts_and_bias(cubes)
        if bias_cubes is None:
            return forecast

        self._check_forecast_bias_consistent(forecast, bias_cubes)
        bias = self._get_mean_bias(bias_cubes)

        corrected_forecast = forecast.copy()
        corrected_forecast.data = self._correction_method(
            forecast, bias, self._fill_masked_bias_values
        )

        if (self._lower_bound is not None) or (self._upper_bound is not None):
            corrected_forecast = clip_cube_data(
                corrected_forecast, self._lower_bound, self._upper_bound
            )

        return corrected_forecast
