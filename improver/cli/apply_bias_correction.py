#!/usr/bin/env python
# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""CLI to apply simple bias correction to ensemble members based on bias from the
reference forecast dataset."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(
    *input_cubes: cli.inputcube,
    lower_bound: float = None,
    upper_bound: float = None,
    fill_masked_bias_data: bool = False,
):
    """Apply simple bias correction to ensemble members based on the bias from the
    reference forecast dataset.

    Where the bias data is evaluated point-by-point, the bias-correction will also
    be applied in this way enabling a form of statistical downscaling where coherent
    biases exist between a coarse forecast dataset and finer truth dataset.

    The bias cube can either be passed in as a series of bias values for individual
    forecasts (from which the mean value is evaluated), or as a single bias value
    evaluated over a series of reference forecasts.

    A lower bound or upper bound can be set to ensure that corrected values are physically
    sensible post-bias correction.

    Args:
        input_cubes (iris.cube.Cube or list of iris.cube.Cube):
            A list of cubes containing:
            - A Cube containing the forecast to be calibrated. The input format is expected
            to be realizations.
            - A cube or cubelist containing forecast bias data over a specified
            set of forecast reference times. If a list of cubes is passed in, each cube
            should represent the forecast error for a single forecast reference time; the
            mean value will then be evaluated over the forecast_reference_time coordinate.
        lower_bound (float):
            Specifies a lower bound below which values will be remapped to.
        upper_bound (float):
            Specifies an upper bound above which values will be remapped to.
        fill_masked_bias_data (bool):
            Flag to specify whether to fill masked areas of the mean bias data with an
            appropriate fill-value.

    Returns:
        iris.cube.Cube:
            Forecast cube with bias correction applied on a per member basis.
    """
    import warnings

    import iris

    from improver.calibration import add_warning_comment, split_forecasts_and_bias_files
    from improver.calibration.simple_bias_correction import ApplyBiasCorrection

    forecast_cube, bias_cubes = split_forecasts_and_bias_files(input_cubes)

    # Check whether bias data supplied, if not then return unadjusted input cube.
    # This behaviour is to allow spin-up of the bias-correction terms.
    if not bias_cubes:
        msg = (
            "There are no forecast_error (bias) cubes provided for calibration. "
            "The uncalibrated forecast will be returned."
        )
        warnings.warn(msg)
        forecast_cube = add_warning_comment(forecast_cube)
        return forecast_cube
    else:
        bias_cubes = iris.cube.CubeList(bias_cubes)
        plugin = ApplyBiasCorrection()
        return plugin.process(
            forecast_cube, bias_cubes, lower_bound, upper_bound, fill_masked_bias_data
        )
