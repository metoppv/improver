# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""CLI to apply rainforests calibration."""

import warnings

from improver import cli


@cli.clizefy
@cli.with_output
def process(
    forecast: cli.inputcube,
    *features: cli.inputcube,
    model_config: cli.inputjson,
    output_thresholds: cli.comma_separated_list_of_float = None,
    output_threshold_config: cli.inputjson = None,
    threshold_units: str = None,
    threads: int = 1,
    bin_data: bool = False,
):
    """
    Calibrate a forecast cube using the Rainforests method.

    Ensemble forecasts must be in realization representation. Deterministic forecasts
    can be processed to produce a pseudo-ensemble; a realization dimension will be added
    to deterministic forecast cubes if one is not already present.

    This calibration is done in a situation dependent fashion using a series of
    decision-tree models to construct representative error distributions which are
    then used to map each input ensemble member onto an error distribution. The
    error distributions are averaged in probability space, and interpolated to the
    output thresholds.

    It is assumed that the models have been trained using the `>=` comparator; i.e.
    they predict the probability that the error is greater than or equal to the various
    error thresholds. The output probability cube also uses the `>=` comparator.

    Args:
        forecast_cube (iris.cube.Cube):
            Cube containing the forecast to be calibrated; must be as realizations.
        feature_cubes (iris.cube.Cubelist):
            Cubelist containing the feature variables (physical parameters) used as inputs
            to the tree-models for the generation of the associated error distributions.
            Feature cubes are expected to have the same dimensions as forecast_cube, with
            the exception of the realization dimension. Where the feature_cube contains a
            realization dimension this is expected to be consistent, otherwise the cube will
            be broadcast along the realization dimension.
        model_config (dict):
            Dictionary containing RainForests model configuration data.
        output_thresholds (list):
            List of thresholds at which to evaluate output probabilities.
        output_threshold_config (dict):
            Threshold configuration dictionary where the keys are strings representing
            thresholds. The threshold config should follow the same format as that of
            the threshold cli, however here only the threshold keys are used and the
            threshold values are disregarded.
        threshold_units (str):
            Units in which threshold_values are specified. If not provided the units are
            assumed to be the same as those of the input cube. Specifying the units here
            will allow a suitable conversion to match the input units of forecast_cube.
        threads (int):
            Number of threads to use during prediction with tree-model objects.
        bin_data:
            Bin data according to splits used in models. This speeds up prediction
            if there are many data points which fall into the same bins for all threshold models.
            Limits the calculation of common feature values by only calculating them once.

    Returns:
        iris.cube.Cube:
            The forecast cube following calibration.
    """
    from iris.cube import CubeList

    from improver.calibration.rainforest_calibration import ApplyRainForestsCalibration

    if output_threshold_config and output_thresholds:
        raise ValueError(
            "--output-threshold-config and --output-thresholds are mutually exclusive "
            "- please set one or the other, not both"
        )
    if (not output_threshold_config) and (not output_thresholds):
        raise ValueError(
            "One of --output-threshold-config and --output-thresholds must be specified"
        )

    if output_threshold_config:
        message = "Fuzzy bounds are not supported. Values of output-threshold-config \
            will be ignored."
        warnings.warn(message)
        thresholds = [float(key) for key in output_threshold_config.keys()]
    else:
        thresholds = [float(x) for x in output_thresholds]
    return ApplyRainForestsCalibration(
        model_config_dict=model_config, threads=threads, bin_data=bin_data
    ).process(
        forecast,
        CubeList(features),
        output_thresholds=thresholds,
        threshold_units=threshold_units,
    )
