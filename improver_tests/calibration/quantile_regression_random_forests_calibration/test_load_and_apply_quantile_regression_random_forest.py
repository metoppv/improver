# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the LoadAndApplyQRF plugin."""

import numpy as np
import pytest
from iris.cube import Cube

from improver.calibration.load_and_apply_quantile_regression_random_forest import (
    LoadAndApplyQRF,
)
from improver.ensemble_copula_coupling.ensemble_copula_coupling import (
    RebadgeRealizationsAsPercentiles,
)
from improver.utilities.save import save_netcdf
from improver_tests.calibration.quantile_regression_random_forests_calibration.test_quantile_regression_random_forest import (
    _add_day_of_training_period,
    _create_ancil_file,
    _create_forecasts,
    _run_train_qrf,
)


@pytest.mark.parametrize("percentile_input", [True, False])
@pytest.mark.parametrize(
    "n_estimators,max_depth,random_state,compression,transformation,pre_transform_addition,extra_kwargs,include_static,quantiles,expected",
    [
        (2, 2, 55, 5, None, 0, {}, False, [0.5], [5.15, 5.65]),  # noqa Basic test case
        (100, 2, 55, 5, None, 0, {}, False, [0.5, 0.9], [[4.1, 5.1], [4.2, 5.1]]),  # noqa Multiple quantiles
        (1, 1, 55, 5, None, 0, {}, False, [0.5], [6.2, 6.2]),  # noqa Fewer estimators and reduced depth
        (1, 1, 73, 5, None, 0, {}, False, [0.5], [4.2, 6.2]),  # Different random state
        (2, 2, 55, 5, "log", 10, {}, False, [0.5], [4.1, 5.1]),  # Log transformation
        (
            2,
            2,
            55,
            5,
            "log10",
            10,
            {},
            False,
            [0.5],
            [4.1, 5.1],
        ),  # Log10 transformation
        (
            2,
            2,
            55,
            5,
            "sqrt",
            10,
            {},
            False,
            [0.5],
            [4.1, 5.1],
        ),  # Square root transformation
        (
            2,
            2,
            55,
            5,
            "cbrt",
            10,
            {},
            False,
            [0.5],
            [4.1, 5.1],
        ),  # Cube root transformation
        (2, 2, 55, 5, None, 0, {"max_samples_leaf": 0.5}, False, [0.5], [5.15, 6.2]),  # noqa # Different criterion
        (2, 5, 55, 5, None, 0, {}, True, [0.5], [5.15, 5.65]),  # Include static data
    ],
)
def test_load_and_apply_qrf(
    tmp_path,
    percentile_input,
    n_estimators,
    max_depth,
    random_state,
    compression,
    transformation,
    pre_transform_addition,
    extra_kwargs,
    include_static,
    quantiles,
    expected,
):
    """Test the LoadAndApplyQRF plugin."""
    feature_config = {"wind_speed_at_10m": ["mean", "std", "latitude", "longitude"]}

    model_output = _run_train_qrf(
        tmp_path,
        feature_config,
        n_estimators,
        max_depth,
        random_state,
        compression,
        transformation,
        pre_transform_addition,
        extra_kwargs,
        include_static,
        forecast_reference_times=[
            "20170101T0000Z",
            "20170102T0000Z",
        ],
        validity_times=[
            "20170101T1200Z",
            "20170102T1200Z",
        ],
        day_of_training_period=[0, 1],
        realization_data=[2, 6, 10],
        truth_data=[[4.2, 6.2], [4.1, 5.1]],
    )

    frt = "20170103T0000Z"
    vt = "20170103T1200Z"
    day_of_training_period = 2
    data = np.arange(6, (len(quantiles) * 6) + 1, 6)

    forecast_cube = _create_forecasts(frt, vt, data)
    forecast_cube = _add_day_of_training_period(
        forecast_cube, day_of_training_period, "forecast_reference_time"
    )
    if percentile_input:
        forecast_cube = RebadgeRealizationsAsPercentiles()(forecast_cube)

    features_dir = tmp_path / "features"
    features_dir.mkdir(parents=True)
    forecast_filepath = str(features_dir / "forecast.nc")
    save_netcdf(forecast_cube, forecast_filepath)

    file_paths = [model_output, forecast_filepath]

    if include_static:
        ancil_cube = _create_ancil_file()
        ancil_filepath = features_dir / "ancil.nc"
        save_netcdf(ancil_cube, ancil_filepath)
        file_paths.append(str(ancil_filepath))

    plugin = LoadAndApplyQRF(
        feature_config,
        "wind_speed_at_10m",
        transformation=transformation,
        pre_transform_addition=pre_transform_addition,
    )
    result = plugin.process(file_paths=file_paths)
    assert isinstance(result, Cube)

    assert result.data.shape == (1, 1, len(quantiles), 2)
    assert np.allclose(result.data, expected, rtol=1e-2)

    # Check that the metadata is as expected
    assert result.name() == "wind_speed_at_10m"
    assert result.units == "m s-1"


@pytest.mark.parametrize("exception", ["no_model_output", "no_features"])
def test_exceptions(
    tmp_path,
    exception,
):
    """Test the LoadAndApplyQRF plugin."""
    feature_config = {"wind_speed_at_10m": ["mean", "std", "latitude", "longitude"]}

    n_estimators = 2
    max_depth = 2
    random_state = 55
    compression = 5
    transformation = None
    pre_transform_addition = 0
    extra_kwargs = {}
    include_static = False
    quantiles = [0.5]

    model_output = _run_train_qrf(
        tmp_path,
        feature_config,
        n_estimators,
        max_depth,
        random_state,
        compression,
        transformation,
        pre_transform_addition,
        extra_kwargs,
        include_static,
        forecast_reference_times=[
            "20170101T0000Z",
            "20170102T0000Z",
        ],
        validity_times=[
            "20170101T1200Z",
            "20170102T1200Z",
        ],
        day_of_training_period=[0, 1],
        realization_data=[2, 6, 10],
        truth_data=[[4.2, 6.2], [4.1, 5.1]],
    )

    frt = "20170103T0000Z"
    vt = "20170103T1200Z"
    day_of_training_period = 2
    data = np.arange(6, (len(quantiles) * 6) + 1, 6)

    forecast_cube = _create_forecasts(frt, vt, data)
    forecast_cube = _add_day_of_training_period(
        forecast_cube, day_of_training_period, "forecast_reference_time"
    )

    features_dir = tmp_path / "features"
    features_dir.mkdir(parents=True)
    forecast_filepath = str(features_dir / "forecast.nc")
    save_netcdf(forecast_cube, forecast_filepath)

    plugin = LoadAndApplyQRF(
        feature_config,
        "wind_speed_at_10m",
        transformation=transformation,
        pre_transform_addition=pre_transform_addition,
    )

    if exception == "no_qrf_model":
        file_paths = [forecast_filepath]
        with pytest.raises(ValueError, match="No QRF model found"):
            plugin.process(file_paths=file_paths)

    if exception == "no_features":
        file_paths = [model_output]
        with pytest.raises(ValueError, match="No features found"):
            plugin.process(file_paths=file_paths)
