# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the PrepareAndApplyQRF plugin."""

import numpy as np
import pytest
from iris.coords import AuxCoord
from iris.cube import Cube, CubeList

from improver.calibration.load_and_apply_quantile_regression_random_forest import (
    PrepareAndApplyQRF,
)
from improver.ensemble_copula_coupling.ensemble_copula_coupling import (
    RebadgeRealizationsAsPercentiles,
)
from improver_tests.calibration.quantile_regression_random_forests_calibration.test_quantile_regression_random_forest import (
    _create_ancil_file,
    _create_forecasts,
    _run_train_qrf,
)

pytest.importorskip("quantile_forest")


def _add_day_of_training_period_to_cube(cube, day_of_training_period, secondary_coord):
    """Add day of training period coordinate to the cube.
    Args:
        cube: Cube to which the day of training period coordinate will be added.
        day_of_training_period: Day of training period to be added.
        secondary_coord: Coordinate to associate the day of training period with.
    Returns:
        Cube with the day of training period coordinate added.
    """
    dims = cube.coord_dims(secondary_coord)
    day_of_training_period_coord = AuxCoord(
        np.array(day_of_training_period, dtype=np.int32),
        long_name="day_of_training_period",
        units="1",
    )
    cube.add_aux_coord(day_of_training_period_coord, data_dims=dims)
    return cube


@pytest.mark.parametrize("percentile_input", [True, False])
@pytest.mark.parametrize(
    "n_estimators,max_depth,random_state,compression,transformation,pre_transform_addition,extra_kwargs,include_static,quantiles,expected",
    [
        (2, 2, 55, 5, None, 0, {}, False, [0.5], [4.1, 5.65]),  # noqa Basic test case
        (100, 2, 55, 5, None, 0, {}, False, [1 / 3, 2 / 3], [[4.1, 5.1], [5.1, 5.1]]),  # noqa Multiple quantiles
        (1, 1, 55, 5, None, 0, {}, False, [0.5], [4.1, 6.2]),  # noqa Fewer estimators and reduced depth
        (1, 1, 73, 5, None, 0, {}, False, [0.5], [4.2, 6.2]),  # Different random state
        (2, 2, 55, 5, "log", 10, {}, False, [0.5], [4.1, 5.64]),  # Log transformation
        (2, 2, 55, 5, "log10", 10, {}, False, [0.5], [4.1, 5.64]),  # noqa Log10 transformation
        (2, 2, 55, 5, "sqrt", 10, {}, False, [0.5], [4.1, 5.64]),  # noqa Square root transformation
        (2, 2, 55, 5, "cbrt", 10, {}, False, [0.5], [4.1, 5.64]),  # noqa Cube root transformation
        (2, 2, 55, 5, None, 0, {"max_samples_leaf": 0.5}, False, [0.5], [4.1, 6.2]),  # noqa # Different criterion
        (2, 5, 55, 5, None, 0, {}, True, [0.5], [4.1, 5.65]),  # noqa Include an additional static feature
    ],
)
def test_prepare_and_apply_qrf(
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
    """Test the PrepareAndApplyQRF plugin."""
    feature_config = {"wind_speed_at_10m": ["mean", "std", "latitude", "longitude"]}

    qrf_model = _run_train_qrf(
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
        realization_data=[2, 6, 10],
        truth_data=[4.2, 6.2, 4.1, 5.1],
    )

    frt = "20170103T0000Z"
    vt = "20170103T1200Z"
    data = np.arange(6, (len(quantiles) * 6) + 1, 6)
    day_of_training_period = 2
    cube_inputs = CubeList()
    forecast_cube = _create_forecasts(frt, vt, data, return_cube=True)

    forecast_cube = _add_day_of_training_period_to_cube(
        forecast_cube, day_of_training_period, "forecast_reference_time"
    )

    if percentile_input:
        forecast_cube = RebadgeRealizationsAsPercentiles()(forecast_cube)

    cube_inputs.append(forecast_cube)
    if include_static:
        ancil_cube = _create_ancil_file(return_cube=True)
        cube_inputs.append(ancil_cube)

    # cube_inputs, qrf_model = LoadForApplyQRF()(file_paths)
    result = PrepareAndApplyQRF(
        feature_config,
        "wind_speed_at_10m",
        transformation=transformation,
        pre_transform_addition=pre_transform_addition,
    )(cube_inputs, qrf_model)
    assert isinstance(result, Cube)

    assert result.data.shape == (len(quantiles), 2)
    assert np.allclose(result.data, expected, rtol=1e-2)

    # Check that the metadata is as expected
    assert result.name() == "wind_speed_at_10m"
    assert result.units == "m s-1"

    if percentile_input:
        percentiles = np.array(quantiles, dtype=np.float32) * 100
        assert result.coords("percentile")
        assert result.coord("percentile").units == "%"
        assert np.allclose(result.coord("percentile").points, percentiles)
    else:
        assert result.coords("realization")
        assert result.coord("realization").units == "1"
        assert np.allclose(result.coord("realization").points, range(len(quantiles)))


@pytest.mark.parametrize(
    "exception",
    [
        "no_model_output",
        "no_features",
        "missing_target_feature",
        "missing_static_feature",
        "missing_dynamic_feature",
        "no_quantile_forest_package",
    ],
)
def test_unexpected(
    exception,
):
    """Test PrepareAndApplyQRF plugin behaviour in atypical situations."""
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

    qrf_model = _run_train_qrf(
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
        realization_data=[2, 6, 10],
        truth_data=[4.2, 6.2, 4.1, 5.1],
    )

    frt = "20170103T0000Z"
    vt = "20170103T1200Z"
    data = np.arange(6, (len(quantiles) * 6) + 1, 6)
    day_of_training_period = 2
    forecast_cube = _create_forecasts(frt, vt, data, return_cube=True)
    forecast_cube = _add_day_of_training_period_to_cube(
        forecast_cube, day_of_training_period, "forecast_reference_time"
    )
    cube_inputs = CubeList([forecast_cube])
    ancil_cube = _create_ancil_file(return_cube=True)
    cube_inputs.append(ancil_cube)

    plugin = PrepareAndApplyQRF(
        feature_config,
        "wind_speed_at_10m",
        transformation=transformation,
        pre_transform_addition=pre_transform_addition,
    )

    if exception == "no_model_output":
        result = plugin(cube_inputs, qrf_model=None)
        assert isinstance(result, Cube)
        assert result.name() == "wind_speed_at_10m"
        assert result.units == "m s-1"
        assert result.data.shape == forecast_cube.data.shape
        assert np.allclose(result.data, forecast_cube.data)
    elif exception == "no_features":
        with pytest.raises(ValueError, match="No target forecast provided."):
            plugin(CubeList(), qrf_model=qrf_model)
    elif exception == "missing_target_feature":
        with pytest.raises(ValueError, match="No target forecast provided."):
            plugin(CubeList([ancil_cube]), qrf_model=qrf_model)
    elif exception == "missing_static_feature":
        feature_config = {
            "wind_speed_at_10m": ["mean", "std"],
            "distance_to_water": ["static"],
        }
        plugin.feature_config = feature_config
        with pytest.raises(ValueError, match="The number of cubes loaded."):
            plugin(CubeList([forecast_cube]), qrf_model=qrf_model)
    elif exception == "missing_dynamic_feature":
        feature_config = {
            "wind_speed_at_10m": ["mean", "std"],
            "air_temperature": ["mean", "std"],
        }
        plugin.feature_config = feature_config
        with pytest.raises(ValueError, match="The number of cubes loaded."):
            plugin(CubeList([forecast_cube]), qrf_model=qrf_model)
    elif exception == "no_quantile_forest_package":
        plugin.quantile_forest_installed = False
        result = plugin(CubeList([forecast_cube]), qrf_model=qrf_model)
        assert isinstance(result, Cube)
        assert result.name() == "wind_speed_at_10m"
        assert result.units == "m s-1"
        assert result.data.shape == forecast_cube.data.shape
        assert np.allclose(result.data, forecast_cube.data)
    else:
        raise ValueError(f"Unknown exception type: {exception}")
