# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the PrepareAndApplyQRF plugin."""

import numpy as np
import pandas as pd
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


@pytest.fixture
def set_up_for_unexpected():
    """Set up common elements for the unexpected tests."""
    feature_config = {"wind_speed_at_10m": ["mean", "std", "latitude", "longitude"]}

    n_estimators = 2
    max_depth = 2
    random_state = 55
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
    )
    return (
        qrf_model,
        transformation,
        pre_transform_addition,
        cube_inputs,
        forecast_cube,
        ancil_cube,
        plugin,
    )


def set_up_for_expected(
    feature_config,
    n_estimators,
    max_depth,
    random_state,
    transformation,
    pre_transform_addition,
    extra_kwargs,
    include_dynamic,
    include_static,
    include_nans,
    include_latlon_nans,
    percentile_input,
    site_id,
    quantiles,
    alt_cycletime,
):
    """Set up common elements for the tests where the expected test result will be
    checked.

    Args:
        feature_config: Feature configuration dictionary.
        n_estimators: Number of trees for the random forest.
        max_depth: Maximum depth of each tree.
        random_state: Random seed for reproducibility.
        transformation: Transformation to be applied to the data before fitting.
        pre_transform_addition: Value to be added before transformation.
        extra_kwargs: Extra keyword arguments for the random forest.
        include_dynamic: Whether to include an additional dynamic feature.
        include_static: Whether to include an additional static feature.
        include_nans: Whether to include NaNs in the input data.
        include_latlon_nans: Whether to include NaNs in the latitude and longitude.
        percentile_input: Whether the input forecast cube uses percentiles.
        site_id: List of strings defining the unique site ID keys.
        quantiles: List of quantiles to be predicted.
        alt_cycletime: Alternative cycletime to be used for the dynamic feature.
    """
    if not alt_cycletime:
        alt_cycletime = "20170103T0000Z"

    qrf_model = _run_train_qrf(
        feature_config,
        n_estimators,
        max_depth,
        random_state,
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
        site_id=site_id,
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
    if include_dynamic:
        dynamic_cube = _create_forecasts(
            alt_cycletime,
            vt,
            data + 0.5,  # Slightly different data to the target feature
            return_cube=True,
        )
        dynamic_cube.rename("air_temperature")
        dynamic_cube = _add_day_of_training_period_to_cube(
            dynamic_cube, day_of_training_period, "forecast_reference_time"
        )
        cube_inputs.append(dynamic_cube)

    if include_static:
        ancil_cube = _create_ancil_file(return_cube=True)
        cube_inputs.append(ancil_cube)

    if include_nans:
        # Add some NaNs to the input data to check that they are handled
        for cube in cube_inputs:
            if cube.name() == "distance_to_water":
                cube.data[0] = np.nan
            else:
                cube.data[0, 0] = np.nan

    if include_latlon_nans:
        # Add some NaNs to the latitude and longitude to check that they are handled
        for cube in cube_inputs:
            cube.coord("latitude").points[1] = np.nan
            cube.coord("longitude").points[1] = np.nan
    return (
        qrf_model,
        cube_inputs,
    )


# Disable ruff formatting to keep the parameter combinations aligned for readability.
# fmt: off
@pytest.mark.parametrize("percentile_input", [True, False])
@pytest.mark.parametrize(
    "site_id", [["wmo_id"], ["station_id"], ["latitude", "longitude", "altitude"]]
)
@pytest.mark.parametrize(
    "n_estimators,max_depth,random_state,transformation,pre_transform_addition,extra_kwargs,include_dynamic,include_static,include_nans,include_latlon_nans,quantiles,expected",
    [
        (2, 2, 55, None, 0, {}, False, False, False, False, [0.5], [4.1, 5.65]),  # Basic test case
        (100, 2, 55, None, 0, {}, False, False, False, False, [1 / 3, 2 / 3], [[4.1, 5.1], [5.1, 5.1]]),  # Multiple quantiles
        (1, 1, 55, None, 0, {}, False, False, False, False, [0.5], [4.1, 6.2]),  # Fewer estimators and reduced depth
        (1, 1, 73, None, 0, {}, False, False, False, False, [0.5], [4.2, 6.2]),  # Different random state
        (2, 2, 55, "log", 10, {}, False, False, False, False, [0.5], [4.1, 5.64]),  # Log transformation
        (2, 2, 55, "log10", 10, {}, False, False, False, False, [0.5], [4.1, 5.64]),  # Log10 transformation
        (2, 2, 55, "sqrt", 10, {}, False, False, False, False, [0.5], [4.1, 5.64]),  # Square root transformation
        (2, 2, 55, "cbrt", 10, {}, False, False, False, False, [0.5], [4.1, 5.64]),  # Cube root transformation
        (2, 2, 55, None, 0, {"max_samples_leaf": 0.5}, False, False, False, False, [0.5], [4.1, 6.2]),  # Different criterion
        (2, 5, 55, None, 0, {}, True, False, False, False, [0.5], [4.1, 4.6]),  # Include an additional dynamic feature
        (2, 5, 55, None, 0, {}, False, True, False, False, [0.5], [4.1, 5.65]),  # Include an additional static feature
        (2, 5, 55, None, 0, {}, True, True, False, False, [0.5], [4.1, 4.6]),  # Include an additional dynamic and static feature
        (2, 2, 55, None, 0, {}, False, False, True, False, [0.5], [4.1, 5.65]),  # NaNs in input data
        (2, 2, 55, None, 0, {}, False, False, False, True, [0.5], [4.1, 5.65]),  # NaNs in lat/lon
        (2, 2, 55, None, 0, {}, True, False, False, True, [0.5], [4.1, 4.6]),  # NaNs in lat/lon and dynamic feature
        (2, 2, 55, None, 0, {}, False, True, False, True, [0.5], [4.1, 5.65]),  # NaNs in lat/lon and static feature
        (2, 2, 55, None, 0, {}, True, True, False, True, [0.5], [4.1, 4.6]),  # NaNs in lat/lon, dynamic and static feature
        (2, 2, 55, None, 0, {}, True, True, True, True, [0.5], [4.6, 4.6]),  # NaNs in lat/lon, dynamic and static feature and input data
    ],
)
# fmt: on
def test_prepare_and_apply_qrf(
    percentile_input,
    site_id,
    n_estimators,
    max_depth,
    random_state,
    transformation,
    pre_transform_addition,
    extra_kwargs,
    include_dynamic,
    include_static,
    include_nans,
    include_latlon_nans,
    quantiles,
    expected
):
    """Test the PrepareAndApplyQRF plugin."""
    feature_config = {"wind_speed_at_10m": ["mean", "std", "latitude", "longitude"]}

    qrf_model, cube_inputs = set_up_for_expected(
        feature_config, n_estimators, max_depth, random_state,
        transformation, pre_transform_addition, extra_kwargs,
        include_dynamic, include_static, include_nans, include_latlon_nans,
        percentile_input, site_id, quantiles)

    result = PrepareAndApplyQRF(
        feature_config,
        "wind_speed_at_10m",
        unique_site_id_keys=site_id,
    )(cube_inputs, (qrf_model, transformation, pre_transform_addition))

    assert isinstance(result, Cube)
    assert result.data.shape == (len(quantiles), 2)

    if include_latlon_nans and site_id == ["latitude", "longitude", "altitude"]:
        assert np.allclose(result.data, expected, rtol=1)
    else:
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
    "n_estimators,max_depth,random_state,cycletime,include_dynamic,include_static,expected",
    [
        (2, 2, 55, None, False, False, [4.1, 5.65]),  # Basic test case
        (2, 2, 55, "20170103T0000Z", False, False, [4.1, 5.65]),  # Specify cycletime
        (2, 2, 55, "20170102T2300Z", True, False, [4.6, 5.1]),  # Cycletime with dynamic feature
        (2, 2, 55, "20170102T2300Z", False, True, [4.1, 5.65]),  # Cycletime with static feature
        (2, 2, 55, "20170102T2300Z", True, True, [4.6, 5.1]),  # Cycletime with dynamic and static feature
    ],
)
def test_mismatching_temporal_coordinates(
    n_estimators,
    max_depth,
    random_state,
    cycletime,
    include_dynamic,
    include_static,
    expected
):
    """Test the PrepareAndApplyQRF plugin."""
    feature_config = {"wind_speed_at_10m": ["mean", "std", "latitude", "longitude"]}
    transformation = None
    pre_transform_addition = 0
    extra_kwargs = {}
    include_nans = False
    include_latlon_nans = False
    percentile_input = False
    site_id = ["wmo_id"]
    quantiles = [0.5]
    if not cycletime or cycletime == "20170103T0000Z":
        forecast_period = 12  # hours
    else:
        forecast_period = 13  # hours

    if include_dynamic:
        feature_config["air_temperature"] = ["mean", "std"]
    if include_static:
        feature_config["distance_to_water"] = ["static"]

    qrf_model, cube_inputs = set_up_for_expected(
        feature_config, n_estimators, max_depth, random_state,
        transformation, pre_transform_addition, extra_kwargs,
        include_dynamic, include_static, include_nans, include_latlon_nans,
        percentile_input, site_id, quantiles, cycletime)

    result = PrepareAndApplyQRF(
        feature_config,
        "wind_speed_at_10m",
        unique_site_id_keys=site_id,
        cycletime=cycletime,
        forecast_period=forecast_period*3600,
    )(cube_inputs, (qrf_model, transformation, pre_transform_addition))
    import pdb
    pdb.set_trace()
    assert isinstance(result, Cube)
    assert result.data.shape == (len(quantiles), 2)
    assert np.allclose(result.data, expected, rtol=1e-2)

    # Check that the metadata is as expected
    assert result.name() == "wind_speed_at_10m"
    assert result.units == "m s-1"
    assert result.coords("realization")
    assert result.coord("realization").units == "1"
    assert np.allclose(result.coord("realization").points, range(len(quantiles)))
    assert result.coord("forecast_reference_time").cell(0) == pd.Timestamp("20170103T0000Z")
    assert result.coord("forecast_period").points[0] == 12 * 3600


def test_no_model_output(set_up_for_unexpected):
    """Test PrepareAndApplyQRF plugin behaviour when no model is provided."""
    (
        qrf_model,
        transformation,
        pre_transform_addition,
        cube_inputs,
        forecast_cube,
        ancil_cube,
        plugin,
    ) = set_up_for_unexpected

    with pytest.warns(UserWarning, match="Unable to apply Quantile Regression Random Forest model."):
        result = plugin(cube_inputs, qrf_descriptors=None)
    assert isinstance(result, Cube)
    assert result.name() == "wind_speed_at_10m"
    assert result.units == "m s-1"
    assert result.data.shape == forecast_cube.data.shape
    assert np.allclose(result.data, forecast_cube.data)


def test_no_features(set_up_for_unexpected):
    """Test PrepareAndApplyQRF plugin behaviour when no features are provided."""
    (
        qrf_model,
        transformation,
        pre_transform_addition,
        cube_inputs,
        forecast_cube,
        ancil_cube,
        plugin,
    ) = set_up_for_unexpected

    qrf_descriptors = (qrf_model, transformation, pre_transform_addition)
    with pytest.raises(ValueError, match="No target forecast provided."):
        plugin(CubeList(), qrf_descriptors=qrf_descriptors)

def test_missing_target_feature(set_up_for_unexpected):
    """Test PrepareAndApplyQRF plugin behaviour when the target feature is missing."""
    (
        qrf_model,
        transformation,
        pre_transform_addition,
        cube_inputs,
        forecast_cube,
        ancil_cube,
        plugin,
    ) = set_up_for_unexpected

    qrf_descriptors = (qrf_model, transformation, pre_transform_addition)
    with pytest.raises(ValueError, match="No target forecast provided."):
        plugin(
            CubeList([ancil_cube]),
            qrf_descriptors=qrf_descriptors,
        )

def test_missing_static_feature(set_up_for_unexpected):
    """Test PrepareAndApplyQRF plugin behaviour when a static feature is missing."""
    (
        qrf_model,
        transformation,
        pre_transform_addition,
        cube_inputs,
        forecast_cube,
        ancil_cube,
        plugin,
    ) = set_up_for_unexpected

    qrf_descriptors = (qrf_model, transformation, pre_transform_addition)
    feature_config = {
        "wind_speed_at_10m": ["mean", "std"],
        "distance_to_water": ["static"],
    }
    plugin.feature_config = feature_config
    with pytest.raises(ValueError, match="The number of cubes loaded."):
        plugin(CubeList([forecast_cube]), qrf_descriptors=qrf_descriptors)

def test_unused_static_feature(set_up_for_unexpected):
    """Test PrepareAndApplyQRF plugin behaviour when a static feature is unused.
    This test is to show that the plugin will ignore features that are provided
    but not specified in the feature_config."""
    (
        qrf_model,
        transformation,
        pre_transform_addition,
        cube_inputs,
        forecast_cube,
        ancil_cube,
        plugin,
    ) = set_up_for_unexpected

    qrf_descriptors = (qrf_model, transformation, pre_transform_addition)
    feature_config = {
        "wind_speed_at_10m": ["mean", "std", "latitude", "longitude"],
    }
    plugin.feature_config = feature_config
    result = plugin(CubeList([forecast_cube, ancil_cube]), qrf_descriptors=qrf_descriptors)
    assert isinstance(result, Cube)
    assert result.name() == "wind_speed_at_10m"
    assert result.units == "m s-1"
    assert result.data.shape == forecast_cube.data.shape

def test_missing_dynamic_feature(set_up_for_unexpected):
    """Test PrepareAndApplyQRF plugin behaviour when a dynamic feature is missing."""
    (
        qrf_model,
        transformation,
        pre_transform_addition,
        cube_inputs,
        forecast_cube,
        ancil_cube,
        plugin,
    ) = set_up_for_unexpected

    qrf_descriptors = (qrf_model, transformation, pre_transform_addition)
    feature_config = {
        "wind_speed_at_10m": ["mean", "std"],
        "air_temperature": ["mean", "std"],
    }
    plugin.feature_config = feature_config
    with pytest.raises(ValueError, match="The number of cubes loaded."):
        plugin(CubeList([forecast_cube]), qrf_descriptors=qrf_descriptors)

def test_no_quantile_forest_package(set_up_for_unexpected):
    """Test PrepareAndApplyQRF plugin behaviour when the quantile_forest package is not installed."""
    (
        qrf_model,
        transformation,
        pre_transform_addition,
        cube_inputs,
        forecast_cube,
        ancil_cube,
        plugin,
    ) = set_up_for_unexpected

    qrf_descriptors = (qrf_model, transformation, pre_transform_addition)
    plugin.quantile_forest_installed = False
    with pytest.warns(UserWarning, match="Unable to apply Quantile Regression Random Forest model."):
        result = plugin(CubeList([forecast_cube]), qrf_descriptors=qrf_descriptors)
    assert isinstance(result, Cube)
    assert result.name() == "wind_speed_at_10m"
    assert result.units == "m s-1"
    assert result.data.shape == forecast_cube.data.shape
    assert np.allclose(result.data, forecast_cube.data)
