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


@pytest.mark.parametrize("percentile_input", [True])
@pytest.mark.parametrize(
    "site_id", [["wmo_id"], ["station_id"], ["latitude", "longitude", "altitude"]]
)
@pytest.mark.parametrize(
    "n_estimators,max_depth,random_state,transformation,pre_transform_addition,extra_kwargs,include_dynamic,include_static,include_nans,include_latlon_nans,quantiles,expected,expected2",
    [
        (2, 2, 55, None, 0, {}, False, False, False, False, [0.5], [4.1, 5.65], None),  # noqa: E501 Basic test case
        (
            100,
            2,
            55,
            None,
            0,
            {},
            False,
            False,
            False,
            False,
            [1 / 3, 2 / 3],
            [[4.1, 5.1], [5.1, 5.1]],
            None,
        ),  # noqa: E501 Multiple quantiles
        (1, 1, 55, None, 0, {}, False, False, False, False, [0.5], [4.1, 6.2], None),  # noqa: E501 Fewer estimators and reduced depth
        (1, 1, 73, None, 0, {}, False, False, False, False, [0.5], [4.2, 6.2], None),  # noqa: E501 Different random state
        (2, 2, 55, "log", 10, {}, False, False, False, False, [0.5], [4.1, 5.64], None),  # noqa: E501 Log transformation
        (
            2,
            2,
            55,
            "log10",
            10,
            {},
            False,
            False,
            False,
            False,
            [0.5],
            [4.1, 5.64],
            None,
        ),  # noqa: E501 Log10 transformation
        (
            2,
            2,
            55,
            "sqrt",
            10,
            {},
            False,
            False,
            False,
            False,
            [0.5],
            [4.1, 5.64],
            None,
        ),  # noqa: E501 Square root transformation
        (
            2,
            2,
            55,
            "cbrt",
            10,
            {},
            False,
            False,
            False,
            False,
            [0.5],
            [4.1, 5.64],
            None,
        ),  # noqa: E501 Cube root transformation
        (
            2,
            2,
            55,
            None,
            0,
            {"max_samples_leaf": 0.5},
            False,
            False,
            False,
            False,
            [0.5],
            [4.1, 6.2],
            None,
        ),  # noqa: E501 Different criterion
        (2, 5, 55, None, 0, {}, True, False, False, False, [0.5], [4.1, 4.6], None),  # noqa: E501 Include an additional dynamic feature
        (2, 5, 55, None, 0, {}, False, True, False, False, [0.5], [4.1, 5.65], None),  # noqa: E501 Include an additional static feature
        (2, 5, 55, None, 0, {}, True, True, False, False, [0.5], [4.1, 4.6], None),  # noqa: E501 Include an additional dynamic and static feature
        (2, 2, 55, None, 0, {}, False, False, True, False, [0.5], [4.1, 5.65], None),  # noqa: E501 NaNs in input data
        (
            2,
            2,
            55,
            None,
            0,
            {},
            False,
            False,
            False,
            True,
            [0.5],
            [4.1, 5.65],
            [4.1, 5.1],
        ),  # noqa: E501  NaNs in lat/lon
        (
            2,
            2,
            55,
            None,
            0,
            {},
            True,
            False,
            False,
            True,
            [0.5],
            [4.1, 4.6],
            [4.1, 5.1],
        ),  # noqa: E501 NaNs in lat/lon and dynamic feature
        (
            2,
            2,
            55,
            None,
            0,
            {},
            False,
            True,
            False,
            True,
            [0.5],
            [4.1, 5.65],
            [4.1, 5.1],
        ),  # noqa: E501 NaNs in lat/lon and static feature
        (2, 2, 55, None, 0, {}, True, True, False, True, [0.5], [4.1, 4.6], [4.1, 5.1]),  # noqa: E501 NaNs in lat/lon, dynamic and static feature
        (2, 2, 55, None, 0, {}, True, True, True, True, [0.5], [4.6, 4.6], [4.6, 5.1]),  # noqa: E501 NaNs in lat/lon, dynamic and static feature and input data
    ],
)
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
    expected,
    expected2,
):
    """Test the PrepareAndApplyQRF plugin."""
    feature_config = {"wind_speed_at_10m": ["mean", "std", "latitude", "longitude"]}

    if include_dynamic:
        feature_config["air_temperature"] = ["mean", "std"]
    if include_static:
        feature_config["distance_to_water"] = ["static"]

    unique_site_id_key = site_id[0]
    if site_id == ["latitude", "longitude", "altitude"]:
        unique_site_id_key = "station_id"

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
        site_id=unique_site_id_key,
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
            frt,
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

    result = PrepareAndApplyQRF(
        feature_config,
        "wind_speed_at_10m",
        unique_site_id_keys=site_id,
    )(cube_inputs, (qrf_model, transformation, pre_transform_addition))
    assert isinstance(result, Cube)

    assert result.data.shape == (len(quantiles), 2)

    if (
        expected2
        and include_latlon_nans
        and site_id == ["latitude", "longitude", "altitude"]
    ):
        assert np.allclose(result.data, expected2, rtol=1e-2)
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

    if exception == "no_model_output":
        result = plugin(cube_inputs, qrf_descriptors=None)
        assert isinstance(result, Cube)
        assert result.name() == "wind_speed_at_10m"
        assert result.units == "m s-1"
        assert result.data.shape == forecast_cube.data.shape
        assert np.allclose(result.data, forecast_cube.data)
    elif exception == "no_features":
        qrf_descriptors = (qrf_model, transformation, pre_transform_addition)
        with pytest.raises(ValueError, match="No target forecast provided."):
            plugin(CubeList(), qrf_descriptors=qrf_descriptors)
    elif exception == "missing_target_feature":
        qrf_descriptors = (qrf_model, transformation, pre_transform_addition)
        with pytest.raises(ValueError, match="No target forecast provided."):
            plugin(
                CubeList([ancil_cube]),
                qrf_descriptors=qrf_descriptors,
            )
    elif exception == "missing_static_feature":
        qrf_descriptors = (qrf_model, transformation, pre_transform_addition)
        feature_config = {
            "wind_speed_at_10m": ["mean", "std"],
            "distance_to_water": ["static"],
        }
        plugin.feature_config = feature_config
        with pytest.raises(ValueError, match="The number of cubes loaded."):
            plugin(CubeList([forecast_cube]), qrf_descriptors=qrf_descriptors)
    elif exception == "missing_dynamic_feature":
        qrf_descriptors = (qrf_model, transformation, pre_transform_addition)
        feature_config = {
            "wind_speed_at_10m": ["mean", "std"],
            "air_temperature": ["mean", "std"],
        }
        plugin.feature_config = feature_config
        with pytest.raises(ValueError, match="The number of cubes loaded."):
            plugin(CubeList([forecast_cube]), qrf_descriptors=qrf_descriptors)
    elif exception == "no_quantile_forest_package":
        qrf_descriptors = (qrf_model, transformation, pre_transform_addition)
        plugin.quantile_forest_installed = False
        result = plugin(CubeList([forecast_cube]), qrf_descriptors=qrf_descriptors)
        assert isinstance(result, Cube)
        assert result.name() == "wind_speed_at_10m"
        assert result.units == "m s-1"
        assert result.data.shape == forecast_cube.data.shape
        assert np.allclose(result.data, forecast_cube.data)
    else:
        raise ValueError(f"Unknown exception type: {exception}")
