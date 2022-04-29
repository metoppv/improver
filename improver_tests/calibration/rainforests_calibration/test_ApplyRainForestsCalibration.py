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
"""Unit tests for the ApplyRainForestsCalibration class."""
import sys

import numpy as np
import pytest
from iris.coords import DimCoord
from iris.cube import CubeList

from improver.calibration.rainforest_calibration import ApplyRainForestsCalibration
from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube
from improver.utilities.cube_manipulation import compare_attributes, compare_coords

try:
    import treelite_runtime
except ModuleNotFoundError:
    TREELITE_ENABLED = False
else:
    TREELITE_ENABLED = True

lightgbm = pytest.importorskip("lightgbm")

EMPTY_COMPARSION_DICT = [{}, {}]


class MockBooster:
    def __init__(self, model_file, **kwargs):
        self.model_class = "lightgbm-Booster"
        self.model_file = model_file

    def reset_parameter(self, params):
        self.threads = params.get("num_threads")
        return self


class MockPredictor:
    def __init__(self, libpath, nthread, **kwargs):
        self.model_class = "treelite-Predictor"
        self.threads = nthread
        self.model_file = libpath


@pytest.mark.parametrize("lightgbm_keys", (True, False))
@pytest.mark.parametrize("ordered_inputs", (True, False))
@pytest.mark.parametrize("treelite_model", (TREELITE_ENABLED, False))
@pytest.mark.parametrize("treelite_file", (True, False))
def test__init__(
    lightgbm_keys,
    ordered_inputs,
    treelite_model,
    treelite_file,
    monkeypatch,
    model_config,
    error_thresholds,
):
    """Test treelite models are loaded if model_config correctly defines them. If all thresholds
    contain treelite model AND the treelite module is available, treelite Predictor is returned,
    otherwise return lightgbm Boosters. Checks outputs are ordered when inputs can be unordered.
    If neither treelite nor lightgbm configs are complete, a ValueError is expected."""
    if treelite_model:
        monkeypatch.setattr(treelite_runtime, "Predictor", MockPredictor)
    else:
        monkeypatch.setitem(sys.modules, "treelite_runtime", None)
    monkeypatch.setattr(lightgbm, "Booster", MockBooster)

    if not treelite_file:
        # Model type should default to lightgbm if there are any treelite models
        # missing across any thresholds
        model_config["0.0000"].pop("treelite_model", None)
    if not ordered_inputs:
        tmp_value = model_config.pop("0.0000", None)
        model_config["0.0000"] = tmp_value
    if not lightgbm_keys:
        for t, d in model_config.items():
            d.pop("lightgbm_model")

    if treelite_model and treelite_file:
        expected_class = "treelite-Predictor"
    elif lightgbm_keys:
        expected_class = "lightgbm-Booster"
    else:
        with pytest.raises(ValueError, match="Path to lightgbm model missing"):
            ApplyRainForestsCalibration(model_config, threads=8)
        return

    result = ApplyRainForestsCalibration(model_config, threads=8)

    for model in result.tree_models:
        assert model.model_class == expected_class
        assert model.threads == 8
    assert result.treelite_enabled is treelite_model
    assert np.all(result.error_thresholds == error_thresholds)
    for threshold, model in zip(result.error_thresholds, result.tree_models):
        assert f"{threshold:06.4f}" in model.model_file


def test__align_feature_variables_ensemble(ensemble_features, ensemble_forecast):
    """Check cube alignment when using feature and forecast variables when realization
    coordinate present in some cube variables."""

    (aligned_features, aligned_forecast,) = ApplyRainForestsCalibration(
        model_config_dict={}, threads=1
    )._align_feature_variables(ensemble_features, ensemble_forecast)

    input_cubes = CubeList([*ensemble_features, ensemble_forecast])
    output_cubes = CubeList([*aligned_features, aligned_forecast])

    # Check that the realization dimension is the outer-most dimension
    assert np.all([cube.coord_dims("realization") == (0,) for cube in output_cubes])

    # Check that all cubes have consistent shape
    assert np.all(
        [cube.data.shape == output_cubes[0].data.shape for cube in output_cubes[1:]]
    )

    # Check the other properties of the cubes are unchanged.
    for input_cube, output_cube in zip(input_cubes, output_cubes):
        assert (
            compare_coords([output_cube, input_cube], ignored_coords="realization")
            == EMPTY_COMPARSION_DICT
        )
        assert compare_attributes([output_cube, input_cube]) == EMPTY_COMPARSION_DICT


def test__align_feature_variables_deterministic(
    deterministic_features, deterministic_forecast
):
    """Check cube alignment when using feature and forecast variables when no realization
    coordinate present in any of the cube variables."""

    (aligned_features, aligned_forecast,) = ApplyRainForestsCalibration(
        model_config_dict={}, threads=1
    )._align_feature_variables(deterministic_features, deterministic_forecast)

    input_cubes = CubeList([*deterministic_features, deterministic_forecast])
    output_cubes = CubeList([*aligned_features, aligned_forecast])

    # Check that all cubes have realization dimension of length 1
    assert np.all([cube.coord("realization").shape == (1,) for cube in output_cubes])

    # Check that the realization dimension is the outer-most dimension
    assert np.all([cube.coord_dims("realization") == (0,) for cube in output_cubes])

    # Check that all cubes have consistent shape
    assert np.all(
        [cube.data.shape == output_cubes[0].data.shape for cube in output_cubes[1:]]
    )

    # Check the other properties of the cubes are unchanged.
    for input_cube, output_cube in zip(input_cubes, output_cubes):
        assert (
            compare_coords([output_cube, input_cube], ignored_coords="realization")
            == EMPTY_COMPARSION_DICT
        )
        assert compare_attributes([output_cube, input_cube]) == EMPTY_COMPARSION_DICT


def test__align_feature_variables_misaligned_dim_coords(ensemble_features):
    """Check ValueError raised when feature/forecast cubes have differing dimension
    coordinates."""
    # Test case where non-realization dimension differ.
    misaligned_forecast_cube = set_up_variable_cube(
        np.maximum(0, np.random.normal(0.002, 0.001, (5, 10, 15))).astype(np.float32),
        name="lwe_thickness_of_precipitation_amount",
        units="m",
        realizations=np.arange(5),
    )

    with pytest.raises(ValueError):
        ApplyRainForestsCalibration(
            model_config_dict={}, threads=1
        )._align_feature_variables(ensemble_features, misaligned_forecast_cube)

    # Test case where realization dimension differ.
    misaligned_forecast_cube = set_up_variable_cube(
        np.maximum(0, np.random.normal(0.002, 0.001, (10, 10, 10))).astype(np.float32),
        name="lwe_thickness_of_precipitation_amount",
        units="m",
        realizations=np.arange(10),
    )

    with pytest.raises(ValueError):
        ApplyRainForestsCalibration(
            model_config_dict={}, threads=1
        )._align_feature_variables(ensemble_features, misaligned_forecast_cube)


@pytest.mark.parametrize(
    "new_dim_location, copy_metadata",
    [(None, True), (None, False), (0, True), (1, True)],
)
def test_add_coordinate(
    deterministic_forecast, new_dim_location, copy_metadata,
):
    """Test adding dimension to input_cube"""

    realization_coord = DimCoord(np.arange(0, 5), standard_name="realization", units=1)

    output_cube = ApplyRainForestsCalibration(
        model_config_dict={}, threads=1
    )._add_coordinate_to_cube(
        deterministic_forecast,
        realization_coord,
        new_dim_location=new_dim_location,
        copy_metadata=copy_metadata,
    )

    # Test all but added coord are consistent
    assert (
        compare_coords(
            [output_cube, deterministic_forecast], ignored_coords="realization"
        )
        == EMPTY_COMPARSION_DICT
    )
    if copy_metadata:
        assert (
            compare_attributes([output_cube, deterministic_forecast])
            == EMPTY_COMPARSION_DICT
        )
    else:
        assert (
            compare_attributes([output_cube, deterministic_forecast])
            != EMPTY_COMPARSION_DICT
        )

    # Test realization coord
    output_realization_coord = output_cube.coord("realization")
    assert np.allclose(output_realization_coord.points, realization_coord.points)
    assert output_realization_coord.standard_name == realization_coord.standard_name
    assert output_realization_coord.units == realization_coord.units

    # Test data values
    consistent_data = [
        np.allclose(realization.data, deterministic_forecast.data)
        for realization in output_cube.slices_over("realization")
    ]
    assert np.all(consistent_data)

    # Check dim is in the correct place
    if new_dim_location is None:
        assert output_cube.coord_dims("realization") == (output_cube.ndim - 1,)
    else:
        assert output_cube.coord_dims("realization") == (new_dim_location,)
