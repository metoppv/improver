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
"""Fixtures for rainforests calibration."""
import numpy as np
import pytest
from iris.cube import CubeList

from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube


@pytest.fixture
def error_thresholds():
    return np.array(
        [-0.0010, -0.0001, 0.0000, 0.0001, 0.0010, 0.0100], dtype=np.float32
    )


@pytest.fixture
def model_config(error_thresholds):
    return {
        f"{threshold:06.4f}": {
            "lightgbm_model": f"lightgbm_model_dir/test_model_{threshold:06.4f}.txt",
            "treelite_model": f"treelite_model_dir/test_model_{threshold:06.4f}.so",
        }
        for threshold in error_thresholds
    }


def gen_forecast_cubes(realizations):
    np.random.seed(0)
    if realizations is None:
        data_shape = (10, 10)
    else:
        data_shape = (len(realizations), 10, 10)
    return set_up_variable_cube(
        np.maximum(0, np.random.normal(0.002, 0.001, data_shape)).astype(np.float32),
        name="lwe_thickness_of_precipitation_amount",
        units="m",
        realizations=realizations,
        attributes={"title": "Test forecast cube"},
    )


def gen_feature_cubes(realizations):
    np.random.seed(0)
    if realizations is None:
        data_shape = (10, 10)
    else:
        data_shape = (len(realizations), 10, 10)
    np.random.seed(0)
    cape = set_up_variable_cube(
        np.maximum(0, np.random.normal(15, 5, data_shape)).astype(np.float32),
        name="cape",
        units="J kg-1",
        realizations=realizations,
    )
    precipitation_accumulation_from_convection = set_up_variable_cube(
        np.maximum(0, np.random.normal(0.001, 0.001, data_shape)).astype(np.float32),
        name="lwe_thickness_of_convective_precipitation_amount",
        units="m",
        realizations=realizations,
    )
    precipitation_accumulation = set_up_variable_cube(
        np.maximum(0, np.random.normal(0.002, 0.001, data_shape)).astype(np.float32),
        name="lwe_thickness_of_precipitation_amount",
        units="m",
        realizations=realizations,
    )
    wind_speed = set_up_variable_cube(
        np.maximum(0, np.random.normal(5, 5, data_shape)).astype(np.float32),
        name="wind_speed",
        units="m s-1",
        realizations=realizations,
    )
    clearsky_solar_rad = set_up_variable_cube(
        np.maximum(0, np.random.normal(5000000, 2000000, (10, 10))).astype(np.float32),
        name="clearsky_solar_radiation",
        units="J m-2",
    )
    return CubeList(
        [
            cape,
            precipitation_accumulation_from_convection,
            precipitation_accumulation,
            clearsky_solar_rad,
            wind_speed,
        ]
    )


@pytest.fixture
def ensemble_forecast():
    return gen_forecast_cubes(realizations=np.arange(5))


@pytest.fixture
def ensemble_features():
    return gen_feature_cubes(realizations=np.arange(5))


@pytest.fixture
def deterministic_forecast():
    return gen_forecast_cubes(realizations=None)


@pytest.fixture
def deterministic_features():
    return gen_feature_cubes(realizations=None)
