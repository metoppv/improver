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
"""Unit tests for the GenerateClearskySolarRadiation plugin."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from iris.cube import Cube

from improver.generate_ancillaries.generate_derived_solar_fields import (
    CLEARSKY_SOLAR_RADIATION_CF_NAME,
    GenerateClearskySolarRadiation,
)
from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube


@pytest.fixture
def target_grid() -> Cube:
    return set_up_variable_cube(
        data=np.ones((10, 8), dtype=np.float32), name="template",
    )


@pytest.fixture
def surface_altitude() -> Cube:
    return set_up_variable_cube(
        data=np.ones((10, 8), dtype=np.float32), name="surface_altitude", units="m",
    )


@pytest.fixture
def linke_turbidity() -> Cube:
    return set_up_variable_cube(
        data=np.ones((10, 8), dtype=np.float32), name="linke_turbidity", units="1",
    )


@pytest.fixture
def surface_altitude_on_alternate_grid() -> Cube:
    return set_up_variable_cube(
        data=np.ones((12, 10), dtype=np.float32), name="surface_altitude", units="m",
    )


@pytest.fixture
def linke_turbidity_on_alternate_grid() -> Cube:
    return set_up_variable_cube(
        data=np.ones((12, 10), dtype=np.float32), name="linke_turbidity", units="1",
    )


def test__initialise_input_cubes(
    target_grid,
    surface_altitude,
    linke_turbidity,
    surface_altitude_on_alternate_grid,
    linke_turbidity_on_alternate_grid,
):
    """Test initialisation of input cubes."""
    # Check arguments remained unchanged when valid cubes are passed in.
    (
        initialised_surface_altitude,
        initialised_linke_turbidity,
    ) = GenerateClearskySolarRadiation()._initialise_input_cubes(
        target_grid, surface_altitude, linke_turbidity
    )
    assert initialised_surface_altitude == surface_altitude
    assert initialised_linke_turbidity == linke_turbidity
    # Check default cubes are returned None is passed in.
    (
        initialised_surface_altitude,
        initialised_linke_turbidity,
    ) = GenerateClearskySolarRadiation()._initialise_input_cubes(
        target_grid, None, None
    )
    # Check surface_altitude cube is initialised when None is passed in.
    assert initialised_surface_altitude.coords() == target_grid.coords()
    assert np.all(initialised_surface_altitude.data == 0.0)
    assert initialised_surface_altitude.data.dtype == np.float32
    assert initialised_surface_altitude.name() == "surface_altitude"
    assert initialised_surface_altitude.units == "m"
    # Check linke_turbidity cube is initialised when None is passed in.
    assert initialised_linke_turbidity.coords() == target_grid.coords()
    assert np.all(initialised_linke_turbidity.data == 3.0)
    assert initialised_linke_turbidity.data.dtype == np.float32
    assert initialised_linke_turbidity.name() == "linke_turbidity"
    assert initialised_linke_turbidity.units == "1"
    # Should fail when inconsistent surface_altitude cube is passed in.
    with pytest.raises(ValueError):
        GenerateClearskySolarRadiation()._initialise_input_cubes(
            target_grid, surface_altitude_on_alternate_grid, None
        )
    # Should fail when inconsistent linke_turbidity cube is passed in.
    with pytest.raises(ValueError):
        GenerateClearskySolarRadiation()._initialise_input_cubes(
            target_grid, None, linke_turbidity_on_alternate_grid
        )

def test__get_irradiance_times():

    time = datetime(2022, 1, 1, 00, 00, tzinfo=timezone.utc)
    accumulation_period = 3  # in hours
    temporal_spacing = 60  # in mins

    expected_times = [
        datetime(2021, 12, 31, 21, 00, tzinfo=timezone.utc),
        datetime(2021, 12, 31, 22, 00, tzinfo=timezone.utc),
        datetime(2021, 12, 31, 23, 00, tzinfo=timezone.utc),
        datetime(2022, 1, 1, 00, 00, tzinfo=timezone.utc),
    ]
    result = GenerateClearskySolarRadiation()._get_irradiance_times(
        time, accumulation_period, temporal_spacing
    )
    assert np.all(result == expected_times)

    accumulation_period = 1
    expected_times = [
        datetime(2021, 12, 31, 23, 00, tzinfo=timezone.utc),
        datetime(2022, 1, 1, 00, 00, tzinfo=timezone.utc),
    ]
    result = GenerateClearskySolarRadiation()._get_irradiance_times(
        time, accumulation_period, temporal_spacing
    )
    assert np.all(result == expected_times)

    misaligned_temporal_spcaing = 19
    with pytest.raises(ValueError, match="must be integer multiple"):
        GenerateClearskySolarRadiation()._get_irradiance_times(
            time, accumulation_period, misaligned_temporal_spcaing
        )


@pytest.mark.parametrize("accumulation_period", [1, 3, 24])
@pytest.mark.parametrize("surface_altitude", [None, "surface_altitude_data"])
@pytest.mark.parametrize("linke_turbidity", [None, "linke_turbidity_data"])
def test_process(request, target_grid, accumulation_period, surface_altitude, linke_turbidity):
    """Test process method returns cubes with correct structure."""
    time = datetime(2022, 1, 1, 00, 00, tzinfo=timezone.utc)

    optional_vars = {}
    if surface_altitude is not None:
        optional_vars["surface_altitude"] = request.getfixturevalue(surface_altitude)
    if linke_turbidity is not None:
        optional_vars["linke_turbidity"] = request.getfixturevalue(linke_turbidity)

    result = GenerateClearskySolarRadiation()(
        target_grid, time, accumulation_period, **optional_vars
    )

    # Check vertical coordinate
    if surface_altitude is None:
        assert np.isclose(result.coord("altitude").points[0], 0.0)
    else:
        assert np.isclose(result.coord("height").points[0], 0.0)

    # Check time value match inputs
    assert result.coord("time").points[0] == time.timestamp()
    assert timedelta(
        seconds=int(
            result.coord("time").bounds[0, 1] - result.coord("time").bounds[0, 0]
        )
    ) == timedelta(hours=accumulation_period)

    # Check variable attributes
    assert result.name() == CLEARSKY_SOLAR_RADIATION_CF_NAME
    assert result.units == "W s m-2"

