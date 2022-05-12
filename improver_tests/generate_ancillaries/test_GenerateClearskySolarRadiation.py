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

import numpy as np
import pytest
from iris.cube import Cube

from improver.generate_ancillaries.generate_derived_solar_fields import (
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
