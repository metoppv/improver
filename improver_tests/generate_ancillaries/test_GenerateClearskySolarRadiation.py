# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
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
from improver.utilities.solar import calc_solar_elevation
from improver.utilities.spatial import get_grid_y_x_values

ATTRIBUTES = {
    "source": "IMRPOVER tests",
    "institution": "Australian Bureau of Meteorology",
    "title": "Test data on sample grid",
}


@pytest.fixture
def target_grid() -> Cube:
    return set_up_variable_cube(
        data=np.ones((10, 8), dtype=np.float32), name="template", attributes=ATTRIBUTES
    )


@pytest.fixture
def target_grid_equal_area() -> Cube:
    return set_up_variable_cube(
        data=np.ones((10, 8), dtype=np.float32),
        name="template",
        spatial_grid="equalarea",
        attributes=ATTRIBUTES,
    )


@pytest.fixture
def surface_altitude() -> Cube:
    return set_up_variable_cube(
        data=np.ones((10, 8), dtype=np.float32),
        name="surface_altitude",
        units="m",
        attributes=ATTRIBUTES,
    )


@pytest.fixture
def linke_turbidity() -> Cube:
    return set_up_variable_cube(
        data=np.ones((10, 8), dtype=np.float32),
        name="linke_turbidity",
        units="1",
        attributes=ATTRIBUTES,
    )


@pytest.fixture
def surface_altitude_on_alternate_grid() -> Cube:
    return set_up_variable_cube(
        data=np.ones((12, 10), dtype=np.float32),
        name="surface_altitude",
        units="m",
        attributes=ATTRIBUTES,
    )


@pytest.fixture
def linke_turbidity_on_alternate_grid() -> Cube:
    return set_up_variable_cube(
        data=np.ones((12, 10), dtype=np.float32),
        name="linke_turbidity",
        units="1",
        attributes=ATTRIBUTES,
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


def test__irradiance_times():
    """Test returned irradiance times are equispaced time-steps
    on the specified interval, with spacing temporal_spacing.
    Where temporal_spacing does not fit evenly into the total interval,
    a ValueError should be raised."""
    time = datetime(2022, 1, 1, 0, 0, tzinfo=timezone.utc)
    accumulation_period = 3  # in hours
    temporal_spacing = 60  # in mins

    expected_times = [
        datetime(2021, 12, 31, 21, 0, tzinfo=timezone.utc),
        datetime(2021, 12, 31, 22, 0, tzinfo=timezone.utc),
        datetime(2021, 12, 31, 23, 0, tzinfo=timezone.utc),
        datetime(2022, 1, 1, 0, 0, tzinfo=timezone.utc),
    ]
    result = GenerateClearskySolarRadiation()._irradiance_times(
        time, accumulation_period, temporal_spacing
    )
    assert np.all(result == expected_times)

    accumulation_period = 1
    expected_times = [
        datetime(2021, 12, 31, 23, 0, tzinfo=timezone.utc),
        datetime(2022, 1, 1, 0, 0, tzinfo=timezone.utc),
    ]
    result = GenerateClearskySolarRadiation()._irradiance_times(
        time, accumulation_period, temporal_spacing
    )
    assert np.all(result == expected_times)

    misaligned_temporal_spacing = 19
    with pytest.raises(ValueError, match="must be integer multiple"):
        GenerateClearskySolarRadiation()._irradiance_times(
            time, accumulation_period, misaligned_temporal_spacing
        )


@pytest.mark.parametrize(
    "zenith, expected_value",
    (
        (-100.0, 0.0),
        (-90.0, 0.0),
        (-85.0, 11.46028),
        (-60.0, 1.99948),
        (-30.0, 1.15445),
        (0.0, 0.99971),
        (30.0, 1.15399),
        (60.0, 1.99429),
        (85.0, 10.30579),
        (90.0, 0.0),
        (100.0, 0.0),
    ),
)
def test__calc_optical_air_mass(zenith, expected_value):
    """Test calc air mass function over a range of zenith angles."""
    values = GenerateClearskySolarRadiation()._calc_optical_air_mass(zenith)
    assert np.allclose(values, expected_value)


@pytest.mark.parametrize(
    "day_of_year, surface_altitude, linke_turbidity, expected_values",
    (
        (0, 0, 3, np.array([1091.9529, 486.4359, 0.0])),
        (180, 0, 3, np.array([1022.2189, 455.3712, 0.0])),
        (0, 0, 1, np.array([1179.8004, 567.6262, 0.0])),
        (0, 1000, 3, np.array([1130.1020, 492.2152, 0.0])),
        (0, 8000, 3, np.array([1412.8340, 694.0237, 0.0])),
    ),
)
def test__calc_clearsky_ineichen_values(
    day_of_year, surface_altitude, linke_turbidity, expected_values
):
    """Test irradiance calc over a range of sample values. Note
    all values here have been evaluated by hand."""
    zenith = np.array([0, 60, 90])

    result = GenerateClearskySolarRadiation()._calc_clearsky_ineichen(
        zenith_angle=zenith,
        day_of_year=day_of_year,
        surface_altitude=surface_altitude,
        linke_turbidity=linke_turbidity,
    )
    assert np.allclose(result, expected_values)


def test__calc_clearsky_ineichen_grid_properties(target_grid):
    """Test irradiance values vary over grid as expected."""
    lats, lons = get_grid_y_x_values(target_grid)

    zenith_angle = 90.0 - calc_solar_elevation(lats, lons, day_of_year=0, utc_hour=12)
    result = GenerateClearskySolarRadiation()._calc_clearsky_ineichen(
        zenith_angle=zenith_angle, day_of_year=0, surface_altitude=0, linke_turbidity=3
    )
    # For constant surface_altitude, check that max irradiance occurs for minimum zenith_angle
    assert np.unravel_index(
        np.argmax(result, axis=None), result.shape
    ) == np.unravel_index(np.argmin(zenith_angle, axis=None), zenith_angle.shape)
    # For constant surface_altitude, check that larger irradiance value at adjacent sites,
    # occurs for the location with the smaller zenith angle.
    assert np.all(
        (result[:, 1:] - result[:, :-1] > 0)
        == (zenith_angle[:, 1:] - zenith_angle[:, :-1] < 0)
    )
    assert np.all(
        (result[1:, :] - result[:-1, :] > 0)
        == (zenith_angle[1:, :] - zenith_angle[:-1, :] < 0)
    )


def test__calc_clearsky_solar_radiation_data(
    target_grid, surface_altitude, linke_turbidity
):
    """Test evaluation of solar radiation data. The expected value is evaluated
    with irradiance data returned from _calc_clearsky_ineichen for each
    irradiance_time and then aggregated to give the expected solar radiation value."""
    irradiance_times = np.array(
        [
            datetime(2021, 12, 31, 21, 00, tzinfo=timezone.utc),
            datetime(2021, 12, 31, 22, 30, tzinfo=timezone.utc),
            datetime(2022, 1, 1, 00, 00, tzinfo=timezone.utc),
        ]
    )
    result = GenerateClearskySolarRadiation()._calc_clearsky_solar_radiation_data(
        target_grid, irradiance_times, surface_altitude.data, linke_turbidity.data, 90
    )
    expected_values = np.array(
        [
            [462277.2, 126636.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [243679.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [46386.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    # Check expected array properties
    assert result.shape == (10, 8)
    assert result.dtype == np.float32
    # Check results are sensible
    assert np.allclose(result, expected_values)


@pytest.mark.parametrize("at_mean_sea_level", (True, False))
@pytest.mark.parametrize("new_title", (None, "IMPROVER ancillary on sample grid"))
def test__create_solar_radiation_cube(target_grid, at_mean_sea_level, new_title):

    solar_radiation_data = np.zeros_like(target_grid.data)
    time = datetime(2022, 1, 1, 0, 0)
    accumulation_period = 24

    result = GenerateClearskySolarRadiation()._create_solar_radiation_cube(
        solar_radiation_data,
        target_grid,
        time,
        accumulation_period,
        at_mean_sea_level,
        new_title,
    )

    # Check vertical coordinate
    if at_mean_sea_level:
        assert np.isclose(result.coord("altitude").points[0], 0.0)
    else:
        assert np.isclose(result.coord("height").points[0], 0.0)
    # Check time value match inputs
    assert (
        result.coord("time").points[0] == time.replace(tzinfo=timezone.utc).timestamp()
    )
    assert timedelta(
        seconds=int(
            result.coord("time").bounds[0, 1] - result.coord("time").bounds[0, 0]
        )
    ) == timedelta(hours=accumulation_period)
    # Check that the dim coords are the spatial coords only, matching those from target_grid
    assert result.coords(dim_coords=True) == [
        target_grid.coord(axis="Y"),
        target_grid.coord(axis="X"),
    ]
    # Check variable attributes
    assert result.name() == CLEARSKY_SOLAR_RADIATION_CF_NAME
    assert result.units == "W s m-2"

    assert result.attributes["source"] == "IMPROVER"
    assert result.attributes.get("title") == new_title
    assert result.attributes["institution"] == target_grid.attributes["institution"]


def test_process_lat_lon(target_grid, surface_altitude):
    """Test process method returns cubes with correct structure."""
    time = datetime(2022, 1, 1, 0, 0)
    accumulation_period = 24

    # Check that default behaviour results in cube with altitude for z-coord.
    result = GenerateClearskySolarRadiation()(target_grid, time, accumulation_period,)
    assert np.isclose(result.coord("altitude").points[0], 0.0)

    # Check cube has same spatial coords as target_grid
    assert result.coords(dim_coords=True) == target_grid.coords(dim_coords=True)
    # Check data is sensible
    assert result.dtype == np.float32
    assert np.all(result.data >= 0.0)
    assert np.all(np.isfinite(result.data))

    # Check that non-zero surface_altitude results in cube with height for z-coord.
    result = GenerateClearskySolarRadiation()(
        target_grid, time, accumulation_period, surface_altitude=surface_altitude
    )
    assert np.isclose(result.coord("height").points[0], 0.0)
    # Check data is sensible
    assert result.dtype == np.float32
    assert np.all(result.data >= 0.0)
    assert np.all(np.isfinite(result.data))


def test_process_equal_area(target_grid_equal_area):
    """Test process method returns cubes with correct structure."""
    time = datetime(2022, 1, 1, 0, 0)
    accumulation_period = 24

    # Check that default behaviour results in cube with altitude for z-coord.
    result = GenerateClearskySolarRadiation()(
        target_grid_equal_area, time, accumulation_period,
    )
    # Check cube has same spatial coords as target_grid
    assert result.coords(dim_coords=True) == target_grid_equal_area.coords(
        dim_coords=True
    )
    # Check data is sensible
    assert result.dtype == np.float32
    assert np.all(result.data >= 0.0)
    assert np.all(np.isfinite(result.data))
