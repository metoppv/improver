import numpy as np
import pytest
from iris.coords import DimCoord
from iris.cube import Cube, CubeList

from improver.constants import R_DRY_AIR
from improver.psychrometric_calculations.air_density import AirDensity  # update this import


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def make_temperature_cube(on_pressure_levels=True):
    """Create a simple virtual temperature cube.

       Args:
           on_pressure_levels: create on pressure or height levels

       Result:
           a virtual temperature cube

    """
    data = np.full((2, 3, 3), 300.0)

    if on_pressure_levels:
        vertical = DimCoord(
            np.array([1000, 900]),  # hPa
            standard_name="air_pressure",
            units="hPa",
        )
    else:
        vertical = DimCoord(
            np.array([100, 200]),  # metres
            standard_name="height",
            units="m",
        )

    y = DimCoord(np.arange(3), long_name="y" )
    x = DimCoord(np.arange(3), long_name="x" )

    return Cube(
        data,
        standard_name="virtual_temperature",
        units="K",
        dim_coords_and_dims=[(vertical, 0), (y, 1), (x, 2)],
    )


def make_pressure_cube(on_pressure_levels=True):
    """Create a pressure cube on same grid as above

       Args:
           on_pressure_levels: create on pressure or height levels

       Result:
           a virtual temperature cube

    """
    data = np.array([100000.0, 90000.0])[:, None, None] * np.ones((2, 3, 3))

    if on_pressure_levels:
        vertical = DimCoord(
            np.array([1000, 900]),  # hPa
            standard_name="air_pressure",
            units="hPa",
        )
    else:
        vertical = DimCoord(
            np.array([100, 200]),  # meters
            standard_name="height",
            units="m",
        )

    y = DimCoord(np.arange(3), long_name="y")
    x = DimCoord(np.arange(3), long_name="x")

    cube = Cube(
        data,
        standard_name="air_pressure",
        units="Pa",
        dim_coords_and_dims=[(vertical, 0), (y, 1), (x, 2)],
    )

    return cube


# ---------------------------------------------------------------------
# Core functionality tests
# ---------------------------------------------------------------------

def test_single_cube_with_pressure_coord():
    """Test Cube input with pressure levels work"""
    cube = make_temperature_cube(on_pressure_levels=True)
    plugin = AirDensity()

    result = plugin.process(cube) # single cube argument

    pressure = np.array([1000, 900]) * 100  # hPa → Pa
    expected = pressure[:, None, None] / (R_DRY_AIR * cube.data)

    np.testing.assert_allclose(result.data, expected)
    assert result.shape == cube.shape
    assert result.name() == "air_density"


def test_cubelist_with_explicit_pressure():
    """Explicit pressure cube is used when provided."""
    temp = make_temperature_cube(on_pressure_levels=True)
    pressure_cube = make_pressure_cube()

    cubes = CubeList([temp, pressure_cube])
    plugin = AirDensity()

    result = plugin.process(cubes)

    expected = pressure_cube.data / (R_DRY_AIR * temp.data)
    np.testing.assert_allclose(result.data, expected)


def test_height_levels_with_pressure_cube():
    """Height-level temperature requires explicit pressure cube."""
    temp = make_temperature_cube(on_pressure_levels=False) # on height levels
    pressure_cube = make_pressure_cube()

    cubes = CubeList([temp, pressure_cube])
    plugin = AirDensity()

    result = plugin.process(cubes)

    expected = pressure_cube.data / (R_DRY_AIR * temp.data)
    np.testing.assert_allclose(result.data, expected)


# ---------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------

def test_error_no_pressure_information():
    """Error when no pressure cube and no pressure coord."""
    temp = make_temperature_cube(on_pressure_levels=False)

    plugin = AirDensity()

    with pytest.raises(ValueError, match="No pressure information"):
        plugin.process(temp)


def test_error_missing_temperature_cube():
    """Error if cubelist has no temperature cube."""
    pressure_cube = make_pressure_cube()

    plugin = AirDensity()

    with pytest.raises(ValueError, match="virtual_temperature"):
        plugin.process(CubeList([pressure_cube]))


# ---------------------------------------------------------------------
# Unit handling
# ---------------------------------------------------------------------

def test_temperature_unit_conversion():
    """Ensure temperature in Celsius is handled correctly."""
    temp = make_temperature_cube(on_pressure_levels=True)
    temp.convert_units("Celsius")

    plugin = AirDensity()
    result = plugin.process(temp)

    # Convert manually for expected result
    Tv_K = temp.copy()
    Tv_K.convert_units("K")

    pressure = np.array([1000, 900]) * 100
    expected = pressure[:, None, None] / (R_DRY_AIR * Tv_K.data)

    np.testing.assert_allclose(result.data, expected)


def test_pressure_unit_conversion():
    """Ensure pressure coord in hPa is converted."""
    temp = make_temperature_cube(on_pressure_levels=True)

    plugin = AirDensity()
    result = plugin.process(temp)

    pressure_pa = np.array([1000, 900]) * 100
    expected = pressure_pa[:, None, None] / (R_DRY_AIR * temp.data)

    np.testing.assert_allclose(result.data, expected)


# ---------------------------------------------------------------------
# Shape and metadata consistency
# ---------------------------------------------------------------------

def test_output_metadata():
    """Check output metadata is correct."""
    temp = make_temperature_cube(on_pressure_levels=True)
    plugin = AirDensity()

    result = plugin.process(temp)

    assert result.name() == "air_density"
    assert str(result.units) == "kg m-3"


def test_output_shape_consistency():
    """Ensure output shape matches input."""
    temp = make_temperature_cube(on_pressure_levels=True)
    plugin = AirDensity()

    result = plugin.process(temp)

    assert result.shape == temp.shape


# ---------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------

def test_multiple_irrelevant_cubes():
    """Ignore irrelevant cubes in CubeList."""
    temp = make_temperature_cube()
    pressure = make_pressure_cube()

    junk_cube = Cube(np.zeros((2, 3, 3)), long_name="dummy")

    cubes = CubeList([junk_cube, temp, pressure])
    plugin = AirDensity()

    result = plugin.process(cubes)

    expected = pressure.data / (R_DRY_AIR * temp.data)
    np.testing.assert_allclose(result.data, expected)


def test_explicit_pressure_on_height_takes_priority():
    """Explicit pressure overrides pressure coord.

    on height levels we use the explicit pressure argument

    """
    temp = make_temperature_cube(on_pressure_levels=False)

    # Deliberately create different pressure cube
    before_pressure_cube = make_pressure_cube(on_pressure_levels=False)

    after_pressure_cube = before_pressure_cube.copy()
    after_pressure_cube.data = after_pressure_cube.data * 2.0

    cubes = CubeList([temp, after_pressure_cube])
    plugin = AirDensity()

    result = plugin.process(cubes)

    not_expected =  before_pressure_cube.data / (R_DRY_AIR * temp.data)  # before doubling
    b1 = np.allclose(result.data, not_expected)
    assert not b1

    expected = after_pressure_cube.data / (R_DRY_AIR * temp.data) # after doublibg
    b2 = np.allclose(result.data, expected)
    assert b2


def test_explicit_pressure_on_pressure_takes_priority():
    """Explicit pressure overrides pressure coord.

    on pressure levels we use the explicit pressure argument

    """
    temp = make_temperature_cube(on_pressure_levels=True)

    pressure_cube_before  = make_pressure_cube()

    pressure_cube_after   = pressure_cube_before.copy()
    pressure_cube_after.data = pressure_cube_before.data * 2.0

    cubes = CubeList([temp, pressure_cube_after])

    plugin = AirDensity()

    result = plugin.process(cubes)

    not_expected = pressure_cube_before.data / (R_DRY_AIR * temp.data)   # before doubling
    b1 = np.allclose(result.data, not_expected)
    assert not b1

    expected = pressure_cube_after.data / (R_DRY_AIR * temp.data) # after doubling
    b2 = np.allclose(result.data, expected)
    assert b2
