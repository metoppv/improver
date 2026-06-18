import numpy as np
import pytest
import iris
from iris.coords import DimCoord
from iris.cube import Cube

from improver.constants import R_DRY_AIR
from improver.psychrometric_calculations.air_density import AirDensity


def create_virtual_temperature_cube():
    """Helper to create a simple test cube."""
    # Create temperature data (2 levels, 3x3 grid)
    data = np.full((2, 3, 3), 300.0)  # K

    # Pressure levels (in hPa deliberately for conversion test)
    pressure = DimCoord(
        np.array([1000, 900]),  # hPa
        standard_name="air_pressure",
        units="hPa",
    )

    # Spatial coords (dummy)
    y = DimCoord(np.arange(3), long_name="y")
    x = DimCoord(np.arange(3), long_name="x")

    cube = Cube(
        data,
        standard_name="virtual_temperature",
        units="K",
        dim_coords_and_dims=[
            (pressure, 0),
            (y, 1),
            (x, 2),
        ],
    )
    return cube


def test_air_density_basic():
    """Test density calculation is correct."""
    cube = create_virtual_temperature_cube()
    plugin = AirDensity()

    result = plugin.process(cube)

    # Convert pressure to Pa for expected calculation
    pressure = np.array([1000, 900]) * 100  # hPa -> Pa
    pressure_3d = pressure[:, None, None]

    expected = pressure_3d / (R_DRY_AIR * cube.data)

    np.testing.assert_allclose(result.data, expected)

    assert result.name() == "air_density"
    assert str(result.units) == "kg m-3"


def test_pressure_broadcasting_shape():
    """Ensure output shape matches input."""
    cube = create_virtual_temperature_cube()
    plugin = AirDensity()

    result = plugin.process(cube)

    assert result.shape == cube.shape


def test_missing_pressure_coord():
    """Test error raised if no air_pressure coordinate."""
    cube = create_virtual_temperature_cube()
    cube.remove_coord("air_pressure")

    plugin = AirDensity()

    with pytest.raises(ValueError, match="air_pressure"):
        plugin.process(cube)


def test_temperature_unit_conversion():
    """Ensure temperature conversion to Kelvin works."""
    cube = create_virtual_temperature_cube()
    cube.convert_units("Celsius")  # now ~26.85°C

    plugin = AirDensity()
    result = plugin.process(cube)

    # Convert back manually for expectation
    Tv_K = cube.copy()
    Tv_K.convert_units("K")

    pressure = np.array([1000, 900]) * 100
    expected = pressure[:, None, None] / (R_DRY_AIR * Tv_K.data)

    np.testing.assert_allclose(result.data, expected)
