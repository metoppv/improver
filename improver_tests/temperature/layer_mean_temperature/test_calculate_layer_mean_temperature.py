# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
import iris
import numpy as np

from improver.temperature.layer_mean_temperature import CalculateLayerMeanTemperature


def make_layer_cube(data):
    """Create a simple 3D temperature cube with shape (height, y, x) for testing.

    Args:
        data (np.ndarray): Temperature data with shape (height, y, x).

    Heights are in metres:
        - 100 m
        - 200 m
        - 300 m
    """
    # produces height arrays of the form [ 100.0, 200.0, 300.0, 400.0, ...]
    height_points = np.array(
        np.arange(start=100.0, stop=(data.shape[0] + 1) * 100, step=100),
        dtype=np.float32,
    )
    # produces y-coordinate arrays of the form [ 0.0, 1.0, 2.0, 3.0, ...]
    y_points = np.array(range(data.shape[1]), dtype=np.float32)
    # produces x-coordinate arrays of the form [ 0.0, 1.0, 2.0, 3.0, ...]
    x_points = np.array(range(data.shape[2]), dtype=np.float32)
    cube = iris.cube.Cube(
        data,
        dim_coords_and_dims=[
            (iris.coords.DimCoord(height_points, standard_name="height", units="m"), 0),
            (iris.coords.DimCoord(y_points, "projection_y_coordinate", units="m"), 1),
            (iris.coords.DimCoord(x_points, "projection_x_coordinate", units="m"), 2),
        ],
        var_name="air_temperature",
        units="K",
    )
    # Add scalar coordinates expected by CalculateLayerMeanTemperature
    cube.add_aux_coord(
        iris.coords.AuxCoord(0, standard_name="forecast_period", units="seconds")
    )
    cube.add_aux_coord(
        iris.coords.AuxCoord(
            0, standard_name="forecast_reference_time", units="seconds since epoch"
        )
    )
    cube.add_aux_coord(
        iris.coords.AuxCoord(0, standard_name="time", units="seconds since epoch")
    )
    return cube


def test_layer_mean_temperature_uniform():
    """Test that CalculateLayerMeanTemperature returns the correct mean
    for a uniform temperature field.

    If all temperature values are constant (280 K) at every height level,
    the weighted mean must also be 280 K regardless of height spacing or weighting.
    """
    # All data values set to 280 K across all heights and grid points
    data = np.full((3, 2, 2), 280, dtype=np.float32)
    cube = make_layer_cube(data)
    plugin = CalculateLayerMeanTemperature()
    result = plugin.process(cube, verbosity=0)

    # Mean of a uniform field must equal the uniform value
    np.testing.assert_allclose(result.data, 280, rtol=1e-5)


def test_layer_mean_temperature_output_shape_and_coordinates_exists():
    """Test that CalculateLayerMeanTemperature returns a 2D cube with correct
    dimension and auxiliary coordinates.

    Checks:
        - Output cube is 2D (height dimension collapsed).
        - projection_y_coordinate and projection_x_coordinate are present.
        - Scalar coordinates (forecast_period, time, forecast_reference_time) are preserved.
    """
    data = np.full((3, 2, 2), 280, dtype=np.float32)
    cube = make_layer_cube(data)
    plugin = CalculateLayerMeanTemperature()
    result = plugin.process(cube, verbosity=0)

    # Check output is 2D - height dimension has been collapsed
    assert result.shape == (2, 2)

    # Check dimension coordinates are present
    assert result.coords("projection_y_coordinate")
    assert result.coords("projection_x_coordinate")

    # Check height coordinate is no longer a dimension coordinate
    assert not result.coords("height", dim_coords=True)

    # Check scalar coordinates are preserved
    assert result.coords("forecast_period")
    assert result.coords("forecast_reference_time")
    assert result.coords("time")

    # Check mean values are as expected
    assert np.allclose(result.data, 280)


def test_layer_mean_temperature_scalar_coordinate_values_preserved():
    """Test that CalculateLayerMeanTemperature preserves scalar coordinate
    values from the input cube in the output cube.

    Checks that forecast_period, forecast_reference_time and time coordinate
    values are unchanged between input and output.
    """
    data = np.full((3, 2, 2), 280, dtype=np.float32)
    cube = make_layer_cube(data)
    plugin = CalculateLayerMeanTemperature()
    result = plugin.process(cube, verbosity=0)

    # Check scalar coordinate values are preserved from input to output
    assert (
        result.coord("forecast_period").points == cube.coord("forecast_period").points
    )
    assert (
        result.coord("forecast_reference_time").points
        == cube.coord("forecast_reference_time").points
    )
    assert result.coord("time").points == cube.coord("time").points


def test_layer_mean_temperature_values():
    """Test that CalculateLayerMeanTemperature returns
    the correct mean temperature values where the temperature
    field is varying

    Reference data was collected from a prior run.

    """

    # create a varying temperature field, so the resulting
    # layer mean temperature will have varying values too
    # the input is deterministically calculated (i.e. no random numbers)
    data = np.fromfunction(
        lambda layer, y, x: 280.0 + 10.0 * np.sin(x) - 5.0 * np.cos(y) + np.sqrt(layer),
        (3, 2, 2),
        dtype=np.float32,
    )

    cube = make_layer_cube(data)
    plugin = CalculateLayerMeanTemperature()

    actual_result = plugin.process(cube, verbosity=0)

    expected_result = [
        [275.8535546875, 284.26826171875],
        [278.15205078125, 286.5667529296875],
    ]

    assert np.allclose(actual_result.data, expected_result)


def test_verbosity_layer_mean_temperature(capsys):
    """Test that CalculateLayerMeanTemperature prints expected output
    when verbosity is set to 1.

    Checks that the layer mean temperature array is printed to stdout.
    """
    data = np.full((3, 2, 2), 280, dtype=np.float32)
    cube = make_layer_cube(data)
    plugin = CalculateLayerMeanTemperature()
    plugin.process(cube, verbosity=1)
    captured = capsys.readouterr()
    assert "Layer mean temperature array" in captured.out
