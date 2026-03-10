# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
import iris
import numpy as np
import pytest

from improver.temperature.layer_mean_temperature import LayerTemperatureInterpolation

METRES_TO_FT = 3.28084


def make_test_cube():
    """Create a simple 3D temperature cube with shape (height, y, x) for testing.

    Heights are in metres:
        - 200 m (~656 ft)
        - 220 m (~722 ft)
        - 240 m (~787 ft)
    """
    data = np.array(
        [
            [[280, 281], [282, 283]],  # height = 200 m
            [[285, 286], [287, 288]],  # height = 220 m
            [[290, 291], [292, 293]],  # height = 240 m
        ],
        dtype=np.float32,
    )
    height_points = np.array([200, 220, 240], dtype=np.float32)
    y_points = np.array([0, 1], dtype=np.float32)
    x_points = np.array([0, 1], dtype=np.float32)
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
    return cube


def test_extract_and_interpolate_layer():
    """Test that LayerTemperatureInterpolation correctly extracts and interpolates
    temperature levels within a specified layer.

    Layer bounds: 600 ft (≈182.88 m) to 800 ft (≈243.84 m).

    Expected output:
        - Interpolated base at 600 ft (≈182.88 m)
        - Interior levels at 200 m, 220 m, 240 m
        - Interpolated top at 800 ft (≈243.84 m)
        - Total: 5 height levels, shape (5, 2, 2)
    """
    cube = make_test_cube()
    plugin = LayerTemperatureInterpolation()
    # bottom=600 ft converts to ~182.88 m, top=800 ft converts to ~243.84 m
    # All three data levels (200, 220, 240 m) fall within the layer bounds,
    # so the result should contain 5 heights: interpolated base + 3 interior + interpolated top
    result = plugin.process(cube, bottom=600, top=800, verbosity=0)
    # Should include interpolated base (600ft), all interior levels, and interpolated top (800ft)
    # In this synthetic example, all heights are present, so result should have 5 heights
    assert result.shape == (5, 2, 2)
    # Check height coordinate values
    expected_heights = np.array([182.88, 200, 220, 240, 243.84])
    np.testing.assert_allclose(
        result.coord("height").points, expected_heights, rtol=1e-2
    )


def test_extract_and_interpolate_layer_coordinates():
    """Test that LayerTemperatureInterpolation preserves and returns
    the correct coordinates in the output cube.

    Checks:
        - Height coordinate exists with correct units.
        - projection_y_coordinate exists and matches input.
        - projection_x_coordinate exists and matches input.
    """
    cube = make_test_cube()
    plugin = LayerTemperatureInterpolation()
    result = plugin.process(cube, bottom=600, top=800, verbosity=0)

    # Check height coordinate exists and has correct units
    height_coord = result.coord("height")
    assert height_coord.units == "m"

    # Check y coordinate exists and matches input
    np.testing.assert_array_equal(
        result.coord("projection_y_coordinate").points,
        cube.coord("projection_y_coordinate").points,
    )

    # Check x coordinate exists and matches input
    np.testing.assert_array_equal(
        result.coord("projection_x_coordinate").points,
        cube.coord("projection_x_coordinate").points,
    )


def test_interpolated_base_and_top_values():
    """Test that LayerTemperatureInterpolation correctly interpolates
    temperature values at the base and top of the layer.

    Layer bounds: 600 ft (≈182.88 m) to 800 ft (≈243.84 m).

    Expected values are derived directly from iris.analysis.Linear() interpolation
    to ensure we are testing our plugin's behaviour, not reimplementing the science.
    """
    cube = make_test_cube()
    plugin = LayerTemperatureInterpolation()
    result = plugin.process(cube, bottom=600, top=800, verbosity=0)

    # Use Iris directly to get the expected interpolated values at base and top
    expected_base = cube.interpolate(
        [("height", np.array([600 / METRES_TO_FT], dtype=np.float32))],
        iris.analysis.Linear(),
    )
    expected_top = cube.interpolate(
        [("height", np.array([800 / METRES_TO_FT], dtype=np.float32))],
        iris.analysis.Linear(),
    )

    # Check base temperature values match Iris interpolation at all grid points
    np.testing.assert_allclose(
        result.data[0, :, :], expected_base.data[0, :, :], rtol=1e-5
    )

    # Check top temperature values match Iris interpolation at all grid points
    np.testing.assert_allclose(
        result.data[-1, :, :], expected_top.data[0, :, :], rtol=1e-5
    )


def test_verbosity_layer_extraction(capsys):
    """Test that LayerTemperatureInterpolation prints expected output
    when verbosity is set to 1.

    Checks that the bottom and top layer bounds are printed to stdout.
    """
    cube = make_test_cube()
    plugin = LayerTemperatureInterpolation()
    plugin.process(cube, bottom=600, top=800, verbosity=1)
    captured = capsys.readouterr()
    assert "600" in captured.out
    assert "800" in captured.out


def test_layer_bounds_match_data_level():
    """Test that LayerTemperatureInterpolation handles the case where
    the layer bounds are both very close to an existing data level.

    Layer bounds: 656 ft (≈200 m) to 787 ft (≈240 m).
    Both 200 m and 240 m are exact data levels in the test cube.

    Expected output:
        - No duplicate height points in the output cube.
    """
    cube = make_test_cube()
    plugin = LayerTemperatureInterpolation()

    # Pass bounds directly in feet - the plugin handles the conversion internally
    result = plugin.process(cube, bottom=656, top=787, verbosity=0)

    # Check no duplicate height points exist in the output
    height_points = result.coord("height").points
    assert len(height_points) == len(
        np.unique(height_points)
    ), f"Duplicate height points found: {height_points}"

def test_no_interior_levels():
    """Test that LayerTemperatureInterpolation returns at least the interpolated
    base and top when no interior levels exist within the layer bounds.

    Layer bounds: 610 ft (≈185 m) to 640 ft (≈195 m).
    No data levels fall between 185 m and 195 m in the test cube
    (nearest levels are 200 m, 220 m, 240 m).

    Expected output:
        - Interpolated base at 610 ft (≈185 m)
        - Interpolated top at 640 ft (≈195 m)
        - Total: 2 height levels, shape (2, 2, 2)
    """
    cube = make_test_cube()
    plugin = LayerTemperatureInterpolation()

    # These bounds have no data levels between them
    result = plugin.process(cube, bottom=610, top=640, verbosity=0)

    # Should have exactly 2 levels: interpolated base and top only
    assert result.shape == (2, 2, 2)


def test_bottom_greater_than_top_raises_error():
    """Test that LayerTemperatureInterpolation raises a ValueError when
    the bottom bound is greater than the top bound.

    This is physically nonsensical and should be caught early with a
    clear error message rather than producing garbage output silently.
    """
    cube = make_test_cube()
    plugin = LayerTemperatureInterpolation()

    with pytest.raises(ValueError, match="Bottom .* must be less than top"):
        plugin.process(cube, bottom=800, top=600, verbosity=0)
