# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.

"""Unit tests for SpotManipulation class"""

import warnings

import iris
import numpy as np
import pytest
from iris.coords import AuxCoord, DimCoord
from numpy.testing import assert_array_equal

from improver.metadata.utilities import create_coordinate_hash
from improver.spotdata.spot_manipulation import SpotManipulation
from improver.synthetic_data.set_up_test_cubes import (
    add_coordinate,
    set_up_percentile_cube,
    set_up_probability_cube,
    set_up_spot_percentile_cube,
    set_up_spot_probability_cube,
    set_up_spot_variable_cube,
    set_up_variable_cube,
)

ATTRIBUTES = {"mosg__grid_domain": "global", "mosg__grid_type": "standard"}


def gridded_variable(forecast_data):
    """Create a gridded variable cube from which to extract spot forecasts."""
    height_crd = DimCoord(
        np.array([1.5], dtype=np.float32), standard_name="height", units="m"
    )
    return set_up_variable_cube(
        forecast_data, include_scalar_coords=[height_crd], attributes=ATTRIBUTES,
    )


def spot_variable(forecast_data):
    """Create a spot variable cube to subset with a neighbour cube."""
    height_crd = DimCoord(
        np.array([1.5], dtype=np.float32), standard_name="height", units="m"
    )
    return set_up_spot_variable_cube(
        forecast_data, include_scalar_coords=[height_crd], attributes=ATTRIBUTES,
    )


@pytest.fixture
def gridded_lapse_rate(lapse_rates):
    """Create a gridded lapse_rate cube."""
    height_crd = DimCoord(
        np.array([1.5], dtype=np.float32), standard_name="height", units="m"
    )
    return set_up_variable_cube(
        lapse_rates,
        name="air_temperature_lapse_rate",
        units="K m-1",
        include_scalar_coords=[height_crd],
        attributes=ATTRIBUTES,
    )


def gridded_percentiles(forecast_data):
    """Create a gridded percentile cube from which to extract spot forecasts."""
    n_percentiles = forecast_data.shape[0]
    percentiles = np.linspace(20, 80, n_percentiles)
    return set_up_percentile_cube(forecast_data, percentiles, attributes=ATTRIBUTES,)


def spot_percentiles(forecast_data):
    """Create a spot percentile cube to subset with a neighbour cube."""
    n_percentiles = forecast_data.shape[0]
    percentiles = np.linspace(20, 80, n_percentiles)
    return set_up_spot_percentile_cube(
        forecast_data, percentiles, attributes=ATTRIBUTES,
    )


def gridded_probabilities(forecast_data):
    """Create a gridded probability cube from which to extract spot forecasts."""
    n_thresholds = forecast_data.shape[0]
    thresholds = np.linspace(273, 283, n_thresholds)
    return set_up_probability_cube(forecast_data, thresholds, attributes=ATTRIBUTES,)


def spot_probabilities(forecast_data):
    """Create a spot probability cube to subset with a neighbour cube."""
    n_thresholds = forecast_data.shape[0]
    thresholds = np.linspace(273, 283, n_thresholds)
    return set_up_spot_probability_cube(
        forecast_data, thresholds, attributes=ATTRIBUTES,
    )


@pytest.fixture
def neighbour_cube(neighbour_data):
    """Takes in a 3D array. The leading dimension corresponds to each type
    of neighbour selection. The second dimension corresponds to the different
    attributes, these being x_index, y_index, and vertical_displacement. The
    final dimension corresponds to the different sites."""

    n_methods = neighbour_data.shape[0]
    # Generate a cube with the right number of sites to which we can add
    # further coordinates.
    data_1d = neighbour_data[0, 0]

    cube = set_up_spot_variable_cube(data_1d)
    methods = [
        "nearest",
        "nearest_land",
        "nearest_minimum_dz",
        "nearest_land_minimum_dz",
    ]
    method_name = AuxCoord(
        methods[0:n_methods], long_name="neighbour_selection_method_name"
    )
    grid_attributes_key = AuxCoord(
        ["x_index", "y_index", "vertical_displacement"], long_name="grid_attributes_key"
    )
    cube = add_coordinate(cube, [0, 1, 2], "grid_attributes", dtype="int32")
    cube = add_coordinate(
        cube, np.arange(n_methods), "neighbour_selection_method", dtype="int32"
    )
    if n_methods == 1:
        cube = iris.util.new_axis(cube, "neighbour_selection_method")
    cube.add_aux_coord(method_name, 0)
    cube.add_aux_coord(grid_attributes_key, 1)
    cube.data = neighbour_data
    return cube


def add_grid_hash(target, source):
    """Add a grid hash attribute to the target cube that has been generated
    using the coordinates on the source cube. This allows a neighbour cube
    to be used with a gridded cube by ensuring the hash test passes in the
    spot extraction plugin."""

    grid_hash = create_coordinate_hash(source)
    target.attributes["model_grid_hash"] = grid_hash


@pytest.mark.parametrize(
    "ftype,forecast_data,neighbour_data,kwargs,expected",
    [
        # Simple extraction of forecast variable from grid.
        (
            gridded_variable,
            np.arange(273, 282).reshape(3, 3),
            np.array([[[1, 2], [0, 0], [5, 10]]]),
            {},
            np.array([274, 275]),
        ),
        # Another simple extraction for a different point.
        (
            gridded_variable,
            np.arange(273, 282).reshape(3, 3),
            np.array([[[1, 2], [2, 2], [5, 10]]]),
            {},
            np.array([280, 281]),
        ),
        # Extract existing percentiles from a percentile input cube.
        (
            gridded_percentiles,
            np.arange(273, 291).reshape(2, 3, 3),
            np.array([[[1, 2], [0, 0], [5, 10]]]),
            {"extract_percentiles": [20, 80]},
            np.array([[274, 275], [283, 284]]),
        ),
        # Resample existing percentiles to extract target percentile.
        (
            gridded_percentiles,
            np.arange(273, 291).reshape(2, 3, 3),
            np.array([[[1, 2], [0, 0], [5, 10]]]),
            {"extract_percentiles": [50]},
            np.array([278.5, 279.5]),
        ),
        # Extract single percentile from a probability input cube.
        (
            gridded_probabilities,
            np.stack(
                [
                    np.full((3, 3), 1.0, dtype=np.float32),
                    np.full((3, 3), 0.75, dtype=np.float32),
                    np.full((3, 3), 0.5, dtype=np.float32),
                    np.full((3, 3), 0.25, dtype=np.float32),
                    np.full((3, 3), 0.0, dtype=np.float32),
                ]
            ),
            np.array([[[1, 2], [0, 0], [5, 10]]]),
            {"extract_percentiles": [50]},
            np.array([278, 278]),
        ),
        # Extract multiple percentiles from a probability input cube.
        (
            gridded_probabilities,
            np.stack(
                [
                    np.full((3, 3), 1.0, dtype=np.float32),
                    np.full((3, 3), 0.75, dtype=np.float32),
                    np.full((3, 3), 0.5, dtype=np.float32),
                    np.full((3, 3), 0.25, dtype=np.float32),
                    np.full((3, 3), 0.0, dtype=np.float32),
                ]
            ),
            np.array([[[1, 2], [0, 0], [5, 10]]]),
            {"extract_percentiles": [40, 60]},
            np.array([[277, 277], [279, 279]]),
        ),
        # Extraction including realization collapse
        (
            gridded_variable,
            np.arange(273, 291).reshape(2, 3, 3),
            np.array([[[1, 2], [0, 0], [5, 10]]]),
            {"realization_collapse": True},
            np.array([278.5, 279.5]),
        ),
        # Percentile extraction derived from realizations
        (
            gridded_variable,
            np.arange(273, 300).reshape(3, 3, 3),
            np.array([[[1, 2], [0, 0], [5, 10]]]),
            {"extract_percentiles": [50]},
            np.array([283, 284]),
        ),
        # Percentile extraction derived from realizations; masked type, no mask
        (
            gridded_variable,
            np.ma.arange(273, 300).reshape(3, 3, 3),
            np.array([[[1, 2], [0, 0], [5, 10]]]),
            {"extract_percentiles": [50]},
            np.array([283, 284]),
        ),
        # Percentile extraction derived from realizations; masked type, masking
        (
            gridded_variable,
            np.ma.masked_array(
                [
                    np.arange(273, 282).reshape(3, 3),
                    np.arange(282, 291).reshape(3, 3),
                    np.arange(291, 300).reshape(3, 3),
                    np.arange(300, 309).reshape(3, 3),
                ],
                mask=[
                    np.zeros((3, 3)),
                    np.zeros((3, 3)),
                    np.zeros((3, 3)),
                    np.ones((3, 3)),
                ],
            ),
            np.array([[[1, 2], [0, 0], [5, 10]]]),
            {"extract_percentiles": [50]},
            np.array([283, 284]),
        ),
        # Subset a spot variable cube.
        (
            spot_variable,
            np.arange(273, 282),
            np.array([[[1, 2, 3], [0, 0, 0], [5, 10, 15]]]),
            {"subset_coord": "wmo_id"},
            np.array([273, 274, 275]),
        ),
        # Subset a spot percentile cube and extract percentiles.
        (
            spot_percentiles,
            np.arange(273, 285).reshape(2, 6),
            np.array([[[1, 2], [0, 0], [5, 10]]]),
            {"subset_coord": "wmo_id", "extract_percentiles": [20, 80]},
            np.array([[273, 274], [279, 280]]),
        ),
        # Subset a spot percentile cube and resample percentiles.
        (
            spot_percentiles,
            np.arange(273, 285).reshape(2, 6),
            np.array([[[1, 2], [0, 0], [5, 10]]]),
            {"subset_coord": "wmo_id", "extract_percentiles": [50]},
            np.array([276, 277]),
        ),
        # Subset a spot probability cube and extract a percentile.
        (
            spot_probabilities,
            np.stack(
                [
                    np.linspace(0.9, 1.0, 6, dtype=np.float32),
                    np.linspace(0.65, 0.75, 6, dtype=np.float32),
                    np.linspace(0.4, 0.5, 6, dtype=np.float32),
                    np.linspace(0.15, 0.25, 6, dtype=np.float32),
                    np.zeros(6, dtype=np.float32),
                ]
            ),
            np.array([[[1, 2], [0, 0], [5, 10]]]),
            {"subset_coord": "wmo_id", "extract_percentiles": [50]},
            np.array([277, 277.2], dtype=np.float32),
        ),
        # Subset a spot probability cube and extract multiple percentiles.
        (
            spot_probabilities,
            np.stack(
                [
                    np.linspace(0.9, 1.0, 6, dtype=np.float32),
                    np.linspace(0.65, 0.75, 6, dtype=np.float32),
                    np.linspace(0.4, 0.5, 6, dtype=np.float32),
                    np.linspace(0.15, 0.25, 6, dtype=np.float32),
                    np.zeros(6, dtype=np.float32),
                ]
            ),
            np.array([[[1, 2], [0, 0], [5, 10]]]),
            {"subset_coord": "wmo_id", "extract_percentiles": [40, 60]},
            np.array([[276, 276.2], [278, 278.2]], dtype=np.float32),
        ),
    ],
)
def test_extraction(ftype, forecast_data, neighbour_cube, kwargs, expected):
    """Test extraction options from simple variable cubes, percentile cubes,
    and probability cubes."""

    forecast = ftype(forecast_data)

    add_grid_hash(neighbour_cube, forecast)
    result = SpotManipulation(**kwargs)([forecast, neighbour_cube])
    assert_array_equal(result.data, expected)
    assert result.attributes.get("model_grid_hash", None) is None


@pytest.mark.parametrize("neighbour_data", [np.array([[[1, 2], [0, 0], [5, 10]]])])
def test_unknown_prob_type_warning(neighbour_cube):
    """Test a warning is raised if percentiles are requested from a cube from
    which they cannot be extracted and that all data is returned in spot
    format. In this case that is the 20th and 80th percentiles are both
    returned rather than the requested 50th percentile. If the
    suppress_warnings option is set then test that no warning is raised."""

    expected = np.array([[274, 275], [283, 284]])
    forecast = gridded_percentiles(np.arange(273, 291).reshape(2, 3, 3))
    forecast.coord("percentile").rename("kittens")
    add_grid_hash(neighbour_cube, forecast)

    # Test with warning suppression
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = SpotManipulation(extract_percentiles=[50], suppress_warnings=True)(
            [forecast, neighbour_cube]
        )
    assert_array_equal(result.data, expected)

    # Test without warning suppression
    with pytest.warns(
        UserWarning, match="Diagnostic cube is not a known probabilistic type"
    ):
        result = SpotManipulation(extract_percentiles=[50])([forecast, neighbour_cube])
    assert_array_equal(result.data, expected)


@pytest.mark.parametrize(
    "forecast_data,lapse_rates,neighbour_data,kwargs,expected",
    [
        (
            np.arange(273, 282).reshape(3, 3),
            np.full((3, 3), 0.1, dtype=np.float32),
            np.array([[[1, 2], [0, 0], [5, 10]]]),
            {"apply_lapse_rate_correction": True},
            np.array([274.5, 276]),
        ),
        (
            np.arange(273, 282).reshape(3, 3),
            np.full((3, 3), 0, dtype=np.float32),
            np.array([[[1, 2], [0, 0], [5, 10]]]),
            {"apply_lapse_rate_correction": True, "fixed_lapse_rate": 0.2},
            np.array([275, 277]),
        ),
    ],
)
def test_lapse_rate_correction(
    forecast_data, gridded_lapse_rate, neighbour_cube, kwargs, expected
):
    """Test the application of lapse rates using a lapse rate cube and a fixed
    lapse rate value."""

    forecast = gridded_variable(forecast_data)
    add_grid_hash(neighbour_cube, forecast)
    inputs = [forecast, gridded_lapse_rate, neighbour_cube]
    if "fixed_lapse_rate" in kwargs.keys():
        inputs = [forecast, neighbour_cube]

    result = SpotManipulation(**kwargs)(inputs)
    assert_array_equal(result.data, expected)


@pytest.mark.parametrize(
    "forecast_data,neighbour_data,kwargs,expected",
    [
        (
            np.arange(273, 282).reshape(3, 3),
            np.array([[[1, 2], [0, 0], [5, 10]]]),
            {"apply_lapse_rate_correction": True},
            np.array([274, 275]),
        ),
        (
            np.arange(273, 282).reshape(3, 3),
            np.array([[[1, 2], [0, 0], [5, 10]]]),
            {"apply_lapse_rate_correction": True, "suppress_warnings": True},
            np.array([274, 275]),
        ),
    ],
)
def test_missing_lapse_rate_warning(forecast_data, neighbour_cube, kwargs, expected):
    """Test a warning is raised if no lapse rate is specified whilst the
    apply_lapse_rate_correction kwarg is set to True. If the
    suppress_warnings option is set then test that no warning is raised."""

    forecast = gridded_variable(forecast_data)
    add_grid_hash(neighbour_cube, forecast)

    if "suppress_warnings" in kwargs.keys():
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = SpotManipulation(**kwargs)([forecast, neighbour_cube])
    else:
        with pytest.warns(
            UserWarning, match="A lapse rate cube or fixed lapse rate was not provided"
        ):
            result = SpotManipulation(**kwargs)([forecast, neighbour_cube])

    assert_array_equal(result.data, expected)


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({}, "nearest"),
        ({"land_constraint": True}, "nearest_land"),
        ({"similar_altitude": True}, "nearest_minimum_dz"),
        (
            {"land_constraint": True, "similar_altitude": True},
            "nearest_land_minimum_dz",
        ),
    ],
)
def test_neighbour_selection_method_setting(kwargs, expected):
    """Test that the neighbour_selection_method argument is set correctly
    within the __init__ method."""
    plugin = SpotManipulation(**kwargs)
    assert plugin.neighbour_selection_method == expected
