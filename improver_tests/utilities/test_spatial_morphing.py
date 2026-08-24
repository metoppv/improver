# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Tests for the SpatialMorphing plugin."""

import json

import numpy as np
import pytest
from iris.cube import Cube, CubeList

from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube
from improver.utilities.spatial_morphing import SpatialMorphing


def make_test_cube(
    data_value=1.0,
    shape=(2, 5, 5),
    model_id="uk_ens",
    realizations=None,
    cluster_sources=None,
):
    """Create a test cube with specified properties.

    Args:
        data_value: Scalar value to fill cube data.
        shape: Tuple of (n_realizations, dim1, dim2).
        model_id: Model ID attribute value.
        realizations: Optional array of realization indices.
        cluster_sources: Optional dict for cluster_sources attribute.

    Returns:
        Iris Cube with specified configuration.
    """
    if realizations is None:
        realizations = np.arange(shape[0])

    data = np.full(shape, data_value, dtype=np.float32)
    cube = set_up_variable_cube(
        data,
        name="air_temperature",
        units="K",
        spatial_grid="equalarea",
        realizations=realizations,
    )
    cube.attributes["mosg__model_configuration"] = model_id

    if cluster_sources is not None:
        cube.attributes["cluster_sources"] = json.dumps(cluster_sources)

    return cube


# ============================================================================
# Initialization tests
# ============================================================================


@pytest.mark.parametrize(
    "model_id_attr,cluster_sources_attr,window_minutes",
    [
        ("mosg__model_configuration", "cluster_sources", 180),
        ("custom_model_attr", "custom_sources", 360),
        ("model_id", "sources", 120),
    ],
)
def test_init_custom_attributes(model_id_attr, cluster_sources_attr, window_minutes):
    """Test initialization with custom attributes."""
    plugin = SpatialMorphing(
        model_id_attr=model_id_attr,
        cluster_sources_attribute=cluster_sources_attr,
        interpolation_window_in_minutes=window_minutes,
    )
    assert plugin.model_id_attr == model_id_attr
    assert plugin.cluster_sources_attribute == cluster_sources_attr
    assert plugin.interpolation_window_in_minutes == window_minutes


def test_init_default_parameters():
    """Test initialization with default parameters."""
    plugin = SpatialMorphing()
    assert plugin.model_id_attr == "mosg__model_configuration"
    assert plugin.cluster_sources_attribute == "cluster_sources"
    assert plugin.interpolation_window_in_minutes == 180
    assert plugin.interpolation_window_by_source_pair == {}


@pytest.mark.parametrize(
    "windows,expected_count",
    [
        ({"uk_ens|gl_ens": 360}, 1),
        ({"uk_ens|gl_ens": 360, "uk_det,ec_det": 120}, 2),
        ({"a|b": 60, "c,d": 90, "e|f": 180}, 3),
    ],
)
def test_init_source_pair_windows(windows, expected_count):
    """Test initialization with source-pair transition windows."""
    plugin = SpatialMorphing(
        interpolation_window_by_source_pair=windows,
    )
    assert len(plugin.interpolation_window_by_source_pair) == expected_count


@pytest.mark.parametrize(
    "invalid_input,error_match",
    [
        ("not_a_dict", "must be a dictionary"),
        ({"uk_ens|gl_ens": -1}, "must be positive integers"),
        ({"uk_ens|uk_ens": 60}, "two distinct source names"),
        ({"uk_ens": 60}, "must contain exactly two source names"),
    ],
)
def test_init_invalid_window_configuration(invalid_input, error_match):
    """Test that invalid window configuration raises appropriate errors."""
    with pytest.raises(ValueError, match=error_match):
        SpatialMorphing(interpolation_window_by_source_pair=invalid_input)


# ============================================================================
# Source pair key parsing tests
# ============================================================================


@pytest.mark.parametrize(
    "key,expected",
    [
        ("uk_ens|gl_ens", frozenset(["uk_ens", "gl_ens"])),
        ("uk_ens, gl_ens", frozenset(["uk_ens", "gl_ens"])),
        ("gl_ens|uk_ens", frozenset(["uk_ens", "gl_ens"])),  # Order insensitive
        ("source_a|source_b", frozenset(["source_a", "source_b"])),
    ],
)
def test_prepare_source_pair_key(key, expected):
    """Test parsing of source-pair keys with various delimiters."""
    plugin = SpatialMorphing()
    result = plugin._prepare_source_pair_key(key)
    assert result == expected


@pytest.mark.parametrize(
    "invalid_key,error_match",
    [
        ("single_source", "exactly two source names"),
        ("uk_ens|uk_ens", "two distinct source names"),
        ("uk_ens||gl_ens", "exactly two source names"),
        ("uk_ens, gl_ens, another", "exactly two source names"),
    ],
)
def test_prepare_source_pair_key_invalid(invalid_key, error_match):
    """Test that invalid source-pair keys raise errors."""
    plugin = SpatialMorphing()
    with pytest.raises(ValueError, match=error_match):
        plugin._prepare_source_pair_key(invalid_key)


# ============================================================================
# Cluster sources parsing tests
# ============================================================================


@pytest.mark.parametrize(
    "cluster_sources,should_parse",
    [
        ({"0": {"uk_ens": [3600, 21600], "gl_ens": [86400]}}, True),
        ({"0": {"uk_det": [3600]}, "1": {"uk_ens": [21600]}}, True),
        ({}, True),  # Empty dict is valid
    ],
)
def test_parse_cluster_sources_valid(cluster_sources, should_parse):
    """Test parsing valid cluster_sources attribute."""
    plugin = SpatialMorphing()
    cube = make_test_cube(cluster_sources=cluster_sources)

    result = plugin._parse_cluster_sources(cube)
    if should_parse:
        assert result == cluster_sources


def test_parse_cluster_sources_missing_attribute():
    """Test that missing cluster_sources returns empty dict."""
    plugin = SpatialMorphing()
    cube = make_test_cube()
    # Explicitly remove cluster_sources
    cube.attributes.pop("cluster_sources", None)

    result = plugin._parse_cluster_sources(cube)
    assert result == {}


@pytest.mark.parametrize(
    "invalid_json,error_match",
    [
        ("not valid json {", "Failed to parse cluster sources JSON"),
        ('{"0": [}', "Failed to parse cluster sources JSON"),
    ],
)
def test_parse_cluster_sources_invalid_json(invalid_json, error_match):
    """Test that invalid JSON raises error."""
    plugin = SpatialMorphing()
    cube = make_test_cube()
    cube.attributes["cluster_sources"] = invalid_json

    with pytest.raises(ValueError, match=error_match):
        plugin._parse_cluster_sources(cube)


# ============================================================================
# Source pair identification tests (internal logic via process)
# ============================================================================


@pytest.mark.parametrize(
    "cluster_sources,query_fp,expected_transition",
    [
        # Single source - no transition
        ({"0": {"uk_ens": [3600, 21600, 86400]}}, 21600, False),
        # Exact match - no transition
        ({"0": {"uk_det": [3600], "uk_ens": [21600]}}, 3600, False),
        # Between two sources - transition
        ({"0": {"uk_det": [3600], "uk_ens": [21600]}}, 14400, True),
        # Midpoint between sources
        ({"0": {"uk_det": [3600], "uk_ens": [21600]}}, 12600, True),
    ],
)
def test_identify_source_pair_for_validity_time(
    cluster_sources, query_fp, expected_transition
):
    """Test source pair identification at various validity times."""
    plugin = SpatialMorphing()
    result = plugin._identify_source_pair_for_validity_time(
        cluster_sources, 0, query_fp
    )

    if expected_transition:
        assert result is not None
        source_a, fp_a, source_b, fp_b, weight = result
        assert isinstance(weight, (int, float))
        assert 0.0 <= weight <= 1.0
    else:
        assert result is None


@pytest.mark.parametrize(
    "query_fp,expected_weight",
    [
        (3600, 0.0),  # At fp_a
        (21600, 1.0),  # At fp_b
        (12600, 0.5),  # Midpoint
        (14400, (14400 - 3600) / (21600 - 3600)),  # 2/3 of the way
    ],
)
def test_weight_calculation(query_fp, expected_weight):
    """Test weight calculation between two source forecast periods."""
    plugin = SpatialMorphing()
    cluster_sources = {"0": {"uk_det": [3600], "uk_ens": [21600]}}

    result = plugin._identify_source_pair_for_validity_time(
        cluster_sources, 0, query_fp
    )

    if result is not None:
        _, _, _, _, weight = result
        assert np.isclose(
            weight, expected_weight, rtol=1e-5
        ), f"Weight {weight} != {expected_weight}"
    else:
        # If exact match at boundaries, no transition returned
        assert expected_weight in (0.0, 1.0)


# ============================================================================
# Cube extraction tests (internal logic via process)
# ============================================================================


@pytest.mark.parametrize(
    "n_realizations,query_realization,should_exist",
    [
        (2, 0, True),
        (2, 1, True),
        (5, 3, True),
        (2, 99, False),
        (1, 0, True),
        (1, 1, False),
    ],
)
def test_extract_source_cube_for_realization(
    n_realizations, query_realization, should_exist
):
    """Test extracting cubes for specific realization from source."""
    plugin = SpatialMorphing()
    cube = make_test_cube(shape=(n_realizations, 5, 5), model_id="uk_ens")
    cubes = CubeList([cube])

    result = plugin._extract_source_cube_for_realization(
        cubes, query_realization, "uk_ens"
    )

    if should_exist:
        assert result is not None
        assert result.shape == (5, 5)
    else:
        assert result is None


def test_extract_source_cube_nonexistent_source():
    """Test extracting from non-existent source returns None."""
    plugin = SpatialMorphing()
    cube = make_test_cube(model_id="uk_ens")
    cubes = CubeList([cube])

    result = plugin._extract_source_cube_for_realization(cubes, 0, "gl_ens")
    assert result is None


# ============================================================================
# Process method tests (public interface)
# ============================================================================


@pytest.mark.parametrize(
    "data_values,source_ids,expected_output_value",
    [
        ([100.0], ["uk_ens"], 100.0),  # Single cube
        ([100.0, 100.0], ["uk_ens", "uk_ens"], 100.0),  # Duplicate source
    ],
)
def test_process_single_source_passthrough(
    data_values, source_ids, expected_output_value
):
    """Test that single-source realizations pass through unchanged."""
    plugin = SpatialMorphing()

    cubes = CubeList()
    cluster_sources = {}

    for idx, (value, source_id) in enumerate(zip(data_values, source_ids)):
        cube = make_test_cube(
            data_value=value,
            model_id=source_id,
            cluster_sources={"0": {source_id: [21600]}},
        )
        cubes.append(cube)
        cluster_sources["0"] = {source_id: [21600]}

    result = plugin.process(cubes)

    assert isinstance(result, Cube)
    assert result.shape == (2, 5, 5)
    assert np.allclose(result.data, expected_output_value)
    assert "mosg__model_configuration" not in result.attributes


def test_process_removes_model_id_attribute():
    """Test that model_id_attr is removed from output."""
    plugin = SpatialMorphing()
    cube = make_test_cube(
        model_id="uk_ens",
        cluster_sources={"0": {"uk_ens": [21600]}, "1": {"uk_ens": [21600]}},
    )

    result = plugin.process(CubeList([cube]))

    assert "mosg__model_configuration" not in result.attributes


def test_process_preserves_other_attributes():
    """Test that other attributes are preserved."""
    plugin = SpatialMorphing()
    cube = make_test_cube(
        model_id="uk_ens",
        cluster_sources={"0": {"uk_ens": [21600]}, "1": {"uk_ens": [21600]}},
    )
    cube.attributes["custom_attr"] = "custom_value"

    result = plugin.process(CubeList([cube]))

    assert result.attributes.get("custom_attr") == "custom_value"


def test_process_mismatched_validity_times_raises():
    """Test that mismatched validity times raise error."""
    plugin = SpatialMorphing()

    cube1 = make_test_cube(model_id="uk_ens")
    cube1.coord("forecast_period").points = [21600]

    cube2 = make_test_cube(model_id="gl_ens")
    cube2.coord("forecast_period").points = [43200]

    with pytest.raises(ValueError, match="same validity time"):
        plugin.process(CubeList([cube1, cube2]))


def test_process_no_cluster_sources_warning():
    """Test that missing cluster_sources generates warning."""
    plugin = SpatialMorphing()
    cube = make_test_cube(model_id="uk_ens")
    # Remove cluster_sources to trigger warning
    cube.attributes.pop("cluster_sources", None)

    with pytest.warns(UserWarning, match="No cluster_sources"):
        result = plugin.process(CubeList([cube]))

    assert isinstance(result, Cube)


@pytest.mark.parametrize(
    "n_realizations,n_sources",
    [
        (2, 1),
        (5, 2),
        (10, 3),
    ],
)
def test_process_multiple_realizations(n_realizations, n_sources):
    """Test that process handles multiple realizations correctly."""
    plugin = SpatialMorphing()

    # Create cube with multiple realizations
    cube = make_test_cube(
        shape=(n_realizations, 5, 5),
        model_id="uk_ens",
        cluster_sources={str(i): {"uk_ens": [21600]} for i in range(n_realizations)},
    )

    result = plugin.process(CubeList([cube]))

    assert isinstance(result, Cube)
    assert result.shape[0] == n_realizations  # Realizations preserved


# ============================================================================
# Transition window tests
# ============================================================================


@pytest.mark.parametrize(
    "windows,source_pair,expected_seconds",
    [
        (
            {"uk_ens|gl_ens": 360},
            frozenset(["uk_ens", "gl_ens"]),
            360 * 60,
        ),
        (
            {"uk_ens|gl_ens": 360, "uk_det|ec_det": 120},
            frozenset(["uk_det", "ec_det"]),
            120 * 60,
        ),
    ],
)
def test_get_transition_window_specific_pair(windows, source_pair, expected_seconds):
    """Test that specific source-pair window is used if configured."""
    plugin = SpatialMorphing(
        interpolation_window_in_minutes=180,
        interpolation_window_by_source_pair=windows,
    )
    window = plugin._get_transition_window_in_seconds(
        frozenset([list(source_pair)[0]]),
        frozenset([list(source_pair)[1]]),
    )
    assert window == expected_seconds


def test_get_transition_window_default_fallback():
    """Test that default window is used if pair not configured."""
    plugin = SpatialMorphing(
        interpolation_window_in_minutes=180,
        interpolation_window_by_source_pair={"uk_ens|gl_ens": 360},
    )
    window = plugin._get_transition_window_in_seconds(
        frozenset(["uk_det"]),
        frozenset(["ec_det"]),
    )
    assert window == 180 * 60  # Default 180 minutes in seconds


@pytest.mark.parametrize(
    "default_window",
    [60, 120, 180, 360],
)
def test_get_transition_window_various_defaults(default_window):
    """Test transition window with various default values."""
    plugin = SpatialMorphing(
        interpolation_window_in_minutes=default_window,
    )
    window = plugin._get_transition_window_in_seconds(
        frozenset(["source_a"]),
        frozenset(["source_b"]),
    )
    assert window == default_window * 60


# ============================================================================
# Integration tests through process method
# ============================================================================


def test_process_end_to_end_single_source():
    """Integration test: process with single source end-to-end."""
    plugin = SpatialMorphing()

    cube = make_test_cube(
        data_value=42.0,
        shape=(3, 10, 10),
        model_id="uk_ens",
        cluster_sources={
            "0": {"uk_ens": [3600]},
            "1": {"uk_ens": [3600]},
            "2": {"uk_ens": [3600]},
        },
    )

    result = plugin.process(CubeList([cube]))

    assert isinstance(result, Cube)
    assert result.shape == (3, 10, 10)
    assert np.allclose(result.data, 42.0)
    assert "mosg__model_configuration" not in result.attributes


def test_process_with_realizations_and_cluster_sources():
    """Integration test: process with cluster_sources metadata."""
    plugin = SpatialMorphing()

    # Create cubes with cluster_sources indicating source transitions
    cluster_sources = {
        "0": {"uk_det": [3600], "uk_ens": [21600]},
        "1": {"uk_ens": [3600, 21600]},
    }

    cube = make_test_cube(
        data_value=100.0,
        shape=(2, 8, 8),
        model_id="uk_ens",
        cluster_sources=cluster_sources,
    )

    result = plugin.process(CubeList([cube]))

    assert isinstance(result, Cube)
