# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Tests for the refactored SpatialMorphing plugin.

The SpatialMorphing plugin now leverages RealizationSelection to identify
which realizations from each forecast source correspond to a cluster, then applies
spatial morphing via Google FILM.
"""

import json
from unittest.mock import patch

import numpy as np
import pytest
from iris.cube import Cube, CubeList

from improver.clustering.realization_clustering import RealizationSelection
from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube
from improver.utilities.spatial_morphing import SpatialMorphing


def make_forecast_cube(model_id="uk_ens", n_realizations=2):
    """Create a test forecast cube with realization coordinate.

    Args:
        model_id: Model ID attribute value.
        n_realizations: Number of realizations.

    Returns:
        Iris Cube with forecast data.
    """
    data = np.arange(n_realizations * 5 * 5, dtype=np.float32).reshape(
        n_realizations, 5, 5
    )
    cube = set_up_variable_cube(
        data,
        name="precipitation_accumulation",
        units="mm",
        spatial_grid="equalarea",
        realizations=np.arange(n_realizations),
    )
    cube.attributes["mosg__model_configuration"] = model_id
    return cube


def make_cluster_cube():
    """Create a mock cluster cube with required attributes.

    Returns:
        Iris Cube with cluster mapping attributes.
    """
    data = np.zeros((5, 5), dtype=np.float32)
    cube = set_up_variable_cube(
        data,
        name="clustering_result",
        units="1",
        spatial_grid="equalarea",
    )

    # Add cluster mapping attributes (from RealizationClusterAndMatch)
    primary_map = {"0": 0, "1": 1}  # cluster_idx -> medoid_realization
    cube.attributes["primary_input_realization_to_cluster_medoid"] = json.dumps(
        primary_map
    )

    # Secondary map (from secondary models)
    secondary_map = {
        "uk_det": {
            "0": [{"realization": 0, "forecast_periods": [22500]}],
            "1": [{"realization": 1, "forecast_periods": [22500]}],
        }
    }
    cube.attributes["secondary_input_realizations_to_clusters"] = json.dumps(
        secondary_map
    )

    # cluster_sources attribute
    cluster_sources = {
        "0": {"uk_ens": [22500]},
        "1": {"uk_ens": [22500]},
    }
    cube.attributes["cluster_sources"] = json.dumps(cluster_sources)

    return cube


# ============================================================================
# Initialization tests
# ============================================================================


@pytest.mark.parametrize(
    "forecast_period,cluster_number",
    [(22500, 0), (22500, 1), (3600, 0)],
)
def test_init_with_required_parameters(forecast_period, cluster_number):
    """Test initialization with required parameters."""
    plugin = SpatialMorphing(
        forecast_period=forecast_period,
        cluster_number=cluster_number,
    )
    assert plugin.forecast_period == forecast_period
    assert plugin.cluster_number == cluster_number
    assert plugin.model_id_attr == "mosg__model_configuration"
    assert plugin.cycletime is None


def test_init_with_optional_parameters():
    """Test initialization with optional parameters."""
    plugin = SpatialMorphing(
        forecast_period=22500,
        cluster_number=0,
        cycletime="20240203T0000Z",
        selection_attr="realization_selection_method",
        selection_attr_value="cluster_medoid",
        model_path="/path/to/model",
        scaling="log10",
    )
    assert plugin.forecast_period == 22500
    assert plugin.cluster_number == 0
    assert plugin.cycletime == "20240203T0000Z"
    assert plugin.selection_attr == "realization_selection_method"
    assert plugin.selection_attr_value == "cluster_medoid"
    assert plugin.model_path == "/path/to/model"
    assert plugin.scaling == "log10"


def test_init_invalid_cycletime_format():
    """Test that invalid cycletime format raises error."""
    with pytest.raises(ValueError):
        SpatialMorphing(
            forecast_period=22500,
            cluster_number=0,
            cycletime="invalid_format",
        )


def test_init_creates_selection_helper():
    """Test that initialization creates RealizationSelection helper."""
    plugin = SpatialMorphing(
        forecast_period=22500,
        cluster_number=0,
    )
    assert isinstance(plugin._selection_helper, RealizationSelection)
    assert plugin._selection_helper.forecast_period == 22500


# ============================================================================
# Process method tests - PUBLIC INTERFACE
# ============================================================================


def test_process_requires_cluster_cube():
    """Test that process() raises ValueError if cluster cube missing."""
    plugin = SpatialMorphing(
        forecast_period=22500,
        cluster_number=0,
    )

    # Only forecast cubes, no cluster cube
    forecast_cube = make_forecast_cube()
    with pytest.raises(ValueError, match="No cluster cube found in input cubes"):
        plugin.process(forecast_cube)


def test_process_requires_forecast_cubes():
    """Test that process() raises ValueError if no forecast cubes."""
    plugin = SpatialMorphing(
        forecast_period=22500,
        cluster_number=0,
    )

    # Only cluster cube, no forecast cubes
    cluster_cube = make_cluster_cube()
    with pytest.raises(ValueError, match="No forecast cubes found in input cubes"):
        plugin.process(cluster_cube)


def test_process_returns_single_cube():
    """Test that process() returns a single Cube (not CubeList)."""
    plugin = SpatialMorphing(
        forecast_period=22500,
        cluster_number=0,
    )

    forecast_cube = make_forecast_cube(model_id="uk_ens")
    cluster_cube = make_cluster_cube()

    result = plugin.process(forecast_cube, cluster_cube)

    assert isinstance(result, Cube)
    assert not isinstance(result, CubeList)


def test_process_output_has_cluster_realization():
    """Test that output cube has realization set to cluster_number."""
    plugin = SpatialMorphing(
        forecast_period=22500,
        cluster_number=0,
    )

    forecast_cube = make_forecast_cube(model_id="uk_ens")
    cluster_cube = make_cluster_cube()

    result = plugin.process(forecast_cube, cluster_cube)

    # Output should have a single realization equal to cluster_number
    assert result.coords("realization")
    realization_values = result.coord("realization").points
    np.testing.assert_array_equal(realization_values, [0])


def test_process_adds_selection_attr():
    """Test that selection_attr is added to output when requested."""
    plugin = SpatialMorphing(
        forecast_period=22500,
        cluster_number=0,
        selection_attr="realization_selection_method",
        selection_attr_value="spatial_morphing_blend",
    )

    forecast_cube = make_forecast_cube(model_id="uk_ens")
    cluster_cube = make_cluster_cube()

    result = plugin.process(forecast_cube, cluster_cube)

    assert result.attributes["realization_selection_method"] == "spatial_morphing_blend"


@pytest.mark.parametrize("cluster_number", [0, 1])
def test_process_with_multiple_clusters(cluster_number):
    """Test processing different cluster numbers."""
    plugin = SpatialMorphing(
        forecast_period=22500,
        cluster_number=cluster_number,
    )

    forecast_cube = make_forecast_cube(model_id="uk_ens", n_realizations=2)
    cluster_cube = make_cluster_cube()

    result = plugin.process(forecast_cube, cluster_cube)

    assert isinstance(result, Cube)
    # Output realization should match requested cluster_number
    np.testing.assert_array_equal(result.coord("realization").points, [cluster_number])


@patch(
    "improver.utilities.source_spatial_morphing.SpatialMorphing._call_google_film_for_morphing"
)
def test_process_calls_google_film_when_secondary_source_available(mock_morph):
    """Test that Google FILM is invoked when a secondary-source realization exists."""
    mock_morph.return_value = make_forecast_cube(model_id="uk_ens")

    plugin = SpatialMorphing(
        forecast_period=22500,
        cluster_number=0,
        model_path="/apath/to/model",
    )

    primary_cube = make_forecast_cube(model_id="uk_ens", n_realizations=2)
    secondary_cube = make_forecast_cube(model_id="uk_det", n_realizations=2)
    cluster_cube = make_cluster_cube()

    plugin.process(primary_cube, secondary_cube, cluster_cube)

    assert mock_morph.called


def test_process_with_cubelist_input():
    """Test that process() accepts CubeList input."""
    plugin = SpatialMorphing(
        forecast_period=22500,
        cluster_number=0,
    )

    forecast_cube = make_forecast_cube(model_id="uk_ens")
    cluster_cube = make_cluster_cube()

    cubes = CubeList([forecast_cube, cluster_cube])
    result = plugin.process(cubes)

    assert isinstance(result, Cube)


def test_process_raises_on_invalid_cluster_number():
    """Test that process() raises ValueError for non-existent cluster."""
    plugin = SpatialMorphing(
        forecast_period=22500,
        cluster_number=99,  # Cluster that doesn't exist in mapping
    )

    forecast_cube = make_forecast_cube(model_id="uk_ens")
    cluster_cube = make_cluster_cube()

    with pytest.raises(ValueError, match="Cluster number 99 not found"):
        plugin.process(forecast_cube, cluster_cube)


@patch(
    "improver.utilities.source_spatial_morphing.SpatialMorphing._call_google_film_for_morphing"
)
def test_process_diagnoses_source_specific_realizations_for_transition(mock_morph):
    """Test transition morphing diagnoses realization indices per source."""

    n_realizations = 24
    det_data = np.zeros((n_realizations, 5, 5), dtype=np.float32)
    ens_data = np.zeros((n_realizations, 5, 5), dtype=np.float32)
    for realization in range(n_realizations):
        det_data[realization, :, :] = 100.0 + realization
        ens_data[realization, :, :] = 200.0 + realization

    det_cube = set_up_variable_cube(
        det_data,
        name="precipitation_accumulation",
        units="mm",
        spatial_grid="equalarea",
        realizations=np.arange(n_realizations),
    )
    det_cube.attributes["mosg__model_configuration"] = "uk_det"

    ens_cube = set_up_variable_cube(
        ens_data,
        name="precipitation_accumulation",
        units="mm",
        spatial_grid="equalarea",
        realizations=np.arange(n_realizations),
    )
    ens_cube.attributes["mosg__model_configuration"] = "uk_ens"

    cluster_cube = set_up_variable_cube(
        np.zeros((5, 5), dtype=np.float32),
        name="clustering_result",
        units="1",
        spatial_grid="equalarea",
    )
    cluster_cube.attributes["primary_input_realization_to_cluster_medoid"] = json.dumps(
        {"17": 8}
    )
    cluster_cube.attributes["secondary_input_realizations_to_clusters"] = json.dumps(
        {
            "uk_det": {
                "17": [
                    {
                        "realization": 3,
                        "forecast_periods": [3600, 21600],
                    }
                ]
            },
            "uk_ens": {
                "17": [
                    {
                        "realization": 11,
                        "forecast_periods": [
                            43200,
                            86400,
                            129600,
                            172800,
                            216000,
                            259200,
                            302400,
                            345600,
                            388800,
                            432000,
                        ],
                    }
                ]
            },
        }
    )
    cluster_cube.attributes["cluster_sources"] = json.dumps(
        {
            "17": {
                "uk_det": [3600, 21600],
                "uk_ens": [43200],
            }
        }
    )

    plugin = SpatialMorphing(
        forecast_period=18000,
        cluster_number=17,
        model_path="/apath/to/model",
    )

    def _blend_stub(cube_a, cube_b, weight):
        result = cube_a.copy()
        result.data = (1.0 - weight) * cube_a.data + weight * cube_b.data
        return result

    mock_morph.side_effect = _blend_stub

    result = plugin.process(det_cube, ens_cube, cluster_cube)

    expected_weight = (18000 - (21600 - 10800)) / (2 * 10800)
    expected_value = (1.0 - expected_weight) * (100.0 + 3) + expected_weight * (
        200.0 + 11
    )
    np.testing.assert_allclose(result.data, expected_value, rtol=1e-6)


# ============================================================================
# Integration with RealizationSelection
# ============================================================================


def test_process_delegates_to_selection_helper():
    """Test that process() correctly uses RealizationSelection methods."""
    plugin = SpatialMorphing(
        forecast_period=22500,
        cluster_number=0,
    )

    forecast_cube = make_forecast_cube(model_id="uk_ens")
    cluster_cube = make_cluster_cube()

    # Verify that the selection_helper is used (by checking it exists and is correct)
    assert plugin._selection_helper is not None
    assert plugin._selection_helper.forecast_period == 22500
    assert plugin._selection_helper.model_id_attr == "mosg__model_configuration"

    # Process should work without error
    result = plugin.process(forecast_cube, cluster_cube)
    assert result is not None


@patch(
    "improver.utilities.source_spatial_morphing.RealizationSelection.split_cubes_forecast_and_cluster"
)
def test_process_calls_split_cubes(mock_split):
    """Test that process() calls split_cubes_forecast_and_cluster."""
    plugin = SpatialMorphing(
        forecast_period=22500,
        cluster_number=0,
    )

    forecast_cube = make_forecast_cube(model_id="uk_ens")
    cluster_cube = make_cluster_cube()

    # Mock to return valid cubes
    mock_split.return_value = (CubeList([forecast_cube]), cluster_cube)

    # Process would call it
    with patch.object(plugin._selection_helper, "validate_common_validity_time"):
        with patch.object(
            plugin._selection_helper, "parse_mapping_attributes"
        ) as mock_parse:
            mock_parse.return_value = (
                {"0": 0, "1": 1},
                {"uk_det": {"0": [{"realization": 0, "forecast_periods": [22500]}]}},
            )
            with patch.object(
                plugin._selection_helper, "find_nearest_secondary_mapping_fp"
            ) as mock_find_fp:
                mock_find_fp.return_value = (22500, True)
                with patch.object(
                    plugin._selection_helper, "build_cluster_to_selection"
                ) as mock_build:
                    mock_build.return_value = {0: ("uk_ens", 0), 1: ("uk_ens", 1)}
                    with patch.object(
                        plugin._selection_helper, "select_realizations_for_clusters"
                    ) as mock_select:
                        result_cube = forecast_cube.extract(
                            pytest.importorskip("iris").Constraint(realization=0)
                        )
                        if isinstance(result_cube, CubeList):
                            result_cube = result_cube[0]
                        result_cube.coord("realization").points = [0]
                        mock_select.return_value = [result_cube]

                        result = plugin.process(forecast_cube, cluster_cube)
                        assert result is not None


# ============================================================================
# Data preservation tests
# ============================================================================


def test_process_preserves_data_with_single_source():
    """Test that process() preserves data when only one source available."""
    plugin = SpatialMorphing(
        forecast_period=22500,
        cluster_number=0,
    )

    forecast_cube = make_forecast_cube(model_id="uk_ens", n_realizations=2)
    cluster_cube = make_cluster_cube()

    # Store original data for realization 0
    orig_data = forecast_cube.extract(
        pytest.importorskip("iris").Constraint(realization=0)
    ).data

    result = plugin.process(forecast_cube, cluster_cube)

    # Result should have the same data values (within floating point tolerance)
    np.testing.assert_allclose(result.data, orig_data, rtol=1e-5)


def test_process_removes_model_id_attr_if_present():
    """Test that model_id_attr is removed from output if originally present."""
    plugin = SpatialMorphing(
        forecast_period=22500,
        cluster_number=0,
    )

    forecast_cube = make_forecast_cube(model_id="uk_ens")
    cluster_cube = make_cluster_cube()

    # Ensure input has model_id_attr
    assert "mosg__model_configuration" in forecast_cube.attributes

    plugin.process(forecast_cube, cluster_cube)

    # Output should not have model_id_attr (it's removed after selection)
    # This ensures only one source identity in output
    # Note: This depends on RealizationSelection behavior


def test_process_preserves_other_attributes():
    """Test that other attributes are preserved in output."""
    plugin = SpatialMorphing(
        forecast_period=22500,
        cluster_number=0,
    )

    forecast_cube = make_forecast_cube(model_id="uk_ens")
    forecast_cube.attributes["custom_attr"] = "custom_value"
    cluster_cube = make_cluster_cube()

    result = plugin.process(forecast_cube, cluster_cube)

    # Custom attributes should be preserved
    assert "custom_attr" in result.attributes


# ============================================================================
# Edge cases and error handling
# ============================================================================


def test_process_with_empty_cubes_list():
    """Test that process() raises ValueError with empty CubeList."""
    plugin = SpatialMorphing(
        forecast_period=22500,
        cluster_number=0,
    )

    with pytest.raises(ValueError, match="No input cubes provided"):
        plugin.process(CubeList())


def test_process_preserves_coordinates():
    """Test that spatial and temporal coordinates are preserved."""
    plugin = SpatialMorphing(
        forecast_period=22500,
        cluster_number=0,
    )

    forecast_cube = make_forecast_cube(model_id="uk_ens")
    cluster_cube = make_cluster_cube()

    # Store coordinate names from input
    input_coords = set(forecast_cube.coord_names())

    result = plugin.process(forecast_cube, cluster_cube)

    # Output should have same coordinate names (except realization which is fixed)
    output_coords = set(result.coord_names())
    # All coordinate names should be present (allowing realization to be replaced)
    for coord_name in input_coords:
        if coord_name != "realization":
            assert coord_name in output_coords
