# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Tests for the SpatialMorphing plugin."""

import json
from unittest.mock import patch

import numpy as np
import pytest
from iris.cube import Cube, CubeList

from improver.clustering.realization_clustering import RealizationSelection
from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube
from improver.utilities.spatial_morphing import SpatialMorphing


def make_forecast_cube(model_id="uk_ens", n_realizations=2, base_value=0.0):
    """Create a forecast cube with a realization coordinate."""
    data = np.zeros((n_realizations, 5, 5), dtype=np.float32)
    for realization in range(n_realizations):
        data[realization, :, :] = base_value + realization

    cube = set_up_variable_cube(
        data,
        name="precipitation_accumulation",
        units="mm",
        spatial_grid="equalarea",
        realizations=np.arange(n_realizations),
    )
    cube.attributes["mosg__model_configuration"] = model_id
    return cube


def make_cluster_cube(include_uk_ens_secondary=False):
    """Create a mock cluster cube with mapping attributes."""
    cube = set_up_variable_cube(
        np.zeros((5, 5), dtype=np.float32),
        name="clustering_result",
        units="1",
        spatial_grid="equalarea",
    )

    cube.attributes["primary_input_realization_to_cluster_medoid"] = json.dumps(
        {"0": 0, "1": 1, "17": 8}
    )

    secondary_map = {
        "uk_ens": {
            "0": [{"realization": 0, "forecast_periods": [22500]}],
            "1": [{"realization": 1, "forecast_periods": [22500]}],
            "17": [{"realization": 3, "forecast_periods": [3600, 21600]}],
        }
    }
    if include_uk_ens_secondary:
        secondary_map["uk_ens"] = {
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
        }
    cube.attributes["secondary_input_realizations_to_clusters"] = json.dumps(
        secondary_map
    )

    cube.attributes["cluster_sources"] = json.dumps(
        {
            "0": {"uk_ens": [22500]},
            "1": {"uk_ens": [22500]},
            "17": {"uk_ens": [43200], "uk_det": [3600, 21600]},
        }
    )
    return cube


def make_transitions():
    """Create a representative explicit transition definition."""
    return {
        "transitions": [
            {
                "source_a": "uk_det",
                "source_b": "uk_ens",
                "start_forecast_period_minutes": 300,
                "end_forecast_period_minutes": 420,
            }
        ]
    }


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
    assert plugin.transitions == []


def test_init_with_optional_parameters():
    """Test initialization with optional parameters."""
    plugin = SpatialMorphing(
        forecast_period=22500,
        cluster_number=0,
        cycletime="20240203T0000Z",
        selection_attr="realization_selection_method",
        selection_attr_value="cluster_medoid",
        transitions=make_transitions(),
        model_path="/path/to/model",
        scaling="log10",
    )
    assert plugin.forecast_period == 22500
    assert plugin.cluster_number == 0
    assert plugin.cycletime == "20240203T0000Z"
    assert plugin.selection_attr == "realization_selection_method"
    assert plugin.selection_attr_value == "cluster_medoid"
    assert plugin.transitions == [
        {
            "source_a": "uk_det",
            "source_b": "uk_ens",
            "start_forecast_period_seconds": 18000,
            "end_forecast_period_seconds": 25200,
        }
    ]
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


@pytest.mark.parametrize(
    "transitions,error_match",
    [
        ({"bad": []}, "transitions dictionary"),
        ({"transitions": [{}]}, "missing required keys"),
        (
            {
                "transitions": [
                    {
                        "source_a": "uk_det",
                        "source_b": "uk_ens",
                        "start_forecast_period_minutes": 420,
                        "end_forecast_period_minutes": 300,
                    }
                ]
            },
            "start < end",
        ),
    ],
)
def test_init_invalid_transitions(transitions, error_match):
    """Test that invalid transition definitions raise errors."""
    with pytest.raises(ValueError, match=error_match):
        SpatialMorphing(
            forecast_period=22500,
            cluster_number=0,
            transitions=transitions,
        )


def test_init_creates_selection_helper():
    """Test that initialization creates RealizationSelection helper."""
    plugin = SpatialMorphing(
        forecast_period=22500,
        cluster_number=0,
    )
    assert isinstance(plugin._selection_helper, RealizationSelection)
    assert plugin._selection_helper.forecast_period == 22500


def test_find_active_transition_returns_matching_entry():
    """Test active transition lookup uses explicit bounds."""
    plugin = SpatialMorphing(
        forecast_period=22500,
        cluster_number=0,
        transitions=make_transitions(),
    )
    assert plugin._find_active_transition(22500) == {
        "source_a": "uk_det",
        "source_b": "uk_ens",
        "start_forecast_period_seconds": 18000,
        "end_forecast_period_seconds": 25200,
    }


@pytest.mark.parametrize(
    "forecast_period,expected_weight",
    [
        (18000, 0.0),
        (21600, 0.5),
        (25200, 1.0),
        (30000, 1.0),
    ],
)
def test_calculate_transition_weight(forecast_period, expected_weight):
    """Test weight calculation from explicit start/end transition bounds."""
    weight = SpatialMorphing._calculate_transition_weight(forecast_period, 18000, 25200)
    assert np.isclose(weight, expected_weight)


# ============================================================================
# Process method tests - PUBLIC INTERFACE
# ============================================================================


def test_process_requires_cluster_cube():
    """Test that process() raises ValueError if cluster cube missing."""
    plugin = SpatialMorphing(forecast_period=22500, cluster_number=0)
    forecast_cube = make_forecast_cube()
    with pytest.raises(ValueError, match="No cluster cube found in input cubes"):
        plugin.process(forecast_cube)


def test_process_requires_forecast_cubes():
    """Test that process() raises ValueError if no forecast cubes."""
    plugin = SpatialMorphing(forecast_period=22500, cluster_number=0)
    cluster_cube = make_cluster_cube()
    with pytest.raises(ValueError, match="No forecast cubes found in input cubes"):
        plugin.process(cluster_cube)


def test_process_returns_single_cube():
    """Test that process() returns a single Cube (not CubeList)."""
    plugin = SpatialMorphing(forecast_period=22500, cluster_number=0)
    result = plugin.process(make_forecast_cube(model_id="uk_ens"), make_cluster_cube())
    assert isinstance(result, Cube)
    assert not isinstance(result, CubeList)


def test_process_output_has_cluster_realization():
    """Test that output cube has realization set to cluster_number."""
    plugin = SpatialMorphing(forecast_period=22500, cluster_number=0)
    result = plugin.process(make_forecast_cube(model_id="uk_ens"), make_cluster_cube())
    assert result.coords("realization")
    np.testing.assert_array_equal(result.coord("realization").points, [0])


def test_process_adds_selection_attr():
    """Test that selection_attr is added to output when requested."""
    plugin = SpatialMorphing(
        forecast_period=22500,
        cluster_number=0,
        selection_attr="realization_selection_method",
        selection_attr_value="spatial_morphing_blend",
    )
    result = plugin.process(make_forecast_cube(model_id="uk_ens"), make_cluster_cube())
    assert result.attributes["realization_selection_method"] == "spatial_morphing_blend"


@pytest.mark.parametrize("cluster_number", [0, 1])
def test_process_with_multiple_clusters(cluster_number):
    """Test processing different cluster numbers."""
    plugin = SpatialMorphing(forecast_period=22500, cluster_number=cluster_number)
    result = plugin.process(
        make_forecast_cube(model_id="uk_ens", n_realizations=2), make_cluster_cube()
    )
    assert isinstance(result, Cube)
    np.testing.assert_array_equal(result.coord("realization").points, [cluster_number])


def test_process_with_cubelist_input():
    """Test that process() accepts CubeList input."""
    plugin = SpatialMorphing(forecast_period=22500, cluster_number=0)
    cubes = CubeList([make_forecast_cube(model_id="uk_ens"), make_cluster_cube()])
    result = plugin.process(cubes)
    assert isinstance(result, Cube)


def test_process_raises_on_invalid_cluster_number():
    """Test that process() raises ValueError for non-existent cluster."""
    plugin = SpatialMorphing(forecast_period=22500, cluster_number=99)
    with pytest.raises(ValueError, match="Cluster number 99 not found"):
        plugin.process(make_forecast_cube(model_id="uk_ens"), make_cluster_cube())


def test_process_preserves_data_with_single_source():
    """Test that process() preserves data when only one source available."""
    plugin = SpatialMorphing(forecast_period=22500, cluster_number=0)
    forecast_cube = make_forecast_cube(model_id="uk_ens", n_realizations=2)
    orig_data = forecast_cube.extract(
        pytest.importorskip("iris").Constraint(realization=0)
    ).data
    result = plugin.process(forecast_cube, make_cluster_cube())
    np.testing.assert_allclose(result.data, orig_data, rtol=1e-5)


def test_process_removes_model_id_attr_if_present():
    """Test that model_id_attr is removed from output if originally present."""
    plugin = SpatialMorphing(forecast_period=22500, cluster_number=0)
    forecast_cube = make_forecast_cube(model_id="uk_ens")
    assert "mosg__model_configuration" in forecast_cube.attributes
    result = plugin.process(forecast_cube, make_cluster_cube())
    assert "mosg__model_configuration" not in result.attributes


def test_process_preserves_other_attributes():
    """Test that other attributes are preserved in output."""
    plugin = SpatialMorphing(forecast_period=22500, cluster_number=0)
    forecast_cube = make_forecast_cube(model_id="uk_ens")
    forecast_cube.attributes["custom_attr"] = "custom_value"
    result = plugin.process(forecast_cube, make_cluster_cube())
    assert "custom_attr" in result.attributes


def test_process_with_empty_cubes_list():
    """Test that process() raises ValueError with empty CubeList."""
    plugin = SpatialMorphing(forecast_period=22500, cluster_number=0)
    with pytest.raises(ValueError, match="No cluster cube found in input cubes"):
        plugin.process(CubeList())


def test_process_preserves_coordinates():
    """Test that spatial and temporal coordinates are preserved."""
    plugin = SpatialMorphing(forecast_period=22500, cluster_number=0)
    forecast_cube = make_forecast_cube(model_id="uk_ens")
    result = plugin.process(forecast_cube, make_cluster_cube())
    input_coords = {coord.name() for coord in forecast_cube.coords()}
    output_coords = {coord.name() for coord in result.coords()}
    for coord_name in input_coords:
        if coord_name != "realization":
            assert coord_name in output_coords


# ============================================================================
# Explicit transition morphing
# ============================================================================


@patch(
    "improver.utilities.spatial_morphing.SpatialMorphing._call_google_film_for_morphing"
)
def test_process_diagnoses_source_specific_realizations_for_transition(mock_morph):
    """Test transition morphing diagnoses realization indices per source."""
    det_cube = make_forecast_cube(
        model_id="uk_det", n_realizations=24, base_value=100.0
    )
    ens_cube = make_forecast_cube(
        model_id="uk_ens", n_realizations=24, base_value=200.0
    )

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
            "uk_det": {"17": [{"realization": 3, "forecast_periods": [3600, 21600]}]},
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
        {"17": {"uk_det": [3600, 21600], "uk_ens": [43200]}}
    )

    plugin = SpatialMorphing(
        forecast_period=21600,
        cluster_number=17,
        transitions=make_transitions(),
        model_path="/apath/to/model",
    )

    def _blend_stub(cube_a, cube_b, weight):
        result = cube_a.copy()
        result.data = (1.0 - weight) * cube_a.data + weight * cube_b.data
        return result

    mock_morph.side_effect = _blend_stub

    result = plugin.process(det_cube, ens_cube, cluster_cube)

    expected_weight = 0.5
    expected_value = (1.0 - expected_weight) * (100.0 + 3) + expected_weight * (
        200.0 + 11
    )
    np.testing.assert_allclose(result.data, expected_value, rtol=1e-6)
