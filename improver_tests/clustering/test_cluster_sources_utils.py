# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the standalone cluster_sources utility functions."""

import numpy as np
import pytest
from iris.cube import Cube

from improver.clustering.cluster_sources_utils import (
    find_nearest_forecast_period_gte,
    get_source_for_forecast_period,
    parse_cluster_sources_attribute,
)


@pytest.fixture
def cluster_sources():
    """Return a simple mapping of realization indices to model sources."""
    return {
        "0": {
            "uk_ens": [3600, 21600, 43200, 86400, 129600, 172800],
            "gl_ens": [432000, 475200, 518400],
        },
        "1": {"ecgl_ens": [3600, 43200, 86400]},
    }


@pytest.mark.parametrize(
    "attr_value, expected",
    [
        (None, {}),
        ({"0": {"uk_ens": [3600]}}, {"0": {"uk_ens": [3600]}}),
        (
            '{"0": {"uk_ens": [3600, 43200], "gl_ens": [86400]} }',
            {"0": {"uk_ens": [3600, 43200], "gl_ens": [86400]}},
        ),
    ],
)
def test_parse_cluster_sources_attribute(attr_value, expected):
    """Test parsing cube attributes into a cluster source dictionary."""
    cube = Cube(np.array([1.0]))
    if attr_value is not None:
        cube.attributes["cluster_sources"] = attr_value

    assert parse_cluster_sources_attribute(cube) == expected


@pytest.mark.parametrize(
    "invalid_value",
    [
        "{not valid json}",
        ["not", "a", "dict"],
        123,
    ],
)
def test_parse_cluster_sources_attribute_raises_for_invalid_values(invalid_value):
    """Malformed cluster source attributes should raise ValueError."""
    cube = Cube(np.array([1.0]))
    cube.attributes["cluster_sources"] = invalid_value

    with pytest.raises(ValueError):
        parse_cluster_sources_attribute(cube)


@pytest.mark.parametrize(
    "forecast_periods,target_fp,expected",
    [
        (None, 1000, (1000, False)),
        ({3600, 7200}, 3600, (3600, True)),
        ({3600, 7200}, 5000, (7200, True)),
        ({3600, 7200}, 100000, (100000, False)),
    ],
)
def test_find_nearest_forecast_period_gte(forecast_periods, target_fp, expected):
    """Test selection of the nearest forecast period >= target."""
    assert find_nearest_forecast_period_gte(forecast_periods, target_fp) == expected


@pytest.mark.parametrize(
    "realization_idx,fp_seconds,expected_source",
    [
        (0, 3600, "uk_ens"),
        (0, 432000, "gl_ens"),
        (0, 450000, "gl_ens"),
        (1, 86400, "ecgl_ens"),
        (99, 3600, None),
    ],
)
def test_get_source_for_forecast_period(
    cluster_sources, realization_idx, fp_seconds, expected_source
):
    """Test source selection for exact and transition periods."""
    assert (
        get_source_for_forecast_period(cluster_sources, realization_idx, fp_seconds)
        == expected_source
    )
