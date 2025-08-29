# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.

"""Unit tests for spot data utilities"""

import numpy as np
import pytest

from improver.spotdata.build_spotdata_cube import build_spotdata_cube
from improver.spotdata.utilities import (
    extract_site_json,
    get_neighbour_finding_method_name,
)


@pytest.mark.parametrize(
    "land_constraint, minimum_dz, expected",
    (
        (False, False, "nearest"),
        (True, False, "nearest_land"),
        (True, True, "nearest_land_minimum_dz"),
        (False, True, "nearest_minimum_dz"),
    ),
)
def test_get_neighbour_finding_method_name(land_constraint, minimum_dz, expected):
    """Test the function for generating the name that describes the neighbour
    finding method."""

    result = get_neighbour_finding_method_name(land_constraint, minimum_dz)
    assert result == expected


@pytest.mark.parametrize(
    "wmo_ids",
    [
        np.arange(4),
        np.array(["00000", "00001", "00002", "00003"]),
        np.array(["00000", "00001", "00002", "None"]),
    ],
)
def test_extract_site_json(wmo_ids):
    """Test the function for extracting site JSON from the neighbour cube."""

    # Neighbour cube copied from spot_extraction tests
    neighbours = np.array(
        [
            [[0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 2.0, 2.0], [0.0, -1.0, 0.0, 1.0]],
            [[1.0, 1.0, 2.0, 2.0], [1.0, 1.0, 2.0, 2.0], [-1.0, 0.0, 0.0, 1.0]],
        ]
    )
    altitudes = np.array([0, 1, 3, 2])
    latitudes = np.array([10, 10, 20, 20])
    longitudes = np.array([10, 10, 20, 20])
    unique_site_id = wmo_ids
    unique_site_id_key = "met_office_site_id"
    grid_attributes = ["x_index", "y_index", "vertical_displacement"]
    neighbour_methods = ["nearest", "nearest_land"]
    neighbour_cube = build_spotdata_cube(
        neighbours,
        "grid_neighbours",
        1,
        altitudes,
        latitudes,
        longitudes,
        wmo_ids,
        unique_site_id=unique_site_id,
        unique_site_id_key=unique_site_id_key,
        grid_attributes=grid_attributes,
        neighbour_methods=neighbour_methods,
    )

    result = extract_site_json(neighbour_cube)

    expected_wmo = []
    for id in wmo_ids:
        expected_wmo.append(int(id) if id != "None" else None)

    assert isinstance(result, list)
    assert all(isinstance(site, dict) for site in result)
    assert [item["altitude"] for item in result] == altitudes.tolist()
    assert [item["latitude"] for item in result] == latitudes.tolist()
    assert [item["longitude"] for item in result] == longitudes.tolist()
    assert [item["wmo_id"] for item in result] == expected_wmo
    assert [item[unique_site_id_key] for item in result] == expected_wmo
