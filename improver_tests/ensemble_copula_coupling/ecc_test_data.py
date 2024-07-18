# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Module defining input data for ECC tests"""

import numpy as np
from iris.coords import DimCoord

from improver.spotdata.build_spotdata_cube import build_spotdata_cube

ECC_TEMPERATURE_REALIZATIONS = np.array(
    [
        [[226.15, 237.4, 248.65], [259.9, 271.15, 282.4], [293.65, 304.9, 316.15]],
        [[230.15, 241.4, 252.65], [263.9, 275.15, 286.4], [297.65, 308.9, 320.15]],
        [[232.15, 243.4, 254.65], [265.9, 277.15, 288.4], [299.65, 310.9, 322.15]],
    ],
    dtype=np.float32,
)


ECC_TEMPERATURE_PROBABILITIES = np.array(
    [
        [[1.0, 0.9, 1.0], [0.8, 0.9, 0.5], [0.5, 0.2, 0.0]],
        [[1.0, 0.5, 1.0], [0.5, 0.5, 0.3], [0.2, 0.0, 0.0]],
        [[1.0, 0.2, 0.5], [0.2, 0.0, 0.1], [0.0, 0.0, 0.0]],
    ],
    dtype=np.float32,
)


ECC_TEMPERATURE_THRESHOLDS = np.array([8, 10, 12], dtype=np.float32)


ECC_SPOT_TEMPERATURES = np.array(
    [
        [226.15, 237.4, 248.65, 259.9, 271.15, 282.4, 293.65, 304.9, 316.15],
        [230.15, 241.4, 252.65, 263.9, 275.15, 286.4, 297.65, 308.9, 320.15],
        [232.15, 243.4, 254.65, 265.9, 277.15, 288.4, 299.65, 310.9, 322.15],
    ],
    dtype=np.float32,
)


ECC_SPOT_PROBABILITIES = np.array(
    [
        [1.0, 0.9, 1.0, 0.8, 0.9, 0.5, 0.5, 0.2, 0.0],
        [1.0, 0.5, 1.0, 0.5, 0.5, 0.3, 0.2, 0.0, 0.0],
        [1.0, 0.2, 0.5, 0.2, 0.0, 0.1, 0.0, 0.0, 0.0],
    ],
    dtype=np.float32,
)


def set_up_spot_test_cube(cube_type="realization"):
    """Use spotdata code to build a test cube with the expected spot metadata,
    with dummy values for the coordinates which are not used in ECC tests

    Args:
        cube_type (str):
            Options "probability", "realization" or "percentile"

    Returns:
        iris.cube.Cube:
            Spotdata cube conforming to expected IMPROVER structure
    """
    dummy_point_locations = np.arange(9).astype(np.float32)
    dummy_string_ids = [f"{i}" for i in range(9)]

    if cube_type == "probability":
        return _build_spot_probability_cube(dummy_point_locations, dummy_string_ids)

    if cube_type == "realization":
        leading_coord = DimCoord(
            np.arange(3).astype(np.int32), standard_name="realization", units="1"
        )
    elif cube_type == "percentile":
        leading_coord = DimCoord(
            np.array([10, 50, 90], dtype=np.float32), long_name="percentile", units="%"
        )

    return build_spotdata_cube(
        ECC_SPOT_TEMPERATURES,
        name="air_temperature",
        units="Kelvin",
        altitude=dummy_point_locations,
        latitude=dummy_point_locations,
        longitude=dummy_point_locations,
        wmo_id=dummy_string_ids,
        additional_dims=[leading_coord],
    )


def _build_spot_probability_cube(dummy_point_locations, dummy_string_ids):
    """Set up a spot cube with an IMPROVER-style threshold coordinate and
    suitable data"""
    threshold_coord = DimCoord(
        ECC_TEMPERATURE_THRESHOLDS,
        standard_name="air_temperature",
        var_name="threshold",
        units="degC",
        attributes={"spp__relative_to_threshold": "above"},
    )
    return build_spotdata_cube(
        ECC_SPOT_PROBABILITIES,
        name="probability_of_air_temperature_above_threshold",
        units="1",
        altitude=dummy_point_locations,
        latitude=dummy_point_locations,
        longitude=dummy_point_locations,
        wmo_id=dummy_string_ids,
        additional_dims=[threshold_coord],
    )
