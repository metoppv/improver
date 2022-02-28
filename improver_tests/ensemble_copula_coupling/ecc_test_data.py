# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2022 Met Office.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
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
