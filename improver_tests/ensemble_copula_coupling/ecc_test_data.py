# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2020 Met Office.
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

ECC_SPOT_TEMPERATURES = np.array(
    [
        [226.15, 237.4, 248.65, 259.9, 271.15, 282.4, 293.65, 304.9, 316.15],
        [230.15, 241.4, 252.65, 263.9, 275.15, 286.4, 297.65, 308.9, 320.15],
        [232.15, 243.4, 254.65, 265.9, 277.15, 288.4, 299.65, 310.9, 322.15],
    ],
    dtype=np.float32,
)


def set_up_spot_test_cube():
    """Use spotdata code to build a test cube with the expected spot metadata,
    with dummy values for the coordinates which are not used in ECC tests"""
    dummy_point_locations = np.arange(9).astype(np.float32)
    dummy_string_ids = [f"{i}" for i in range(9)]
    realization_coord = DimCoord(
        np.arange(3).astype(np.int32), units="1", standard_name="realization"
    )
    return build_spotdata_cube(
        ECC_SPOT_TEMPERATURES,
        name="screen_temperature",
        units="Kelvin",
        altitude=dummy_point_locations,
        latitude=dummy_point_locations,
        longitude=dummy_point_locations,
        wmo_id=dummy_string_ids,
        additional_dims=[realization_coord],
    )
