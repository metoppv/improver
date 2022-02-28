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
"""Utilities for use by precipitation_type plugins / functions."""

from iris.cube import Cube
from iris.exceptions import CoordinateNotFoundError

from improver.metadata.constants import FLOAT_DTYPE
from improver.metadata.probabilistic import find_threshold_coordinate


def make_shower_condition_cube(cube: Cube, in_place: bool = False) -> Cube:
    """
    Modify the input cube's metadata and coordinates to produce a shower
    condition proxy. The input cube is expected to possess a single valued
    threshold coordinate.

    Args:
        cube:
            A thresholded diagnostic to be used as a proxy for showery conditions.
            The threshold coordinate should contain only one value, which denotes
            the key threshold that above which conditions are showery, and below
            which precipitation is more likely dynamic.
        in_place:
            If set true the cube is modified in place. By default a modified
            copy is returned.

    Returns:
        A shower condition probability cube that is an appropriately renamed
        version of the input with an updated threshold coordinate representing
        the probability of shower conditions occurring.

    Raises:
        CoordinateNotFoundError: Input has no threshold coordinate.
        ValueError: Input cube's threshold coordinate is multi-valued.
    """

    if not in_place:
        cube = cube.copy()

    shower_condition_name = "shower_condition"
    cube.rename(f"probability_of_{shower_condition_name}_above_threshold")
    try:
        shower_threshold = find_threshold_coordinate(cube)
    except CoordinateNotFoundError as err:
        msg = "Input has no threshold coordinate and cannot be used"
        raise CoordinateNotFoundError(msg) from err

    try:
        (_,) = shower_threshold.points
    except ValueError as err:
        msg = (
            "Expected a single valued threshold coordinate, but threshold "
            f"contains multiple points : {shower_threshold.points}"
        )
        raise ValueError(msg) from err

    cube.coord(shower_threshold).rename(shower_condition_name)
    cube.coord(shower_condition_name).var_name = "threshold"
    cube.coord(shower_condition_name).points = FLOAT_DTYPE(1.0)
    cube.coord(shower_condition_name).units = 1

    return cube
