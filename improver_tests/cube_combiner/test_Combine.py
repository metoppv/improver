# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2021 Met Office.
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
"""Tests for the cube_combiner.Combine plugin."""

import pytest

from improver.cube_combiner import Combine, CubeCombiner, CubeMultiplier
from improver_tests.utilities.test_FilterRealizations import realization_cubes_fixture


@pytest.mark.parametrize("broadcast_to_threshold", (False, True))
@pytest.mark.parametrize("minimum_realizations", (None, 1))
@pytest.mark.parametrize(
    "operation, expected_instance", (("+", CubeCombiner), ("*", CubeMultiplier)),
)
def test_init(
    operation, expected_instance, minimum_realizations, broadcast_to_threshold
):
    """Ensure the class initialises as expected"""
    result = Combine(
        operation,
        new_name="name",
        minimum_realizations=minimum_realizations,
        broadcast_to_threshold=broadcast_to_threshold,
    )
    assert isinstance(result.plugin, expected_instance)
    assert result.new_name == "name"
    assert result.minimum_realizations == minimum_realizations
    assert result.broadcast_to_threshold == broadcast_to_threshold
    if broadcast_to_threshold and isinstance(result.plugin, CubeMultiplier):
        assert result.plugin.broadcast_to_threshold == broadcast_to_threshold


@pytest.mark.parametrize(
    "minimum_realizations, msg",
    (
        (0, "Minimum realizations must be at least 1, not 0"),
        (-1, "Minimum realizations must be at least 1, not -1"),
        (5, "After filtering, number of realizations 4 is less than 5"),
    ),
)
def test_minimum_realizations_exceptions(minimum_realizations, msg, realization_cubes):
    """Ensure specifying too few realizations will raise an error"""
    with pytest.raises(ValueError, match=msg):
        Combine("+", minimum_realizations=minimum_realizations)(realization_cubes)
