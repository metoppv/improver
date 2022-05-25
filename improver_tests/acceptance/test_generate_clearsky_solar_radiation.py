# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown copyright. The Met Office.
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
"""Tests for the generate-clearsky-solar-radiation CLI."""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


def test_basic(tmp_path):
    """Test basic generation of clearsky solar radiation derived field."""
    kgo_dir = acc.kgo_root() / "generate-clearsky-solar-radiation"
    kgo_path = kgo_dir / "basic" / "kgo.nc"
    input_path = kgo_dir / "surface_altitude.nc"  # Use this as target_grid
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        "--time",
        "20220506T0000Z",
        "--accumulation-period",
        "24",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_with_altitude_and_lt(tmp_path):
    """Test generation of clearsky solar radiation derived field with input
    surface_altitude and linke_turbidity."""
    kgo_dir = acc.kgo_root() / "generate-clearsky-solar-radiation"
    kgo_path = kgo_dir / "with_altitude_and_lt" / "kgo.nc"
    input_path = kgo_dir / "surface_altitude.nc"  # Use this as target_grid
    surface_altitude_path = kgo_dir / "surface_altitude.nc"
    linke_turbidity_path = kgo_dir / "linke_turbidity.nc"
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        surface_altitude_path,
        linke_turbidity_path,
        "--time",
        "20220506T0000Z",
        "--accumulation-period",
        "24",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_temporal_spacing(tmp_path):
    """Test generation of clearsky solar radiation derived field with input
    temporal-spacing."""
    kgo_dir = acc.kgo_root() / "generate-clearsky-solar-radiation"
    kgo_path = kgo_dir / "basic" / "kgo.nc"
    input_path = kgo_dir / "surface_altitude.nc"  # Use this as target_grid
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        "--time",
        "20220506T0000Z",
        "--accumulation-period",
        "24",
        "--temporal-spacing",
        "60",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path, rtol=0.005)
