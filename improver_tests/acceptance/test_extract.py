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
"""
Tests for the extract CLI
"""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


def test_basic(tmp_path):
    """Test basic extraction"""
    kgo_dir = acc.kgo_root() / "extract/basic"
    kgo_path = kgo_dir / "kgo.nc"
    input_path = kgo_dir / "input.nc"
    output_path = tmp_path / "output.nc"
    args = [input_path, "--constraints", "realization=1", "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_change_units(tmp_path):
    """Test extraction and unit conversion"""
    kgo_dir = acc.kgo_root() / "extract/change_units"
    kgo_path = kgo_dir / "kgo.nc"
    input_path = kgo_dir / "../basic/input.nc"
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        "--constraints",
        "wind_speed=10000",
        "--units",
        "mm s-1",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_multiple_constraints(tmp_path):
    """Test extraction with multiple constraints"""
    kgo_dir = acc.kgo_root() / "extract/multiple_constraints"
    kgo_path = kgo_dir / "kgo.nc"
    input_path = kgo_dir / "../basic/input.nc"
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        "--constraints",
        "wind_speed=20",
        "--constraints",
        "realization=2",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_multiple_constraints_units(tmp_path):
    """Test extraction with multiple constraints and unit conversion"""
    kgo_dir = acc.kgo_root() / "extract/multiple_constraints_units"
    kgo_path = kgo_dir / "kgo.nc"
    input_path = kgo_dir / "../basic/input.nc"
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        "--constraints",
        "wind_speed=20000",
        "--constraints",
        "realization=2",
        "--units",
        "mm s-1,None",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_invalid_constraints(tmp_path):
    """Test extraction with invalid constraints"""
    kgo_dir = acc.kgo_root() / "extract/basic"
    input_path = kgo_dir / "input.nc"
    output_path = tmp_path / "output.nc"
    args = [input_path, "--constraints", "realization=6", "--output", output_path]
    with pytest.raises(ValueError, match=".*Constraint.*"):
        run_cli(args)


def test_list_constraints(tmp_path):
    """Test extraction with a list of constraints"""
    kgo_dir = acc.kgo_root() / "extract/list_constraints"
    kgo_path = kgo_dir / "kgo.nc"
    input_path = kgo_dir / "../basic/input.nc"
    output_path = tmp_path / "output.nc"
    args = [input_path, "--constraints", "wind_speed=[20, 30]", "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_range_constraints(tmp_path):
    """Test extraction with range constraints"""
    kgo_dir = acc.kgo_root() / "extract/range_constraints"
    kgo_path = kgo_dir / "kgo.nc"
    input_path = kgo_dir / "../basic/input.nc"
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        "--constraints",
        "projection_y_coordinate=[-400000:-158000]",
        "--constraints",
        "projection_x_coordinate=[-204000:16000]",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path, recreate=False)


def test_spot(tmp_path):
    """Test extraction from spot file"""
    kgo_dir = acc.kgo_root() / "extract/sites"
    kgo_path = kgo_dir / "kgo_spot.nc"
    input_path = kgo_dir / "input_spot.nc"
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        "--constraints",
        "wmo_id=['700', '845', '996', '3346', '3382']",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_uk_grid(tmp_path):
    """Test subsetting of UK standard gridded data"""
    kgo_dir = acc.kgo_root() / "extract/grids"
    kgo_path = kgo_dir / "kgo_grid_uk.nc"
    input_path = kgo_dir / "input_grid_uk.nc"
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        "--constraints",
        "projection_x_coordinate=[-102000:150000:5]",
        "--constraints",
        "projection_y_coordinate=[-102000:200000:5]",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_lat_lon_grid(tmp_path):
    """Test subsetting of data on a lat-lon grid"""
    kgo_dir = acc.kgo_root() / "extract/grids"
    kgo_path = kgo_dir / "kgo_grid_latlon.nc"
    input_path = kgo_dir / "input_grid_latlon.nc"
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        "--constraints",
        "latitude=[45.0:52:2]",
        "--constraints",
        "longitude=[-2:6:2]",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)
