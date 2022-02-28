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
"""Tests for the generate-metadata-cube CLI."""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)
kgo_dir = acc.kgo_root() / "generate-metadata-cube"
mandatory_attributes_json = kgo_dir / "mandatory_attributes.json"


def test_default(tmp_path):
    """Test default metadata cube generation"""
    kgo_path = kgo_dir / "kgo_default.nc"
    output_path = tmp_path / "output.nc"
    args = [
        mandatory_attributes_json,
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_ensemble_members(tmp_path):
    """Test creating variable cube with all options set"""
    kgo_path = kgo_dir / "kgo_ensemble_members_all_options.nc"
    output_path = tmp_path / "output.nc"
    args = [
        mandatory_attributes_json,
        "--name",
        "air_temperature",
        "--units",
        "degrees",
        "--spatial-grid",
        "equalarea",
        "--time-period",
        "120",
        "--ensemble-members",
        "4",
        "--grid-spacing",
        "1000",
        "--domain-corner",
        "0,0",
        "--npoints",
        "50",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_json_all_inputs(tmp_path):
    """Test creating variable cube with all options set"""
    kgo_path = kgo_dir / "kgo_variable_cube_json_inputs.nc"
    json_input_path = kgo_dir / "variable_cube_all_inputs.json"
    output_path = tmp_path / "output.nc"
    args = [
        mandatory_attributes_json,
        "--json-input",
        json_input_path,
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_realization_json(tmp_path):
    """Test variable/realization metadata cube generated using realization
    coordinate defined in the json input"""
    kgo_path = kgo_dir / "kgo_realization.nc"
    realizations_path = kgo_dir / "realizations.json"
    output_path = tmp_path / "output.nc"
    args = [
        mandatory_attributes_json,
        "--json-input",
        realizations_path,
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_percentile_cube(tmp_path):
    """Test percentile metadata cube generated using using percentile
    coordinate defined in the json input"""
    kgo_path = kgo_dir / "kgo_percentile.nc"
    percentiles_path = kgo_dir / "percentiles.json"
    output_path = tmp_path / "output.nc"
    args = [
        mandatory_attributes_json,
        "--json-input",
        percentiles_path,
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_probability_cube(tmp_path):
    """Test probability metadata cube generated using using threshold
    coordinate defined in the json input"""
    kgo_path = kgo_dir / "kgo_probability.nc"
    thresholds_path = kgo_dir / "thresholds.json"
    output_path = tmp_path / "output.nc"
    args = [
        mandatory_attributes_json,
        "--json-input",
        thresholds_path,
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_height_levels(tmp_path):
    """Test metadata cube generated with height levels from json"""
    kgo_path = kgo_dir / "kgo_height_levels.nc"
    height_levels_path = kgo_dir / "height_levels.json"
    output_path = tmp_path / "output.nc"
    args = [
        mandatory_attributes_json,
        "--json-input",
        height_levels_path,
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_single_height_level(tmp_path):
    """Test metadata cube generation giving single value (rather than comma separated
    list) for height levels option demotes height to scalar coordinate"""
    kgo_path = kgo_dir / "kgo_single_height_level.nc"
    height_level_path = kgo_dir / "single_height_level.json"
    output_path = tmp_path / "output.nc"
    args = [
        mandatory_attributes_json,
        "--json-input",
        height_level_path,
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_pressure_levels(tmp_path):
    """Test metadata cube generated with pressure in Pa instead of height in metres"""
    kgo_path = kgo_dir / "kgo_pressure_levels.nc"
    pressure_levels_path = kgo_dir / "pressure_levels.json"
    output_path = tmp_path / "output.nc"
    args = [
        mandatory_attributes_json,
        "--json-input",
        pressure_levels_path,
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)
