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
"""Tests for the generate-metadata-cube CLI."""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


def test_default(tmp_path):
    """Test default metadata cube generation"""
    kgo_dir = acc.kgo_root() / "generate-metadata-cube"
    kgo_path = kgo_dir / "kgo_default.nc"
    output_path = tmp_path / "output.nc"
    args = [
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_variable_cube(tmp_path):
    """Test metadata cube generation creating a variable/realization cube setting
    values for all options except pressure and dimension json inputs"""
    kgo_dir = acc.kgo_root() / "generate-metadata-cube"
    kgo_path = kgo_dir / "kgo_variable.nc"
    attributes_path = kgo_dir / "attributes.json"
    output_path = tmp_path / "output.nc"
    args = [
        "--name",
        "air_temperature",
        "--units",
        "degrees",
        "--spatial-grid",
        "equalarea",
        "--time",
        "20200102T0400Z",
        "--time-period",
        "120",
        "--frt",
        "20200101T0400Z",
        "--ensemble-members",
        "4",
        "--leading-dimension",
        "1,4,8,12",
        "--attributes",
        attributes_path,
        "--resolution",
        "5000",
        "--domain-corner",
        "0,0",
        "--npoints",
        "50",
        "--height-levels",
        "1.5,3.0,4.5,6.0",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_realization_json(tmp_path):
    """Test variable/realization metadata cube generated using leading dimension json input"""
    kgo_dir = acc.kgo_root() / "generate-metadata-cube"
    kgo_path = kgo_dir / "kgo_realization_json.nc"
    realizations_path = kgo_dir / "realizations.json"
    output_path = tmp_path / "output.nc"
    args = ["--leading-dimension-json", realizations_path, "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_percentile_cube(tmp_path):
    """Test percentile metadata cube generated"""
    kgo_dir = acc.kgo_root() / "generate-metadata-cube"
    kgo_path = kgo_dir / "kgo_percentile.nc"
    output_path = tmp_path / "output.nc"
    args = [
        "--leading-dimension",
        "30, 50, 80",
        "--percentile",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_percentile_cube_json(tmp_path):
    """Test percentile metadata cube generated using leading dimension json input"""
    kgo_dir = acc.kgo_root() / "generate-metadata-cube"
    kgo_path = kgo_dir / "kgo_percentile_json.nc"
    percentiles_path = kgo_dir / "percentiles.json"
    output_path = tmp_path / "output.nc"
    args = [
        "--leading-dimension-json",
        percentiles_path,
        "--percentile",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_percentile_cube_realization_json(tmp_path):
    """Test error raised if json provided without percentiles key"""
    kgo_dir = acc.kgo_root() / "generate-metadata-cube"
    realizations_path = kgo_dir / "realizations.json"
    output_path = tmp_path / "output.nc"
    args = [
        "--leading-dimension-json",
        realizations_path,
        "--percentile",
        "--output",
        output_path,
    ]
    with pytest.raises(KeyError, match="'percentiles'"):
        run_cli(args)


def test_probability_cube(tmp_path):
    """Test probability metadata cube generated"""
    kgo_dir = acc.kgo_root() / "generate-metadata-cube"
    kgo_path = kgo_dir / "kgo_probability.nc"
    output_path = tmp_path / "output.nc"
    args = [
        "--leading-dimension",
        "275.0,275.5,276.0",
        "--probability",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_probability_cube_json(tmp_path):
    """Test probability metadata cube generated using leading dimension json input"""
    kgo_dir = acc.kgo_root() / "generate-metadata-cube"
    kgo_path = kgo_dir / "kgo_probability_json.nc"
    thresholds_path = kgo_dir / "thresholds.json"
    output_path = tmp_path / "output.nc"
    args = [
        "--leading-dimension-json",
        thresholds_path,
        "--probability",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_both_leading_dimension_json(tmp_path):
    """Test metadata cube generated with leading_dimension input rather than
    leading_dimension_json input if both provided"""
    kgo_dir = acc.kgo_root() / "generate-metadata-cube"
    kgo_path = kgo_dir / "kgo_leading_dimension.nc"
    realizations_path = kgo_dir / "realizations.json"
    output_path = tmp_path / "output.nc"
    args = [
        "--leading-dimension",
        "1, 5, 8",
        "--leading-dimension-json",
        realizations_path,
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_single_height_level(tmp_path):
    """Test metadata cube generation giving single value (rather than comma separated
    list) for height levels option demotes height to scalar coordinate"""
    kgo_dir = acc.kgo_root() / "generate-metadata-cube"
    kgo_path = kgo_dir / "kgo_single_height_level.nc"
    output_path = tmp_path / "output.nc"
    args = ["--height-levels", "1.5", "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_height_levels_json(tmp_path):
    """Test metadata cube generated with height levels from json"""
    kgo_dir = acc.kgo_root() / "generate-metadata-cube"
    kgo_path = kgo_dir / "kgo_height_levels_json.nc"
    height_levels_path = kgo_dir / "height_levels.json"
    output_path = tmp_path / "output.nc"
    args = ["--height-levels-json", height_levels_path, "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_pressure_levels(tmp_path):
    """Test metadata cube generated with pressure in Pa instead of height in metres"""
    kgo_dir = acc.kgo_root() / "generate-metadata-cube"
    kgo_path = kgo_dir / "kgo_pressure_levels.nc"
    height_levels_path = kgo_dir / "height_levels.json"
    output_path = tmp_path / "output.nc"
    args = [
        "--height-levels-json",
        height_levels_path,
        "--pressure",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_both_height_levels_input(tmp_path):
    """Test metadata cube generated with height_levels input rather than
    height_levels_json input if both provided"""
    kgo_dir = acc.kgo_root() / "generate-metadata-cube"
    kgo_path = kgo_dir / "kgo_single_height_level.nc"
    height_levels_path = kgo_dir / "height_levels.json"
    output_path = tmp_path / "output.nc"
    args = [
        "--height-levels",
        "1.5",
        "--height-levels-json",
        height_levels_path,
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)
