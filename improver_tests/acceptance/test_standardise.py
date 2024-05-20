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
"""
Tests for the standardise CLI
"""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


ATTRIBUTES_PATH = acc.kgo_root() / "standardise/metadata/metadata.json"


def test_change_metadata(tmp_path):
    """Test applying a JSON metadata file"""
    kgo_dir = acc.kgo_root() / "standardise/metadata"
    kgo_path = kgo_dir / "kgo.nc"
    input_path = kgo_dir / "input.nc"
    metadata_path = kgo_dir / "radar_metadata.json"
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        "--output",
        output_path,
        "--new-name",
        "lwe_precipitation_rate",
        "--new-units",
        "m s-1",
        "--attributes-config",
        metadata_path,
        "--coords-to-remove",
        "height",
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_fix_float64(tmp_path):
    """Test conversion of float64 data to float32"""
    kgo_dir = acc.kgo_root() / "standardise/float64"
    kgo_path = kgo_dir / "kgo.nc"
    input_path = kgo_dir / "float64_data.nc"
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        "--attributes-config",
        ATTRIBUTES_PATH,
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_nimrod_radarrate_basic(tmp_path):
    """Test updating a file with Nimrod-format Radarnet data"""
    kgo_dir = acc.kgo_root() / "standardise/radarnet"
    kgo_path = kgo_dir / "kgo_preciprate.nc"
    input_path = kgo_dir / "input_preciprate.nimrod"
    output_path = tmp_path / "output.nc"
    args = [input_path, "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_nimrod_radarcoverage_basic(tmp_path):
    """Test updating a file with Nimrod-format Radarnet data"""
    kgo_dir = acc.kgo_root() / "standardise/radarnet"
    kgo_path = kgo_dir / "kgo_coverage.nc"
    input_path = kgo_dir / "input_coverage.nimrod"
    output_path = tmp_path / "output.nc"
    args = [input_path, "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_change_scalar_coord(tmp_path):
    """Test applying a JSON coord_modification file"""
    kgo_dir = acc.kgo_root() / "standardise/modification"
    kgo_path = kgo_dir / "kgo.nc"
    input_path = acc.kgo_root() / "standardise/metadata" / "input.nc"
    modification_path = kgo_dir / "scalar_change.json"
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        "--output",
        output_path,
        "--coord-modification",
        modification_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)
