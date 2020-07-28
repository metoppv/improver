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


def test_all_options(tmp_path):
    """Test metadata cube generation setting values for all options"""
    kgo_dir = acc.kgo_root() / "generate-metadata-cube"
    kgo_path = kgo_dir / "kgo_all.nc"
    attributes_path = kgo_dir / "attributes.json"
    output_path = tmp_path / "output.nc"
    args = [
        "--name",
        "air_pressure",
        "--units",
        "pascal",
        "--spatial-grid",
        "equalarea",
        "--time",
        "20200102T0400Z",
        "--frt",
        "20200101T0400Z",
        "--ensemble-members",
        "4",
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


def test_single_height_level(tmp_path):
    """Test metadata cube generation giving single value (rather than comma separated list) for height levels option"""
    kgo_dir = acc.kgo_root() / "generate-metadata-cube"
    kgo_path = kgo_dir / "kgo_single_height_level.nc"
    output_path = tmp_path / "output.nc"
    args = ["--height-levels", "1.5", "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_error_invalid_domain_corner(tmp_path):
    """Test error is raised invalid domain corner is set"""
    output_path = tmp_path / "output.nc"
    args = ["--domain-corner", "0", "--output", output_path]
    with pytest.raises(
        TypeError, match="Domain corner must be a comma separated list of length 2"
    ):
        run_cli(args)
