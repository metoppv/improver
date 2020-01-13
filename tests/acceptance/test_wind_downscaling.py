# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2019 Met Office.
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
"""Tests for the wind-downscaling CLI"""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


@pytest.mark.slow
def test_basic(tmp_path):
    """Test basic wind downscaling"""
    kgo_dir = acc.kgo_root() / "wind_downscaling/basic"
    kgo_path = kgo_dir / "kgo.nc"
    input_paths = [kgo_dir / f"{p}.nc"
                   for p in ("input", "sigma", "highres_orog", "standard_orog",
                             "a_over_s")]
    output_path = tmp_path / "output.nc"
    args = [*input_paths,
            "--model-resolution", "1500",
            "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


@pytest.mark.slow
def test_vegetation(tmp_path):
    """Test wind downscaling with vegetation roughness"""
    kgo_dir = acc.kgo_root() / "wind_downscaling/veg"
    kgo_path = kgo_dir / "kgo.nc"
    input_paths = [kgo_dir / f"{p}.nc"
                   for p in ("input", "sigma", "highres_orog", "standard_orog",
                             "a_over_s")]
    veg_path = kgo_dir / "veg.nc"
    output_path = tmp_path / "output.nc"
    args = [*input_paths, veg_path,
            "--model-resolution", "1500",
            "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


@pytest.mark.slow
def test_realization(tmp_path):
    """Test wind downscaling with realization coordinate"""
    kgo_dir = acc.kgo_root() / "wind_downscaling/with_realization"
    kgo_path = kgo_dir / "kgo.nc"
    input_paths = [kgo_dir / f"{p}.nc"
                   for p in ("input", "sigma", "highres_orog", "standard_orog",
                             "a_over_s")]
    output_path = tmp_path / "output.nc"
    args = [*input_paths,
            "--model-resolution", "1500",
            "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_single_level_output(tmp_path):
    """Test downscaling with output at a single specified level"""
    kgo_dir = acc.kgo_root() / "wind_downscaling/single_level"
    kgo_path = kgo_dir / "kgo.nc"
    input_dir = kgo_dir / "../basic"
    input_paths = [input_dir / f"{p}.nc"
                   for p in ("input", "sigma", "highres_orog", "standard_orog",
                             "a_over_s")]
    output_path = tmp_path / "output.nc"
    args = [*input_paths,
            "--model-resolution", "1500",
            "--output-height-level", "10",
            "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_unavailable_level(tmp_path):
    """Test attempting to downscale to unavailable height level"""
    kgo_dir = acc.kgo_root() / "wind_downscaling/basic"
    input_paths = [kgo_dir / f"{p}.nc"
                   for p in ("input", "sigma", "highres_orog", "standard_orog",
                             "a_over_s")]
    output_path = tmp_path / "output.nc"
    args = [*input_paths,
            "--model-resolution", "1500",
            "--output-height-level", "9",
            "--output-height-level-units", "m",
            "--output", output_path]
    with pytest.raises(ValueError, match=".*height level.*"):
        run_cli(args)


def test_single_level_units(tmp_path):
    """Test downscaling with specified level and units"""
    kgo_dir = acc.kgo_root() / "wind_downscaling/single_level"
    kgo_path = kgo_dir / "kgo.nc"
    input_dir = kgo_dir / "../basic"
    input_paths = [input_dir / f"{p}.nc"
                   for p in ("input", "sigma", "highres_orog", "standard_orog",
                             "a_over_s")]
    output_path = tmp_path / "output.nc"
    args = [*input_paths,
            "--model-resolution", "1500",
            "--output-height-level", "1000",
            "--output-height-level-units", "cm",
            "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)
