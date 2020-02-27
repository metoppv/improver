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
"""
Tests for the temp-lapse-rate CLI
"""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


@pytest.mark.parametrize(
    "extra,match",
    ((["--max-lapse-rate", "-1", "--min-lapse-rate", "1"], "lapse rate"),
     (["--max-height-diff", "-1"], "height difference"),
     (["--nbhood-radius", "-1"], "radius")))
def test_bad_params(tmp_path, extra, match):
    """Test use of incorrect parameters"""
    kgo_dir = acc.kgo_root() / "temp-lapse-rate/basic"
    input_path = kgo_dir / "temperature_at_screen_level.nc"
    orography_path = kgo_dir / "ukvx_orography.nc"
    landmask_path = kgo_dir / "ukvx_landmask.nc"
    output_path = tmp_path / "output.nc"
    args = [input_path, orography_path, landmask_path,
            *extra,
            "--output", output_path]
    with pytest.raises(ValueError, match=f".*{match}.*"):
        run_cli(args)


def test_no_orog_or_mask(tmp_path):
    """Test basic temperature lapse rate"""
    kgo_dir = acc.kgo_root() / "temp-lapse-rate/basic"
    input_path = kgo_dir / "temperature_at_screen_level.nc"
    output_path = tmp_path / "output.nc"
    args = [input_path,
            "--output", output_path]
    with pytest.raises(RuntimeError, match="orography.*land mask"):
        run_cli(args)


def test_basic(tmp_path):
    """Test basic temperature lapse rate"""
    kgo_dir = acc.kgo_root() / "temp-lapse-rate/basic"
    kgo_path = kgo_dir / "kgo.nc"
    input_path = kgo_dir / "temperature_at_screen_level.nc"
    orography_path = kgo_dir / "ukvx_orography.nc"
    landmask_path = kgo_dir / "ukvx_landmask.nc"
    output_path = tmp_path / "output.nc"
    args = [input_path, orography_path, landmask_path,
            "--output", output_path,
            "--model-id-attr", "mosg__model_configuration"]
    run_cli(args)
    acc.compare(output_path, kgo_path)


@pytest.mark.slow
def test_realizations(tmp_path):
    """Test temperature lapse rate with realizations"""
    kgo_dir = acc.kgo_root() / "temp-lapse-rate/realizations"
    kgo_path = kgo_dir / "kgo.nc"
    input_path = kgo_dir / "enukx_temperature.nc"
    orography_path = kgo_dir / "enukx_orography.nc"
    landmask_path = kgo_dir / "enukx_landmask.nc"
    output_path = tmp_path / "output.nc"
    args = [input_path, orography_path, landmask_path,
            "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path, rtol=0.0)


def test_options(tmp_path):
    """Test temperature lapse rate with additional options"""
    kgo_dir = acc.kgo_root() / "temp-lapse-rate/options"
    kgo_path = kgo_dir / "kgo.nc"
    input_dir = kgo_dir / "../basic"
    input_path = input_dir / "temperature_at_screen_level.nc"
    orography_path = input_dir / "ukvx_orography.nc"
    landmask_path = input_dir / "ukvx_landmask.nc"
    output_path = tmp_path / "output.nc"
    args = [input_path, orography_path, landmask_path,
            "--max-height-diff", "10",
            "--nbhood-radius", "3",
            "--max-lapse-rate", "0.06",
            "--min-lapse-rate", "-0.01",
            "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_dalr(tmp_path):
    """Test dry adiabatic lapse rate"""
    kgo_dir = acc.kgo_root() / "temp-lapse-rate/dalr"
    kgo_path = kgo_dir / "kgo.nc"
    input_dir = kgo_dir / "../basic"
    input_path = input_dir / "temperature_at_screen_level.nc"
    output_path = tmp_path / "output.nc"
    args = [input_path,
            "--dry-adiabatic",
            "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)
