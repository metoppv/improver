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
Tests for the phase-probability CLI
"""

import pytest

from improver_tests.acceptance import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


@pytest.mark.parametrize(
    "kgo_name,input_file",
    [
        ("snow_kgo", "snow_sleet_input"),
        ("rain_kgo", "sleet_rain_input"),
        ("hail_kgo", "hail_rain_input"),
    ],
)
def test_phase_probabilities(tmp_path, kgo_name, input_file):
    """Test phase probability calculations for snow->sleet, sleet->rain and hail->rain"""
    kgo_dir = acc.kgo_root() / f"{CLI}/basic"
    kgo_path = kgo_dir / f"{kgo_name}.nc"
    output_path = tmp_path / "output.nc"
    input_paths = [
        acc.kgo_root() / x
        for x in (
            "phase-change-level/orog.nc",
            "phase-probability/basic/" f"{input_file}.nc",
        )
    ]
    args = [*input_paths, "--radius", "10000", "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_rain_large_radius(tmp_path):
    """Test prob(rain) calculation"""
    kgo_dir = acc.kgo_root() / f"{CLI}/large_radius"
    kgo_path = kgo_dir / "rain_kgo.nc"
    output_path = tmp_path / "output.nc"
    input_paths = [
        acc.kgo_root() / x
        for x in (
            "phase-change-level/orog.nc",
            "phase-probability/basic/sleet_rain_input.nc",
        )
    ]
    args = [*input_paths, "--radius", "20000", "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_non_equal_area_projection(tmp_path):
    """Test prob(snow) calculation on a non-equal areas projection for
    which no radius is used. Without a radius the falling level in each
    cell is compared directly with the orography."""
    kgo_dir = acc.kgo_root() / f"{CLI}/global"
    kgo_path = kgo_dir / "snow_kgo.nc"
    output_path = tmp_path / "output.nc"
    input_paths = [
        kgo_dir / "orography.nc",
        kgo_dir / "snow_sleet_input.nc",
    ]
    args = [*input_paths, "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)
