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
"""Tests for the wet-bulb-temperature CLI"""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


def test_basic(tmp_path):
    """Test basic wet bulb temperature calculation"""
    kgo_dir = acc.kgo_root() / "wet-bulb-temperature/basic"
    kgo_path = kgo_dir / "kgo.nc"
    input_paths = [kgo_dir / f"enukx_{p}.nc"
                   for p in ("temperature", "relative_humidity", "pressure")]
    output_path = tmp_path / "output.nc"
    args = [*input_paths, "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_multilevel(tmp_path):
    """Test wet bulb temperature on multiple levels"""
    kgo_dir = acc.kgo_root() / "wet-bulb-temperature/multi_level"
    kgo_path = kgo_dir / "kgo.nc"
    input_paths = [kgo_dir / f"enukx_multilevel_{p}.nc"
                   for p in ("temperature", "relative_humidity", "pressure")]
    output_path = tmp_path / "output.nc"
    args = [*input_paths,
            "--convergence-condition", "0.005",
            "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_global(tmp_path):
    """Test wet bulb temperature calculation on global domain"""
    kgo_dir = acc.kgo_root() / "wet-bulb-temperature/global"
    kgo_path = kgo_dir / "kgo.nc"
    input_paths = [kgo_dir / f"{p}_input.nc"
                   for p in ("temperature", "relative_humidity", "pressure")]
    output_path = tmp_path / "output.nc"
    args = [*input_paths, "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)
