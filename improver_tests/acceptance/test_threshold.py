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
Tests for the threshold CLI
"""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


def test_basic(tmp_path):
    """Test basic thresholding"""
    kgo_dir = acc.kgo_root() / "threshold/basic"
    kgo_path = kgo_dir / "kgo.nc"
    input_path = kgo_dir / "input.nc"
    output_path = tmp_path / "output.nc"
    args = [input_path,
            "--output", output_path,
            "--threshold-values", "280"]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_below_threshold(tmp_path):
    """Test thresholding with specified operator"""
    kgo_dir = acc.kgo_root() / "threshold/below_threshold"
    kgo_path = kgo_dir / "kgo.nc"
    input_path = kgo_dir / "../basic/input.nc"
    output_path = tmp_path / "output.nc"
    args = [input_path,
            "--output", output_path,
            "--threshold-values", "280",
            "--comparison-operator", "<="]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_multiple_thresholds(tmp_path):
    """Test thresholding with multiple thresholds"""
    kgo_dir = acc.kgo_root() / "threshold/multiple_thresholds"
    kgo_path = kgo_dir / "kgo.nc"
    input_path = kgo_dir / "../basic/input.nc"
    output_path = tmp_path / "output.nc"
    args = [input_path,
            "--output", output_path,
            "--threshold-values", "270,280,290"]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_threshold_units(tmp_path):
    """Test thresholding with specified units"""
    kgo_dir = acc.kgo_root() / "threshold/threshold_units"
    kgo_path = kgo_dir / "kgo.nc"
    input_path = kgo_dir / "../basic/input.nc"
    output_path = tmp_path / "output.nc"
    args = [input_path,
            "--output", output_path,
            "--threshold-values", "6.85",
            "--threshold-units", "celsius"]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_fuzzy_factor(tmp_path):
    """Test thresholding with fuzzy factor"""
    kgo_dir = acc.kgo_root() / "threshold/fuzzy_factor"
    kgo_path = kgo_dir / "kgo.nc"
    input_path = kgo_dir / "../basic/input.nc"
    output_path = tmp_path / "output.nc"
    args = [input_path,
            "--output", output_path,
            "--threshold-values", "280",
            "--fuzzy-factor", "0.99"]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_fuzzy_bounds(tmp_path):
    """Test thresholding with fuzzy bounds configuration file"""
    kgo_dir = acc.kgo_root() / "threshold/fuzzy_factor"
    kgo_path = kgo_dir / "kgo.nc"
    threshold_path = kgo_dir / "../fuzzy_bounds/threshold_config.json"
    input_path = kgo_dir / "../basic/input.nc"
    output_path = tmp_path / "output.nc"
    args = [input_path,
            "--output", output_path,
            "--threshold-config", threshold_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_threshold_units_fuzzy(tmp_path):
    """Test thresholding with specified units and fuzzy factor"""
    kgo_dir = acc.kgo_root() / "threshold/threshold_units_fuzzy_factor"
    kgo_path = kgo_dir / "kgo.nc"
    input_path = kgo_dir / "../basic/input.nc"
    output_path = tmp_path / "output.nc"
    args = [input_path,
            "--output", output_path,
            "--threshold-values", "6.85",
            "--threshold-units", "celsius",
            "--fuzzy-factor", "0.2"]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_collapse_realization(tmp_path):
    """Test thresholding with collapsing realizations"""
    kgo_dir = acc.kgo_root() / "threshold/coord_collapse"
    kgo_path = kgo_dir / "kgo.nc"
    input_path = kgo_dir / "../basic/input.nc"
    output_path = tmp_path / "output.nc"
    args = [input_path,
            "--output", output_path,
            "--threshold-values", "280",
            "--collapse-coord", "realization"]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_vicinity(tmp_path):
    """Test thresholding with vicinity"""
    kgo_dir = acc.kgo_root() / "threshold/vicinity"
    kgo_path = kgo_dir / "kgo.nc"
    input_path = kgo_dir / "input.nc"
    output_path = tmp_path / "output.nc"
    args = [input_path,
            "--output", output_path,
            "--threshold-values", "0.03,0.1,1.0",
            "--threshold-units", "mm hr-1",
            "--vicinity", "10000"]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_vicinity_collapse(tmp_path):
    """Test thresholding with vicinity and collapse"""
    kgo_dir = acc.kgo_root() / "threshold/vicinity"
    kgo_path = kgo_dir / "kgo_collapsed.nc"
    input_path = kgo_dir / "input.nc"
    output_path = tmp_path / "output.nc"
    args = [input_path,
            "--output", output_path,
            "--threshold-values", "0.03,0.1,1.0",
            "--threshold-units", "mm hr-1",
            "--vicinity", "10000",
            "--collapse-coord=realization"]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_vicinity_masked(tmp_path):
    """Test thresholding with vicinity and masked precipitation"""
    kgo_dir = acc.kgo_root() / "threshold/vicinity"
    kgo_path = kgo_dir / "kgo_masked.nc"
    input_path = kgo_dir / "masked_precip.nc"
    output_path = tmp_path / "output.nc"
    args = [input_path,
            "--output", output_path,
            "--threshold-values", "0.03,0.1,1.0",
            "--threshold-units", "mm hr-1",
            "--vicinity", "10000"]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_threshold_config(tmp_path):
    """Test basic thresholding using configuration file"""
    kgo_dir = acc.kgo_root() / "threshold/basic"
    kgo_path = kgo_dir / "kgo.nc"
    input_path = kgo_dir / "input.nc"
    output_path = tmp_path / "output.nc"
    config_path = kgo_dir / "../json/threshold_config.json"
    args = [input_path,
            "--output", output_path,
            "--threshold-config", config_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)
