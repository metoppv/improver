# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2021 Met Office.
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
Tests for the generate-orographic-smoothing-coefficients CLI
"""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


def test_basic(tmp_path):
    """Test basic generate orographic smoothing coefficients processing"""
    kgo_dir = acc.kgo_root() / "generate-orographic-smoothing-coefficients"
    input_path = kgo_dir / "orography.nc"
    kgo_path = kgo_dir / "basic" / "kgo.nc"
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        "--max-gradient-smoothing-coefficient",
        "0.",
        "--min-gradient-smoothing-coefficient",
        "0.5",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_altered_limits(tmp_path):
    """Test generation of smoothing coefficients with different limiting values."""
    kgo_dir = acc.kgo_root() / "generate-orographic-smoothing-coefficients"
    input_path = kgo_dir / "orography.nc"
    kgo_path = kgo_dir / "basic" / "kgo_different_limits.nc"
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        "--max-gradient-smoothing-coefficient",
        "0.",
        "--min-gradient-smoothing-coefficient",
        "0.25",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_altered_power(tmp_path):
    """Test generation of smoothing coefficients with different power."""
    kgo_dir = acc.kgo_root() / "generate-orographic-smoothing-coefficients"
    input_path = kgo_dir / "orography.nc"
    kgo_path = kgo_dir / "basic" / "kgo_different_power.nc"
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        "--power",
        "0.5",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_mask_edges(tmp_path):
    """Test generation of orographic smoothing coefficients with a zero value
    along mask edges, which in this case is the coastline."""
    kgo_dir = acc.kgo_root() / "generate-orographic-smoothing-coefficients"
    input_path = kgo_dir / "orography.nc"
    input_mask = kgo_dir / "landmask.nc"
    kgo_path = kgo_dir / "mask_boundary" / "kgo.nc"
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        input_mask,
        "--max-gradient-smoothing-coefficient",
        "0.",
        "--min-gradient-smoothing-coefficient",
        "0.5",
        "--use-mask-boundary",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_mask_area_zeroed(tmp_path):
    """Test generation of orographic smoothing coefficients with a zero value
    under all the masked regions and along their edges."""
    kgo_dir = acc.kgo_root() / "generate-orographic-smoothing-coefficients"
    input_path = kgo_dir / "orography.nc"
    input_mask = kgo_dir / "landmask.nc"
    kgo_path = kgo_dir / "mask_zeroed" / "kgo.nc"
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        input_mask,
        "--max-gradient-smoothing-coefficient",
        "0.",
        "--min-gradient-smoothing-coefficient",
        "0.5",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_inverse_mask_area_zeroed(tmp_path):
    """Test generation of orographic smoothing coefficients with a zero value
    under all the unmasked regions and along their edges."""
    kgo_dir = acc.kgo_root() / "generate-orographic-smoothing-coefficients"
    input_path = kgo_dir / "orography.nc"
    input_mask = kgo_dir / "landmask.nc"
    kgo_path = kgo_dir / "inverse_mask_zeroed" / "kgo.nc"
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        input_mask,
        "--max-gradient-smoothing-coefficient",
        "0.",
        "--min-gradient-smoothing-coefficient",
        "0.5",
        "--invert-mask",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)
