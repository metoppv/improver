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
"""Tests for the calculate-forecast-bias CLI."""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


def test_single_frt(tmp_path):
    """
    Test case where single historical forecast value provided.
    """
    kgo_dir = acc.kgo_root() / "calculate-forecast-bias"
    kgo_path = kgo_dir / "single_frt" / "kgo.nc"
    inputs_path = (kgo_dir / "inputs").glob("20220811T0300Z-PT00*.nc")
    output_path = tmp_path / "output.nc"
    args = [
        *inputs_path,
        "--truth-attribute",
        "mosg__model_configuration=msas_det",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_multiple_frt(tmp_path):
    """
    Test case where multiple historical forecast values provided.
    """
    kgo_dir = acc.kgo_root() / "calculate-forecast-bias"
    kgo_path = kgo_dir / "multiple_frt" / "kgo.nc"
    inputs_path = (kgo_dir / "inputs").glob("202208*T0300Z-PT00*.nc")
    output_path = tmp_path / "output.nc"
    args = [
        *inputs_path,
        "--truth-attribute",
        "mosg__model_configuration=msas_det",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_single_frt_masked_inputs(tmp_path):
    """
    Test case where single historical forecast value provided.
    """
    kgo_dir = acc.kgo_root() / "calculate-forecast-bias"
    kgo_path = kgo_dir / "single_frt_masked_inputs" / "kgo.nc"
    inputs_path = (kgo_dir / "inputs/masked").glob("20220811T0300Z-PT00*.nc")
    output_path = tmp_path / "output.nc"
    args = [
        *inputs_path,
        "--truth-attribute",
        "mosg__model_configuration=msas_det",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_multiple_frt_masked_inputs(tmp_path):
    """
    Test case where multiple historical forecast values provided.
    """
    kgo_dir = acc.kgo_root() / "calculate-forecast-bias"
    kgo_path = kgo_dir / "multiple_frt_masked_inputs" / "kgo.nc"
    inputs_path = (kgo_dir / "inputs/masked").glob("202208*T0300Z-PT00*.nc")
    output_path = tmp_path / "output.nc"
    args = [
        *inputs_path,
        "--truth-attribute",
        "mosg__model_configuration=msas_det",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)
