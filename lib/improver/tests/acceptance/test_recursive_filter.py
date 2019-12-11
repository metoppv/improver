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
Tests for the recursive-filter CLI
"""

import pytest

from improver.tests.acceptance import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


def test_basic_alpha(tmp_path):
    """Test basic recursive filter usage with a numerical alpha value"""
    kgo_dir = acc.kgo_root() / "recursive-filter/basic_alpha"
    kgo_path = kgo_dir / "kgo.nc"
    input_path = kgo_dir / "input.nc"
    output_path = tmp_path / "output.nc"
    args = [input_path, output_path,
            "--alpha_x=0.5", "--alpha_y=0.5", "--iterations=2"]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_alpha_files(tmp_path):
    """Test recursive filter usage with alpha grids"""
    kgo_dir = acc.kgo_root() / "recursive-filter/option_alphas"
    kgo_path = kgo_dir / "kgo.nc"
    input_path = kgo_dir / "input.nc"
    alphasx_path = kgo_dir / "alphasx.nc"
    alphasy_path = kgo_dir / "alphasy.nc"
    output_path = tmp_path / "output.nc"
    args = [input_path, output_path,
            f"--input_filepath_alphas_x={alphasx_path}",
            f"--input_filepath_alphas_y={alphasy_path}",
            "--iterations=2"]
    run_cli(args)
    acc.compare(output_path, kgo_path, rtol=None)


@pytest.mark.slow
def test_external_mask_with_remask(tmp_path):
    """Test recursive filter with input mask file and remasking"""
    kgo_dir = acc.kgo_root() / "recursive-filter/masked_input"
    kgo_path = kgo_dir / "kgo_external_mask_with_re_mask.nc"
    input_path = kgo_dir / "input.nc"
    mask_path = kgo_dir / "mask.nc"
    output_path = tmp_path / "output.nc"
    args = [input_path, output_path,
            "--alpha_x=0.5", "--alpha_y=0.5", "--iterations=2",
            f"--input_mask_filepath={mask_path}",
            "--re_mask"]
    run_cli(args)
    acc.compare(output_path, kgo_path)


@pytest.mark.slow
def test_external_mask_with_no_remask(tmp_path):
    """Test recursive filter with input mask file and no remasking"""
    kgo_dir = acc.kgo_root() / "recursive-filter/masked_input"
    kgo_path = kgo_dir / "kgo_external_mask_no_re_mask.nc"
    input_path = kgo_dir / "input.nc"
    mask_path = kgo_dir / "mask.nc"
    output_path = tmp_path / "output.nc"
    args = [input_path, output_path,
            "--alpha_x=0.5", "--alpha_y=0.5", "--iterations=2",
            f"--input_mask_filepath={mask_path}"]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_internal_mask_with_remask(tmp_path):
    """Test recursive filter with internal mask and remasking"""
    kgo_dir = acc.kgo_root() / "recursive-filter/masked_input"
    kgo_path = kgo_dir / "kgo_internal_mask_with_re_mask.nc"
    input_path = kgo_dir / "input_masked.nc"
    output_path = tmp_path / "output.nc"
    args = [input_path, output_path,
            "--alpha_x=0.5", "--alpha_y=0.5", "--iterations=2", "--re_mask"]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_internal_mask_with_no_remask(tmp_path):
    """Test recursive filter with internal mask and remasking"""
    kgo_dir = acc.kgo_root() / "recursive-filter/masked_input"
    kgo_path = kgo_dir / "kgo_internal_mask_no_re_mask.nc"
    input_path = kgo_dir / "input_masked.nc"
    output_path = tmp_path / "output.nc"
    args = [input_path, output_path,
            "--alpha_x=0.5", "--alpha_y=0.5", "--iterations=2"]
    run_cli(args)
    acc.compare(output_path, kgo_path)
