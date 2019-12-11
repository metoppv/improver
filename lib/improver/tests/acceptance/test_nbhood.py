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
Tests for the nbhood CLI
"""

import pytest

from improver.tests.acceptance import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


def test_basic_circular(tmp_path):
    """Test basic circular neighbourhooding"""
    kgo_dir = acc.kgo_root() / "nbhood/basic"
    kgo_path = kgo_dir / "kgo_circular.nc"
    input_path = kgo_dir / "input_circular.nc"
    output_path = tmp_path / "output.nc"
    args = ["probabilities", "circular", "--radius=20000",
            "--weighted_mode", input_path, output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_basic_square(tmp_path):
    """Test basic square neighbourhooding"""
    kgo_dir = acc.kgo_root() / "nbhood/basic"
    kgo_path = kgo_dir / "kgo_square.nc"
    input_path = kgo_dir / "input_square.nc"
    output_path = tmp_path / "output.nc"
    args = ["probabilities", "square", "--radius=20000",
            "--weighted_mode", input_path, output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_masked_square(tmp_path):
    """Test square neighbourhooding with a mask"""
    kgo_dir = acc.kgo_root() / "nbhood/mask"
    kgo_path = kgo_dir / "kgo_masked.nc"
    input_path = kgo_dir / "input_masked.nc"
    output_path = tmp_path / "output.nc"
    args = ["probabilities", "square", "--radius=20000",
            "--weighted_mode", input_path, output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


@pytest.mark.slow
def test_circular_percentile(tmp_path):
    """Test circular neighbourhooding of percentiles"""
    kgo_dir = acc.kgo_root() / "nbhood/percentile"
    kgo_path = kgo_dir / "kgo_circular_percentile.nc"
    input_path = kgo_dir / "input_circular_percentile.nc"
    output_path = tmp_path / "output.nc"
    args = ["percentiles", "circular", input_path, output_path,
            "--radius=20000", "--percentiles", "25", "50", "75"]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_masked_square_external(tmp_path):
    """Test square neighbourhooding with an external mask"""
    kgo_dir = acc.kgo_root() / "nbhood/mask"
    kgo_path = kgo_dir / "kgo_external_masked.nc"
    input_path = kgo_dir / "input.nc"
    mask_path = kgo_dir / "mask.nc"
    output_path = tmp_path / "output.nc"
    args = ["probabilities", "square", input_path, output_path,
            "--radius=20000", "--input_mask_filepath", mask_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_masked_square_remask(tmp_path):
    """Test square neighbourhooding with an external mask and re-masking"""
    kgo_dir = acc.kgo_root() / "nbhood/mask"
    kgo_path = kgo_dir / "kgo_re_mask.nc"
    input_path = kgo_dir / "input.nc"
    mask_path = kgo_dir / "mask.nc"
    output_path = tmp_path / "output.nc"
    args = ["probabilities", "square", input_path, output_path,
            "--radius=20000", "--input_mask_filepath", mask_path, "--re_mask"]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_recfilter_alpha_basic(tmp_path):
    """Test basic recursive filter"""
    kgo_dir = acc.kgo_root() / "nbhood/recursive"
    kgo_path = kgo_dir / "kgo_recursive_alpha.nc"
    input_path = kgo_dir / "../basic/input_square.nc"
    output_path = tmp_path / "output.nc"
    args = ["probabilities", "square", "--radius=20000",
            input_path, output_path,
            "--apply-recursive-filter", "--alpha_x=0.5", "--alpha_y=0.5",
            "--iterations=2"]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_recfilter_alphas(tmp_path):
    """Test recursive filter with alpha cubes"""
    kgo_dir = acc.kgo_root() / "nbhood/recursive"
    kgo_path = kgo_dir / "kgo_recursive_alphas_gridded.nc"
    input_path = kgo_dir / "../basic/input_square.nc"
    alphax_path = kgo_dir / "alphasx.nc"
    alphay_path = kgo_dir / "alphasy.nc"
    output_path = tmp_path / "output.nc"
    args = ["probabilities", "square", "--radius=20000",
            input_path, output_path,
            "--apply-recursive-filter",
            f"--input_filepath_alphas_x_cube={alphax_path}",
            f"--input_filepath_alphas_y_cube={alphay_path}",
            "--iterations=2"]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_recfilter_alpha_remask(tmp_path):
    """Test recursive filter with re-masking"""
    kgo_dir = acc.kgo_root() / "nbhood/recursive"
    kgo_path = kgo_dir / "kgo_internal_mask_re_masked_recursive_alpha.nc"
    input_path = kgo_dir / "../mask/input_masked.nc"
    output_path = tmp_path / "output.nc"
    args = ["probabilities", "square", "--radius=20000",
            input_path, output_path,
            "--apply-recursive-filter", "--re_mask",
            "--alpha_x=0.5", "--alpha_y=0.5", "--iterations=2"]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_recfilter_alpha_noremask(tmp_path):
    """Test recursive filter without re-masking"""
    kgo_dir = acc.kgo_root() / "nbhood/recursive"
    kgo_path = kgo_dir / "kgo_internal_mask_no_re_mask_recursive_alpha.nc"
    input_path = kgo_dir / "../mask/input_masked.nc"
    output_path = tmp_path / "output.nc"
    args = ["probabilities", "square", "--radius=20000",
            input_path, output_path,
            "--apply-recursive-filter",
            "--alpha_x=0.5", "--alpha_y=0.5", "--iterations=2"]
    run_cli(args)
    acc.compare(output_path, kgo_path)


@pytest.mark.slow
def test_recfilter_alpha_external_remask(tmp_path):
    """Test recursive filter with external mask and re-masking"""
    kgo_dir = acc.kgo_root() / "nbhood/recursive"
    kgo_path = kgo_dir / "kgo_external_mask_with_re_mask_recursive_alpha.nc"
    input_path = kgo_dir / "../mask/input.nc"
    mask_path = kgo_dir / "../mask/mask.nc"
    output_path = tmp_path / "output.nc"
    args = ["probabilities", "square", "--radius=20000",
            input_path, output_path,
            "--apply-recursive-filter", "--re_mask",
            "--alpha_x=0.5", "--alpha_y=0.5", "--iterations=2",
            "--input_mask_filepath", mask_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


@pytest.mark.slow
def test_recfilter_alpha_external_noremask(tmp_path):
    """Test recursive filter with external mask and without re-masking"""
    kgo_dir = acc.kgo_root() / "nbhood/recursive"
    kgo_path = kgo_dir / "kgo_external_mask_no_re_mask_recursive_alpha.nc"
    input_path = kgo_dir / "../mask/input.nc"
    mask_path = kgo_dir / "../mask/mask.nc"
    output_path = tmp_path / "output.nc"
    args = ["probabilities", "square", "--radius=20000",
            input_path, output_path,
            "--apply-recursive-filter",
            "--alpha_x=0.5", "--alpha_y=0.5", "--iterations=2",
            "--input_mask_filepath", mask_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_wind_direction(tmp_path):
    """Test wind direction neighbourhooding"""
    kgo_dir = acc.kgo_root() / "nbhood/wind_direction"
    kgo_path = kgo_dir / "kgo.nc"
    input_path = kgo_dir / "input.nc"
    output_path = tmp_path / "output.nc"
    args = ["probabilities", "square", "--radius=20000",
            input_path, output_path, "--degrees_as_complex"]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_halo_radius(tmp_path):
    """Test neighbourhooding with a halo radius"""
    kgo_dir = acc.kgo_root() / "nbhood/halo"
    kgo_path = kgo_dir / "kgo.nc"
    input_path = kgo_dir / "input.nc"
    output_path = tmp_path / "output.nc"
    args = ["probabilities", "square", "--radius=20000", "--weighted_mode",
            "--halo_radius=162000", input_path, output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)
