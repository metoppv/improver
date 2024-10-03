# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""
Tests for the recursive-filter CLI
"""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


def test_basic(tmp_path):
    """Test basic recursive filter usage"""
    kgo_dir = acc.kgo_root() / "recursive-filter"
    kgo_path = kgo_dir / "kgo_basic.nc"
    input_path = kgo_dir / "input.nc"
    smoothing_coefficients_path = kgo_dir / "smoothing_coefficients.nc"
    output_path = tmp_path / "output.nc"
    args = [input_path, smoothing_coefficients_path, "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_internal_mask(tmp_path):
    """Test recursive filter with masked data"""
    kgo_dir = acc.kgo_root() / "recursive-filter"
    kgo_path = kgo_dir / "kgo_internal_mask_with_re_mask.nc"
    input_path = kgo_dir / "input_masked.nc"
    smoothing_coefficients_path = kgo_dir / "smoothing_coefficients.nc"
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        smoothing_coefficients_path,
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_variable_internal_mask(tmp_path):
    """Test recursive filter with variably masked data"""
    kgo_dir = acc.kgo_root() / "recursive-filter"
    kgo_path = kgo_dir / "kgo_variable_internal_mask_with_re_mask.nc"
    input_path = kgo_dir / "input_variable_masked.nc"
    smoothing_coefficients_path = kgo_dir / "smoothing_coefficients.nc"
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        smoothing_coefficients_path,
        "--variable-mask",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)
