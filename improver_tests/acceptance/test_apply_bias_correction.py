# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Tests for the apply-bias-correction CLI."""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


def test_single_bias_file(tmp_path):
    """
    Test case where bias values are stored in a single file (mean value over
    multiple historic forecasts).
    """
    kgo_dir = acc.kgo_root() / "apply-bias-correction"
    kgo_path = kgo_dir / "single_bias_file" / "kgo.nc"
    fcst_path = kgo_dir / "20220814T0300Z-PT0003H00M-wind_speed_at_10m.nc"
    bias_file_path = (
        kgo_dir
        / "single_bias_file"
        / "bias_data"
        / "20220813T0300Z-PT0003H00M-wind_speed_at_10m.nc"
    )
    output_path = tmp_path / "output.nc"
    args = [fcst_path, bias_file_path, "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_multiple_bias_files(tmp_path):
    """
    Test case where bias values are stored over multiple files (single file
    per historic forecasts).
    """
    kgo_dir = acc.kgo_root() / "apply-bias-correction"
    kgo_path = kgo_dir / "multiple_bias_files" / "kgo.nc"
    fcst_path = kgo_dir / "20220814T0300Z-PT0003H00M-wind_speed_at_10m.nc"
    bias_file_paths = (kgo_dir / "multiple_bias_files" / "bias_data").glob(
        "202208*T0300Z-PT0003H00M-wind_speed_at_10m.nc"
    )
    output_path = tmp_path / "output.nc"
    args = [fcst_path, *bias_file_paths, "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_bounds(tmp_path):
    """
    Test case where lower and upper bounds are supplied.

    Note: we are using an artificially higher lower bound and lower higher bound
    than would use in practice to ensure process works.
    """
    kgo_dir = acc.kgo_root() / "apply-bias-correction"
    kgo_path = kgo_dir / "with_bounds" / "kgo.nc"
    fcst_path = kgo_dir / "20220814T0300Z-PT0003H00M-wind_speed_at_10m.nc"
    bias_file_path = (
        kgo_dir
        / "single_bias_file"
        / "bias_data"
        / "20220813T0300Z-PT0003H00M-wind_speed_at_10m.nc"
    )
    output_path = tmp_path / "output.nc"
    args = [
        fcst_path,
        bias_file_path,
        "--lower-bound",
        "2",
        "--upper-bound",
        "5",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_with_masked_bias_data(tmp_path):
    """
    Test case where bias data contains masked values. Default is for mask to pass
    through to the bias-corrected forecast output.
    """
    kgo_dir = acc.kgo_root() / "apply-bias-correction" / "masked_bias_data"
    kgo_path = kgo_dir / "retain_mask" / "kgo.nc"
    fcst_path = kgo_dir / "20220814T0300Z-PT0003H00M-wind_speed_at_10m.nc"
    bias_file_path = (
        kgo_dir / "bias_data" / "20220813T0300Z-PT0003H00M-wind_speed_at_10m.nc"
    )
    output_path = tmp_path / "output.nc"
    args = [fcst_path, bias_file_path, "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_fill_masked_bias_data(tmp_path):
    """
    Test case where bias data contains masked values, and masked bias data values
    are filled using appropriate fill value.
    """
    kgo_dir = acc.kgo_root() / "apply-bias-correction" / "masked_bias_data"
    kgo_path = kgo_dir / "fill_masked_values" / "kgo.nc"
    fcst_path = kgo_dir / "20220814T0300Z-PT0003H00M-wind_speed_at_10m.nc"
    bias_file_path = (
        kgo_dir / "bias_data" / "20220813T0300Z-PT0003H00M-wind_speed_at_10m.nc"
    )
    output_path = tmp_path / "output.nc"
    args = [
        fcst_path,
        bias_file_path,
        "--fill-masked-bias-data",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)
