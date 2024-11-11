# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""
Tests for the generate-realizations CLI
"""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


def test_percentiles(tmp_path):
    """Test basic percentile to realization conversion"""
    kgo_dir = acc.kgo_root() / "generate-realizations/percentiles_rebadging"
    kgo_path = kgo_dir / "kgo.nc"
    input_path = kgo_dir / "multiple_percentiles_wind_cube.nc"

    output_path = tmp_path / "output.nc"

    args = [input_path, "--realizations-count", "12", "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


@pytest.mark.parametrize(
    "tie_break, kgo",
    (
        ("random", "tie_break_with_random_kgo.nc"),
        ("realization", "tie_break_with_realization_kgo.nc"),
    ),
)
def test_percentiles_reordering(tmp_path, tie_break, kgo):
    """Test percentile to realization conversion with reordering"""
    kgo_dir = acc.kgo_root() / "generate-realizations/percentiles_reordering"
    kgo_path = kgo_dir / kgo
    forecast_path = kgo_dir / "raw_precip_forecast.nc"
    percentiles_path = kgo_dir / "multiple_percentiles_precip_cube.nc"
    output_path = tmp_path / "output.nc"
    args = [
        "--realizations-count",
        "12",
        "--random-seed",
        "0",
        "--tie-break",
        tie_break,
        percentiles_path,
        forecast_path,
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


@pytest.mark.parametrize(
    "bounds_option, kgo",
    (
        ("", "with_ecc_bounds_kgo.nc"),
        ("--skip-ecc-bounds", "without_ecc_bounds_kgo.nc"),
    ),
)
def test_skip_ecc_bounds_extreme_percentiles(tmp_path, bounds_option, kgo):
    """Test percentile to percentile conversion where outputs are more extreme than inputs
    (lowest percentile of inputs is 31, outputs have lowest of 20)"""
    kgo_dir = acc.kgo_root() / "generate-realizations/percentiles_extremes"
    kgo_path = kgo_dir / kgo
    percentiles_path = kgo_dir / "few_percentiles_wind_cube.nc"
    output_path = tmp_path / "output.nc"
    args = ["--realizations-count", "5", percentiles_path, "--output", output_path]
    if bounds_option:
        args += [bounds_option]
    run_cli(args)
    acc.compare(output_path, kgo_path)


@pytest.mark.parametrize(
    "bounds_option, kgo",
    (
        ("", "with_ecc_bounds_kgo.nc"),
        (["--skip-ecc-bounds"], "without_ecc_bounds_kgo.nc"),
    ),
)
def test_skip_ecc_bounds_probabilities(tmp_path, bounds_option, kgo):
    """Test probability to percentile conversion where the percentiles need to sample
    outside of the distribution given by the probabilities.
    (there are three input probabilities with non-zero probabilities, outputs have
    five percentiles)."""
    kgo_dir = acc.kgo_root() / "generate-realizations/skip_ecc_bounds_probabilities"
    kgo_path = kgo_dir / kgo
    percentiles_path = kgo_dir / "input.nc"
    output_path = tmp_path / "output.nc"
    args = [
        "--realizations-count",
        "5",
        percentiles_path,
        "--output",
        output_path,
        *bounds_option,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_probabilities(tmp_path):
    """Test basic probabilities to realization conversion"""
    kgo_dir = acc.kgo_root() / "generate-realizations/probabilities_12_realizations"
    kgo_path = kgo_dir / "kgo.nc"
    input_path = kgo_dir / "input.nc"
    output_path = tmp_path / "output.nc"
    args = [input_path, "--realizations-count", "12", "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_probabilities_reordering(tmp_path):
    """Test probabilities to realization conversion with reordering"""
    kgo_dir = acc.kgo_root() / "generate-realizations/probabilities_reordering"
    kgo_path = kgo_dir / "kgo.nc"
    raw_path = kgo_dir / "raw_ens.nc"
    input_path = kgo_dir / "input.nc"
    output_path = tmp_path / "output.nc"
    args = ["--random-seed", "0", input_path, raw_path, "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_realizations(tmp_path):
    """Test basic null realization to realization conversion"""
    kgo_dir = acc.kgo_root() / "generate-realizations/probabilities_12_realizations"
    kgo_path = kgo_dir / "kgo.nc"
    input_path = kgo_path
    output_path = tmp_path / "output.nc"
    args = [input_path, "--realizations-count", "12", "--output", output_path]
    run_cli(args)
    acc.compare(output_path, input_path)


def test_ecc_bounds_warning(tmp_path):
    """
    Test use of ECC to convert one set of percentiles to another set of
    percentiles, and then rebadge the percentiles to be ensemble realizations.
    Data in this input exceeds the ECC bounds and so tests ecc_bounds_warning
    functionality.
    """
    kgo_dir = acc.kgo_root() / "generate-realizations/ecc_bounds_warning"
    kgo_path = kgo_dir / "kgo.nc"
    input_path = kgo_dir / "multiple_percentiles_wind_cube_out_of_bounds.nc"
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        "--realizations-count",
        "5",
        "--ignore-ecc-bounds-exceedance",
        "--output",
        output_path,
    ]
    with pytest.warns(UserWarning, match="Forecast values exist that fall outside"):
        run_cli(args)
    acc.compare(output_path, kgo_path)


def test_error_no_realizations_count(tmp_path):
    """Test a helpful error is raised if wrong args are set"""
    kgo_dir = acc.kgo_root() / "generate-realizations/probabilities_12_realizations"
    input_path = kgo_dir / "input.nc"
    output_path = tmp_path / "output.nc"
    args = [input_path, "--output", output_path]
    with pytest.raises(ValueError, match=".*realizations_count or raw_cube.*"):
        run_cli(args)


def test_invalid_dataset(tmp_path):
    """Test unhandlable conversion failure"""
    input_dir = acc.kgo_root() / "generate-realizations/invalid/"
    input_path = input_dir / "input.nc"
    output_path = tmp_path / "output.nc"
    args = [input_path, "--output", output_path]
    with pytest.raises(ValueError, match=".*Unable to convert.*"):
        run_cli(args)
