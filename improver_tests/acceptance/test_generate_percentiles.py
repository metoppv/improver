# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of improver and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Tests for the generate-percentiles CLI"""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
run_cli = acc.run_cli("generate-percentiles")


def test_basic(tmp_path):
    """Test basic percentile processing"""
    kgo_dir = acc.kgo_root() / "generate-percentiles/basic"
    kgo_path = kgo_dir / "kgo.nc"
    perc_input = kgo_dir / "input.nc"
    output_path = tmp_path / "output.nc"
    args = [
        perc_input,
        "--output",
        output_path,
        "--coordinates",
        "realization",
        "--percentiles",
        "25.0,50,75.0",
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


@pytest.mark.slow
@pytest.mark.parametrize("count", ("single", "multi"))
def test_probconvert(tmp_path, count):
    """Test probability conversion"""
    kgo_dir = acc.kgo_root() / "generate-percentiles/probability_convert"
    kgo_path = kgo_dir / f"{count}_realization_kgo.nc"
    prob_input = kgo_dir / f"{count}_realization.nc"
    output_path = tmp_path / "output.nc"
    args = [
        prob_input,
        "--output",
        output_path,
        "--coordinates",
        "realization",
        "--percentiles",
        "25,50,75",
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


@pytest.mark.slow
def test_ignore_ecc_bounds(tmp_path,):
    """Test ECC bounds warning option"""
    kgo_dir = acc.kgo_root() / "generate-percentiles/ecc_bounds_warning"
    kgo_path = kgo_dir / "kgo.nc"
    perc_input = kgo_dir / "input.nc"
    output_path = tmp_path / "output.nc"
    args = [
        perc_input,
        "--output",
        output_path,
        "--coordinates",
        "realization",
        "--percentiles",
        "25,50,75",
        "--ignore-ecc-bounds-exceedance",
    ]
    with pytest.warns(UserWarning, match="The calculated threshold values"):
        run_cli(args)
    acc.compare(output_path, kgo_path)


@pytest.mark.slow
@pytest.mark.parametrize(
    "bounds_option,kgo",
    (
        ("", "with_ecc_bounds_kgo.nc"),
        (["--skip-ecc-bounds"], "without_ecc_bounds_kgo.nc"),
    ),
)
def test_skip_ecc_bounds(tmp_path, bounds_option, kgo):
    """Test for when the ECC bounds are skipped."""
    kgo_dir = acc.kgo_root() / "generate-percentiles/skip_ecc_bounds"
    kgo_path = kgo_dir / kgo
    perc_input = kgo_dir / "input.nc"
    output_path = tmp_path / "output.nc"
    args = [
        perc_input,
        "--output",
        output_path,
        "--coordinates",
        "realization",
        "--percentiles",
        "2,50,98",
        *bounds_option,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_masked_percentiles_warning(tmp_path):
    """Test masked_percentiles warning"""
    kgo_dir = acc.kgo_root() / "generate-percentiles/basic"
    kgo_path = kgo_dir / "kgo.nc"
    perc_input = kgo_dir / "input.nc"
    output_path = tmp_path / "output.nc"
    args = [
        perc_input,
        "--output",
        output_path,
        "--coordinates",
        "realization",
        "--percentiles",
        "25,50,75",
        "--mask-percentiles",
    ]
    with pytest.warns(UserWarning, match="mask_percentiles is only implemented"):
        run_cli(args)
    acc.compare(output_path, kgo_path)


def test_masked_percentiles(tmp_path):
    """Test probability conversion when masked_percentiles is True"""
    kgo_dir = acc.kgo_root() / "generate-percentiles/probability_convert"
    kgo_path = kgo_dir / "masked_kgo.nc"
    prob_input = kgo_dir / "masked_input.nc"
    output_path = tmp_path / "output.nc"
    args = [
        prob_input,
        "--output",
        output_path,
        "--percentiles",
        "25,50,75",
        "--mask-percentiles",
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


@pytest.mark.parametrize(
    "identifier", ("flat_rank_histogram_percentiles", "optimal_crps_percentiles")
)
def test_rebadging(tmp_path, identifier):
    """Test rebadging realizations as percentiles."""
    kgo_dir = acc.kgo_root() / "generate-percentiles/basic"
    kgo_path = kgo_dir / "rebadging" / f"{identifier}_kgo.nc"
    perc_input = kgo_dir / "input.nc"
    output_path = tmp_path / "output.nc"
    args = [
        perc_input,
        "--output",
        output_path,
    ]
    if identifier == "optimal_crps_percentiles":
        args += ["--optimal-crps-percentiles"]

    run_cli(args)
    acc.compare(output_path, kgo_path)
