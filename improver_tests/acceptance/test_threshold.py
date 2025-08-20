# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""
Tests for the threshold CLI
"""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


@pytest.mark.parametrize(
    "extra_args,kgo_subdir",
    (
        (["--threshold-values", "280"], "basic"),
        (
            ["--threshold-values", "280", "--comparison-operator", "<="],
            "below_threshold",
        ),
        (["--threshold-values", "270,280,290"], "multiple_thresholds"),
        (
            ["--threshold-values", "6.85", "--threshold-units", "celsius"],
            "threshold_units",
        ),
        (["--threshold-values", "280", "--fuzzy-factor", "0.99"], "fuzzy_factor"),
        (
            [
                "--threshold-config",
                acc.kgo_root() / "threshold" / "fuzzy_bounds" / "threshold_config.json",
            ],
            "fuzzy_factor",
        ),
        (
            [
                "--threshold-values",
                "6.85",
                "--threshold-units",
                "celsius",
                "--fuzzy-factor",
                "0.2",
            ],
            "threshold_units_fuzzy_factor",
        ),
        (
            [
                "--threshold-config",
                acc.kgo_root() / "threshold" / "json" / "threshold_config.json",
            ],
            "basic",
        ),
        (
            [
                "--threshold-values",
                "280",
                "--collapse-coord",
                "realization",
                "--collapse-cell-methods",
                acc.kgo_root() / "threshold" / "cell_method" / "cell_method.json",
            ],
            "cell_method",
        ),
    ),
)
def test_args(tmp_path, extra_args, kgo_subdir):
    """Test thresholding with different argument combinations using temperature data"""
    cli_dir = acc.kgo_root() / "threshold"
    kgo_dir = cli_dir / kgo_subdir
    kgo_path = kgo_dir / "kgo.nc"
    input_path = cli_dir / "basic" / "input.nc"
    output_path = tmp_path / "output.nc"
    args = [input_path, "--output", output_path]
    if extra_args:
        args += extra_args
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_collapse_realization(tmp_path):
    """Test thresholding with collapsing realizations. Ensure that in this case,
    using unmasked data, that no warning is raised in relation to collapsing the
    coordinate."""
    kgo_dir = acc.kgo_root() / "threshold/coord_collapse"
    kgo_path = kgo_dir / "kgo.nc"
    input_path = kgo_dir / "../basic/input.nc"
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        "--output",
        output_path,
        "--threshold-values",
        "280",
        "--collapse-coord",
        "realization",
    ]
    with pytest.warns() as record:
        run_cli(args)
    for msg in record:
        assert "Blending masked data without spatial" not in str(msg.message)
    acc.compare(output_path, kgo_path)


@pytest.mark.parametrize(
    "extra_arg,kgo", (([], "kgo.nc"), (["--fill-masked", "inf"], "kgo_mask_filled.nc"))
)
def test_collapse_realization_masked_data(tmp_path, extra_arg, kgo):
    """Test thresholding and collapsing realizations where the data being
    thresholded is masked."""
    kgo_dir = acc.kgo_root() / "threshold/masked_collapse"
    kgo_path = kgo_dir / kgo
    input_path = kgo_dir / "input.nc"
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        "--output",
        output_path,
        "--threshold-values",
        "500",
        "--collapse-coord",
        "realization",
    ]
    args += extra_arg
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_collapse_percentile(tmp_path):
    """Test thresholding with collapsing percentiles."""
    kgo_dir = acc.kgo_root() / "threshold/percentile_collapse"
    kgo_path = kgo_dir / "kgo.nc"
    input_path = kgo_dir / "input.nc"
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        "--output",
        output_path,
        "--threshold-values",
        "5",
        "--collapse-coord",
        "percentile",
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


@pytest.mark.parametrize(
    "extra_args,kgo",
    (
        (["--vicinity", "10000"], "kgo.nc"),
        (
            ["--vicinity", "10000,20000", "--collapse-coord", "realization"],
            "kgo_multiple_vicinities.nc",
        ),
        (
            ["--vicinity", "10000", "--collapse-coord", "realization"],
            "kgo_collapsed.nc",
        ),
        (
            [
                acc.kgo_root() / "threshold" / "vicinity" / "landmask.nc",
                "--vicinity",
                "10000",
            ],
            "kgo_landmask.nc",
        ),
        (
            [
                acc.kgo_root() / "threshold" / "vicinity" / "landmask.nc",
                "--vicinity",
                "10000",
                "--collapse-coord",
                "realization",
            ],
            "kgo_landmask_collapsed.nc",
        ),
    ),
)
def test_vicinity(tmp_path, extra_args, kgo):
    """Test thresholding with vicinity"""
    kgo_dir = acc.kgo_root() / "threshold/vicinity"
    kgo_path = kgo_dir / kgo
    input_path = kgo_dir / "input.nc"
    output_path = tmp_path / "output.nc"
    args = [input_path]
    if extra_args:
        args += extra_args
    args += [
        "--output",
        output_path,
        "--threshold-values",
        "0.03,0.1,1.0",
        "--threshold-units",
        "mm hr-1",
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_vicinity_masked(tmp_path):
    """Test thresholding with vicinity and masked precipitation"""
    kgo_dir = acc.kgo_root() / "threshold/vicinity"
    kgo_path = kgo_dir / "kgo_masked.nc"
    input_path = kgo_dir / "masked_precip.nc"
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        "--output",
        output_path,
        "--threshold-values",
        "0.03,0.1,1.0",
        "--threshold-units",
        "mm hr-1",
        "--vicinity",
        "10000",
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_nowcast_specific(tmp_path):
    """Test thresholding nowcast data using multiple vicinities and masked
    precipitation"""
    kgo_dir = acc.kgo_root() / "threshold/nowcast"
    kgo_path = kgo_dir / "kgo_masked.nc"
    input_path = kgo_dir / "masked_precip.nc"
    threshold_config = kgo_dir / "precip_accumulation_thresholds.json"
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        "--output",
        output_path,
        "--threshold-config",
        threshold_config,
        "--threshold-units",
        "mm",
        "--vicinity",
        "25000,50000",
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)
