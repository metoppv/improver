# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
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
