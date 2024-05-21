# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""
Tests for the orographic-enhancement CLI
"""

import pytest

from improver.constants import LOOSE_TOLERANCE

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
OE = "orographic_enhancement_high_resolution"
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


@pytest.mark.slow
def test_basic(tmp_path):
    """Test basic orographic enhancement"""
    kgo_dir = acc.kgo_root() / "orographic_enhancement/basic"
    kgo_path = kgo_dir / "kgo_hi_res.nc"
    input_args = [
        kgo_dir / f"{param}.nc"
        for param in (
            "temperature",
            "humidity",
            "pressure",
            "wind_speed",
            "wind_direction",
            "orography_uk-standard_1km",
        )
    ]

    output_path = tmp_path / "output.nc"

    args = [*input_args, "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path, rtol=LOOSE_TOLERANCE)


@pytest.mark.slow
def test_boundary_height(tmp_path):
    """Test orographic enhancement with specified boundary height"""
    kgo_dir = acc.kgo_root() / "orographic_enhancement/boundary_height"
    kgo_path = kgo_dir / "kgo_hi_res.nc"
    input_dir = kgo_dir / "../basic"
    input_args = [
        input_dir / f"{param}.nc"
        for param in (
            "temperature",
            "humidity",
            "pressure",
            "wind_speed",
            "wind_direction",
            "orography_uk-standard_1km",
        )
    ]

    output_path = tmp_path / "output.nc"

    args = [
        *input_args,
        "--boundary-height=500.",
        "--boundary-height-units=m",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path, rtol=LOOSE_TOLERANCE)


@pytest.mark.slow
def test_boundary_height_units(tmp_path):
    """Test orographic enhancement with boundary height unit conversion"""
    kgo_dir = acc.kgo_root() / "orographic_enhancement/boundary_height"
    kgo_path = kgo_dir / "kgo_hi_res.nc"
    input_dir = kgo_dir / "../basic"
    input_args = [
        input_dir / f"{param}.nc"
        for param in (
            "temperature",
            "humidity",
            "pressure",
            "wind_speed",
            "wind_direction",
            "orography_uk-standard_1km",
        )
    ]

    output_path = tmp_path / "output.nc"

    args = [
        *input_args,
        "--boundary-height=1640.41994751",
        "--boundary-height-units=ft",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path, rtol=LOOSE_TOLERANCE)


def test_invalid_boundary_height(tmp_path):
    """Test excessively high boundary height"""
    kgo_dir = acc.kgo_root() / "orographic_enhancement/boundary_height"
    input_dir = kgo_dir / "../basic"
    input_args = [
        input_dir / f"{param}.nc"
        for param in (
            "temperature",
            "humidity",
            "pressure",
            "wind_speed",
            "wind_direction",
            "orography_uk-standard_1km",
        )
    ]

    output_path = tmp_path / "output.nc"

    args = [
        *input_args,
        "--boundary-height=500000.",
        "--boundary-height-units=m",
        "--output",
        output_path,
    ]
    with pytest.raises(ValueError, match=".*height.*"):
        run_cli(args)
