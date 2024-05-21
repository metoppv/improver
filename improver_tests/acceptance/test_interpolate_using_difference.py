# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""
Tests for the interpolate-using-difference CLI
"""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


def test_filling_without_limit(tmp_path):
    """Test filling masked areas using difference interpolation, with no limits
    on the returned values."""
    kgo_dir = acc.kgo_root() / f"{CLI}/basic"
    kgo_path = kgo_dir / "sleet_rain_unlimited_kgo.nc"
    output_path = tmp_path / "output.nc"
    input_paths = [kgo_dir / x for x in ("sleet_rain_unfilled.nc", "snow_sleet.nc")]
    args = [*input_paths, "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_filling_with_maximum_limit(tmp_path):
    """Test filling masked areas using difference interpolation, with an
    orography field used to limit the maximum of returned values."""
    kgo_dir = acc.kgo_root() / f"{CLI}/basic"
    kgo_path = kgo_dir / "sleet_rain_max_limited_kgo.nc"
    output_path = tmp_path / "output.nc"
    input_paths = [
        kgo_dir / x for x in ("sleet_rain_unfilled.nc", "snow_sleet.nc", "orog.nc")
    ]
    args = [*input_paths, "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_filling_with_minimum_limit(tmp_path):
    """Test filling masked areas using difference interpolation, with an
    orography field used to limit the minimum of returned values."""
    kgo_dir = acc.kgo_root() / f"{CLI}/basic"
    kgo_path = kgo_dir / "sleet_rain_min_limited_kgo.nc"
    output_path = tmp_path / "output.nc"
    input_paths = [
        kgo_dir / x for x in ("sleet_rain_unfilled.nc", "snow_sleet.nc", "orog.nc")
    ]
    args = [*input_paths, "--limit-as-maximum", "False", "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_filling_with_nearest_use(tmp_path):
    """Test filling masked areas using difference interpolation, in this case
    with a hole in the corner of the data that requires use of nearest
    neighbour interpolation."""
    pytest.importorskip("stratify")
    kgo_dir = acc.kgo_root() / f"{CLI}/basic"
    kgo_path = kgo_dir / "sleet_rain_nearest_filled_kgo.nc"
    output_path = tmp_path / "output.nc"
    input_paths = [
        kgo_dir / x for x in ("sleet_rain_unfilled_corner.nc", "snow_sleet.nc")
    ]
    args = [*input_paths, "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)
