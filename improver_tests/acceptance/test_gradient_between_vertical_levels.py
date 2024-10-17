# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Tests for the gradient-between-vertical-levels CLI"""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


def test_gradient_vertical_levels(tmp_path):
    """Test gradient between vertical levels calculation returns expected result using
    temperature at screen level and 850hPa, in combination with orography and height of
    pressure levels."""
    kgo_dir = acc.kgo_root() / CLI
    kgo_path = kgo_dir / "kgo.nc"
    temp_at_screen = kgo_dir / "temperature_at_screen_level.nc"
    temp_at_850 = kgo_dir / "temperature_at_850hpa.nc"
    orography = kgo_dir / "orography.nc"
    height_of_pressure_levels = kgo_dir / "height_of_pressure_levels.nc"

    output_path = tmp_path / "output.nc"
    args = [
        temp_at_screen,
        temp_at_850,
        orography,
        height_of_pressure_levels,
        "--output",
        f"{output_path}",
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)
