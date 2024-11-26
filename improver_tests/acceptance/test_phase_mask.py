# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Tests for the snow-fraction CLI"""

import iris
import numpy as np
import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


@pytest.mark.parametrize("phase", ("rain", "sleet", "snow"))
def test_basic(tmp_path, phase):
    """Test basic phase-mask calculation"""
    kgo_dir = acc.kgo_root() / CLI
    kgo_path = kgo_dir / phase / "kgo.nc"
    snow_fraction_path = kgo_dir / "snow_fraction.nc"
    output_path = tmp_path / "output.nc"
    args = [
        snow_fraction_path,
        phase,
        "--model-id-attr",
        "mosg__model_configuration",
        "--output",
        f"{output_path}",
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_kgos():
    """Ensure the KGO for the three phases total exactly 1 everywhere"""
    kgo_dir = acc.kgo_root() / CLI
    rain_kgo = kgo_dir / "rain" / "kgo.nc"
    sleet_kgo = kgo_dir / "sleet" / "kgo.nc"
    snow_kgo = kgo_dir / "snow" / "kgo.nc"
    rain = iris.load_cube(str(rain_kgo))
    sleet = iris.load_cube(str(sleet_kgo))
    snow = iris.load_cube(str(snow_kgo))
    total = rain.data + sleet.data + snow.data
    expected = np.ones_like(total)
    assert np.allclose(total, expected)
