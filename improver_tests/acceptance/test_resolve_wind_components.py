# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of improver and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""
Tests for the resolve-wind-components CLI
"""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
run_cli = acc.run_cli("resolve-wind-components")

WSPD = "wind_speed_on_pressure_levels"
WDIR = "wind_direction_on_pressure_levels"


@pytest.mark.slow
def test_basic(tmp_path):
    """Test basic wind speed/direction to u/v vector conversion"""
    kgo_dir = acc.kgo_root() / "resolve-wind-components/basic"
    kgo_path = kgo_dir / "kgo.nc"
    wspd_path = kgo_dir / f"20181103T1600Z-PT0001H00M-{WSPD}.nc"
    wdir_path = kgo_dir / f"20181103T1600Z-PT0001H00M-{WDIR}.nc"
    output_path = tmp_path / "output.nc"
    args = [wspd_path, wdir_path, "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path, rtol=0.0)
