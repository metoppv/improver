# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""
Tests for the subperiod-selector CLI
"""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


ATTRIBUTES_PATH = acc.kgo_root() / "subperiod-selector/metadata/metadata.json"


def test_basic(tmp_path):
    """Test no mask application to duration is split evenly into shorter
    periods."""
    input_dir = acc.kgo_root() / "subperiod-selector"
    kgo_dir = input_dir / "basic"
    kgo_path = kgo_dir / "kgo.nc"
    input_paths = [
        input_dir / f
        for f in [
            "20250128T0900Z-daily_precipitation.nc",
            "20250127T1200Z-precipitation_accumulation.nc",
            "20250127T1500Z-precipitation_accumulation.nc",
            "20250127T1800Z-precipitation_accumulation.nc",
            "20250127T2100Z-precipitation_accumulation.nc",
            "20250128T0000Z-precipitation_accumulation.nc",
            "20250128T0300Z-precipitation_accumulation.nc",
            "20250128T0600Z-precipitation_accumulation.nc",
            "20250128T0900Z-precipitation_accumulation.nc",
        ]
    ]
    output_path = tmp_path / "output.nc"
    args = input_paths + [
        "--percentile=50",
        "--threshold-kwargs",
        input_dir / "threshold_kwargs.json",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)
