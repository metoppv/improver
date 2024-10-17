# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Tests for the clip CLI"""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


@pytest.mark.parametrize("cube_type", ("spot_data", "gridded_data"))
@pytest.mark.parametrize("min_value,max_value", ((0, 4000), (None, 6000), (1000, None)))
def test_basic(tmp_path, min_value, max_value, cube_type):
    """Test clip functionality with different combinations of min and max values"""
    kgo_dir = acc.kgo_root() / "clip" / cube_type
    kgo_path = kgo_dir / f"kgo_{min_value}_{max_value}.nc"
    output_path = tmp_path / "output.nc"

    args = [kgo_dir / "input.nc", "--output", output_path]
    if min_value is not None:
        args.append(f"--min-value={min_value}")
    if max_value is not None:
        args.append(f"--max-value={max_value}")

    run_cli(args)
    acc.compare(output_path, kgo_path)
