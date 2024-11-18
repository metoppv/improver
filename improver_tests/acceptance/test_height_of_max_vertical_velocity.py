# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Tests the height_of_max_vertical_velocity cli"""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


def test_basic(tmp_path):
    """Test height_of_max_vertical_velocity computation"""
    kgo_dir = acc.kgo_root() / "height-of-max-vertical-velocity"
    input_file1 = kgo_dir / "vertical_velocity_on_height_levels.nc"
    input_file2 = kgo_dir / "max_vertical_velocity.nc"
    output_path = tmp_path / "output.nc"
    args = [input_file1, input_file2, "--output", f"{output_path}"]

    kgo_path = kgo_dir / "kgo.nc"
    run_cli(args)
    acc.compare(output_path, kgo_path)
