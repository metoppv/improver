# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""
Tests for the sleet-probability CLI
"""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


def test_basic(tmp_path):
    """Test basic snow falling level calculation"""
    kgo_dir = acc.kgo_root() / "sleet_probability/basic"
    kgo_path = kgo_dir / "kgo.nc"
    output_path = tmp_path / "output.nc"
    half_input_path = kgo_dir / "half_prob_snow_falling_level.nc"
    tenth_input_path = kgo_dir / "tenth_prob_snow_falling_level.nc"
    args = [half_input_path, tenth_input_path, "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)
