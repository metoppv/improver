# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of improver and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Tests for the max_in_time_window CLI"""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


def test_basic(tmp_path):
    """Test max_in_time_window computation."""
    kgo_dir = acc.kgo_root() / "max-in-time-window"
    kgo_path = kgo_dir / "kgo.nc"
    input_path1 = kgo_dir / "input_PT0029H00M.nc"
    input_path2 = kgo_dir / "input_PT0032H00M.nc"
    output_path = tmp_path / "output.nc"
    args = [input_path1, input_path2, "--output", f"{output_path}"]
    run_cli(args)
    acc.compare(output_path, kgo_path)


@pytest.mark.parametrize("realizations, gives_error", ((4, False), (5, True)))
def test_minimum_realizations(tmp_path, realizations, gives_error):
    """Test specifying a minimum number of realizations."""
    kgo_dir = acc.kgo_root() / "max-in-time-window"
    kgo_path = kgo_dir / "kgo.nc"
    input_path1 = kgo_dir / "input_PT0029H00M.nc"
    input_path2 = kgo_dir / "input_PT0032H00M.nc"
    output_path = tmp_path / "output.nc"
    args = [
        input_path1,
        input_path2,
        "--minimum-realizations",
        f"{realizations}",
        "--output",
        f"{output_path}",
    ]
    if gives_error:
        with pytest.raises(
            ValueError,
            match="After filtering, number of realizations 4 is less than the minimum number "
            rf"of realizations allowed \({realizations}\)",
        ):
            run_cli(args)
    else:
        run_cli(args)
        acc.compare(output_path, kgo_path)
