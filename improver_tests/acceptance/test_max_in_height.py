# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Tests the max_in_height CLI"""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


def test_with_bounds(tmp_path):
    """Test max_in_height computation with specified bounds"""

    kgo_dir = acc.kgo_root() / "max-in-height"
    input_path = kgo_dir / "input.nc"
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        "--upper-height-bound",
        "3000",
        "--lower-height-bound",
        "500",
        "--output",
        f"{output_path}",
    ]

    kgo_path = kgo_dir / "kgo_with_bounds.nc"
    run_cli(args)
    acc.compare(output_path, kgo_path)


@pytest.mark.parametrize("new_name", (None, "max_relative_humidity"))
def test_without_bounds(tmp_path, new_name):
    """Test max_in_height computation without bounds."""

    kgo_dir = acc.kgo_root() / "max-in-height"
    input_path = kgo_dir / "input.nc"
    output_path = tmp_path / "output.nc"
    args = [input_path, "--output", f"{output_path}"]
    if new_name:
        kgo_path = kgo_dir / "kgo_without_bounds_new_name.nc"
        args.extend(["--new-name", f"{new_name}"])
    else:
        kgo_path = kgo_dir / "kgo_without_bounds.nc"
    run_cli(args)
    acc.compare(output_path, kgo_path)
