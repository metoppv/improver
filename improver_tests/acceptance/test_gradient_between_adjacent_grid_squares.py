# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Tests for the gradient between adjacent grid squares CLI"""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


@pytest.mark.parametrize("regrid", (True, False))
def test_basic(tmp_path, regrid):
    """Test gradient between adjacent grid squares plugin, with and without regridding"""
    test_dir = acc.kgo_root() / "gradient-between-adjacent-grid-squares"
    output_path = tmp_path / "output.nc"
    args = [
        test_dir / "input.nc",
        "--output",
        output_path,
    ]
    if regrid:
        args += ["--regrid"]
        kgo_dir = test_dir / "with_regrid"
    else:
        kgo_dir = test_dir / "without_regrid"
    kgo_path = kgo_dir / "kgo.nc"
    run_cli(args)
    acc.compare(output_path, kgo_path)
