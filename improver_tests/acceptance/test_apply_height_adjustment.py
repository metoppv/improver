# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Tests for the apply-height-adjustment CLI"""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


@pytest.mark.parametrize("input_type", ("prob", "realization"))
def test_basic(tmp_path, input_type):
    """Test apply-height-adjustment for a probability and realization input cube"""
    kgo_dir = acc.kgo_root() / "apply-height-adjustment/"
    kgo_path = kgo_dir / f"kgo_{input_type}.nc"
    output_path = tmp_path / "output.nc"
    args = [
        kgo_dir / f"input_{input_type}.nc",
        kgo_dir / "neighbours.nc",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)
