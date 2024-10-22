# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""
Tests for the apply-mask CLI
"""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


@pytest.mark.parametrize("invert", [True, False])
def test_apply_mask(tmp_path, invert):
    """Test apply-mask CLI."""
    kgo_dir = acc.kgo_root() / "apply-mask/"
    kgo_path = kgo_dir / "kgo.nc"
    if invert:
        kgo_path = kgo_dir / "kgo_inverted.nc"

    output_path = tmp_path / "output.nc"
    args = [
        kgo_dir / "wind_speed.nc",
        kgo_dir / "mask.nc",
        "--mask-name",
        "land_binary_mask",
        "--invert-mask",
        f"{invert}",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)
