# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Tests for the turbulence CLI"""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
# Note the possibly undocumented behavior to have a CLI that is a coercion of a file name, changing it from underscores
# to dashes. This also means that there must be a file in the CLI folder (../improver/cli) with this exact name.
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


@pytest.mark.parametrize("with_model_attr", (True, False))
def test_basic(tmp_path, with_model_attr):
    """Test basic invocation"""

    kgo_dir = acc.kgo_root() / "turbulence-index-above-1500m-usaf"
    kgo_path = kgo_dir / "kgo.nc"
    output_path = tmp_path / "output.nc"

    args = [
        kgo_dir / "UWindComponentAt550mb.nc",
        kgo_dir / "VWindComponentAt550mb.nc",
        kgo_dir / "UWindComponentAt500mb.nc",
        kgo_dir / "VWindComponentAt500mb.nc",
        kgo_dir / "GeopotentialHeightAt550.nc",
        kgo_dir / "GeopotentialHeightAt500.nc",
        "--output",
        f"{output_path}",
    ]

    run_cli(args)
    acc.compare(output_path, kgo_path)
