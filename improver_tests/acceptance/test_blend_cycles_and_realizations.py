# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""
Tests for the blend-cycles-and-realizations CLI
"""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


def test_basic(tmp_path):
    """Test basic usage"""
    kgo_dir = acc.kgo_root() / f"{CLI}/basic"
    kgo_path = kgo_dir / "kgo.nc"
    input_dir = kgo_dir
    input_paths = sorted((input_dir.glob("??00Z_precip_rate.nc")))
    output_path = tmp_path / "output.nc"
    args = [
        "--cycletime",
        "20190101T1000Z",
        *input_paths,
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)
