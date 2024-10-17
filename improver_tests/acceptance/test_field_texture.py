# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Tests for the field-texture CLI"""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


def test_basic(tmp_path):
    """Test field texture operation with default arguments."""

    kgo_dir = acc.kgo_root() / "field-texture/basic"
    kgo_path = kgo_dir / "kgo.nc"
    input_path = kgo_dir / "input.nc"
    output_path = tmp_path / "output.nc"

    args = [
        input_path,
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_args(tmp_path):
    """Tests field texture operation with defined arguments"""

    kgo_dir = acc.kgo_root() / "field-texture/args"
    kgo_path = kgo_dir / "kgo.nc"
    input_path = kgo_dir / "../basic/input.nc"
    output_path = tmp_path / "output.nc"

    args = [
        input_path,
        "--nbhood-radius=5000.0",
        "--textural-threshold=0.05",
        "--diagnostic-threshold=0.6145",
        "--model-id-attr=mosg__model_configuration",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)
