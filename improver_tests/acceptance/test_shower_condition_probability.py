# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Tests for the shower-condition-probability CLI"""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


def test_basic(tmp_path):
    """Test shower condition probability is generated using global model
    derived data."""

    kgo_dir = acc.kgo_root() / "shower-condition-probability/"
    kgo_path = kgo_dir / "kgo.nc"
    inputs = [kgo_dir / f for f in ["cloud_input.nc", "convection_input.nc"]]
    output_path = tmp_path / "output.nc"

    args = [
        *inputs,
        "--output",
        output_path,
        "--cloud-threshold",
        "0.8125",
        "--convection-threshold",
        "0.8",
        "--model-id-attr",
        "mosg__model_configuration",
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)
