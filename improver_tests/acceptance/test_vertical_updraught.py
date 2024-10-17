# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Tests for the vertical-updraught CLI"""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


@pytest.mark.parametrize("model_id_attr", (True, False))
def test_basic(tmp_path, model_id_attr):
    """Test vertical-updraught usage, with and without model_id_attr"""
    test_dir = acc.kgo_root() / "vertical-updraught"
    output_path = tmp_path / "output.nc"
    args = [
        test_dir / "cape.nc",
        test_dir / "precip_rate_max.nc",
        "--least-significant-digit",
        "2",
        "--output",
        output_path,
    ]
    if model_id_attr:
        args += ["--model-id-attr", "mosg__model_configuration"]
        kgo_dir = test_dir / "with_id_attr"
    else:
        kgo_dir = test_dir / "without_id_attr"
    kgo_path = kgo_dir / "kgo.nc"
    run_cli(args)
    acc.compare(output_path, kgo_path)
