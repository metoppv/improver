# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Tests for the lightning_from_cape_and_precip CLI"""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


@pytest.mark.parametrize("with_model_attr", (True, False))
def test_basic(tmp_path, with_model_attr):
    """Test basic invocation"""
    kgo_dir = acc.kgo_root() / "lightning-from-cape-and-precip"
    kgo_path = kgo_dir / "kgo.nc"
    cape_path = kgo_dir / "cape.nc"
    precipitation_path = kgo_dir / "precipitation_rate.nc"
    output_path = tmp_path / "output.nc"
    args = [
        cape_path,
        precipitation_path,
        "--output",
        f"{output_path}",
    ]
    if with_model_attr:
        args.append("--model-id-attr=mosg__model_configuration")
        kgo_path = kgo_dir / "kgo_with_model_config.nc"

    run_cli(args)
    acc.compare(output_path, kgo_path)
