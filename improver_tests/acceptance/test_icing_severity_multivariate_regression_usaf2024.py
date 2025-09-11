# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Tests for the lightning_usaf"""

import pytest

from . import acceptance as acc

#pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


@pytest.mark.parametrize("with_model_attr", (True, False))
def test_basic(tmp_path, with_model_attr):
    """Test basic invocation"""
    kgo_dir = acc.kgo_root() / "icing-severity-multivariate-regression-usaf2024" 
    kgo_path = kgo_dir / "kgo.nc"
    output_path = tmp_path / "output.nc"
    
    args = [
        kgo_dir / "rh.nc",
        kgo_dir / "t.nc",
        "--output",
        f"{output_path}",
    ]



    run_cli(args)
    acc.compare(output_path, kgo_path)
