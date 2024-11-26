# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Tests for the snow-fraction CLI"""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


@pytest.mark.parametrize("input_names", (("rain", "snow"), ("snow", "rain")))
def test_basic(tmp_path, input_names):
    """Test basic snow-fraction calculation with inputs in either order"""
    kgo_dir = acc.kgo_root() / CLI / "basic"
    kgo_path = kgo_dir / "kgo.nc"
    first_input_path = kgo_dir / f"{input_names[0]}.nc"
    second_input_path = kgo_dir / f"{input_names[1]}.nc"
    output_path = tmp_path / "output.nc"
    args = [
        first_input_path,
        second_input_path,
        "--model-id-attr",
        "mosg__model_configuration",
        "--output",
        f"{output_path}",
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)
