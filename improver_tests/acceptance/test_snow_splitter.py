# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of improver and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Tests for the snow-splitter CLI"""

import iris
import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


@pytest.mark.parametrize(
    "output_variable,output_is_rain", (("snow", "False"), ("rain", "True"))
)
def test_basic(tmp_path, output_variable, output_is_rain):
    """Test snow-splitter if desired output is rain or snow."""
    test_dir = acc.kgo_root() / "snow-splitter"
    input_paths = iris.cube.CubeList(
        [test_dir / f"{input}.nc" for input in ("rain", "snow", "precip_rate")]
    )
    output_path = tmp_path / "output.nc"
    args = [*input_paths, "--output-is-rain", output_is_rain, "--output", output_path]

    kgo_path = test_dir / f"{output_variable}_kgo.nc"
    run_cli(args)
    acc.compare(output_path, kgo_path)
