# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of improver and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Tests for the freezing-rain CLI"""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


@pytest.mark.parametrize("period", ["one", "three"])
def test_period_results(tmp_path, period):
    """Test freezing rain calculation returns expected result using rain and
    sleet accumulations, in combination with a minimum temperature over the
    accumulation period."""
    kgo_dir = acc.kgo_root() / CLI / f"{period}_hour"
    kgo_path = kgo_dir / "kgo.nc"
    rain_input_path = kgo_dir / "rain_acc.nc"
    sleet_input_path = kgo_dir / "sleet_acc.nc"
    temperature_input_path = kgo_dir / "temperature_min.nc"
    output_path = tmp_path / "output.nc"
    args = [
        rain_input_path,
        sleet_input_path,
        temperature_input_path,
        "--model-id-attr",
        "mosg__model_configuration",
        "--output",
        f"{output_path}",
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)
