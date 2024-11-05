# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""
Tests for the copy-attributes CLI
"""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


def test_change_metadata(tmp_path):
    """Test copying attribute values from a template file"""
    kgo_dir = acc.kgo_root() / "copy-metadata"
    kgo_path = kgo_dir / "kgo.nc"
    input_path = kgo_dir / "input.nc"
    template_path = kgo_dir / "stage_input.nc"
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        template_path,
        "--attributes",
        "mosg__forecast_run_duration,mosg__grid_version",
        "--aux-coord",
        "wind_speed status_flag",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)
