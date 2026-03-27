# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Tests for the deterministic_realization_selector CLI."""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


def test_deterministic_realization(tmp_path):
    """Test with an available target realization."""
    cube_dir = acc.kgo_root() / "deterministic_realization_selection"
    kgo_path = cube_dir / "deterministic_realization_selection_kgo.nc"
    fcst_path = cube_dir / "forecast_input_cube.nc"
    cluster_path = cube_dir / "cluster_input_cube.nc"

    output_path = tmp_path / "output.nc"

    args = [
        fcst_path,
        cluster_path,
        "--target-realization-id",
        "0",
        "--attribute",
        "primary_input_realizations_to_clusters",
        "--output",
        output_path,
    ]

    run_cli(args)
    acc.compare(output_path, kgo_path)
