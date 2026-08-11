# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Tests for the realization-selection CLI"""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


@pytest.mark.parametrize("use_selection_attr", [True, False])
def test_two_forecast_sources(tmp_path, use_selection_attr):
    """Test realization selection for a single lead time."""
    kgo_dir = acc.kgo_root() / "realization-selection"
    kgo_path = (
        kgo_dir / "kgo_with_attr.nc" if use_selection_attr else kgo_dir / "kgo.nc"
    )
    forecast_input1 = kgo_dir / "coarse_resolution_primary_subdomain_PT0079H00M.nc"
    forecast_input2 = kgo_dir / "high_resolution_secondary_subdomain_PT0079H00M.nc"
    cluster_cube_input = kgo_dir / "three_lead_times_input.nc"
    output_path = tmp_path / "output.nc"
    args = [
        forecast_input1,
        forecast_input2,
        cluster_cube_input,
        "--forecast-period",
        "284400",  # 79 hours in seconds
        "--model-id-attr",
        "mosg__model_configuration",
        "--output",
        output_path,
    ]
    if use_selection_attr:
        args.extend(
            [
                "--selection-attr",
                "realization_selection_method",
                "--selection-attr-value",
                "cluster_medoid",
            ]
        )

    run_cli(args)
    acc.compare(output_path, kgo_path)
