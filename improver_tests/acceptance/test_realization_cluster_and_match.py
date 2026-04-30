# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Tests for the realization-cluster-and-match CLI"""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)

pytest.importorskip("esmf_regrid")
pytest.importorskip("kmedoids")


def test_single_lead_time(tmp_path):
    """Test with a single lead time and two input cubes."""
    kgo_dir = (
        acc.kgo_root() / "realization-cluster-and-match" / "two_forecast_period_input"
    )
    central_dir = acc.kgo_root() / "realization-cluster-and-match"
    kgo_path = kgo_dir / "single_lead_time_kgo.nc"
    coarse_resolution_primary_input = (
        kgo_dir / "coarse_resolution_primary_subdomain_PT0006H00M.nc"
    )
    high_resolution_secondary_input = (
        kgo_dir / "high_resolution_secondary_subdomain_PT0006H00M.nc"
    )
    target_grid_input = kgo_dir / "target_grid.nc"
    hierarchy_input = kgo_dir / "hierarchy.json"
    clustering_kwargs_input = central_dir / "clustering_kwargs.json"
    output_path = tmp_path / "output.nc"
    args = [
        coarse_resolution_primary_input,
        high_resolution_secondary_input,
        target_grid_input,
        "--hierarchy",
        hierarchy_input,
        "--n-clusters",
        "2",
        "--model-id-attr",
        "mosg__model_configuration",
        "--target-grid-name",
        "target_grid",
        "--renumber-primary-realizations",
        "False",
        "--clustering-kwargs",
        clustering_kwargs_input,
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_two_lead_times(tmp_path):
    """Test with two lead times and four input cubes."""
    central_dir = acc.kgo_root() / "realization-cluster-and-match"
    kgo_dir = central_dir / "two_forecast_period_input"
    kgo_path = kgo_dir / "two_lead_times_kgo.nc"
    coarse_resolution_primary_6H_input = (
        kgo_dir / "coarse_resolution_primary_subdomain_PT0006H00M.nc"
    )
    high_resolution_secondary_6H_input = (
        kgo_dir / "high_resolution_secondary_subdomain_PT0006H00M.nc"
    )
    coarse_resolution_primary_12H_input = (
        kgo_dir / "coarse_resolution_primary_subdomain_PT0012H00M.nc"
    )
    high_resolution_secondary_12H_input = (
        kgo_dir / "high_resolution_secondary_subdomain_PT0012H00M.nc"
    )
    target_grid_input = kgo_dir / "target_grid.nc"
    hierarchy_input = kgo_dir / "hierarchy.json"
    clustering_kwargs_input = central_dir / "clustering_kwargs.json"
    output_path = tmp_path / "output.nc"
    args = [
        coarse_resolution_primary_6H_input,
        high_resolution_secondary_6H_input,
        coarse_resolution_primary_12H_input,
        high_resolution_secondary_12H_input,
        target_grid_input,
        "--hierarchy",
        hierarchy_input,
        "--n-clusters",
        "2",
        "--model-id-attr",
        "mosg__model_configuration",
        "--target-grid-name",
        "target_grid",
        "--renumber-primary-realizations",
        "False",
        "--clustering-kwargs",
        clustering_kwargs_input,
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_three_lead_times(tmp_path):
    """Test with three lead times and six input cubes."""
    central_dir = acc.kgo_root() / "realization-cluster-and-match"
    kgo_dir = central_dir / "three_forecast_period_input"
    kgo_path = kgo_dir / "three_lead_times_kgo.nc"
    coarse_resolution_primary_78H_input = (
        kgo_dir / "coarse_resolution_primary_subdomain_PT0078H00M.nc"
    )
    high_resolution_secondary_78H_input = (
        kgo_dir / "high_resolution_secondary_subdomain_PT0078H00M.nc"
    )
    coarse_resolution_primary_81H_input = (
        kgo_dir / "coarse_resolution_primary_subdomain_PT0081H00M.nc"
    )
    high_resolution_secondary_81H_input = (
        kgo_dir / "high_resolution_secondary_subdomain_PT0081H00M.nc"
    )
    coarse_resolution_primary_84H_input = (
        kgo_dir / "coarse_resolution_primary_subdomain_PT0084H00M.nc"
    )
    high_resolution_secondary_84H_input = (
        kgo_dir / "high_resolution_secondary_subdomain_PT0084H00M.nc"
    )
    target_grid_input = kgo_dir / "target_grid.nc"
    hierarchy_input = kgo_dir / "hierarchy.json"
    clustering_kwargs_input = central_dir / "clustering_kwargs.json"
    output_path = tmp_path / "output.nc"
    args = [
        coarse_resolution_primary_78H_input,
        high_resolution_secondary_78H_input,
        coarse_resolution_primary_81H_input,
        high_resolution_secondary_81H_input,
        coarse_resolution_primary_84H_input,
        high_resolution_secondary_84H_input,
        target_grid_input,
        "--hierarchy",
        hierarchy_input,
        "--n-clusters",
        "2",
        "--model-id-attr",
        "mosg__model_configuration",
        "--target-grid-name",
        "target_grid",
        "--clustering-kwargs",
        clustering_kwargs_input,
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)
