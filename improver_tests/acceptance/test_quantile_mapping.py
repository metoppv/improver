# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Tests for the quantile-mapping CLI"""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


@pytest.mark.parametrize("method", ["step", "continuous"])
def test_step_no_threshold(tmp_path, method):
    """Test quantile mapping with step method and no preservation threshold."""
    kgo_dir = acc.kgo_root() / "quantile-mapping/basic/"
    kgo_path = kgo_dir / f"kgo_{method}.nc"
    reference_path = acc.kgo_root() / "quantile-mapping/reference.nc"
    forecast_path = acc.kgo_root() / "quantile-mapping/forecast.nc"
    output_path = tmp_path / "output.nc"

    args = [
        reference_path,
        forecast_path,
        "--method",
        method,
        "--reference-attribute",
        "realization_selection_method=cluster_medoid",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


@pytest.mark.parametrize("method", ["step", "continuous"])
def test_step_with_preservation_threshold(tmp_path, method):
    """Test quantile mapping with step method and preservation threshold."""
    kgo_dir = acc.kgo_root() / "quantile-mapping/with_preservation_threshold/"
    kgo_path = kgo_dir / f"kgo_{method}.nc"
    reference_path = acc.kgo_root() / "quantile-mapping/reference.nc"
    forecast_path = acc.kgo_root() / "quantile-mapping/forecast.nc"
    output_path = tmp_path / "output.nc"

    args = [
        reference_path,
        forecast_path,
        "--method",
        method,
        "--preservation-threshold",
        "0.00003",
        "--reference-attribute",
        "realization_selection_method=cluster_medoid",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


@pytest.mark.parametrize("method", ["step", "continuous"])
def test_step_with_occurrence_threshold(tmp_path, method):
    """Test quantile mapping with step method and occurrence threshold."""
    kgo_dir = acc.kgo_root() / "quantile-mapping/with_occurrence_threshold/"
    kgo_path = kgo_dir / f"kgo_{method}.nc"
    reference_path = acc.kgo_root() / "quantile-mapping/reference.nc"
    forecast_path = acc.kgo_root() / "quantile-mapping/forecast.nc"
    output_path = tmp_path / "output.nc"

    args = [
        reference_path,
        forecast_path,
        "--method",
        method,
        "--occurrence-threshold",
        "0.00003",
        "--reference-attribute",
        "realization_selection_method=cluster_medoid",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)
