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

METRES_PER_SECOND_TOLERANCE = 1e-9


@pytest.mark.parametrize("example", ["discrete", "overlapping"])
@pytest.mark.parametrize("method", ["step", "continuous"])
def test_step_no_threshold(tmp_path, method, example):
    """Test quantile mapping with step method and no preservation threshold."""
    kgo_dir = acc.kgo_root() / "quantile-mapping/basic/"
    kgo_path = kgo_dir / f"kgo_{method}_{example}.nc"
    reference_path = acc.kgo_root() / f"quantile-mapping/reference_{example}.nc"
    forecast_path = acc.kgo_root() / f"quantile-mapping/forecast_{example}.nc"
    output_path = tmp_path / "output.nc"

    args = [
        reference_path,
        forecast_path,
        "--method",
        method,
        "--output",
        output_path,
    ]
    if example == "overlapping":
        args += [
            "--reference-attribute",
            "realization_selection_method=cluster_medoid",
        ]
    else:
        args += [
            "--reference-attribute",
            "mosg__model_configuration=uk_det",
        ]
    run_cli(args)
    if example == "overlapping":
        acc.compare(output_path, kgo_path)
    else:
        acc.compare(
            output_path,
            kgo_path,
            rtol=METRES_PER_SECOND_TOLERANCE,
            atol=METRES_PER_SECOND_TOLERANCE,
        )


@pytest.mark.parametrize("example", ["discrete", "overlapping"])
@pytest.mark.parametrize("method", ["step", "continuous"])
def test_step_with_preservation_threshold(tmp_path, method, example):
    """Test quantile mapping with step method and preservation threshold."""
    kgo_dir = acc.kgo_root() / "quantile-mapping/with_preservation_threshold/"
    kgo_path = kgo_dir / f"kgo_{method}_{example}.nc"
    reference_path = acc.kgo_root() / f"quantile-mapping/reference_{example}.nc"
    forecast_path = acc.kgo_root() / f"quantile-mapping/forecast_{example}.nc"
    output_path = tmp_path / "output.nc"

    args = [
        reference_path,
        forecast_path,
        "--method",
        method,
        "--output",
        output_path,
    ]
    if example == "overlapping":
        args += [
            "--preservation-threshold",
            "0.00003",
            "--reference-attribute",
            "realization_selection_method=cluster_medoid",
        ]
    else:
        args += [
            "--preservation-threshold",
            "8.333333e-9",
            "--reference-attribute",
            "mosg__model_configuration=uk_det",
        ]
    run_cli(args)
    if example == "overlapping":
        acc.compare(output_path, kgo_path)
    else:
        acc.compare(
            output_path,
            kgo_path,
            rtol=METRES_PER_SECOND_TOLERANCE,
            atol=METRES_PER_SECOND_TOLERANCE,
        )


@pytest.mark.parametrize("example", ["discrete", "overlapping"])
@pytest.mark.parametrize("method", ["step", "continuous"])
def test_step_with_occurrence_threshold(tmp_path, method, example):
    """Test quantile mapping with step method and occurrence threshold."""
    kgo_dir = acc.kgo_root() / "quantile-mapping/with_occurrence_threshold/"
    kgo_path = kgo_dir / f"kgo_{method}_{example}.nc"
    reference_path = acc.kgo_root() / f"quantile-mapping/reference_{example}.nc"
    forecast_path = acc.kgo_root() / f"quantile-mapping/forecast_{example}.nc"
    output_path = tmp_path / "output.nc"

    args = [
        reference_path,
        forecast_path,
        "--method",
        method,
        "--output",
        output_path,
    ]
    if example == "overlapping":
        args += [
            "--occurrence-threshold",
            "0.00003",
            "--reference-attribute",
            "realization_selection_method=cluster_medoid",
        ]
    else:
        args += [
            "--occurrence-threshold",
            "8.333333e-9",
            "--reference-attribute",
            "mosg__model_configuration=uk_det",
        ]

    run_cli(args)

    if example == "overlapping":
        acc.compare(output_path, kgo_path)
    else:
        acc.compare(
            output_path,
            kgo_path,
            rtol=METRES_PER_SECOND_TOLERANCE,
            atol=METRES_PER_SECOND_TOLERANCE,
        )


def test_step_with_non_occurrence_value(tmp_path):
    """Test quantile mapping with step method and non-occurrence value."""
    kgo_dir = acc.kgo_root() / "quantile-mapping/with_non_occurrence_value/"
    kgo_path = kgo_dir / "kgo.nc"
    reference_path = acc.kgo_root() / "quantile-mapping/reference_overlapping.nc"
    forecast_path = acc.kgo_root() / "quantile-mapping/forecast_overlapping.nc"
    output_path = tmp_path / "output.nc"

    args = [
        reference_path,
        forecast_path,
        "--method",
        "step",
        "--occurrence-threshold",
        "0.00003",
        "--non-occurrence-value",
        "0.00001",
        "--reference-attribute",
        "realization_selection_method=cluster_medoid",
        "--output",
        output_path,
    ]

    run_cli(args)
    acc.compare(output_path, kgo_path)
