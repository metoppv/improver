# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Tests for the enforce-consistent-forecasts CLI."""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


@pytest.mark.parametrize(
    "forecast_type, ref_name, additive_amount, multiplicative_amount, comparison_operator, bounds",
    (
        (
            "probability",
            "probability_of_cloud_area_fraction_above_threshold",
            "0.0",
            "1.0",
            "<=",
            "single_bound",
        ),
        ("percentile", "wind_speed", "0.0", "1.1", ">=", "single_bound"),
        ("realization", "surface_temperature", "0.0", "0.9", "<=", "single_bound"),
        ("percentile", "wind_speed", "0.0,0.0", "1.1,3.0", ">=,<=", "double_bound"),
    ),
)
def test_enforce_consistent_forecasts(
    tmp_path,
    ref_name,
    forecast_type,
    additive_amount,
    multiplicative_amount,
    comparison_operator,
    bounds,
):
    """
    Test enforcement of consistent forecasts between cubes
    """
    kgo_dir = acc.kgo_root() / "enforce-consistent-forecasts"
    kgo_path = kgo_dir / f"{bounds}_{forecast_type}_kgo.nc"

    forecast = kgo_dir / f"{forecast_type}_forecast.nc"
    reference = kgo_dir / f"{forecast_type}_reference.nc"
    output_path = tmp_path / "output.nc"

    if forecast_type == "probability":
        args = [
            forecast,
            reference,
            "--ref-name",
            ref_name,
            "--comparison-operator",
            comparison_operator,
            "--output",
            output_path,
        ]
    else:
        args = [
            forecast,
            reference,
            "--ref-name",
            ref_name,
            "--additive-amount",
            additive_amount,
            "--multiplicative-amount",
            multiplicative_amount,
            "--comparison-operator",
            comparison_operator,
            "--output",
            output_path,
        ]

    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_too_many_cubes(tmp_path):
    """
    Test to ensure an error is raised if too many cubes are provided.
    """
    kgo_dir = acc.kgo_root() / "enforce-consistent-forecasts"

    forecast = kgo_dir / "probability_forecast.nc"
    reference = kgo_dir / "probability_reference.nc"
    output_path = tmp_path / "output.nc"

    args = [
        forecast,
        forecast,
        reference,
        "--ref-name",
        "probability_of_cloud_area_fraction_above_threshold",
        "--output",
        output_path,
    ]
    with pytest.raises(ValueError, match="Exactly two cubes"):
        run_cli(args)


@pytest.mark.parametrize(
    "additive_amount, multiplicative_amount, comparison_operator",
    (("0.0, 0.0", "1.0", ">="), ("0.0, 0.0, 0.0", "1.0, 1.0, 1.0", ">=, >=, <="),),
)
def test_bad_inputs(
    tmp_path, additive_amount, multiplicative_amount, comparison_operator
):
    """
    Test to ensure an error is raised if additive_amount, multiplicative_amount, and
    comparison_operator are not the same length, or each have length greater than 2.
    """
    kgo_dir = acc.kgo_root() / "enforce-consistent-forecasts"

    forecast = kgo_dir / "probability_forecast.nc"
    reference = kgo_dir / "probability_reference.nc"
    output_path = tmp_path / "output.nc"

    args = [
        forecast,
        reference,
        "--ref-name",
        "probability_of_cloud_area_fraction_above_threshold",
        "--additive-amount",
        additive_amount,
        "--multiplicative-amount",
        multiplicative_amount,
        "--comparison-operator",
        comparison_operator,
        "--output",
        output_path,
    ]
    with pytest.raises(ValueError, match="Each of additive_amount"):
        run_cli(args)
