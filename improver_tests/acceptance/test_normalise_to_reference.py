# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of improver and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Tests for the normalise-to-reference CLI."""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


@pytest.mark.parametrize(
    "forecast_type,ignore_zero_total",
    (("probability", False), ("percentile", False), ("percentile", True)),
)
def test_normalise_to_reference(
    tmp_path, forecast_type, ignore_zero_total,
):
    """
    Test input cubes are updated correctly so that their total equals the reference cube
    """
    kgo_dir = acc.kgo_root() / f"normalise-to-reference/{forecast_type}"
    kgo_path = kgo_dir / "kgo.nc"

    output_path = tmp_path / "output.nc"

    if forecast_type == "probability":
        inputs = kgo_dir / "*acc.nc"
        reference_name = (
            "probability_of_lwe_thickness_of_precipitation_amount_above_threshold"
        )
        return_name = "probability_of_thickness_of_rainfall_amount_above_threshold"
    else:
        inputs = kgo_dir / "*rate.nc"
        reference_name = "lwe_precipitation_rate"
        return_name = "rainfall_rate"

    args = [
        inputs,
        "--reference-name",
        reference_name,
        "--return-name",
        return_name,
        "--output",
        output_path,
    ]
    if forecast_type == "probability":
        run_cli(args)
        acc.compare(output_path, kgo_path)
    elif ignore_zero_total:
        args.append("--ignore-zero-total")
        run_cli(args)
        acc.compare(output_path, kgo_path)
    else:
        with pytest.raises(
            ValueError, match="There are instances where the total of input"
        ):
            run_cli(args)


def test_incorrect_reference(tmp_path):
    """
    Test correct error is raised when incorrect number of reference cubes are found.
    """
    kgo_dir = acc.kgo_root() / "normalise-to-reference/percentile"

    inputs = kgo_dir / "input*.nc"
    output_path = tmp_path / "output.nc"

    reference_name = "lwe_precipitation_rate"
    return_name = "rainfall_rate"

    args = [
        inputs,
        "--reference-name",
        reference_name,
        "--return-name",
        return_name,
        "--output",
        output_path,
    ]
    with pytest.raises(ValueError, match="Exactly one cube "):
        run_cli(args)
