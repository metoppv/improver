# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
import pytest

from improver.constants import LOOSE_TOLERANCE

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


@pytest.mark.parametrize(
    "transformation",
    ["without_transformation", "with_transformation"],
)
def test_basic(tmp_path, transformation):
    """Test apply-quantile-regression-random-forest CLI with and without a
    transformation applied."""
    kgo_dir = acc.kgo_root() / "apply-quantile-regression-random-forest/"
    kgo_path = kgo_dir / f"{transformation}_kgo.nc"
    qrf_path = kgo_dir / f"{transformation}_input.pickle"
    forecast_path = kgo_dir / "input_forecast.nc"
    config_path = kgo_dir / "config.json"
    output_path = tmp_path / "output.nc"
    args = [
        forecast_path,
        qrf_path,
        "--feature-config",
        config_path,
        "--target-cf-name",
        "air_temperature",
        "--output",
        output_path,
    ]

    run_cli(args)
    acc.compare(output_path, kgo_path, atol=LOOSE_TOLERANCE)


def test_missing_qrf_model(tmp_path):
    """Test that if no QRF model is provided, the result matches the input forecast
    with the exception of the comment attribute."""
    kgo_dir = acc.kgo_root() / "apply-quantile-regression-random-forest/"
    kgo_path = kgo_dir / "added_comment_kgo.nc"
    forecast_path = kgo_dir / "input_forecast.nc"
    config_path = kgo_dir / "config.json"
    output_path = tmp_path / "output.nc"
    args = [
        forecast_path,
        "--feature-config",
        config_path,
        "--target-cf-name",
        "air_temperature",
        "--output",
        output_path,
    ]

    run_cli(args)
    acc.compare(
        output_path, forecast_path, atol=LOOSE_TOLERANCE, exclude_attributes=["comment"]
    )
    acc.compare(output_path, kgo_path, atol=LOOSE_TOLERANCE)
