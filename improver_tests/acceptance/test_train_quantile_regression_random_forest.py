# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""
Tests for the train-quantile-regression-random-forest CLI

"""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)

for mod in ["joblib", "pyarrow", "quantile_forest"]:
    pytest.importorskip(mod)


@pytest.mark.parametrize(
    "transformation",
    ["without_transformation", "with_transformation"],
)
def test_basic(
    tmp_path,
    transformation,
):
    """
    Test train-quantile-regression-random-forest CLI with and without a transformation
    applied.
    """
    kgo_dir = acc.kgo_root() / CLI
    kgo_path = kgo_dir / f"{transformation}_kgo.pickle"
    history_path = kgo_dir / "spot_calibration_tables"
    truth_path = kgo_dir / "spot_observation_tables"
    config_path = kgo_dir / "config.json"
    output_path = tmp_path / "output.pickle"
    compulsory_args = [history_path, truth_path]
    named_args = [
        "--feature-config",
        config_path,
        "--target-diagnostic-name",
        "temperature_at_screen_level",
        "--target-cf-name",
        "air_temperature",
        "--forecast-periods",
        "6:18:6",
        "--cycletime",
        "20250804T0000Z",
        "--training-length",
        "2",
        "--experiment",
        "mix-latestblend",
        "--n-estimators",
        "10",
        "--max-depth",
        "5",
        "--random-state",
        "42",
        "--compression",
        "5",
        "--output",
        output_path,
    ]
    if transformation == "with_transformation":
        named_args += [
            "--transformation",
            "log",
            "--pre-transform-addition",
            "0.1",
        ]

    run_cli(compulsory_args + named_args)
    acc.compare(output_path, kgo_path, file_type="pickled_forest")
