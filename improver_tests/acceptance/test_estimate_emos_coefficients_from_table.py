# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""
Tests for the estimate-emos-coefficients-from-table CLI

"""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)

for mod in ["fastparquet", "statsmodels"]:
    pytest.importorskip(mod)

# The EMOS estimation tolerance is defined in units of the variable being
# calibrated - not in terms of the EMOS coefficients produced by
# estimate-emos-coefficients and compared against KGOs here.
# See comments and CLI help messages in
# improver/cli/estimate_emos_coefficients.py for more detail.
EST_EMOS_TOLERANCE = 1e-4

# The EMOS coefficients are expected to vary by at most one order of magnitude
# more than the CRPS tolerance specified.
COMPARE_EMOS_TOLERANCE = EST_EMOS_TOLERANCE * 10

# Pre-convert to string for easier use in each test
EST_EMOS_TOL = str(EST_EMOS_TOLERANCE)


@pytest.mark.parametrize(
    "forecast_input,distribution,diagnostic,percentiles,additional_predictor,kgo_name",
    [
        (
            "forecast_table",
            "norm",
            "temperature_at_screen_level",
            "10,20,30,40,50,60,70,80,90",
            None,
            "screen_temperature",
        ),
        (
            "forecast_table",
            "truncnorm",
            "wind_speed_at_10m",
            "20,40,60,80",
            None,
            "wind_speed",
        ),
        (
            "forecast_table_quantiles",
            "norm",
            "temperature_at_screen_level",
            None,
            None,
            "screen_temperature_input_quantiles",
        ),
        (
            "forecast_table",
            "norm",
            "temperature_at_screen_level",
            "10,20,30,40,50,60,70,80,90",
            "altitude.nc",
            "screen_temperature_additional_predictor",
        ),
    ],
)
@pytest.mark.slow
def test_basic(
    tmp_path,
    forecast_input,
    distribution,
    diagnostic,
    percentiles,
    additional_predictor,
    kgo_name,
):
    """
    Test estimate-emos-coefficients-from-table with an example forecast and truth
    table for screen temperature and wind speed.
    """
    kgo_dir = acc.kgo_root() / "estimate-emos-coefficients-from-table/"
    kgo_path = kgo_dir / f"{kgo_name}_kgo.nc"
    history_path = kgo_dir / forecast_input
    truth_path = kgo_dir / "truth_table"
    output_path = tmp_path / "output.nc"
    compulsory_args = [history_path, truth_path]
    named_args = [
        "--diagnostic",
        diagnostic,
        "--cycletime",
        "20210805T2100Z",
        "--forecast-period",
        "86400",
        "--training-length",
        "5",
        "--distribution",
        distribution,
        "--tolerance",
        EST_EMOS_TOL,
        "--output",
        output_path,
    ]
    if additional_predictor:
        compulsory_args += [kgo_dir / additional_predictor]
    if percentiles:
        named_args += ["--percentiles", percentiles]
    run_cli(compulsory_args + named_args)
    acc.compare(
        output_path, kgo_path, atol=COMPARE_EMOS_TOLERANCE, rtol=COMPARE_EMOS_TOLERANCE
    )


@pytest.mark.slow
def test_invalid_truth_filter(tmp_path,):
    """
    Test using an invalid diagnostic name to filter the truth table.
    """
    kgo_dir = acc.kgo_root() / "estimate-emos-coefficients-from-table/"
    history_path = kgo_dir / "forecast_table"
    truth_path = kgo_dir / "truth_table_misnamed_diagnostic"
    output_path = tmp_path / "output.nc"
    args = [
        history_path,
        truth_path,
        "--diagnostic",
        "temperature_at_screen_level",
        "--cycletime",
        "20210805T2100Z",
        "--forecast-period",
        "86400",
        "--training-length",
        "5",
        "--distribution",
        "norm",
        "--tolerance",
        EST_EMOS_TOL,
        "--output",
        output_path,
    ]
    with pytest.raises(
        IOError, match="The requested filepath.*temperature_at_screen_level.*"
    ):
        run_cli(args)


@pytest.mark.slow
def test_return_none(tmp_path,):
    """
    Test that None is returned if a non-existent forecast period is requested.
    """
    kgo_dir = acc.kgo_root() / "estimate-emos-coefficients-from-table/"
    history_path = kgo_dir / "forecast_table_quantiles"
    truth_path = kgo_dir / "truth_table"
    output_path = tmp_path / "output.nc"
    args = [
        history_path,
        truth_path,
        "--diagnostic",
        "temperature_at_screen_level",
        "--cycletime",
        "20210805T2100Z",
        "--forecast-period",
        "3600000",
        "--training-length",
        "5",
        "--distribution",
        "norm",
        "--tolerance",
        EST_EMOS_TOL,
        "--output",
        output_path,
    ]
    assert run_cli(args) is None
