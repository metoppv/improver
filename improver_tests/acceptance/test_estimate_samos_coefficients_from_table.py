# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""
Tests for the estimate-samos-coefficients-from-table CLI
"""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)

for mod in ["pyarrow", "statsmodels"]:
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


@pytest.mark.slow
@pytest.mark.parametrize("adjacent_range,kgo", [(0, "kgo_coordinates.nc"), (1, "kgo_coordinates_adjacent.nc")])
def test_additional_features_coords(tmp_path, adjacent_range, kgo):
    """
    Test estimate-samos-coefficients-from-table with an example forecast and truth
    table for screen temperature.Extra features for the GAMs are provided
    as coordinates on the forecast and truth cubes.
    """
    source_dir = acc.kgo_root() / "estimate-emos-coefficients-from-table/"
    history_path = source_dir / "forecast_table"
    truth_path = source_dir / "truth_table"

    kgo_dir = acc.kgo_root() / "estimate-samos-coefficients-from-table/"
    kgo_path = kgo_dir / kgo
    output_path = tmp_path / "output.nc"

    gam_config = kgo_dir / "gam_coordinates.pkl"

    compulsory_args = [history_path, truth_path, gam_config]
    named_args = [
        "--gam-features",
        "latitude,longitude,altitude",
        "--percentiles",
        "10,20,30,40,50,60,70,80,90",
        "--forecast-period",
        "86400",
        "--training-length",
        "5",
        "--diagnostic",
        "temperature_at_screen_level",
        "--cycletime",
        "20210805T2100Z",
        "--adjacent-range",
        str(adjacent_range),
        "--output",
        output_path,
    ]

    run_cli(compulsory_args + named_args)
    acc.compare(
        output_path, kgo_path, atol=COMPARE_EMOS_TOLERANCE, rtol=COMPARE_EMOS_TOLERANCE
    )


@pytest.mark.slow
@pytest.mark.parametrize("adjacent_range,kgo", [(0, "kgo_gam_cube.nc"), (1, "kgo_gam_cube_adjacent.nc")])
def test_additional_gam_features_cube(tmp_path, adjacent_range, kgo):
    """
    Test estimate-samos-coefficients-from-table with an example forecast and truth
    table for screen temperature. Extra features for the GAMs are provided
    as a cube.
    """
    source_dir = acc.kgo_root() / "estimate-emos-coefficients-from-table/"
    history_path = source_dir / "forecast_table"
    truth_path = source_dir / "truth_table"

    kgo_dir = acc.kgo_root() / "estimate-samos-coefficients-from-table/"
    kgo_path = kgo_dir / kgo
    output_path = tmp_path / "output.nc"
    gam_additional_features = kgo_dir / "distance_to_water.nc"
    gam_config = kgo_dir / "gam_coordinates.pkl"

    compulsory_args = [history_path, truth_path, gam_config, gam_additional_features]
    named_args = [
        "--gam-features",
        "latitude,longitude,distance_to_water",
        "--percentiles",
        "10,20,30,40,50,60,70,80,90",
        "--forecast-period",
        "86400",
        "--training-length",
        "5",
        "--diagnostic",
        "temperature_at_screen_level",
        "--cycletime",
        "20210805T2100Z",
        "--adjacent-range",
        str(adjacent_range),
        "--output",
        output_path,
    ]

    run_cli(compulsory_args + named_args)
    acc.compare(
        output_path, kgo_path, atol=COMPARE_EMOS_TOLERANCE, rtol=COMPARE_EMOS_TOLERANCE
    )


@pytest.mark.slow
def test_no_gam(tmp_path):
    """
    Test estimate-samos-coefficients-from-table when no GAM is provided. The CLI should
    return None in this instance.
    """
    source_dir = acc.kgo_root() / "estimate-emos-coefficients-from-table/"
    history_path = source_dir / "forecast_table"
    truth_path = source_dir / "truth_table"

    output_path = tmp_path / "output.nc"

    compulsory_args = [history_path, truth_path]
    named_args = [
        "--gam-features",
        "latitude,longitude,altitude",
        "--percentiles",
        "10,20,30,40,50,60,70,80,90",
        "--forecast-period",
        "86400",
        "--training-length",
        "5",
        "--diagnostic",
        "temperature_at_screen_level",
        "--cycletime",
        "20210805T2100Z",
        "--output",
        output_path,
    ]
    run_cli(compulsory_args + named_args)
    # Check no file has been written to disk.
    assert not output_path.exists()


@pytest.mark.slow
def test_return_none(tmp_path):
    """
    Test that None is returned if cube cannot be created from table.
    """
    source_dir = acc.kgo_root() / "estimate-emos-coefficients-from-table/"
    history_path = source_dir / "forecast_table_quantiles"
    truth_path = source_dir / "truth_table"

    kgo_dir = acc.kgo_root() / "estimate-samos-coefficients-from-table/"
    output_path = tmp_path / "output.nc"
    gam_config = kgo_dir / "gam_coordinates.pkl"

    compulsory_args = [history_path, truth_path, gam_config]
    named_args = [
        "--gam-features",
        "projection_y_coordinate,projection_x_coordinate,height",
        "--percentiles",
        "10,20,30,40,50,60,70,80,90",
        "--forecast-period",
        "3600000",
        "--training-length",
        "5",
        "--diagnostic",
        "temperature_at_screen_level",
        "--cycletime",
        "20210805T2100Z",
        "--output",
        output_path,
    ]
    run_cli(compulsory_args + named_args)
    # Check no file has been written to disk.
    assert not output_path.exists()
