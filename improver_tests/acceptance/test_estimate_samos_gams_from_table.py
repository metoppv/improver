# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""
Tests for the estimate-samos-gams-from-table CLI

"""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)

for mod in ["pyarrow", "statsmodels"]:
    pytest.importorskip(mod)


@pytest.mark.slow
def test_additional_features_coords(
    tmp_path,
):
    """
    Test estimate-samos-gams-from-table with an example forecast and truth
    table for screen temperature. Extra features for the GAMs are provided
    as coordinates on the forecast and truth cubes.
    """
    source_dir = acc.kgo_root() / "estimate-emos-coefficients-from-table/"
    history_path = source_dir / "forecast_table"
    truth_path = source_dir / "truth_table"

    kgo_dir = acc.kgo_root() / "estimate-samos-gams-from-table/"
    kgo_path = kgo_dir / "kgo_coords.pkl"

    output_path = tmp_path / "output.pkl"
    compulsory_args = [history_path, truth_path]
    named_args = [
        "--diagnostic",
        "temperature_at_screen_level",
        "--cycletime",
        "20210805T2100Z",
        "--forecast-period",
        "86400",
        "--training-length",
        "5",
        "--distribution",
        "normal",
        "--tolerance",
        "1e-4",
        "--gam-features",
        "latitude,longitude,altitude",
        "--model-specification",
        kgo_dir / "samos_model_spec_simple.json",
        "--percentiles",
        "10,20,30,40,50,60,70,80,90",
        "--window-length",
        "2",
        "--required-rolling-window-points",
        "2",
        "--output",
        output_path,
    ]

    run_cli(compulsory_args + named_args)
    # Compare the output with the known good output. This
    # comparison only ensures that the string version of the
    # pickled objects are the same, not the actual objects as
    # there is no function to compare the GAM class objects.
    acc.compare(output_path, kgo_path, file_type="generic_pickle")


@pytest.mark.slow
def test_additional_features_cube(
    tmp_path,
):
    """
    Test estimate-samos-gams-from-table with an example forecast and truth
    table for screen temperature. Extra features for the GAMs are provided
    as a cube.
    """
    source_dir = acc.kgo_root() / "estimate-emos-coefficients-from-table/"
    history_path = source_dir / "forecast_table"
    truth_path = source_dir / "truth_table"

    kgo_dir = acc.kgo_root() / "estimate-samos-gams-from-table/"
    kgo_path = kgo_dir / "kgo_cubes.pkl"
    additional_features = kgo_dir / "distance_to_water.nc"

    output_path = tmp_path / "output.pkl"
    compulsory_args = [history_path, truth_path, additional_features]
    named_args = [
        "--diagnostic",
        "temperature_at_screen_level",
        "--cycletime",
        "20210805T2100Z",
        "--forecast-period",
        "86400",
        "--training-length",
        "5",
        "--distribution",
        "normal",
        "--tolerance",
        "1e-4",
        "--gam-features",
        "latitude,longitude,distance_to_water",
        "--model-specification",
        kgo_dir / "samos_model_spec_simple.json",
        "--percentiles",
        "10,20,30,40,50,60,70,80,90",
        "--window-length",
        "2",
        "--required-rolling-window-points",
        "2",
        "--output",
        output_path,
    ]

    run_cli(compulsory_args + named_args)
    # Compare the output with the known good output. This
    # comparison only ensures that the string version of the
    # pickled objects are the same, not the actual objects as
    # there is no function to compare the GAM class objects.
    acc.compare(output_path, kgo_path, file_type="generic_pickle")


@pytest.mark.slow
def test_additional_features_cubes(
    tmp_path,
):
    """
    Test estimate-samos-gams-from-table with an example forecast and truth
    table for screen temperature. Extra features for the GAMs are provided
    by multiple cubes. This test demonstrates that the CLI can accept
    multiple additional predictor cubes.
    """
    source_dir = acc.kgo_root() / "estimate-emos-coefficients-from-table/"
    history_path = source_dir / "forecast_table"
    truth_path = source_dir / "truth_table"

    kgo_dir = acc.kgo_root() / "estimate-samos-gams-from-table/"
    kgo_path = kgo_dir / "kgo_cubes.pkl"
    additional_features = [kgo_dir / "distance_to_water.nc", kgo_dir / "roughness.nc"]

    output_path = tmp_path / "output.pkl"
    compulsory_args = [history_path, truth_path, *additional_features]
    named_args = [
        "--diagnostic",
        "temperature_at_screen_level",
        "--cycletime",
        "20210805T2100Z",
        "--forecast-period",
        "86400",
        "--training-length",
        "5",
        "--distribution",
        "normal",
        "--tolerance",
        "1e-4",
        "--gam-features",
        "latitude,longitude,distance_to_water,vegetative_roughness_length",
        "--model-specification",
        kgo_dir / "samos_model_spec_simple.json",
        "--percentiles",
        "10,20,30,40,50,60,70,80,90",
        "--window-length",
        "2",
        "--required-rolling-window-points",
        "2",
        "--output",
        output_path,
    ]

    run_cli(compulsory_args + named_args)
    # Compare the output with the known good output. This
    # comparison only ensures that the string version of the
    # pickled objects are the same, not the actual objects as
    # there is no function to compare the GAM class objects.
    acc.compare(output_path, kgo_path, file_type="generic_pickle")


@pytest.mark.slow
def test_no_forecast(
    tmp_path,
):
    """
    Test estimate-samos-gams-from-table returns None when no forecast data is available
     for the given leadtime in the given table.
    """
    source_dir = acc.kgo_root() / "estimate-emos-coefficients-from-table/"
    history_path = source_dir / "forecast_table"
    truth_path = source_dir / "truth_table"

    kgo_dir = acc.kgo_root() / "estimate-samos-gams-from-table/"

    output_path = tmp_path / "output.pkl"
    compulsory_args = [history_path, truth_path]
    named_args = [
        "--diagnostic",
        "temperature_at_screen_level",
        "--cycletime",
        "20210805T2100Z",
        "--forecast-period",
        "3600000",
        "--training-length",
        "5",
        "--distribution",
        "normal",
        "--tolerance",
        "1e-4",
        "--gam-features",
        "latitude,longitude,altitude",
        "--model-specification",
        kgo_dir / "samos_model_spec_simple.json",
        "--percentiles",
        "10,20,30,40,50,60,70,80,90",
        "--window-length",
        "2",
        "--required-rolling-window-points",
        "2",
        "--output",
        output_path,
    ]
    run_cli(compulsory_args + named_args)
    # Check no file has been written to disk.
    assert not output_path.exists()


@pytest.mark.slow
def test_insufficient_data(
    tmp_path,
):
    """
    Test estimate-samos-gams-from-table returns None when insufficient data is
    available at all sites.

    This test provides 3 days of input data but uses a window length of 10 days. This
    will cause the training data at all sites to be considered insufficient to fit the
    GAMs (at least 6 days of data are required). Hence, None should be
    returned.
    """
    source_dir = acc.kgo_root() / "estimate-emos-coefficients-from-table/"
    history_path = source_dir / "forecast_table"
    truth_path = source_dir / "truth_table"

    kgo_dir = acc.kgo_root() / "estimate-samos-gams-from-table/"

    output_path = tmp_path / "output.pkl"
    compulsory_args = [history_path, truth_path]
    named_args = [
        "--diagnostic",
        "temperature_at_screen_level",
        "--cycletime",
        "20210805T2100Z",
        "--forecast-period",
        "86400",
        "--training-length",
        "5",
        "--distribution",
        "normal",
        "--tolerance",
        "1e-4",
        "--gam-features",
        "latitude,longitude,altitude",
        "--model-specification",
        kgo_dir / "samos_model_spec_simple.json",
        "--percentiles",
        "10,20,30,40,50,60,70,80,90",
        "--window-length",
        "10",
        "--required-rolling-window-points",
        "6",
        "--output",
        output_path,
    ]
    run_cli(compulsory_args + named_args)
    # Check no file has been written to disk.
    assert not output_path.exists()
