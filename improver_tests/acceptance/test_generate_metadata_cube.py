# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of improver and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Tests for the generate-metadata-cube CLI."""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)
kgo_dir = acc.kgo_root() / "generate-metadata-cube"
mandatory_attributes_json = kgo_dir / "mandatory_attributes.json"


def test_default(tmp_path):
    """Test default metadata cube generation"""
    kgo_path = kgo_dir / "kgo_default.nc"
    output_path = tmp_path / "output.nc"
    args = [
        mandatory_attributes_json,
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_ensemble_members(tmp_path):
    """Test creating variable cube with all options set"""
    kgo_path = kgo_dir / "kgo_ensemble_members_all_options.nc"
    output_path = tmp_path / "output.nc"
    args = [
        mandatory_attributes_json,
        "--name",
        "air_temperature",
        "--units",
        "degrees",
        "--spatial-grid",
        "equalarea",
        "--time-period",
        "120",
        "--ensemble-members",
        "4",
        "--x-grid-spacing",
        "1000",
        "--y-grid-spacing",
        "1000",
        "--domain-corner",
        "0,0",
        "--npoints",
        "50",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_json_all_inputs(tmp_path):
    """Test creating variable cube with all options set"""
    kgo_path = kgo_dir / "kgo_variable_cube_json_inputs.nc"
    json_input_path = kgo_dir / "variable_cube_all_inputs.json"
    output_path = tmp_path / "output.nc"
    args = [
        mandatory_attributes_json,
        "--json-input",
        json_input_path,
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_realization_json(tmp_path):
    """Test variable/realization metadata cube generated using realization
    coordinate defined in the json input"""
    kgo_path = kgo_dir / "kgo_realization.nc"
    realizations_path = kgo_dir / "realizations.json"
    output_path = tmp_path / "output.nc"
    args = [
        mandatory_attributes_json,
        "--json-input",
        realizations_path,
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_percentile_cube(tmp_path):
    """Test percentile metadata cube generated using using percentile
    coordinate defined in the json input"""
    kgo_path = kgo_dir / "kgo_percentile.nc"
    percentiles_path = kgo_dir / "percentiles.json"
    output_path = tmp_path / "output.nc"
    args = [
        mandatory_attributes_json,
        "--json-input",
        percentiles_path,
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_probability_cube(tmp_path):
    """Test probability metadata cube generated using using threshold
    coordinate defined in the json input"""
    kgo_path = kgo_dir / "kgo_probability.nc"
    thresholds_path = kgo_dir / "thresholds.json"
    output_path = tmp_path / "output.nc"
    args = [
        mandatory_attributes_json,
        "--json-input",
        thresholds_path,
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_height_levels(tmp_path):
    """Test metadata cube generated with height levels from json"""
    kgo_path = kgo_dir / "kgo_height_levels.nc"
    height_levels_path = kgo_dir / "height_levels.json"
    output_path = tmp_path / "output.nc"
    args = [
        mandatory_attributes_json,
        "--json-input",
        height_levels_path,
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_single_height_level(tmp_path):
    """Test metadata cube generation giving single value (rather than comma separated
    list) for height levels option demotes height to scalar coordinate"""
    kgo_path = kgo_dir / "kgo_single_height_level.nc"
    height_level_path = kgo_dir / "single_height_level.json"
    output_path = tmp_path / "output.nc"
    args = [
        mandatory_attributes_json,
        "--json-input",
        height_level_path,
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_pressure_levels(tmp_path):
    """Test metadata cube generated with pressure in Pa instead of height in metres"""
    kgo_path = kgo_dir / "kgo_pressure_levels.nc"
    pressure_levels_path = kgo_dir / "pressure_levels.json"
    output_path = tmp_path / "output.nc"
    args = [
        mandatory_attributes_json,
        "--json-input",
        pressure_levels_path,
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)
