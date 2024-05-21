# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Tests for the categorical CLI"""
import re

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)

ALL_PARAMS = [
    "low_and_medium_type_cloud_area_fraction_above",
    "low_type_cloud_area_fraction_above",
    "lwe_graupel_and_hail_fall_rate_in_vicinity_above",
    "lwe_thickness_of_graupel_and_hail_fall_amount_above",
    "lwe_thickness_of_precipitation_amount_above",
    "lwe_thickness_of_precipitation_amount_in_vicinity_above",
    "lwe_thickness_of_sleetfall_amount_above",
    "lwe_thickness_of_snowfall_amount_above",
    "number_of_lightning_flashes_per_unit_area_in_vicinity_above",
    "shower_condition_above",
    "thickness_of_rainfall_amount_above",
    "visibility_in_air_below",
]


@pytest.mark.slow
@pytest.mark.parametrize(
    "title_option, kgo",
    (
        ("", "kgo"),
        (
            ("--title", "IMPROVER Post-Processed Multi-Model Blend of flavours"),
            "kgo_titled",
        ),
    ),
)
def test_basic(tmp_path, title_option, kgo):
    """Test basic categorical weather symbol processing with and without a user defined title
    attribute."""
    kgo_dir = acc.kgo_root() / "categorical"
    kgo_path = kgo_dir / "basic" / f"{kgo}.nc"
    param_paths = [
        kgo_dir / "basic" / f"probability_of_{p}_threshold.nc" for p in ALL_PARAMS
    ]
    wxtree = kgo_dir / "wx_decision_tree.json"
    output_path = tmp_path / "output.nc"
    args = [
        *param_paths,
        "--decision-tree",
        wxtree,
        "--model-id-attr",
        "mosg__model_configuration",
        "--record-run-attr",
        "mosg__model_run",
        "--target-period",
        "3600",
        *title_option,
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


@pytest.mark.slow
def test_native_units(tmp_path):
    """Test categorical weather symbol processing with non-SI units for threshold coordinates:
    precipitation: mm
    visibility: feet
    """
    kgo_dir = acc.kgo_root() / "categorical"
    kgo_path = kgo_dir / "basic" / "kgo.nc"
    param_paths = [
        kgo_dir / "native_units" / f"probability_of_{p}_threshold.nc"
        for p in ALL_PARAMS
    ]
    wxtree = kgo_dir / "wx_decision_tree.json"
    output_path = tmp_path / "output.nc"

    args = [
        *param_paths,
        "--decision-tree",
        wxtree,
        "--model-id-attr",
        "mosg__model_configuration",
        "--record-run-attr",
        "mosg__model_run",
        "--target-period",
        "3600",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_global(tmp_path):
    """Test global categorical weather symbol processing"""
    kgo_dir = acc.kgo_root() / "categorical"
    kgo_path = kgo_dir / "global" / "kgo.nc"
    params = [param for param in ALL_PARAMS if "hail" not in param]
    param_paths = [
        kgo_dir / "global" / f"probability_of_{p}_threshold.nc" for p in params
    ]
    wxtree = kgo_dir / "wx_decision_tree.json"
    output_path = tmp_path / "output.nc"
    args = [
        *param_paths,
        "--decision-tree",
        wxtree,
        "--model-id-attr",
        "mosg__model_configuration",
        "--target-period",
        "10800",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_insufficient_files(tmp_path):
    """Test categorical processing with insufficient files"""
    kgo_dir = acc.kgo_root() / "categorical"
    params = [
        "low_and_medium_type_cloud_area_fraction_above",
        "low_type_cloud_area_fraction_above",
        "lwe_thickness_of_snowfall_amount_above",
        "thickness_of_rainfall_amount_above",
    ]
    param_paths = [
        kgo_dir / "global" / f"probability_of_{p}_threshold.nc" for p in params
    ]
    wxtree = kgo_dir / "wx_decision_tree.json"
    output_path = tmp_path / "output.nc"
    args = [
        *param_paths,
        "--decision-tree",
        wxtree,
        "--model-id-attr",
        "mosg__model_configuration",
        "--target-period",
        "10800",
        "--output",
        output_path,
    ]
    with pytest.raises(OSError):
        run_cli(args)


@pytest.mark.slow
def test_without_optional_input(tmp_path):
    """Test categorical processing with an optional input absent"""
    kgo_dir = acc.kgo_root() / "categorical"
    kgo_path = kgo_dir / "basic" / "kgo_no_lightning.nc"
    param_paths = [
        kgo_dir / "basic" / f"probability_of_{p}_threshold.nc"
        for p in ALL_PARAMS
        if "lightning" not in p
    ]
    wxtree = kgo_dir / "wx_decision_tree.json"
    output_path = tmp_path / "output.nc"
    args = [
        *param_paths,
        "--decision-tree",
        wxtree,
        "--model-id-attr",
        "mosg__model_configuration",
        "--target-period",
        "3600",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_cubelist_input(tmp_path):
    """Test categorical decision tree with a cubelist input."""
    kgo_dir = acc.kgo_root() / "categorical"
    kgo_path = kgo_dir / "deterministic" / "kgo.nc"

    param_paths = [
        kgo_dir / "deterministic" / "precipitation_rate.nc",
        kgo_dir / "deterministic" / "hail_cubelist.nc",
    ]

    wxtree = kgo_dir / "deterministic_decision_tree.json"
    output_path = tmp_path / "output.nc"
    args = [
        *param_paths,
        "--decision-tree",
        wxtree,
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


@pytest.mark.parametrize(
    "decision_tree,expected",
    (
        ("wx_decision_tree.json", "Decision tree OK\nRequired inputs are:"),
        ("bad_wx_decision_tree.json", "Unreachable node 'unreachable'"),
    ),
)
def test_trees(decision_tree, expected):
    """Test the check-tree option"""
    kgo_dir = acc.kgo_root() / "categorical"
    args = [
        "--decision-tree",
        kgo_dir / decision_tree,
        "--check-tree",
        "--target-period",
        "3600",
    ]
    result = run_cli(args)
    assert re.match(expected, result)
