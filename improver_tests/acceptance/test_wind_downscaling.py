# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Tests for the wind-downscaling CLI"""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


def test_basic(tmp_path):
    """Test basic wind downscaling"""
    kgo_dir = acc.kgo_root() / "wind_downscaling/basic"
    kgo_path = kgo_dir / "kgo.nc"
    input_paths = [
        kgo_dir / f"{p}.nc"
        for p in ("input", "sigma", "highres_orog", "standard_orog", "a_over_s")
    ]
    output_path = tmp_path / "output.nc"
    args = [*input_paths, "--model-resolution", "1500", "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_vegetation(tmp_path):
    """Test wind downscaling with vegetation roughness"""
    kgo_dir = acc.kgo_root() / "wind_downscaling/veg"
    kgo_path = kgo_dir / "kgo.nc"
    input_paths = [
        kgo_dir / f"{p}.nc"
        for p in ("input", "sigma", "highres_orog", "standard_orog", "a_over_s")
    ]
    veg_path = kgo_dir / "veg.nc"
    output_path = tmp_path / "output.nc"
    args = [
        *input_paths,
        veg_path,
        "--model-resolution",
        "1500",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_realization(tmp_path):
    """Test wind downscaling with realization coordinate"""
    kgo_dir = acc.kgo_root() / "wind_downscaling/with_realization"
    kgo_path = kgo_dir / "kgo.nc"
    input_paths = [
        kgo_dir / f"{p}.nc"
        for p in ("input", "sigma", "highres_orog", "standard_orog", "a_over_s")
    ]
    output_path = tmp_path / "output.nc"
    args = [*input_paths, "--model-resolution", "1500", "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_single_level_output(tmp_path):
    """Test downscaling with output at a single specified level"""
    kgo_dir = acc.kgo_root() / "wind_downscaling/single_level"
    kgo_path = kgo_dir / "kgo.nc"
    input_dir = kgo_dir / "../basic"
    input_paths = [
        input_dir / f"{p}.nc"
        for p in ("input", "sigma", "highres_orog", "standard_orog", "a_over_s")
    ]
    output_path = tmp_path / "output.nc"
    args = [
        *input_paths,
        "--model-resolution",
        "1500",
        "--output-height-level",
        "50",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_unavailable_level(tmp_path):
    """Test attempting to downscale to unavailable height level"""
    kgo_dir = acc.kgo_root() / "wind_downscaling/basic"
    input_paths = [
        kgo_dir / f"{p}.nc"
        for p in ("input", "sigma", "highres_orog", "standard_orog", "a_over_s")
    ]
    output_path = tmp_path / "output.nc"
    args = [
        *input_paths,
        "--model-resolution",
        "1500",
        "--output-height-level",
        "9",
        "--output-height-level-units",
        "m",
        "--output",
        output_path,
    ]
    with pytest.raises(ValueError, match=".*height level.*"):
        run_cli(args)


def test_single_level_units(tmp_path):
    """Test downscaling with specified level and units"""
    kgo_dir = acc.kgo_root() / "wind_downscaling/single_level"
    kgo_path = kgo_dir / "kgo.nc"
    input_dir = kgo_dir / "../basic"
    input_paths = [
        input_dir / f"{p}.nc"
        for p in ("input", "sigma", "highres_orog", "standard_orog", "a_over_s")
    ]
    output_path = tmp_path / "output.nc"
    args = [
        *input_paths,
        "--model-resolution",
        "1500",
        "--output-height-level",
        "5000",
        "--output-height-level-units",
        "cm",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)
