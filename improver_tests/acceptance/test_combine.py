# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of improver and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Tests for the combine CLI"""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


def test_basic(tmp_path):
    """Test basic combine operation"""
    kgo_dir = acc.kgo_root() / "combine/basic"
    kgo_path = kgo_dir / "kgo_cloud.nc"
    low_cloud_path = kgo_dir / "low_cloud.nc"
    medium_cloud_path = kgo_dir / "medium_cloud.nc"
    output_path = tmp_path / "output.nc"
    args = [
        "--operation",
        "max",
        "--new-name",
        "cloud_area_fraction",
        low_cloud_path,
        medium_cloud_path,
        "--output",
        f"{output_path}",
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


@pytest.mark.parametrize("minmax", ("min", "max"))
def test_minmax_temperatures(tmp_path, minmax):
    """Test combining minimum and maximum temperatures"""
    kgo_dir = acc.kgo_root() / "combine/bounds"
    kgo_path = kgo_dir / f"kgo_{minmax}.nc"
    temperatures = sorted(kgo_dir.glob(f"*temperature_at_screen_level_{minmax}.nc"))
    output_path = tmp_path / "output.nc"
    args = ["--operation", f"{minmax}", *temperatures, "--output", f"{output_path}"]
    run_cli(args)
    acc.compare(output_path, kgo_path)


@pytest.mark.parametrize("realizations, gives_error", ((3, False), (4, True)))
def test_minimum_realizations(tmp_path, realizations, gives_error):
    """Test combining with the minimum-realizations filter"""
    kgo_dir = acc.kgo_root() / "combine/minimum_realizations"
    kgo_path = kgo_dir / "kgo.nc"
    temperatures = sorted(kgo_dir.glob("*temperature_at_screen_level*.nc"))
    output_path = tmp_path / "output.nc"
    args = [
        "--operation",
        "max",
        "--minimum-realizations",
        f"{realizations}",
        *temperatures,
        "--output",
        f"{output_path}",
    ]
    if gives_error:
        with pytest.raises(
            ValueError,
            match="After filtering, number of realizations 3 is less than the minimum number "
            rf"of realizations allowed \({realizations}\)",
        ):
            run_cli(args)
    else:
        run_cli(args)
        acc.compare(output_path, kgo_path)


def test_combine_accumulation(tmp_path):
    """Test combining precipitation accumulations"""
    kgo_dir = acc.kgo_root() / "combine/accum"
    kgo_path = kgo_dir / "kgo_accum.nc"
    rains = sorted(kgo_dir.glob("*rainfall_accumulation.nc"))
    output_path = tmp_path / "output.nc"
    args = [*rains, "--output", f"{output_path}"]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_mean_temperature(tmp_path):
    """Test combining mean temperature"""
    kgo_dir = acc.kgo_root() / "combine/bounds"
    kgo_path = kgo_dir / "kgo_mean.nc"
    temperatures = sorted(kgo_dir.glob("*temperature_at_screen_level.nc"))
    output_path = tmp_path / "output.nc"
    args = ["--operation", "mean", *temperatures, "--output", f"{output_path}"]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_mean_temperature_cell_method(tmp_path):
    """Test combining mean temperature with a cell_method_coordinate provided."""
    kgo_dir = acc.kgo_root() / "combine/mean_cellmethods"
    kgo_path = kgo_dir / "kgo.nc"
    input_dir = acc.kgo_root() / "combine/bounds"
    temperatures = sorted(input_dir.glob("*temperature_at_screen_level.nc"))
    output_path = tmp_path / "output.nc"
    args = [
        "--operation",
        "mean",
        "--cell-method-coordinate",
        "time",
        *temperatures,
        "--output",
        f"{output_path}",
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_combine_broadcast(tmp_path):
    """Test combining precipitation realizations with phaseprob"""
    kgo_dir = acc.kgo_root() / "combine/broadcast"
    kgo_path = kgo_dir / "kgo.nc"
    inputs = [kgo_dir / f for f in ["input_larger_cube.nc", "input_smaller_cube.nc"]]
    output_path = tmp_path / "output.nc"
    args = [
        *inputs,
        "--operation",
        "multiply",
        "--broadcast",
        "threshold",
        "--new-name",
        "rainfall_rate",
        "--output",
        f"{output_path}",
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_multiplication_cell_methods(tmp_path):
    """Test cell method comments are updated for multiplication"""
    kgo_dir = acc.kgo_root() / "combine/multiplication_cellmethods"
    kgo_path = kgo_dir / "kgo.nc"
    precipaccum = kgo_dir / "precipitation_accumulation-PT01H.nc"
    precipphase = kgo_dir / "precipitation_is_snow.nc"
    output_path = tmp_path / "output.nc"
    args = [
        precipaccum,
        precipphase,
        "--operation",
        "multiply",
        "--broadcast",
        "threshold",
        "--new-name",
        "lwe_thickness_of_snowfall_amount",
        "--output",
        f"{output_path}",
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


@pytest.mark.parametrize("expand_bound", ("True", "False"))
def test_expand_bound(tmp_path, expand_bound):
    """Test combine operation with and without expand_bound"""
    kgo_dir = acc.kgo_root() / "combine/expand_bound"
    kgo_path = kgo_dir / f"kgo_{expand_bound}.nc"
    orography_path = kgo_dir / "orography.nc"
    cloud_base_path = kgo_dir / "input.nc"
    output_path = tmp_path / "output.nc"
    args = [
        "--operation",
        "subtract",
        "--broadcast",
        "realization",
        "--expand-bound",
        expand_bound,
        cloud_base_path,
        orography_path,
        "--output",
        f"{output_path}",
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)
