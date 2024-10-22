# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""
Tests for the nbhood CLI
"""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


def test_basic_circular(tmp_path):
    """Test basic circular neighbourhooding"""
    kgo_dir = acc.kgo_root() / "nbhood/basic"
    kgo_path = kgo_dir / "kgo_circular.nc"
    input_path = kgo_dir / "input.nc"
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        "--neighbourhood-output",
        "probabilities",
        "--neighbourhood-shape",
        "circular",
        "--radii",
        "20000",
        "--weighted-mode",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_basic_square(tmp_path):
    """Test basic square neighbourhooding"""
    kgo_dir = acc.kgo_root() / "nbhood/basic"
    kgo_path = kgo_dir / "kgo_square.nc"
    input_path = kgo_dir / "input.nc"
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        "--neighbourhood-output",
        "probabilities",
        "--neighbourhood-shape",
        "square",
        "--radii",
        "20000",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_masked_square(tmp_path):
    """Test square neighbourhooding with a mask"""
    kgo_dir = acc.kgo_root() / "nbhood/mask"
    kgo_path = kgo_dir / "kgo_masked.nc"
    input_path = kgo_dir / "input_masked.nc"
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        "--neighbourhood-output",
        "probabilities",
        "--neighbourhood-shape",
        "square",
        "--radii",
        "20000",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


@pytest.mark.slow
def test_circular_percentile(tmp_path):
    """Test circular neighbourhooding of percentiles"""
    kgo_dir = acc.kgo_root() / "nbhood/percentile"
    kgo_path = kgo_dir / "kgo_circular_percentile.nc"
    input_path = kgo_dir / "input_circular_percentile.nc"
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        "--neighbourhood-output",
        "percentiles",
        "--neighbourhood-shape",
        "circular",
        "--radii",
        "20000",
        "--percentiles",
        "25, 50, 75",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_masked_square_external(tmp_path):
    """Test square neighbourhooding with an external mask"""
    kgo_dir = acc.kgo_root() / "nbhood/mask"
    kgo_path = kgo_dir / "kgo_external_masked.nc"
    input_path = kgo_dir / "input.nc"
    mask_path = kgo_dir / "mask.nc"
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        mask_path,
        "--neighbourhood-output",
        "probabilities",
        "--neighbourhood-shape",
        "square",
        "--radii",
        "20000",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_wind_direction(tmp_path):
    """Test wind direction neighbourhooding"""
    kgo_dir = acc.kgo_root() / "nbhood/wind_direction"
    kgo_path = kgo_dir / "kgo.nc"
    input_path = kgo_dir / "input.nc"
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        "--neighbourhood-output",
        "probabilities",
        "--radii",
        "20000",
        "--degrees-as-complex",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_halo_radius(tmp_path):
    """Test neighbourhooding with a halo radius"""
    kgo_dir = acc.kgo_root() / "nbhood/halo"
    kgo_path = kgo_dir / "kgo.nc"
    input_path = kgo_dir / "input.nc"
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        "--neighbourhood-output",
        "probabilities",
        "--neighbourhood-shape",
        "square",
        "--radii",
        "100000",
        "--halo-radius=162000",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)
