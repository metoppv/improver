# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""
Tests for the temporal-interpolate CLI
"""

import urllib.parse
import urllib.request

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
T2M = "temperature_at_screen_level"
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


def test_basic(tmp_path):
    """Test basic interpolation"""
    kgo_dir = acc.kgo_root() / "temporal-interpolate/basic"
    kgo_path = kgo_dir / "kgo_t1.nc"
    input_paths = [
        kgo_dir / f"20190116T{v:04}Z-PT{l:04}H00M-{T2M}.nc"
        for v, l in ((900, 33), (1200, 36))
    ]
    output_path = tmp_path / "output.nc"
    args = [*input_paths, "--times", "20190116T1000Z", "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_linear_interval(tmp_path):
    """Test linear time intervals"""
    kgo_dir = acc.kgo_root() / "temporal-interpolate/basic"
    kgo_path = kgo_dir / "kgo.nc"
    input_paths = [
        kgo_dir / f"20190116T{v:04}Z-PT{l:04}H00M-{T2M}.nc"
        for v, l in ((900, 33), (1200, 36))
    ]
    output_path = tmp_path / "output.nc"
    args = [*input_paths, "--interval-in-mins", "60", "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_solar_uv_index(tmp_path):
    """Test interpolation of UV index"""
    kgo_dir = acc.kgo_root() / "temporal-interpolate/uv"
    kgo_path = kgo_dir / "kgo_t1.nc"
    input_paths = [
        kgo_dir / f"20181220T{v:04}Z-PT{l:04}H00M-uv_index.nc"
        for v, l in ((900, 21), (1200, 24))
    ]
    output_path = tmp_path / "output.nc"
    args = [
        *input_paths,
        "--times",
        "20181220T1000Z",
        "--interpolation-method",
        "solar",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_daynight(tmp_path):
    """Test interpolation of UV index with day/night method"""
    kgo_dir = acc.kgo_root() / "temporal-interpolate/uv"
    kgo_path = kgo_dir / "kgo_t1_daynight.nc"
    input_paths = [
        kgo_dir / f"20181220T{v:04}Z-PT{l:04}H00M-uv_index.nc"
        for v, l in ((900, 21), (1200, 24))
    ]
    output_path = tmp_path / "output.nc"
    args = [
        *input_paths,
        "--times",
        "20181220T1000Z",
        "--interpolation-method",
        "daynight",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_period_max(tmp_path):
    """Test interpolation of an period maximum."""
    kgo_dir = acc.kgo_root() / "temporal-interpolate/period"
    kgo_path = kgo_dir / "kgo.nc"
    input_paths = [
        kgo_dir / f"20240217T{v:04}Z-PT{l:04}H00M-wind_gust_at_10m_max-PT03H.nc"
        for v, l in ((300, 16), (600, 19))
    ]
    output_path = tmp_path / "output.nc"
    args = [
        *input_paths,
        "--times",
        "20240217T0430Z",
        "--interpolation-method",
        "linear",
        "--max",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_accumulation(tmp_path):
    """Test interpolation of an accumulation."""
    kgo_dir = acc.kgo_root() / "temporal-interpolate/accumulation"
    kgo_path = kgo_dir / "kgo.nc"
    input_paths = [
        kgo_dir / f"20240217T{v:04}Z-PT{l:04}H00M-precipitation_accumulation-PT03H.nc"
        for v, l in ((1900, 33), (2200, 36))
    ]
    output_path = tmp_path / "output.nc"
    args = [
        *input_paths,
        "--times",
        "20240217T2000Z,20240217T2100Z",
        "--interpolation-method",
        "linear",
        "--accumulation",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


@pytest.mark.slow
def test_google_film(tmp_path):
    """Test interpolation using google_film method with deep learning model."""
    model_url = "https://tfhub.dev/google/film/1"

    try:
        with urllib.request.urlopen(model_url) as response:  # noqa: S310
            if response.status != 200:
                pytest.skip(f"Google FILM model not available at {model_url}")
    except Exception:
        pytest.skip(f"Google FILM model not available at {model_url}")

    kgo_dir = acc.kgo_root() / "temporal-interpolate/google_film"
    kgo_path = kgo_dir / "kgo.nc"
    input_paths = [
        kgo_dir / "20251205T0600Z-PT0006H00M-precip_rate.nc",
        kgo_dir / "20251205T0900Z-PT0009H00M-precip_rate.nc",
    ]
    output_path = tmp_path / "output.nc"
    args = [
        *input_paths,
        "--interval-in-mins",
        "60",
        "--interpolation-method",
        "google_film",
        "--model-path",
        model_url,
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)
