# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Tests for the forecast-trajectory-gap-filler CLI."""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


def test_fill_gaps(tmp_path):
    """
    Test filling gaps in the forecast trajectory using linear interpolation.

    Input cubes at T+3 and T+9 hours are provided. The plugin should
    identify that T+6 is missing and interpolate it from T+3 and T+9.
    """
    kgo_dir = acc.kgo_root() / "forecast-trajectory-gap-filler" / "gap"
    kgo_path = kgo_dir / "kgo.nc"

    # Input files at T+3 and T+9 (T+6 is missing and should be filled)
    input_files = [
        kgo_dir / "20251217T0300Z-PT0003H00M-precip_rate.nc",
        kgo_dir / "20251217T0900Z-PT0009H00M-precip_rate.nc",
    ]

    output_path = tmp_path / "output.nc"
    args = [
        *input_files,
        "--interval-in-mins",
        "180",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


@pytest.mark.slow
def test_regenerate_at_source_transitions(tmp_path):
    """
    Test regenerating forecast periods within a single forecast trajectory
    (i.e. multiple forecast periods with the same forecast reference time)
    at source transitions using cluster sources.

    Input cubes contain a cluster_sources attribute indicating transitions between
    different forecast sources. The plugin should identify these transitions and
    regenerate forecast periods within the specified interpolation window around
    the transition points.
    """
    kgo_dir = acc.kgo_root() / "forecast-trajectory-gap-filler" / "regenerate"
    kgo_path = kgo_dir / "kgo.nc"

    # Input file containing cluster sources attribute
    input_file = kgo_dir / "20251217T0000Z-precip_rate.nc"

    output_path = tmp_path / "output.nc"
    args = [
        input_file,
        "--interval-in-mins",
        "180",
        "--interpolation-method",
        "linear",
        "--cluster-sources-attribute",
        "cluster_sources",
        "--interpolation-window-in-hours",
        "3",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)
