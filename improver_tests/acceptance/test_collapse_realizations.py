# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Tests for the collapse-realizations CLI"""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


def test_basic(tmp_path):
    """
    Test mean aggregation.
    """

    kgo_dir = acc.kgo_root() / "collapse-realizations"
    kgo_path = kgo_dir / "kgo_mean.nc"
    input_path = kgo_dir / "input.nc"
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        "--method",
        "mean",
        "--new-name",
        "ensemble_mean_of_air_temperature",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_no_rename(tmp_path):
    """
    Test mean aggregation with new-name unspecified.
    """

    kgo_dir = acc.kgo_root() / "collapse-realizations"
    kgo_path = kgo_dir / "kgo_no_rename.nc"
    input_path = kgo_dir / "input.nc"
    output_path = tmp_path / "output.nc"
    args = [input_path, "--method", "mean", "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_no_realization_coord(tmp_path):
    """
    Test that an error is raised if there is no realization dimension.
    """

    kgo_dir = acc.kgo_root() / "collapse-realizations"
    input_path = kgo_dir / "input_no_realization.nc"
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        "--method",
        "mean",
        "--new-name",
        "ensemble_mean_of_air_temperature",
        "--output",
        output_path,
    ]
    with pytest.raises(ValueError):
        run_cli(args)
