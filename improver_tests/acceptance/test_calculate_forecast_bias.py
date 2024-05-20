# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of improver and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Tests for the calculate-forecast-bias CLI."""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


def test_single_frt(tmp_path):
    """
    Test case where single historical forecast value provided.
    """
    kgo_dir = acc.kgo_root() / "calculate-forecast-bias"
    kgo_path = kgo_dir / "single_frt" / "kgo.nc"
    inputs_path = (kgo_dir / "inputs").glob("20220811T0300Z-PT00*.nc")
    output_path = tmp_path / "output.nc"
    args = [
        *inputs_path,
        "--truth-attribute",
        "mosg__model_configuration=msas_det",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_multiple_frt(tmp_path):
    """
    Test case where multiple historical forecast values provided.
    """
    kgo_dir = acc.kgo_root() / "calculate-forecast-bias"
    kgo_path = kgo_dir / "multiple_frt" / "kgo.nc"
    inputs_path = (kgo_dir / "inputs").glob("202208*T0300Z-PT00*.nc")
    output_path = tmp_path / "output.nc"
    args = [
        *inputs_path,
        "--truth-attribute",
        "mosg__model_configuration=msas_det",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_single_frt_masked_inputs(tmp_path):
    """
    Test case where single historical forecast value provided.
    """
    kgo_dir = acc.kgo_root() / "calculate-forecast-bias"
    kgo_path = kgo_dir / "single_frt_masked_inputs" / "kgo.nc"
    inputs_path = (kgo_dir / "inputs/masked").glob("20220811T0300Z-PT00*.nc")
    output_path = tmp_path / "output.nc"
    args = [
        *inputs_path,
        "--truth-attribute",
        "mosg__model_configuration=msas_det",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_multiple_frt_masked_inputs(tmp_path):
    """
    Test case where multiple historical forecast values provided.
    """
    kgo_dir = acc.kgo_root() / "calculate-forecast-bias"
    kgo_path = kgo_dir / "multiple_frt_masked_inputs" / "kgo.nc"
    inputs_path = (kgo_dir / "inputs/masked").glob("202208*T0300Z-PT00*.nc")
    output_path = tmp_path / "output.nc"
    args = [
        *inputs_path,
        "--truth-attribute",
        "mosg__model_configuration=msas_det",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)
