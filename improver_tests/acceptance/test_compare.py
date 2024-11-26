# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Tests for the compare CLI"""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI, verbose=False)


def test_same(capsys):
    """Compare identical files, should not produce any output"""
    kgo_dir = acc.kgo_root()
    input_file = kgo_dir / "wind_downscaling/basic/highres_orog.nc"
    matching_file = kgo_dir / "wind_downscaling/veg/highres_orog.nc"
    args = [input_file, matching_file]
    run_cli(args)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_different(capsys):
    """Compare different files, should report differences to stdout"""
    kgo_dir = acc.kgo_root()
    a_file = kgo_dir / "generate-percentiles/basic/input.nc"
    b_file = kgo_dir / "threshold/basic/input.nc"
    args = [a_file, b_file]
    run_cli(args)
    captured = capsys.readouterr()
    assert "different dimension size" in captured.out
    assert "different variables" in captured.out
    assert "different data" in captured.out


def test_ignored_attributes(capsys):
    """Ensure attribute differences are not reported if explicity excluded."""
    kgo_dir = acc.kgo_root()
    a_file = kgo_dir / "spot-extract/inputs/all_methods_uk_unique_ids.nc"
    b_file = kgo_dir / "spot-extract/inputs/all_methods_global.nc"
    args = [a_file, b_file, "--ignored-attributes=model_grid_hash"]
    run_cli(args)
    captured = capsys.readouterr()
    assert "different attribute value" not in captured.out
