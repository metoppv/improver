# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Test for the threshold interpolation CLI."""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


@pytest.mark.parametrize("with_new_thresholds", (True, False))
def test_basic(tmp_path, with_new_thresholds):
    """Test basic invocation with and without threshold argument"""
    kgo_dir = acc.kgo_root() / "threshold-interpolation"
    kgo_path = kgo_dir / "kgo.nc"
    input_path = kgo_dir / "input.nc"
    output_path = tmp_path / "output.nc"
    args = [input_path, "--output", f"{output_path}"]
    if with_new_thresholds:
        args.append("--thresholds=100,150,200,250,300")
        kgo_path = kgo_dir / "kgo_with_new_thresholds.nc"

def test_realization_collapse(tmp_path):
    """Test realization coordinate is collapsed"""
    kgo_dir = acc.kgo_root() / "threshold-interpolation"
    kgo_path = kgo_dir / "kgo_with_realization.nc"
    input_path = kgo_dir / "input_realization.nc"
    output_path = tmp_path / "output.nc"
    args = [input_path, "--output", f"{output_path}"]




    run_cli(args)
    acc.compare(output_path, kgo_path)