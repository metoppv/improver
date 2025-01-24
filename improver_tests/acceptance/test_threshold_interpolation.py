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
    thresholds = (
        "50.0,100.0,150.0,200.0,250.0,300.0,350.0,400.0,450.0,500.0,600.0,"
        "700.0,800.0,900.0,1000.0,1500.0,2000.0,2500.0,3000.0,3500.0,4000.0,"
        "4500.0,5000.0,5500.0,6000.0,6500.0,7000.0,7500.0,8000.0,8500.0,9000.0,"
        "9500.0,10000.0,12000.0,14000.0,16000.0,18000.0,20000.0,25000.0,30000.0,"
        "35000.0,40000.0,45000.0,50000.0"
    )
    kgo_dir = acc.kgo_root() / "threshold-interpolation"
    kgo_path = kgo_dir / "same_thresholds_kgo.nc"
    input_path = kgo_dir / "input.nc"
    output_path = tmp_path / "output.nc"
    args = [input_path, "--output", f"{output_path}"]
    if with_new_thresholds:
        args += ["--thresholds", thresholds]
        kgo_path = kgo_dir / "extra_thresholds_kgo.nc"

    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_realization_collapse(tmp_path):
    """Test realization coordinate is collapsed"""
    kgo_dir = acc.kgo_root() / "threshold-interpolation"
    kgo_path = kgo_dir / "realization_collapse_kgo.nc"
    input_path = kgo_dir / "input_realization.nc"
    output_path = tmp_path / "output.nc"
    args = [input_path, "--output", f"{output_path}"]

    run_cli(args)
    acc.compare(output_path, kgo_path)
