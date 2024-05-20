# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of improver and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Tests for the wet-bulb-temperature CLI"""

import pytest

from improver.constants import LOOSE_TOLERANCE

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


@pytest.mark.parametrize("model_id_attr", (True, False))
def test_basic(tmp_path, model_id_attr):
    """Test basic wet bulb temperature calculation with and without the model_id_attr
    attribute."""
    test_dir = acc.kgo_root() / "wet-bulb-temperature/basic"
    input_paths = [
        test_dir / f"enukx_{p}.nc"
        for p in ("temperature", "relative_humidity", "pressure")
    ]
    output_path = tmp_path / "output.nc"
    args = [*input_paths, "--output", output_path]
    if model_id_attr:
        args += ["--model-id-attr", "mosg__model_configuration"]
        test_dir = test_dir / "with_id_attr"
    else:
        test_dir = test_dir / "without_id_attr"
    kgo_path = test_dir / "kgo.nc"
    run_cli(args)
    acc.compare(output_path, kgo_path, rtol=LOOSE_TOLERANCE)


def test_multilevel(tmp_path):
    """Test wet bulb temperature on multiple levels"""
    kgo_dir = acc.kgo_root() / "wet-bulb-temperature/multi_level"
    kgo_path = kgo_dir / "kgo.nc"
    input_paths = [
        kgo_dir / f"enukx_multilevel_{p}.nc"
        for p in ("temperature", "relative_humidity", "pressure")
    ]
    output_path = tmp_path / "output.nc"
    args = [*input_paths, "--convergence-condition", "0.005", "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path, rtol=LOOSE_TOLERANCE)


def test_global(tmp_path):
    """Test wet bulb temperature calculation on global domain"""
    kgo_dir = acc.kgo_root() / "wet-bulb-temperature/global"
    kgo_path = kgo_dir / "kgo.nc"
    input_paths = [
        kgo_dir / f"{p}_input.nc"
        for p in ("temperature", "relative_humidity", "pressure")
    ]
    output_path = tmp_path / "output.nc"
    args = [*input_paths, "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path, rtol=LOOSE_TOLERANCE)
