# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""
Tests for the wet-bulb-temperature-integral CLI
"""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


@pytest.mark.parametrize("model_id_attr", (True, False))
def test_basic(tmp_path, model_id_attr):
    """Test basic wet bulb temperature integral calculation with and without the
    model_id_attr attribute."""
    test_dir = acc.kgo_root() / "wet-bulb-temperature-integral/basic"
    output_path = tmp_path / "output.nc"
    input_path = test_dir / "input.nc"
    args = [input_path, "--output", output_path]
    if model_id_attr:
        args += ["--model-id-attr", "mosg__model_configuration"]
        test_dir = test_dir / "with_id_attr"
    else:
        test_dir = test_dir / "without_id_attr"
    kgo_path = test_dir / "kgo.nc"
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_realizations(tmp_path):
    """Test wet bulb temperature integral calculation with a realization
    coord on the input cube"""
    kgo_dir = acc.kgo_root() / "wet-bulb-temperature-integral/realizations"
    kgo_path = kgo_dir / "kgo.nc"
    output_path = tmp_path / "output.nc"
    input_path = kgo_dir / "input.nc"
    args = [input_path, "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)
