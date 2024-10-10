#!/usr/bin/env python
# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""
Tests for the manipulate-n-realizations CLI
"""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


@pytest.mark.parametrize(
    "manipulation, kgo", (("extend", "extend_kgo.nc"), ("reduce", "reduce_kgo.nc"),),
)
def test_basic(tmp_path, manipulation, kgo):
    """Test basic extension/reduction of realization dimension"""
    kgo_dir = acc.kgo_root() / "manipulate-n-realizations"
    kgo_path = kgo_dir / kgo
    input_path = kgo_dir / "input.nc"

    if manipulation == "extend":
        n_realizations = "16"
    else:
        n_realizations = "8"

    output_path = tmp_path / "output.nc"

    args = [input_path, n_realizations, "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)
