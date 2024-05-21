# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Tests for the expected-value CLI."""


import iris
import numpy as np
import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


@pytest.mark.parametrize(
    "input_kind,atol,mae,bias",
    [
        ("realization", 0.001, 0.0, 0.0),
        ("percentile", 0.09, 0.02, 0.011),
        ("threshold", 0.156, 0.06, 0.007),
    ],
)
def test_probabilistic(tmp_path, input_kind, atol, mae, bias):
    """Test processing of probabilistic data.

    The same KGO is shared across all three representations as percentile and
    threshold inputs are derived from the realization input.
    """
    kgo_dir = acc.kgo_root() / "expected-value"
    kgo_path = kgo_dir / "kgo.nc"
    input_path = kgo_dir / f"{input_kind}.nc"
    output_path = tmp_path / "output.nc"
    args = [input_path, "--output", output_path]
    run_cli(args)
    # standard comparison, checking the maximum difference using atol
    # KGO comes from realizations, don't recreate if using other kinds of input
    is_realization = input_kind == "realization"
    acc.compare(output_path, kgo_path, rtol=0.0, atol=atol, recreate=is_realization)
    kgo_data = iris.load(str(kgo_path))[0].data
    output_data = iris.load(str(output_path))[0].data
    # custom comparison - mean absolute error
    data_mae = np.mean(np.abs(output_data - kgo_data))
    assert data_mae <= mae
    # custom comparison - bias
    data_bias = np.abs(np.mean(output_data - kgo_data))
    assert data_bias <= bias


def test_deterministic(tmp_path):
    """Test attempting to process deterministic data raises exception."""
    kgo_dir = acc.kgo_root() / "expected-value"
    input_path = kgo_dir / "deterministic.nc"
    output_path = tmp_path / "output.nc"
    args = [input_path, "--output", output_path]
    # iris CoordinateNotFoundError is a KeyError
    with pytest.raises(KeyError, match="realization"):
        run_cli(args)
