# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Tests for the categorical-modes CLI"""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


@pytest.mark.parametrize(
    "test_path",
    [
        "gridded_input",
        "spot_input",
        "gridded_ties",
        "spot_ties",
        "blend_mismatch_inputs",
        "single_input",
    ],
)
@pytest.mark.slow
def test_expected(tmp_path, test_path):
    """Test categorical modal calculation returns the expected results with weather symbol data.
    The tests are:

        - simple gridded / spot data input
        - gridded / spot data input engineered to provide many ties that are
          solved using grouping
        - a night-time code test using spot data
        - spot data where one input has a different blend-time to the rest
        - a single input file rather than multiple
    """
    kgo_dir = acc.kgo_root() / "categorical-modes" / test_path
    kgo_path = kgo_dir / "kgo.nc"
    input_paths = (kgo_dir).glob("202012*.nc")
    wxtree = acc.kgo_root() / "categorical-modes" / "wx_decision_tree.json"
    output_path = tmp_path / "output.nc"
    args = [
        *input_paths,
        "--decision-tree",
        wxtree,
        "--model-id-attr",
        "mosg__model_configuration",
        "--record-run-attr",
        "mosg__model_run",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_no_input(tmp_path):
    """Test an exception is raised by the CLI if no cubes are provided."""
    wxtree = acc.kgo_root() / "categorical-modes" / "wx_decision_tree.json"
    output_path = tmp_path / "output.nc"
    args = [
        "--decision-tree",
        wxtree,
        "--output",
        output_path,
    ]
    with pytest.raises(RuntimeError, match="Not enough input arguments*"):
        run_cli(args)
