# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown copyright. The Met Office.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
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
    output_path = tmp_path / "output.nc"
    args = [
        *input_paths,
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
    """Test an exceptions is raised by the CLI if no cubes are provided."""
    output_path = tmp_path / "output.nc"
    args = [
        "--output",
        output_path,
    ]
    with pytest.raises(RuntimeError, match="Not enough input arguments*"):
        run_cli(args)
