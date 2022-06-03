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

    The same KGO is shared across all representations. Processing realizations is
    straightforward and accurate, so has low error tolerances. Processing percentiles
    is also straightforward, but with some variation in the result due to representation
    differences. Processing thresholds is quite rough, due to the currently implemented
    method of conversion to percentiles. Expect this accuracy to improve if the expected
    value is directly calculcated from the threshold data.
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
    print(data_bias)
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
