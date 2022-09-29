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
"""Tests for the max_in_time_window CLI"""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


def test_basic(tmp_path):
    """Test max_in_time_window computation."""
    kgo_dir = acc.kgo_root() / "max-in-time-window"
    kgo_path = kgo_dir / "kgo.nc"
    input_path1 = kgo_dir / "input_PT0029H00M.nc"
    input_path2 = kgo_dir / "input_PT0032H00M.nc"
    output_path = tmp_path / "output.nc"
    args = [input_path1, input_path2, "--output", f"{output_path}"]
    run_cli(args)
    acc.compare(output_path, kgo_path)


@pytest.mark.parametrize("realizations, gives_error", ((4, False), (5, True)))
def test_minimum_realizations(tmp_path, realizations, gives_error):
    """Test specifying a minimum number of realizations."""
    kgo_dir = acc.kgo_root() / "max-in-time-window"
    kgo_path = kgo_dir / "kgo.nc"
    input_path1 = kgo_dir / "input_PT0029H00M.nc"
    input_path2 = kgo_dir / "input_PT0032H00M.nc"
    output_path = tmp_path / "output.nc"
    args = [
        input_path1,
        input_path2,
        "--minimum-realizations",
        f"{realizations}",
        "--output",
        f"{output_path}",
    ]
    if gives_error:
        with pytest.raises(
            ValueError,
            match="After filtering, number of realizations 4 is less than the minimum number "
            rf"of realizations allowed \({realizations}\)",
        ):
            run_cli(args)
    else:
        run_cli(args)
        acc.compare(output_path, kgo_path)
