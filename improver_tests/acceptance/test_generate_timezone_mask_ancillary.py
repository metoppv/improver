# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2022 Met Office.
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
"""Tests for the generate-timezone-mask-ancillary CLI."""

import pytest

from . import acceptance as acc

pytest.importorskip("timezonefinder")
pytest.importorskip("numba")
pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]

CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)

GRIDS = ["uk", "global"]
TIMES = ["20210615T1200Z", "20211215T1200Z"]


@pytest.mark.parametrize("time", TIMES)
@pytest.mark.parametrize("grid", GRIDS)
def test_ignoring_dst(tmp_path, time, grid):
    """Test masks generated ignoring daylight savings time. The time of year
    should have no impact on the result, which is demonstrated here by use of a
    common kgo for both summer and winter. The kgo is checked excluding the
    validity time as this is necessarily different between summer and winter,
    whilst everything else remains unchanged."""

    kgo_dir = acc.kgo_root() / f"generate-timezone-mask-ancillary/{grid}/"
    kgo_path = kgo_dir / "ignore_dst_kgo.nc"
    input_path = kgo_dir / "input.nc"
    output_path = tmp_path / "output.nc"
    args = [input_path, "--time", f"{time}", "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path, exclude_vars=["time"])


@pytest.mark.parametrize("time", TIMES)
@pytest.mark.parametrize("grid", GRIDS)
def test_with_dst(tmp_path, time, grid):
    """Test masks generated including daylight savings time. In this case the
    time of year chosen will give different results."""

    kgo_dir = acc.kgo_root() / f"generate-timezone-mask-ancillary/{grid}/"
    kgo_path = kgo_dir / f"{time}_with_dst_kgo.nc"
    input_path = kgo_dir / "input.nc"
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        "--time",
        f"{time}",
        "--include-dst",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


@pytest.mark.parametrize("grid", GRIDS)
def test_grouping(tmp_path, grid):
    """Test masks generated with grouping produce the expected output."""

    kgo_dir = acc.kgo_root() / f"generate-timezone-mask-ancillary/{grid}/"
    kgo_path = kgo_dir / "grouped_kgo.nc"
    input_path = kgo_dir / "input.nc"
    groups = kgo_dir / "group_config.json"
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        "--time",
        "20210615T1200Z",
        "--groupings",
        groups,
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)
