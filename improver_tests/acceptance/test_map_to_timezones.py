# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2021 Met Office.
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
"""Tests for the map-to-timezones CLI."""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]

CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)

GRIDS = ["uk", "global"]


@pytest.mark.parametrize("grid", GRIDS)
def test_basic(tmp_path, grid):
    """Test collapsing multiple input times into a single local-time output. For global,
    timezone_mask.nc is a copy of generate-timezone-mask-ancillary/global/grouped_kgo.nc
    which has 2 time-zones (-6 and +6), so only 2 input files required.
    For UK, there are 4 timezones (-2 to +1)."""

    kgo_dir = acc.kgo_root() / f"map-to-timezones/{grid}/"
    kgo_path = kgo_dir / "kgo.nc"
    input_path = kgo_dir / "input_*.nc"
    timezone_path = kgo_dir / "timezone_mask.nc"
    local_time = "20201203T0000"
    output_path = tmp_path / "output.nc"
    args = [timezone_path, local_time, input_path, "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)
