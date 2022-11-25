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
"""Tests for the manipulate-reliability-table CLI."""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


def test_manipulate(tmp_path):
    """
    Test manipulation of a reliability table
    """
    kgo_dir = acc.kgo_root() / "manipulate-reliability-table/basic"
    kgo_path = kgo_dir / "kgo_precip.nc"
    table_path = kgo_dir / "reliability_table_precip.nc"
    output_path = tmp_path / "output.nc"
    args = [table_path, "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_manipulate_minimum_forecast_count(tmp_path):
    """
    Test manipulation of a reliability table with an increased minimum forecast count
    """
    kgo_dir = acc.kgo_root() / "manipulate-reliability-table/basic"
    kgo_path = kgo_dir / "kgo_300_min_count.nc"
    table_path = kgo_dir / "reliability_table_cloud.nc"
    output_path = tmp_path / "output.nc"
    args = [table_path, "--minimum-forecast-count", "300", "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_manipulate_point_by_point(tmp_path):
    """
    Test manipulation of a reliability table using point_by_point functionality
    """
    kgo_dir = acc.kgo_root() / "manipulate-reliability-table/basic"
    kgo_path = kgo_dir / "kgo_point_by_point.nc"
    table_path = kgo_dir / "reliability_table_point_by_point.nc"
    output_path = tmp_path / "output.nc"
    args = [table_path, "--point-by-point", "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)
