# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2019 Met Office.
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
"""Tests for the wind direction CLI"""

import pytest as pt
from improver.cli import wind_direction
from improver.tests import kgo, cli


@pt.mark.cli
def test_no_arguments(capsys):
    """Test usage message with no arguments"""
    with pt.raises(SystemExit):
        wind_direction.main([])
    cli.check_usage_msg(capsys)


@pt.mark.cli
def test_help(capsys):
    """Test help message"""
    with pt.raises(SystemExit):
        wind_direction.main(["-h"])
    cli.check_help_msg(capsys)


@pt.mark.cli
@kgo.skipifmissing
def test_basic(tmp_path):
    """Test basic wind direction operation"""
    kgo_dir = kgo.root() / "wind_direction/basic"
    kgo_path = kgo_dir / "kgo.nc"
    output_path = tmp_path / "output.nc"
    wind_direction.main([str(kgo_dir / "input.nc"),
                         str(output_path)])
    kgo.compare(output_path, kgo_path)


@pt.mark.cli
@kgo.skipifmissing
def test_global(tmp_path):
    """Test global wind direction operation"""
    kgo_dir = kgo.root() / "wind_direction/global"
    kgo_path = kgo_dir / "kgo.nc"
    output_path = tmp_path / "output.nc"
    wind_direction.main(["--backup_method=first_realization",
                         str(kgo_dir / "input.nc"),
                         str(output_path)])
    kgo.compare(output_path, kgo_path)
