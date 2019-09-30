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
"""Checker functions for CLI tests"""

USAGE_LENGTH = 10


def check_usage_msg(capsys):
    """
    Check that a usage style message has been written to stderr.
    Raises assertions to be picked up by test framework.

    Args:
        capsys (_pytest.capture.CaptureFixture): captured output

    Returns:
        None
    """
    captured = capsys.readouterr()
    assert captured.err.startswith("usage: ")
    assert len(captured.err.splitlines()) < USAGE_LENGTH


def check_help_msg(capsys):
    """
    Check that a help style message has been written to stdout.
    Raises assertions to be picked up by test framework.

    Args:
        capsys (_pytest.capture.CaptureFixture): captured output

    Returns:
        None
    """
    captured = capsys.readouterr()
    assert captured.out.startswith("usage: ")
    assert len(captured.out.splitlines()) > USAGE_LENGTH
    assert "positional arguments" in captured.out
    assert "optional arguments:" in captured.out
