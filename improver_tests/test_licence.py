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
"""Licence checks"""

from pathlib import Path


def self_licence():
    """Collect licence text from this file"""
    self_lines = Path(__file__).read_text().splitlines()
    licence_lines = list()
    for line in self_lines:
        if not line.startswith("#"):
            break
        licence_lines.append(line)
    licence = "\n".join(licence_lines)
    return licence


def test_py_licence():
    """
    Check that non-empty python files contain the utf8 header and
    3-clause BSD licence text
    """
    top_level = (Path(__file__).parent / "..").resolve()
    directories_covered = [top_level / "improver", top_level / "improver_tests"]
    failed_files = []
    licence_text = self_licence()
    for directory in directories_covered:
        python_files = list(directory.glob("**/*.py"))
        for file in python_files:
            contents = file.read_text()
            # skip zero-byte empty files such as __init__.py
            if len(contents) > 0 and licence_text not in contents:
                failed_files.append(str(file))
    assert len(failed_files) == 0, "\n".join(failed_files)
