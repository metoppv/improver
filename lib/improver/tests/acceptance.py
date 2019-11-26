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
"""Setup and checking of known good output for CLI tests"""

import os
import pathlib
import shutil
import subprocess
import tempfile

import pytest

TIGHT_TOLERANCE = 1e-5
DEFAULT_TOLERANCE = 1e-4
LOOSE_TOLERANCE = 1e-2


def kgo_recreate():
    """True if KGO should be re-created"""
    return "RECREATE_KGO" in os.environ


def kgo_root():
    """Path to the root of the KGO directories"""
    return pathlib.Path(os.environ["IMPROVER_ACC_TEST_DIR"])


def kgo_exists():
    """True if KGO files exist"""
    return "IMPROVER_ACC_TEST_DIR" in os.environ


def recreate_if_needed(output_path, kgo_path):
    """
    Re-create a file in the KGO, depending on configuration.

    Args:
        output_path (pathlib.Path): Path to output produced by test
        kgo_path (pathlib.Path): Path to KGO file

    Returns:
        None
    """
    if not kgo_recreate():
        return
    if not kgo_path.is_absolute():
        raise IOError("KGO path is not absolute")
    if kgo_root() not in kgo_path.parents:
        raise IOError("Provided KGO path is not within KGO root directory")
    kgo_path.parent.mkdir(exist_ok=True)
    shutil.copyfile(output_path, kgo_path)
    return


def temporary_copy():
    """True if using temporary copy of output/KGO files"""
    return "IMPROVER_ACC_TEST_TEMP" in os.environ


def dechunk_temporary(prefix, input_path, temp_dir):
    """
    Create a copy of a netcdf file, with compression and chunking removed.
    There is no cleanup here - code calling this function should do that.

    Args:
        prefix (str): Filename prefix
        input_path (pathlib.Path): Path to input file
        temp_dir (pathlib.Path): Path to directory where output will be placed

    Returns:
        pathlib.Path: Path to created file in temp_dir
    """
    temp_file = tempfile.mkstemp(prefix=prefix, suffix='.nc',
                                 dir=temp_dir)[1]
    temp_path = pathlib.Path(temp_file)
    cmd = ["ncks", "-O", "-4", "--cnk_plc=unchunk", "--dfl_lvl=0",
           str(input_path), str(temp_path)]
    completion = subprocess.run(cmd, timeout=15,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
    assert completion.returncode == 0
    return temp_path


def nccmp_available():
    """True if nccmp tool is available in path"""
    return shutil.which("nccmp") is not None


def nccmp(output_path, kgo_path,
          atol=DEFAULT_TOLERANCE, rtol=DEFAULT_TOLERANCE):
    """
    Compare output and KGO using nccmp command line tool.
    Raises assertions to be picked up by test framework.

    Args:
        output_path (pathlib.Path): Path to output produced by test
        kgo_path (pathlib.Path): Path to KGO file
        atol (float): absolute tolerance
        rtol (float): relative tolerance

    Returns:
        None
    """
    # nccmp options:
    #    -d means compare data
    #    -m also compare metadata
    #    -N ignore NaN comparisons
    #    -s report 'Files X and Y are identical' if they really are.
    #    -g compares global attributes in the file
    #    -b verbose output

    cmd = ["nccmp", "-dmNsgb", "-t", str(atol), "-T", str(rtol),
           str(output_path), str(kgo_root() / kgo_path)]
    print(" ".join(cmd))
    completion = subprocess.run(cmd,
                                timeout=30,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
    print(completion.stdout.decode())
    assert completion.returncode == 0
    assert b"are identical." in completion.stdout
    return


def compare(output_path, kgo_path,
            atol=DEFAULT_TOLERANCE, rtol=DEFAULT_TOLERANCE):
    """
    Compare output against expected using KGO file with absolute and
    relative tolerances. Also recreates KGO if that setting is enabled.

    Args:
        output_path (pathlib.Path): Path to output produced by test
        kgo_path (pathlib.Path): Path to KGO file
        atol (float): absolute tolerance
        rtol (float): relative tolerance

    Returns:
        None
    """
    assert output_path.is_absolute()
    assert kgo_path.is_absolute()
    recreate_if_needed(output_path, kgo_path)
    if temporary_copy():
        temporary_kgo = dechunk_temporary(
            "kgo", kgo_path, output_path.parent)
        temporary_output = dechunk_temporary(
            "out", output_path, output_path.parent)
        nccmp(temporary_output, temporary_kgo, atol, rtol)
    else:
        nccmp(output_path, kgo_path, atol, rtol)
    return


# Pytest decorator to skip tests if KGO is not available for use
# pylint: disable=invalid-name
skip_if_kgo_missing = pytest.mark.skipif(
    not kgo_exists() and nccmp_available(),
    reason="KGO files and nccmp tool required")
