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

import importlib
import os
import pathlib
import shlex
import shutil
import subprocess
import tempfile

import pytest

from improver import cli

TIGHT_TOLERANCE = 1e-4
DEFAULT_TOLERANCE = 1e-2
LOOSE_TOLERANCE = 1e-1


def run_cli(cli_name):
    """
    Prepare a function for running clize CLIs.
    Use of the returned function avoids writing "improver" and the CLI name in
    each test function.

    Args:
        cli_name (str): name of the CLI

    Returns:
        Callable([Iterable[str], None]): function to run the specified CLI
    """
    def run_function(args):
        cli_args = ["improver", "--verbose", cli_name, *args]
        cli.run_main(cli_args)
    return run_function


def cli_name_with_dashes(dunder_file):
    """
    Convert an acceptance test module name to the corresponding CLI

    Args:
        dunder_file (str): test module name retrieved from __file__

    Returns:
        str: CLI name
    """
    module_name = str(pathlib.Path(dunder_file).stem)
    if module_name.startswith("test_"):
        module_name = module_name[5:]
    module_dashes = module_name.replace("_", "-")
    return module_dashes


def kgo_recreate():
    """True if KGO should be re-created"""
    return "RECREATE_KGO" in os.environ


def kgo_root():
    """Path to the root of the KGO directories"""
    try:
        test_dir = os.environ["IMPROVER_ACC_TEST_DIR"]
    except KeyError:
        pytest.skip()
        return pathlib.Path("/")
    return pathlib.Path(test_dir)


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
    shutil.copyfile(str(output_path), str(kgo_path))
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
    print(' '.join([shlex.quote(x) for x in cmd]))
    completion = subprocess.run(cmd, timeout=15, check=False,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
    print(completion.stdout.decode())
    assert completion.returncode == 0
    return temp_path


def nccmp_available():
    """True if nccmp tool is available in path"""
    return shutil.which("nccmp") is not None


def statsmodels_available():
    """True if statsmodels library is importable"""
    if importlib.util.find_spec('statsmodels'):
        return True
    return False


def nccmp(output_path, kgo_path, exclude_dims=None, options=None,
          atol=DEFAULT_TOLERANCE, rtol=DEFAULT_TOLERANCE):
    """
    Compare output and KGO using nccmp command line tool.
    Raises assertions to be picked up by test framework.

    Args:
        output_path (pathlib.Path): Path to output produced by test
        kgo_path (pathlib.Path): Path to KGO file
        exclude_dims (Optional[List[str]]): dimensions to exclude
        options (Optional[str]): comparison options
        atol (Optional[float]): absolute tolerance
        rtol (Optional[float]): relative tolerance

    Returns:
        None
    """
    # don't show this function in pytest tracebacks
    __tracebackhide__ = True  # pylint: disable=unused-variable
    if atol is not None:
        atol_args = ["-t", str(atol)]
    else:
        atol_args = []
    if rtol is not None:
        rtol_args = ["-T", str(rtol)]
    else:
        rtol_args = []
    # nccmp options:
    #    -d means compare data
    #    -m also compare metadata
    #    -N ignore NaN comparisons
    #    -s report 'Files X and Y are identical' if they really are.
    #    -g compares global attributes in the file
    #    -b verbose output
    if options is None:
        options = "-dmNsg"
    if exclude_dims is not None:
        exclude_args = [f"--exclude={x}" for x in exclude_dims]
    else:
        exclude_args = []
    cmd = ["nccmp", options, *exclude_args, *atol_args, *rtol_args,
           str(output_path), str(kgo_root() / kgo_path)]
    print(' '.join([shlex.quote(x) for x in cmd]))
    completion = subprocess.run(cmd, timeout=300, check=False,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
    print(completion.stdout.decode())
    assert completion.returncode == 0
    assert b"are identical." in completion.stdout


def compare(output_path, kgo_path, recreate=True, **kwargs):
    """
    Compare output against expected using KGO file with absolute and
    relative tolerances. Also recreates KGO if that setting is enabled.

    Args:
        output_path (pathlib.Path): Path to output produced by test
        kgo_path (pathlib.Path): Path to KGO file
        recreate (bool): False to disable KGO recreation, compare only

    Keyword Args:
        exclude_dims (Optional[List[str]]): dimensions to exclude
        options (Optional[str]): comparison options
        atol (Optional[float]): absolute tolerance
        rtol (Optional[float]): relative tolerance

    Returns:
        None
    """
    # don't show this function in pytest tracebacks
    __tracebackhide__ = True  # pylint: disable=unused-variable
    assert output_path.is_absolute()
    assert kgo_path.is_absolute()
    if recreate:
        recreate_if_needed(output_path, kgo_path)
    if temporary_copy():
        temporary_kgo = dechunk_temporary(
            "kgo-", kgo_path, output_path.parent)
        temporary_output = dechunk_temporary(
            "out-", output_path, output_path.parent)
        nccmp(temporary_output, temporary_kgo, **kwargs)
    else:
        nccmp(output_path, kgo_path, **kwargs)


# Pytest decorator to skip tests if KGO is not available for use
# pylint: disable=invalid-name
skip_if_kgo_missing = pytest.mark.skipif(
    not kgo_exists() and nccmp_available(),
    reason="KGO files and nccmp tool required")


# Pytest decorator to skip tests if statsmodels is available
# pylint: disable=invalid-name
skip_if_statsmodels = pytest.mark.skipif(
    statsmodels_available(), reason="statsmodels library is available")

# Pytest decorator to skip tests if statsmodels is not available
# pylint: disable=invalid-name
skip_if_no_statsmodels = pytest.mark.skipif(
    not statsmodels_available(), reason="statsmodels library is not available")
