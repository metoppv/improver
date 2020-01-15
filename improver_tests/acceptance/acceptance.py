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

import pytest

from improver import cli
from improver.utilities.compare import DEFAULT_TOLERANCE, compare_netcdfs

RECREATE_DIR = "RECREATE_KGO"
ACC_TEST_DIR = "IMPROVER_ACC_TEST_DIR"


def run_cli(cli_name, verbose=True):
    """
    Prepare a function for running clize CLIs.
    Use of the returned function avoids writing "improver" and the CLI name in
    each test function.

    Args:
        cli_name (str): name of the CLI
        verbose (bool): pass verbose option to CLI

    Returns:
        Callable([Iterable[str], None]): function to run the specified CLI
    """
    def run_function(args):
        cli.main("improver", cli_name, *args, verbose=verbose)
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
    return RECREATE_DIR in os.environ


def kgo_root():
    """Path to the root of the KGO directories"""
    try:
        test_dir = os.environ[ACC_TEST_DIR]
    except KeyError:
        pytest.skip()
        return pathlib.Path("/")
    return pathlib.Path(test_dir)


def kgo_exists():
    """True if KGO files exist"""
    return ACC_TEST_DIR in os.environ


def recreate_if_needed(output_path, kgo_path):
    """
    Re-create a file in the KGO, depending on configuration.

    Args:
        output_path (pathlib.Path): Path to output produced by test
        kgo_path (pathlib.Path): Path to expected/original KGO file

    Returns:
        bool: True if KGO was recreated
    """
    if not kgo_recreate():
        return False
    if not kgo_path.is_absolute():
        raise IOError("KGO path is not absolute")
    kgo_root_dir = kgo_root()
    recreate_dir = pathlib.Path(os.environ[RECREATE_DIR])
    if kgo_root_dir not in kgo_path.parents:
        raise IOError("KGO path for test is not within KGO root directory")
    if not recreate_dir.is_absolute():
        raise IOError("Recreate KGO path is not absolute")
    print("Comparison found differences - recreating KGO for this test")
    print(f"Original KGO file is at {kgo_path}")
    kgo_relative = kgo_path.relative_to(kgo_root_dir)
    recreate_path = recreate_dir / kgo_relative
    recreate_path.parent.mkdir(exist_ok=True, parents=True)
    if recreate_path.exists():
        recreate_path.unlink()
    shutil.copyfile(str(output_path), str(recreate_path))
    print(f"Updated KGO file is at {recreate_path}")
    if recreate_path.resolve() == kgo_path.resolve():
        print("KGO updated in place, rerunning this test should now pass")
    else:
        print(f"Put the updated KGO file in {ACC_TEST_DIR} to make this"
              f" test pass. For example:")
        quoted_kgo = shlex.quote(str(kgo_path))
        quoted_recreate = shlex.quote(str(recreate_path))
        print(f"cp {quoted_recreate} {quoted_kgo}")
    return True


def statsmodels_available():
    """True if statsmodels library is importable"""
    if importlib.util.find_spec('statsmodels'):
        return True
    return False


def compare(output_path, kgo_path, recreate=True,
            atol=DEFAULT_TOLERANCE, rtol=DEFAULT_TOLERANCE, exclude_vars=None):
    """
    Compare output against expected using KGO file with absolute and
    relative tolerances. Also recreates KGO if that setting is enabled.

    Args:
        output_path (pathlib.Path): Path to output produced by test
        kgo_path (pathlib.Path): Path to KGO file
        recreate (bool): False to disable KGO recreation, compare only
        atol (float): Absolute tolerance
        rtol (float): Relative tolerance
        exclude_vars (Iterable[str]): Variables to exclude from comparison

    Returns:
        None
    """
    # don't show this function in pytest tracebacks
    __tracebackhide__ = True  # pylint: disable=unused-variable
    assert output_path.is_absolute()
    assert kgo_path.is_absolute()
    if not isinstance(atol, (int, float)):
        raise ValueError("atol")
    if not isinstance(rtol, (int, float)):
        raise ValueError("rtol")

    difference_found = False
    message = ''

    def message_recorder(exception_message):
        nonlocal difference_found
        nonlocal message
        difference_found = True
        message = exception_message

    compare_netcdfs(output_path, kgo_path, atol=atol, rtol=rtol,
                    exclude_vars=exclude_vars, reporter=message_recorder)
    if difference_found:
        if recreate:
            recreate_if_needed(output_path, kgo_path)
        raise AssertionError(message)


# Pytest decorator to skip tests if KGO is not available for use
# pylint: disable=invalid-name
skip_if_kgo_missing = pytest.mark.skipif(
    not kgo_exists(), reason="KGO files required")

# Pytest decorator to skip tests if statsmodels is available
# pylint: disable=invalid-name
skip_if_statsmodels = pytest.mark.skipif(
    statsmodels_available(), reason="statsmodels library is available")

# Pytest decorator to skip tests if statsmodels is not available
# pylint: disable=invalid-name
skip_if_no_statsmodels = pytest.mark.skipif(
    not statsmodels_available(), reason="statsmodels library is not available")
