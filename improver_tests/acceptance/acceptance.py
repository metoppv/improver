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
"""Setup and checking of known good output for CLI tests"""

import functools
import hashlib
import importlib
import os
import pathlib
import shlex
import shutil

import pytest

from improver import cli
from improver.calibration.utilities import statsmodels_available
from improver.constants import DEFAULT_TOLERANCE
from improver.utilities.compare import compare_netcdfs

RECREATE_DIR_ENVVAR = "RECREATE_KGO"
ACC_TEST_DIR_ENVVAR = "IMPROVER_ACC_TEST_DIR"
IGNORE_CHECKSUMS = "IMPROVER_IGNORE_CHECKSUMS"
ACC_TEST_DIR_MISSING = pathlib.Path("/dev/null")
DEFAULT_CHECKSUM_FILE = pathlib.Path(__file__).parent / "SHA256SUMS"
IGNORED_ATTRIBUTES = ["history", "Conventions"]


def run_cli(cli_name, verbose=True):
    """
    Prepare a function for running clize CLIs.
    Use of the returned function avoids writing "improver" and the CLI name in
    each test function.
    Checksums of input files are verified before the clize CLI is run.

    Args:
        cli_name (str): name of the CLI
        verbose (bool): pass verbose option to CLI

    Returns:
        Callable([Iterable[str], None]): function to run the specified CLI
    """

    def run_function(args):
        if not checksum_ignore():
            verify_checksums(args)
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


@functools.lru_cache()
def acceptance_checksums(checksum_path=DEFAULT_CHECKSUM_FILE):
    """
    Retrieve a list of checksums from file in text list format, as produced by
    the sha256sum command line tool.

    Args:
        checksum_path (pathlib.Path): Path to checksum file. File
            should be plain text in the format produced by the sha256sum
            command line tool. Paths listed in the file should be relative to
            the KGO root directory found by kgo_root().

    Returns:
        Dict[pathlib.Path, str]: Dict with keys being relative paths and
            values being hexadecimal checksums
    """
    if checksum_path is None:
        checksum_path = DEFAULT_CHECKSUM_FILE
    with open(checksum_path, mode="r") as checksum_file:
        checksum_lines = checksum_file.readlines()
    checksums = {}
    for line in checksum_lines:
        parts = line.strip().split("  ", maxsplit=1)
        csum = parts[0]
        path = pathlib.Path(parts[1])
        checksums[path] = csum
    return checksums


def verify_checksum(kgo_path, checksums=None, checksum_path=DEFAULT_CHECKSUM_FILE):
    """
    Verify an individual KGO file's checksum.

    Args:
        kgo_path (pathlib.Path): Path to file in KGO directory
        checksums (Optional[Dict[pathlib.Path, str]]): Lookup dictionary
            mapping from paths to hexadecimal checksums. If provided, used in
            preference to checksum_path.
        checksum_path (pathlib.Path): Path to checksum file, used if checksums is
            None. File should be plain text in the format produced by the
            sha256sum command line tool.

    Raises:
        KeyError: File being verified is not found in checksum dict/file
        ValueError: Checksum does not match value in checksum dict/file
    """
    if checksums is None:
        checksums_dict = acceptance_checksums(checksum_path=checksum_path)
        checksums_source = checksum_path
    else:
        checksums_dict = checksums
        checksums_source = "lookup dict"
    kgo_csum = calculate_checksum(kgo_path)
    kgo_norm_path = pathlib.Path(os.path.normpath(kgo_path))
    kgo_rel_path = kgo_norm_path.relative_to(kgo_root())
    try:
        expected_csum = checksums_dict[kgo_rel_path]
    except KeyError:
        msg = f"Checksum for {kgo_rel_path} missing from {checksums_source}"
        raise KeyError(msg)
    if kgo_csum != expected_csum:
        msg = (
            f"Checksum for {kgo_rel_path} is {kgo_csum}, "
            f"expected {expected_csum} in {checksums_source}"
        )
        raise ValueError(msg)


def calculate_checksum(path):
    """
    Calculate SHA256 hash/checksum of a file

    Args:
        path (pathlib.Path): Path to file

    Returns:
        str: checksum as hexadecimal string
    """
    hasher = hashlib.sha256()
    with open(path, mode="rb") as kgo_file:
        while True:
            # read 1 megabyte binary chunks from file and feed them to hasher
            kgo_chunk = kgo_file.read(2 ** 20)
            if not kgo_chunk:
                break
            hasher.update(kgo_chunk)
    checksum = hasher.hexdigest()
    return checksum


def verify_checksums(cli_arglist):
    """
    Verify input file checksums based on input arguments to a CLI.
    Intended for use inside acceptance tests, so raises exceptions to report
    various issues that should result in a test failure.

    Args:
        cli_arglist (List[Union[str,pathlib.Path]]): list of arguments being
            passed to a CLI such as via improver.cli.main function.
    """
    # copy the arglist as it will be edited to remove output args
    arglist = cli_arglist.copy()
    # if there is an --output argument, remove the path in the following argument
    try:
        output_idx = cli_arglist.index("--output")
        arglist.pop(output_idx + 1)
    except ValueError:
        pass
    # drop arguments of the form --output=file
    arglist = [
        arg
        for arg in arglist
        if not isinstance(arg, str) or not arg.startswith("--output=")
    ]
    # check for non-path-type arguments that refer to KGOs
    kgo_dir = str(kgo_root())
    path_strs = [arg for arg in arglist if isinstance(arg, str) and kgo_dir in arg]
    if path_strs:
        msg = (
            f"arg list contains KGO paths as strings {path_strs}, "
            "expected paths to be pathlib.Path objects"
        )
        raise ValueError(msg)
    # verify checksums of remaining path-type arguments
    path_args = [arg for arg in arglist if isinstance(arg, pathlib.Path)]
    for arg in path_args:
        # expand any globs in the argument and verify each of them
        arg_globs = list(arg.parent.glob(arg.name))
        for arg_glob in arg_globs:
            verify_checksum(arg_glob)


def checksum_ignore():
    """True if CHECKSUMs should be checked"""
    return os.getenv(IGNORE_CHECKSUMS, "false").lower() == "true"


def kgo_recreate():
    """True if KGO should be re-created"""
    return RECREATE_DIR_ENVVAR in os.environ


def kgo_root():
    """Path to the root of the KGO directories"""
    try:
        test_dir = os.environ[ACC_TEST_DIR_ENVVAR]
    except KeyError:
        return ACC_TEST_DIR_MISSING
    return pathlib.Path(test_dir)


def kgo_exists():
    """True if KGO files exist"""
    return not kgo_root().samefile(ACC_TEST_DIR_MISSING)


def recreate_if_needed(output_path, kgo_path, recreate_dir_path=None):
    """
    Re-create a file in the KGO, depending on configuration.

    Args:
        output_path (pathlib.Path): Path to output produced by test
        kgo_path (pathlib.Path): Path to expected/original KGO file
        recreate_dir_path (Optional[pathlib.Path]): Path to directory where
            recreated KGOs will be placed. Default is environment variable
            specified in RECREATE_DIR_ENVVAR constant.

    Returns:
        bool: True if KGO was recreated
    """
    if not kgo_recreate():
        return False
    if not kgo_path.is_absolute():
        raise IOError("KGO path is not absolute")
    if not output_path.is_file():
        raise IOError("Expected output file not created by running test")
    if recreate_dir_path is None:
        recreate_dir_path = pathlib.Path(os.environ[RECREATE_DIR_ENVVAR])
    kgo_root_dir = kgo_root()
    if kgo_root_dir not in kgo_path.parents:
        raise IOError("KGO path for test is not within KGO root directory")
    if not recreate_dir_path.is_absolute():
        raise IOError("Recreate KGO path is not absolute")
    print("Comparison found differences - recreating KGO for this test")
    if kgo_path.exists():
        print(f"Original KGO file is at {kgo_path}")
    else:
        print("Original KGO file does not exist")
    kgo_relative = kgo_path.relative_to(kgo_root_dir)
    recreate_file_path = recreate_dir_path / kgo_relative
    if recreate_file_path == kgo_path:
        err = (
            f"Recreate KGO path {recreate_file_path} must be different from"
            f" original KGO path {kgo_path} to avoid overwriting"
        )
        raise IOError(err)
    recreate_file_path.parent.mkdir(exist_ok=True, parents=True)
    if recreate_file_path.exists():
        recreate_file_path.unlink()
    shutil.copyfile(str(output_path), str(recreate_file_path))
    print(f"Updated KGO file is at {recreate_file_path}")
    print(
        f"Put the updated KGO file in {ACC_TEST_DIR_ENVVAR} to make this"
        f" test pass. For example:"
    )
    quoted_kgo = shlex.quote(str(kgo_path))
    quoted_recreate = shlex.quote(str(recreate_file_path))
    print(f"cp {quoted_recreate} {quoted_kgo}")
    return True


def iris_nimrod_patch_available():
    """True if iris_nimrod_patch library is importable"""
    if importlib.util.find_spec("iris_nimrod_patch"):
        return True
    return False


def compare(
    output_path,
    kgo_path,
    recreate=True,
    atol=DEFAULT_TOLERANCE,
    rtol=DEFAULT_TOLERANCE,
    exclude_vars=None,
):
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
    # pylint: disable=unused-variable
    __tracebackhide__ = True
    assert output_path.is_absolute()
    assert kgo_path.is_absolute()
    if not isinstance(atol, (int, float)):
        raise ValueError("atol")
    if not isinstance(rtol, (int, float)):
        raise ValueError("rtol")

    difference_found = False
    message = ""

    def message_recorder(exception_message):
        nonlocal difference_found
        nonlocal message
        difference_found = True
        message = exception_message

    compare_netcdfs(
        output_path,
        kgo_path,
        atol=atol,
        rtol=rtol,
        exclude_vars=exclude_vars,
        reporter=message_recorder,
        ignored_attributes=IGNORED_ATTRIBUTES,
    )
    if difference_found:
        if recreate:
            recreate_if_needed(output_path, kgo_path)
        raise AssertionError(message)
    if not checksum_ignore():
        verify_checksum(kgo_path)


# Pytest decorator to skip tests if KGO is not available for use
# pylint: disable=invalid-name
skip_if_kgo_missing = pytest.mark.skipif(not kgo_exists(), reason="KGO files required")

# Pytest decorator to skip tests if statsmodels is available
# pylint: disable=invalid-name
skip_if_statsmodels = pytest.mark.skipif(
    statsmodels_available(), reason="statsmodels library is available"
)

# Pytest decorator to skip tests if statsmodels is not available
# pylint: disable=invalid-name
skip_if_no_statsmodels = pytest.mark.skipif(
    not statsmodels_available(), reason="statsmodels library is not available"
)

# Pytest decorator to skip tests if iris_nimrod_patch is not available
# pylint: disable=invalid-name
skip_if_no_iris_nimrod_patch = pytest.mark.skipif(
    not iris_nimrod_patch_available(),
    reason="iris_nimrod_patch library is not available",
)
