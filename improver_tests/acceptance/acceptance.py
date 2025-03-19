# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Setup and checking of known good output for CLI tests"""

import functools
import hashlib
import os
import pathlib
import shlex
import shutil

import pytest

from improver import cli
from improver.constants import DEFAULT_TOLERANCE
from improver.utilities.compare import compare_netcdfs

RECREATE_DIR_ENVVAR = "RECREATE_KGO"
ACC_TEST_DIR_ENVVAR = "IMPROVER_ACC_TEST_DIR"
IGNORE_CHECKSUMS = "IMPROVER_IGNORE_CHECKSUMS"
ACC_TEST_DIR_MISSING = pathlib.Path("/dev/null")
DEFAULT_CHECKSUM_FILE = pathlib.Path(__file__).parent / "SHA256SUMS"
IGNORED_ATTRIBUTES = ["history", "Conventions"]
RESULT_PATH = pathlib.Path(__file__).parent / "results"


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
        return cli.main("improver", cli_name, *args, verbose=verbose)

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
    Verify the checksum of the provided KGO path.

    Args:
        kgo_path (pathlib.Path): Either a path to a KGO file or a path
            to a directory containing a KGO dataset.
        checksums (Optional[Dict[pathlib.Path, str]]): Lookup dictionary
            mapping from paths to hexadecimal checksums. If provided, used in
            preference to checksum_path.
        checksum_path (pathlib.Path): Path to checksum file, used if checksums is
            None. File should be plain text in the format produced by the
            sha256sum command line tool.
    """
    if checksums is None:
        checksums_dict = acceptance_checksums(checksum_path=checksum_path)
        checksums_source = checksum_path
    else:
        checksums_dict = checksums
        checksums_source = "lookup dict"

    if kgo_path.is_dir():
        for kgo_filepath in kgo_path.rglob("*"):
            if kgo_filepath.is_file():
                verify_file_checksum(kgo_filepath, checksums_dict, checksums_source)
    else:
        verify_file_checksum(kgo_path, checksums_dict, checksums_source)


def verify_file_checksum(kgo_path, checksums_dict, checksums_source):
    """
    Verify an individual KGO file's checksum.

    Args:
        kgo_path (pathlib.Path): A path to a KGO file.
        checksums_dict (Dict[pathlib.Path, str]): Dict with keys being
            relative paths and values being hexadecimal checksums.
        checksums_source (Union[pathlib.Path, str]): Path or string
            identifying the source of the checksum information.

    Raises:
        KeyError: File being verified is not found in checksum dict/file
        ValueError: Checksum does not match value in checksum dict/file
    """

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
            kgo_chunk = kgo_file.read(2**20)
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


def compare(
    output_path,
    kgo_path,
    recreate=True,
    atol=DEFAULT_TOLERANCE,
    rtol=DEFAULT_TOLERANCE,
    exclude_vars=None,
    exclude_attributes=None,
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
        exclude_attributes (Iterable[str]): Attributes to exclude from comparison

    Returns:
        None
    """
    # don't show this function in pytest tracebacks
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

    if exclude_attributes:
        if isinstance(exclude_attributes, str):
            exclude_attributes = [exclude_attributes]
        exclude_attributes.extend(IGNORED_ATTRIBUTES)
    else:
        exclude_attributes = IGNORED_ATTRIBUTES

    compare_netcdfs(
        output_path,
        kgo_path,
        atol=atol,
        rtol=rtol,
        exclude_vars=exclude_vars,
        reporter=message_recorder,
        ignored_attributes=exclude_attributes,
    )
    if difference_found:
        if recreate:
            recreate_if_needed(output_path, kgo_path)
        raise AssertionError(message)
    if not checksum_ignore():
        verify_checksum(kgo_path)


# Pytest decorator to skip tests if KGO is not available for use
skip_if_kgo_missing = pytest.mark.skipif(not kgo_exists(), reason="KGO files required")


# Default perceptual hash size.
_HASH_SIZE = 16
# Default maximum perceptual hash hamming distance.
_HAMMING_DISTANCE = 2


import imagehash
import tempfile
import inspect
from PIL import Image
import io
import warnings


def get_result_path(relative_path):
    """
    Returns the absolute path to a result file when given the relative path
    as a string, or sequence of strings.
    """
    if not isinstance(relative_path, str):
        relative_path = os.path.join(*relative_path)
    return os.path.abspath(os.path.join(RESULT_PATH, relative_path))


def check_graphic():
    """
    Compare current matplotlib.pyplot figure to a reference image.
    Checks the hamming distance between the current computed
    matplotlib.pyplot figure hash, and that computed from a reference
    image, then closes the figure.
    By default, if the reference image does not exist, the test will raise
    the typical exception associated with a missing file.
    If the environment variable ANTS_TEST_CREATE_MISSING is non-empty, the
    reference file is created if it doesn't exist.
    See Also
    --------
    http://www.hackerfactor.com/blog/index.php?/archives/529-Kind-of-Like-That.html
    """
    # Inspired by:
    # - Filename handling from https://github.com/SciTools/iris/blob/\
    # 576952d883f0118722e5334a410a176dd8072aef/lib/iris/tests/__init__.py\
    # #L626
    # - imagehash usage https://github.com/SciTools/iris/pull/2206
    def compare_images(figure, expected_filename):
        # Use imagehash to compare images fast and reliably.
        img_buffer = io.BytesIO()
        figure.savefig(img_buffer, format="png")
        img_buffer.seek(0)
        gen_phash = imagehash.phash(Image.open(img_buffer), hash_size=_HASH_SIZE)
        exp_phash = imagehash.phash(
            Image.open(expected_fname), hash_size=_HASH_SIZE
        )
        distance = abs(gen_phash - exp_phash)
        problem = distance > _HAMMING_DISTANCE
        msg = None
        if problem:
            fh = tempfile.NamedTemporaryFile(suffix=".png")
            fh.close()
            figure.savefig(fh.name, format="png")
            msg = "Bad phash {} with hamming distance {} for {} ({})"
            msg = msg.format(gen_phash, distance, expected_filename, fh.name)
        assert distance <= _HAMMING_DISTANCE, msg

    # get the pytest test id inc. py. mod. path.
    unique_id = os.environ.get('PYTEST_CURRENT_TEST', '').split(' ')[0].replace('/', '.').replace('::', '.')
    assert "improver_tests" in unique_id, "This function is intended for improver tests"
    unique_id = unique_id.split('.'.join(__name__.split('.')[:-1]))[-1][1:] # trim away package path improver_tests.acceptance

    try:
        expected_fname = get_result_path(unique_id + ".png")
        import matplotlib.pyplot as plt
        figure = plt.gcf()
        if not os.path.exists(expected_fname):
            if not os.path.isdir(os.path.dirname(expected_fname)):
                os.makedirs(os.path.dirname(expected_fname))
            msg = (
                f"Reference image '{expected_fname}' did not exist.  Reference file "
                "generated.  Commit this new file to include it."
            )
            figure.savefig(expected_fname, format="png")
            raise RuntimeError(msg)
        else:
            compare_images(figure, expected_fname)
    finally:
        plt.close()