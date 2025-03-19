# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Extends IrisTest class with additional useful tests."""
import iris
from iris.cube import Cube, CubeList
from iris.tests import IrisTest
import imagehash
import tempfile
import inspect
from PIL import Image
import io
import os
import warnings
import pathlib

# Default perceptual hash size.
_HASH_SIZE = 16
# Default maximum perceptual hash hamming distance.
_HAMMING_DISTANCE = 2

RESULT_PATH = pathlib.Path(__file__).parent / "results"


def get_result_path(relative_path):
    """
    Returns the absolute path to a result file when given the relative path
    as a string, or sequence of strings.
    """
    if not isinstance(relative_path, str):
        relative_path = os.path.join(*relative_path)
    return os.path.abspath(os.path.join(RESULT_PATH, relative_path))


def get_test_id():
    """Get the pytest test id for the current test."""
    unique_id = os.environ.get('PYTEST_CURRENT_TEST', '').split(' ')[0].replace('/', '.')
    assert "improver_tests" in unique_id, "This function is intended for improver tests"
    unique_id = unique_id.split('.'.join(__name__.split('.')))[-1][1:] # trim away parent package dot path
    return unique_id


def assertGraphic():
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
    - http://www.hackerfactor.com/blog/index.php?/archives/529-Kind-of-Like-That.html
    - https://github.com/SciTools/iris/pull/2206

    """
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

    try:
        unique_id = get_test_id()
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


# def assertCML(cubes, reference_filename=None, checksum=True):
#     """Test that the CML for the given cubes matches the contents of
#     the reference file."""
#     if isinstance(cubes, iris.cube.Cube):
#         cubes = [cubes]
#     if reference_filename is None:
#         reference_filename = self.result_path(None, "cml")


class ImproverTest(IrisTest):
    """Extends IrisTest with a method for comparing cubes and cubelists"""

    def assertCubeEqual(self, cube_a: Cube, cube_b: Cube):
        """Uses Cube.xml method to create an easily-comparable string containing all
        meta-data and data"""
        self.assertEqual(
            cube_a.xml(checksum=True, order=False, byteorder=False),
            cube_b.xml(checksum=True, order=False, byteorder=False),
        )

    def assertCubeListEqual(self, cubelist_a: CubeList, cubelist_b: CubeList):
        """Uses CubeList.xml method to create an easily-comparable string containing all
        meta-data and data"""
        self.assertEqual(
            cubelist_a.xml(checksum=True, order=False, byteorder=False),
            cubelist_b.xml(checksum=True, order=False, byteorder=False),
        )
