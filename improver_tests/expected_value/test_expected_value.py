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
"""Unit tests for the ExpectedValue plugin."""
import numpy as np
import pytest
from iris.coords import CellMethod
from numpy.testing import assert_allclose

from improver.expected_value import ExpectedValue
from improver.synthetic_data.set_up_test_cubes import (
    set_up_percentile_cube,
    set_up_probability_cube,
    set_up_variable_cube,
)


@pytest.fixture
def realizations_cube():
    data = np.array(
        [range(0, 9), range(10, 19), range(30, 39)], dtype=np.float32
    ).reshape([3, 3, 3])
    return set_up_variable_cube(data, realizations=[0, 1, 2])


@pytest.fixture
def percentile_cube():
    data = np.array(
        [range(10, 19), range(20, 29), range(40, 49)], dtype=np.float32
    ).reshape([3, 3, 3])
    return set_up_percentile_cube(data, percentiles=[25, 50, 75])


@pytest.fixture
def threshold_cube():
    data = np.zeros([3, 4, 5], dtype=np.float32)
    return set_up_probability_cube(data, thresholds=[270, 280, 290])


def test_process_realizations_basic(realizations_cube):
    """Check that the expected value of realisations calculates the mean and
    appropriately updates metadata."""
    expval = ExpectedValue().process(realizations_cube)
    # coords should be the same, but with the first (realization) dimcoord removed
    assert expval.coords() == realizations_cube.coords()[1:]
    # a cell method indicating mean over realizations should be added
    assert expval.cell_methods == (CellMethod("mean", "realization"),)
    # mean of the realisation coord (was the first dim of data)
    expected_data = np.mean(realizations_cube.data, axis=0)
    assert_allclose(expval.data, expected_data)


def test_process_percentile_basic(percentile_cube):
    """Check that percentiles are converted to realisations and the mean is
    calculated."""
    expval = ExpectedValue().process(percentile_cube)
    # coords should be the same, but with the first (percentile) dimcoord removed
    assert expval.coords() == percentile_cube.coords()[1:]
    # a cell method indicating mean over realizations should be added
    assert expval.cell_methods == (CellMethod("mean", "realization"),)
    # this works out to be a mean over percentiles
    # since the percentiles are equally spaced
    expected_data = np.linspace(23 + 1.0 / 3.0, 31 + 1.0 / 3.0, 9).reshape([3, 3])
    assert_allclose(expval.data, expected_data)


def test_process_non_probabilistic(realizations_cube):
    """Check that attempting to process non-probabilistic data raises an exception."""
    realizations_cube.remove_coord("realization")
    with pytest.raises(Exception, match="realization"):
        ExpectedValue().process(realizations_cube)


def test_process_threshold_basic(threshold_cube):
    """Check that attempting to process threshold data (not implemented yet) raises
    an exception."""
    with pytest.raises(NotImplementedError):
        print(threshold_cube)
        ExpectedValue().process(threshold_cube)
