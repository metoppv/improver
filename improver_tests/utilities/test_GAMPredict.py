# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the GAMPredict class within statistical.py"""

import numpy as np
import pytest
from pygam import GAM, f, s
from pygam.datasets import wage

from improver.utilities.statistical import GAMPredict


@pytest.mark.parametrize(
    "X_new,expected",
    [
        (
            [[2006, 32, 1]],
            [92.13292996],
        ),  # Test prediction of a single new value
        (
            [[2006, 32, 1], [2009, 18, 1], [2004, 78, 4]],
            [92.13292996, 66.06155349, 132.48350915],
        ),  # Test prediction of multiple new values
        (
            [[2010, 90, 4], [2050, 160, 2]],
            [160.20340934, 284.10127629],
        ),  # Test prediction of new values where the continuous inputs are greater than those used in training to
            # demonstrate that we can extrapolate beyond the bounds of the training dataset.
        (
            [[2002, 15, 0], [1950, 1, 2]],
            [43.40042416, -161.03952442],
        ),  # Test prediction of new values where the continuous inputs are less than those used in training to
            # demonstrate that we can extrapolate beyond the bounds of the training dataset. This test also demonstrates
            # that extrapolation can lead to nonsensical results, such as a negative wage.
    ],
)
def test_process(X_new, expected):
    """Test that the process method returns the expected results. Uses an example of a fitted model from pyGAM quick
    start documentation: https://pygam.readthedocs.io/en/latest/notebooks/quick_start.html#Fit-a-Model."""
    X, y = wage()

    gam = GAM(s(0) + s(1) + f(2)).fit(X, y)
    result = GAMPredict().process(gam, X_new)

    np.testing.assert_array_almost_equal(result, expected)
