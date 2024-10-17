# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""
Unit tests for the
`ensemble_copula_coupling._scipy_continuous_distns` scipy truncnorm workaround.

"""
import unittest

import numpy as np
import pytest
from scipy.stats import truncnorm as scipytruncnorm

from improver.ensemble_copula_coupling._scipy_continuous_distns import truncnorm

LINSPACE = np.linspace(0, 1, 10)
ARANGE = list(range(-20, 20))


@pytest.mark.parametrize(
    "method,x",
    [
        ("ppf", LINSPACE),
        ("cdf", ARANGE),
        ("sf", ARANGE),
        ("pdf", ARANGE),
        ("logpdf", ARANGE),
    ],
)
def test_method(method, x):
    """
    Test each method available for scipy truncnorm.

    Test is between the scipy v1.3.3 truncnorm and the scipy truncnorm
    within the Python environment.

    """
    loc = 0
    scale = 3
    a = -1
    b = 3
    scipy_tnorm = scipytruncnorm(a, b, loc, scale)
    our_tnorm = truncnorm(a, b, loc, scale)
    target = getattr(scipy_tnorm, method)(x)
    result = getattr(our_tnorm, method)(x)
    np.testing.assert_allclose(result, target, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
