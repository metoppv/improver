# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2022 Met Office.
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
