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
"""
Unit tests for the
`ensemble_copula_coupling._scipy_continuous_distns` scipy truncnorm workaround.

"""
import unittest

import numpy as np
from iris.tests import IrisTest
from scipy.stats import truncnorm as scipytruncnorm

from improver.ensemble_copula_coupling._scipy_continuous_distns import truncnorm


class Test_truncnorm(IrisTest):
    def run_method(self, x, method):
        loc = 0
        scale = 3
        a = -1
        b = 3
        scipy_tnorm = scipytruncnorm(a, b, loc, scale)
        our_tnorm = truncnorm(a, b, loc, scale)
        tar = getattr(scipy_tnorm, method)(x)
        res = getattr(our_tnorm, method)(x)
        self.assertArrayAlmostEqual(res, tar)

    def test_ppf(self):
        x = np.linspace(0, 1, 10)
        self.run_method(x, "ppf")

    def test_cdf(self):
        x = list(range(-20, 20))
        self.run_method(x, "cdf")

    def test_sf(self):
        x = list(range(-20, 20))
        self.run_method(x, "sf")

    def test_pdf(self):
        x = list(range(-20, 20))
        self.run_method(x, "pdf")

    def test_logpdf(self):
        x = list(range(-20, 20))
        self.run_method(x, "logpdf")


if __name__ == "__main__":
    unittest.main()
