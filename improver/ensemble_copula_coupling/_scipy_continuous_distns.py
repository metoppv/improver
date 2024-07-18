# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""
This module defines the truncnorm as per scipy v1.3.3 to overcome performance
issue introduced in later versions:
- https://github.com/scipy/scipy/issues/12370
- https://github.com/scipy/scipy/issues/12733

"""
import numpy as np
import scipy.special as sc
from scipy.stats._distn_infrastructure import rv_continuous

# ============================================================================
# |                        Copyright SciPy                                   |
# | Code from this point unto the termination banner is copyright SciPy.     |
# |                                                                          |
# | Copyright © 2001, 2002 Enthought, Inc.                                   |
# | All rights reserved.                                                     |
# |                                                                          |
# | Copyright © 2003-2019 SciPy Developers.                                  |
# | All rights reserved.                                                     |
# |                                                                          |
# | Redistribution and use in source and binary forms, with or without       |
# | modification, are permitted provided that the following conditions are   |
# | met:                                                                     |
# |                                                                          |
# | Redistributions of source code must retain the above copyright notice,   |
# | this list of conditions and the following disclaimer.                    |
# |                                                                          |
# | - Redistributions in binary form must reproduce the above copyright      |
# |   notice, this list of conditions and the following disclaimer in the    |
# |   documentation and/or other materials provided with the distribution.   |
# | - Neither the name of Enthought nor the names of the SciPy Developers    |
# |   may be used to endorse or promote products derived from this software  |
# |   without specific prior written permission.                             |
# |                                                                          |
# | THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS      |
# | “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT        |
# | LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A  |
# | PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR      |
# | CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,    |
# | EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,      |
# | PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR       |
# | PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF   |
# | LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING     |
# | NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS       |
# | SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.             |
# |                                                                          |
# | Further details can be found at scipy.org/scipylib/license.html          |
# ============================================================================

# Source: https://github.com/scipy/scipy/blob/v1.3.3/scipy/stats/_continuous_\
# distns.py


_norm_pdf_C = np.sqrt(2 * np.pi)
_norm_pdf_logC = np.log(_norm_pdf_C)


def _norm_pdf(x):
    return np.exp(-(x ** 2) / 2.0) / _norm_pdf_C


def _norm_logpdf(x):
    return -(x ** 2) / 2.0 - _norm_pdf_logC


def _norm_cdf(x):
    return sc.ndtr(x)


def _norm_ppf(q):
    return sc.ndtri(q)


def _norm_sf(x):
    return _norm_cdf(-x)


def _norm_isf(q):
    return -_norm_ppf(q)


class truncnorm_gen(rv_continuous):
    r"""A truncated normal continuous random variable.

    %(before_notes)s

    Notes
    -----
    The standard form of this distribution is a standard normal truncated to
    the range [a, b] --- notice that a and b are defined over the domain of the
    standard normal.  To convert clip values for a specific mean and standard
    deviation, use::

        a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std

    `truncnorm` takes :math:`a` and :math:`b` as shape parameters.

    %(after_notes)s

    %(example)s

    """

    def _argcheck(self, a, b):
        return a < b

    def _get_support(self, a, b):
        return a, b

    def _get_norms(self, a, b):
        _nb = _norm_cdf(b)
        _na = _norm_cdf(a)
        _sb = _norm_sf(b)
        _sa = _norm_sf(a)
        _delta = np.where(a > 0, _sa - _sb, _nb - _na)
        with np.errstate(divide="ignore"):
            return _na, _nb, _sa, _sb, _delta, np.log(_delta)

    def _pdf(self, x, a, b):
        ans = self._get_norms(a, b)
        _delta = ans[4]
        return _norm_pdf(x) / _delta

    def _logpdf(self, x, a, b):
        ans = self._get_norms(a, b)
        _logdelta = ans[5]
        return _norm_logpdf(x) - _logdelta

    def _cdf(self, x, a, b):
        ans = self._get_norms(a, b)
        _na, _delta = ans[0], ans[4]
        return (_norm_cdf(x) - _na) / _delta

    def _ppf(self, q, a, b):
        # XXX Use _lazywhere...
        ans = self._get_norms(a, b)
        _na, _nb, _sa, _sb = ans[:4]
        ppf = np.where(
            a > 0,
            _norm_isf(q * _sb + _sa * (1.0 - q)),
            _norm_ppf(q * _nb + _na * (1.0 - q)),
        )
        return ppf


truncnorm = truncnorm_gen(name="truncnorm")


# ============================================================================
# |                        END SciPy copyright                               |
# ============================================================================
