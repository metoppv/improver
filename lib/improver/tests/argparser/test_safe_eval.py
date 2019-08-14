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
"""Unit tests for argparser.safe_eval."""


import unittest

import cartopy.crs as ccrs
import iris

from improver.argparser import safe_eval


class Test_safe_eval(unittest.TestCase):

    """Test function for safely using the eval command."""

    def test_iris_coords(self):
        """Test the return of an iris.coords component."""
        allowed = ['coords']
        result = safe_eval('coords', iris, allowed=allowed)
        self.assertEqual(result, iris.coords)

    def test_cartopy_projection(self):
        """Test the return of a cartopy projection."""
        allowed = ['Mercator', 'Miller']
        result = safe_eval('Mercator', ccrs, allowed=allowed)
        self.assertEqual(result, ccrs.Mercator)

    def test_unallowed_cartopy(self):
        """Test the raising of an error when requesting a projection not in the
        allowed list."""
        allowed = ['Mercator']
        msg = 'Function/method/object "Miller" not available in module'
        with self.assertRaisesRegex(TypeError, msg):
            safe_eval('Miller', ccrs, allowed=allowed)


if __name__ == '__main__':
    unittest.main()
