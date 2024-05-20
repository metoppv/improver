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
import pytest
from iris.cube import Cube, CubeList

from improver.utilities.common_input_handle import as_cubelist


def test_cubelist_as_cubelist():
    cube = Cube([0])
    res = as_cubelist(cube)
    assert isinstance(res, CubeList)
    assert id(res[0]) == id(cube)


def test_cube_as_cubelist():
    cube = Cube([0])
    res = as_cubelist(cube)
    assert isinstance(res, CubeList)
    assert id(res[0]) == id(cube)


def test_cube_cubelist_mixture_as_cubelist():
    cube = Cube([0])
    cubes = CubeList([cube])
    res = as_cubelist(cube, cubes)
    assert isinstance(res, CubeList)
    assert id(res[0]) == id(cube)
    assert id(res[1]) == id(cube)


def test_argument_provided():
    msg = "One or more cube should be provided."
    with pytest.raises(ValueError, match=msg):
        as_cubelist(None)


def test_no_cube_provided():
    msg = "One or more cube should be provided."
    with pytest.raises(ValueError, match=msg):
        as_cubelist([])
