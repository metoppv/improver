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
"""Test nowcast_accumulate name_constraint function"""

import dask.array as da
import pytest
from iris import Constraint
from iris.cube import Cube, CubeList

from improver.cli.nowcast_accumulate import name_constraint


def test_all():
    """Check that cubes returned using the 'name_constraint' are not lazy"""
    constraint = name_constraint(["dummy1"])
    dummy1_cube = Cube(da.zeros((1, 1), chunks=(1, 1)), long_name="dummy2")
    dummy2_cube = Cube(da.zeros((1, 1), chunks=(1, 1)), long_name="dummy1")
    assert dummy1_cube.has_lazy_data()
    assert dummy2_cube.has_lazy_data()

    res = CubeList([dummy1_cube, dummy2_cube]).extract_cube(
        Constraint(cube_func=constraint)
    )
    assert res.name() == "dummy1"
    assert not res.has_lazy_data()


if __name__ == "__main__":
    pytest.main()
