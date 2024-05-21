# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
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
