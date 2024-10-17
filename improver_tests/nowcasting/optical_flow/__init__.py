# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Set up function for dummy cube with correct metadata"""

from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube


def set_up_test_cube(data, name, units, time):
    """Template for cube metadata with 2 km coordinate spacing and zero
    forecast period

    Args:
        data (numpy.ndarray)
        name (str)
        units (str)
        time (datetime.datetime)

    Returns:
        iris.cube.Cube
    """
    return set_up_variable_cube(
        data,
        name=name,
        units=units,
        spatial_grid="equalarea",
        time=time,
        frt=time,
        x_grid_spacing=2000,
        y_grid_spacing=2000,
        domain_corner=(0, 0),
    )
