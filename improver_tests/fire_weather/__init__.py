# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Common test utilities for fire weather index tests."""

from datetime import datetime

import numpy as np
from iris.cube import Cube

from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube

# Default times for test cubes
DEFAULT_FRT = datetime(2017, 11, 10, 0, 0)
DEFAULT_TIME = datetime(2017, 11, 10, 12, 0)
DEFAULT_TIME_BOUNDS = (datetime(2017, 11, 10, 0, 0), datetime(2017, 11, 10, 12, 0))


def make_cube(
    data: np.ndarray,
    name: str,
    units: str,
    add_time_coord: bool = False,
) -> Cube:
    """Create a test cube for fire weather index tests.

    This is a wrapper around set_up_variable_cube for concise cube creation
    with consistent time coordinates across all fire weather tests.

    Args:
        data:
            The data array for the cube.
        name:
            The variable name for the cube (can be standard_name or long_name).
        units:
            The units for the cube.
        add_time_coord:
            Whether to add time bounds (for accumulation periods).

    Returns:
        Iris Cube with the given properties, including forecast_reference_time
        and optionally time coordinates with bounds.
    """
    time_bounds = DEFAULT_TIME_BOUNDS if add_time_coord else None
    return set_up_variable_cube(
        data.astype(np.float32),
        name=name,
        units=units,
        frt=DEFAULT_FRT,
        time=DEFAULT_TIME,
        time_bounds=time_bounds,
    )


def make_input_cubes(
    cube_specs: list[tuple[str, float | np.ndarray, str, bool]],
    shape: tuple[int, ...] = (5, 5),
) -> tuple[Cube, ...]:
    """Create a list of test cubes for fire weather index tests.

    This is a convenience function for creating multiple input cubes with
    a consistent shape and default values.

    Args:
        cube_specs:
            List of tuples, each containing:
            (name, value, units, add_time_coord)
            - name: Variable name (standard_name or long_name)
            - value: Scalar value or ndarray to fill the cube
            - units: Units for the cube
            - add_time_coord: Whether to add time bounds
        shape:
            Shape of the grid for each cube.

    Returns:
        Tuple of Iris Cubes with the specified properties.

    Example:
        >>> cubes = make_input_cubes(
        ...     [
        ...         ("air_temperature", 20.0, "Celsius", False),
        ...         ("lwe_thickness_of_precipitation_amount", 1.0, "mm", True),
        ...     ]
        ... )
    """
    return tuple(
        make_cube(
            np.full(shape, value) if isinstance(value, (int, float)) else value,
            name,
            units,
            add_time_coord,
        )
        for name, value, units, add_time_coord in cube_specs
    )
