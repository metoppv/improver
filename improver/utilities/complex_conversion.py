# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
from typing import Union

import numpy as np
from numpy import ndarray


def deg_to_complex(
    angle_deg: Union[ndarray, float], radius: Union[ndarray, float] = 1
) -> Union[ndarray, float]:
    """Converts degrees to complex values.

    The radius argument can be used to weight values. Defaults to 1.

    Args:
        angle_deg:
            3D array or float - direction angles in degrees.
        radius:
            3D array or float - radius value for each point, default=1.

    Returns:
        3D array or float - direction translated to
        complex numbers.
    """
    # Convert from degrees to radians.
    angle_rad = np.deg2rad(angle_deg)
    # Derive real and imaginary components (also known as a and b)
    real = radius * np.cos(angle_rad)
    imag = radius * np.sin(angle_rad)

    # Combine components into a complex number and return.
    return real + 1j * imag


def complex_to_deg(complex_in: ndarray) -> ndarray:
    """Converts complex to degrees.

    The "np.angle" function returns negative numbers when the input
    is greater than 180. Therefore additional processing is needed
    to ensure that the angle is between 0-359.

    Args:
        complex_in:
            3D array - direction angles in complex number form.

    Returns:
        3D array - direction in angle form

    Raises:
        TypeError: If complex_in is not an array.
    """

    if not isinstance(complex_in, np.ndarray):
        msg = "Input data is not a numpy array, but {}"
        raise TypeError(msg.format(type(complex_in)))

    angle = np.angle(complex_in, deg=True)

    # Convert angles so they are in the range [0, 360)
    angle = np.mod(np.float32(angle), 360)

    # We don't need 64 bit precision.
    angle = angle.astype(np.float32)

    return angle
