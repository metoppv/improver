# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Rounding utility"""

from typing import Type, Union

import numpy as np
from numpy import ndarray


def round_close(
    data: Union[float, ndarray], dtype: Type = np.int64
) -> Union[int, ndarray]:
    """Casts input data to the nearest integer value, where the input
    data is expected to be very close to the nearest integer.

    Args:
        data
            Data that is very close to the nearest integer value
        dtype:
            Required integer datatype

    Returns:
        Rounded data value

    Raises:
        ValueError: If rounding would significantly change the input value
    """
    new_data = np.around(data).astype(dtype)
    if not np.allclose(np.array(data), np.array(new_data), atol=1e-7):
        msg = "Input to 'round_close' {} is not integer-equivalent"
        raise ValueError(msg.format(data))
    return new_data
