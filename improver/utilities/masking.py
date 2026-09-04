# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
from typing import Union

import numpy as np
from iris.cube import Cube


def as_masked_array(data_like: Union[Cube, np.ndarray]) -> np.ma.MaskedArray:
    """Convert the data from a Cube or ndarray into a MaskedArray, masking invalid.

    Args:
        data_like:
            Input data as an Iris Cube or a numpy ndarray.
    Returns:
        Masked array with invalid values masked.
    """
    if isinstance(data_like, Cube):
        data = data_like.data
    else:
        data = data_like
    if not isinstance(data, np.ma.MaskedArray):
        return np.ma.masked_invalid(data)
    return data
