# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
""" Provides support utilities for flattening."""

from typing import List, Tuple, Union


def flatten(nested_list: Union[List, Tuple]) -> List:
    """Flatten an arbitrarily nested iterable.

    Args:
        nested_list:
            An arbitrarily nested iterable to be flattened.

    Returns:
        A list containing a flattened version of the arbitrarily nested input.
    """
    flat_list = []
    if not isinstance(nested_list, (list, tuple)):
        raise ValueError(
            f"Expected object of type list or tuple, not {type(nested_list)}"
        )
    for item in nested_list:
        if isinstance(item, (list, tuple)):
            flat_list.extend(flatten(item))
        else:
            flat_list.append(item)
    return flat_list
