# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
""" Provides support utilities for flattening."""

from typing import Any, Generator, Iterable, Union


def flatten(nested_iterable: Union[Iterable[Any], Any]) -> Generator[Any, None, None]:
    """
    Flatten an arbitrarily nested iterable.

    Args:
        nested_iterable:
            The nested iterable to be flattened.

    Yields:
        The flattened items from the nested iterable.
    """
    if not isinstance(nested_iterable, Iterable) or isinstance(
        nested_iterable, (str, bytes)
    ):
        yield nested_iterable
        return
    for item in nested_iterable:
        if isinstance(item, Iterable) and not isinstance(item, (str, bytes)):
            yield from flatten(item)
        else:
            yield item
