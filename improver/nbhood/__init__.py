# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""init for nbhood"""

from typing import List, Optional, Tuple, Union

from improver.utilities.common_input_handle import as_iterable


def radius_by_lead_time(
    radii: Union[str, List[str]], lead_times: Optional[List[str]] = None
) -> Tuple[Union[float, List[float], Optional[List[int]]]]:
    """
    Parse radii and lead_times provided to CLIs that use neighbourhooding.
    If no lead times are provided, return the first radius for use at all
    lead times. If lead times are provided, ensure there are sufficient
    radii to assign one to each lead time. If so return two lists, else raise
    an exception.

    Args:
        radii:
            Radii as a list provided by clize.
        lead_times:
            Lead times as a list provided by clize, or None if not set.

    Returns:
        - Radii as a float or list of floats.
        - Lead times in hours as a list of ints or None.

    Raises:
        ValueError: If multiple radii are provided without any lead times.
        ValueError: If radii and lead_times lists are on unequal lengths.
    """
    if lead_times is not None:
        lead_times = as_iterable(lead_times)
    if radii is not None:
        radii = as_iterable(radii)

    if lead_times is None:
        if not len(radii) == 1:
            raise ValueError(
                "Multiple radii have been supplied but no associated lead times."
            )
        radius_or_radii = float(radii[0])
    else:
        if not len(radii) == len(lead_times):
            raise ValueError(
                "If leadtimes are supplied, it must be a list"
                " of equal length to a list of radii."
            )
        radius_or_radii = [float(x) for x in radii]
        lead_times = [int(x) for x in lead_times]

    return radius_or_radii, lead_times
