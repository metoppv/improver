# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.

"""Spot data utilities."""


def get_neighbour_finding_method_name(land_constraint: bool, minimum_dz: bool) -> str:
    """
    Create a name to describe the neighbour method based on the constraints
    provided.

    Returns:
        A string that describes the neighbour finding method employed.
        This is essentially a concatenation of the options.
    """
    method_name = "{}{}{}".format(
        "nearest",
        "_land" if land_constraint else "",
        "_minimum_dz" if minimum_dz else "",
    )
    return method_name
