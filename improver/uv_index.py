# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Module for calculating the uv index using radiation flux in UV downward
at the surface."""

from typing import Optional

import numpy as np
from iris.cube import Cube

from improver.metadata.utilities import (
    create_new_diagnostic_cube,
    generate_mandatory_attributes,
)


def calculate_uv_index(
    uv_downward: Cube, scale_factor: float = 3.6, model_id_attr: Optional[str] = None,
) -> Cube:
    """
    A plugin to calculate the uv index using radiation flux in UV downward
    at the surface and a scaling factor.
    The scaling factor is configurable by the user.

    Args:
        uv_downward:
            A cube of the radiation flux in UV downward at surface.
            This is a UM diagnostic produced by the UM radiation scheme
            see above or the paper referenced for more details.(W m-2)
        scale_factor:
            The uv scale factor. Default is 3.6 (m2 W-1). This factor has
            been empirically derived and should not be
            changed except if there are scientific reasons to
            do so. For more information see section 2.1.1 of the paper
            referenced below.
        model_id_attr:
            Name of the attribute used to identify the source model for
            blending.

    Returns:
        A cube of the calculated UV index.

    Raises:
        ValueError: If uv_downward is not named correctly.
        ValueError: If uv_downward contains values that are negative or
        not a number.

    References:
        Turner, E.C, Manners, J. Morcrette, C. J, O'Hagan, J. B,
        & Smedley, A.R.D. (2017): Toward a New UV Index Diagnostic
        in the Met Office's Forecast Model. Journal of Advances in
        Modeling Earth Systems 9, 2654-2671.

    """

    if uv_downward.name() != "surface_downwelling_ultraviolet_flux_in_air":
        msg = (
            "The radiation flux in UV downward has the wrong name, "
            "it should be "
            "surface_downwelling_ultraviolet_flux_in_air "
            "but is {}".format(uv_downward.name())
        )
        raise ValueError(msg)

    if np.any(uv_downward.data < 0) or np.isnan(uv_downward.data).any():
        msg = (
            "The radiation flux in UV downward contains data "
            "that is negative or NaN. Data should be >= 0."
        )
        raise ValueError(msg)

    uv_downward.convert_units("W m-2")
    uv_data = uv_downward.data * scale_factor
    attributes = generate_mandatory_attributes(
        [uv_downward], model_id_attr=model_id_attr
    )
    uv_index = create_new_diagnostic_cube(
        "ultraviolet_index", "1", uv_downward, attributes, data=uv_data
    )

    return uv_index
