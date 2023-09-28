# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown copyright. The Met Office.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
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
    uv_downward: Cube,
    scale_factor: float = 3.6,
    model_id_attr: Optional[str] = None,
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
