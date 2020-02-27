# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2019 Met Office.
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
at surface and radiation flux in UV upward at the surface."""

from improver.metadata.utilities import (
    create_new_diagnostic_cube, generate_mandatory_attributes)


def calculate_uv_index(uv_upward, uv_downward, scale_factor=3.6,
                       model_id_attr=None):
    """
    A plugin to calculate the uv index using radiation flux in UV downward
    at surface, radiation flux UV upward at surface and a scaling factor.
    The scaling factor is configurable by the user.

    Args:
        uv_upward (iris.cube.Cube):
            A cube of the radiation flux in UV upward at surface. This is a
            UM diagnostic produced by the UM radiation scheme.
            This band covers 200-320 nm and uses six absorption coefficients
            for ozone and one Rayleigh scattering coefficient(W m-2)
        uv_downward (iris.cube.Cube):
            A cube of the radiation flux in UV downward at surface.
            This is a UM diagnostic produced by the UM radiation scheme
            see above or the paper referenced for more details.(W m-2)
        scale_factor (float):
            The uv scale factor. Default is 3.6. This factor has
            been empirically derived and should not be
            changed except if there are scientific reasons to
            do so. For more information see section 2.1.1 of the paper
            referenced below (no units)
        model_id_attr (str):
            Name of the attribute used to identify the source model for
            blending.

    Returns:
        iris.cube.Cube:
            A cube of the calculated UV index.

    Raises:
        ValueError: If uv_upward is not named correctly.
        ValueError: If uv_downward is not named correctly.
        ValueError: If units do not match.

    References:
        Turner, E.C, Manners, J. Morcette, C. J, O'Hagan, J. B,
        & Smedley, A.R.D. (2017): Toward a New UV Index Diagnostic
        in the Met Office's Forecast Model. Journal of Advances in
        Modeling Earth Systems 9, 2654-2671.

    """
    if uv_upward.name() != 'surface_upwelling_ultraviolet_flux_in_air':
        msg = ("The radiation flux in UV upward has the wrong name, "
               "it should be "
               "surface_upwelling_ultraviolet_flux_in_air "
               "but is {}".format(uv_upward.name()))
        raise ValueError(msg)
    if uv_downward.name() != 'surface_downwelling_ultraviolet_flux_in_air':
        msg = ("The radiation flux in UV downward has the wrong name, "
               "it should be "
               "surface_downwelling_ultraviolet_flux_in_air "
               "but is {}".format(uv_downward.name()))
        raise ValueError(msg)
    if uv_upward.units != uv_downward.units:
        msg = "The input uv files do not have the same units."
        raise ValueError(msg)

    uv_data = (uv_upward.data + uv_downward.data) * scale_factor
    attributes = generate_mandatory_attributes([
        uv_upward, uv_downward], model_id_attr=model_id_attr)
    uv_index = create_new_diagnostic_cube(
        "ultraviolet_index", "1", uv_upward, attributes, data=uv_data)

    return uv_index
