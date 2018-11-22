# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2018 Met Office.
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

import iris


def uv_index(uv_upward, uv_downward, scale_factor=3.6):
    """
    A plugin to calculate the uv index using radiation flux in UV downward
    at surface, radiation flux UV upward at surface and a scaling factor.
    The scaling factor is currently set at 3.6 (for Global data) but needs
    to be configurable, since it may need to be changed after testing.

    NOTE- The STASH codes and global model files don't specify a unit for
    the upward and downward uv fluxes, so we have suggested W/m^2 for now.

    Args:
        uv_upward (iris.cube.Cube):
            A cube of the radiation flux in UV upward at surface (W/m^2)
        uv_downward (iris.cube.Cube):
            A cube of the radiation flux in UV downward at surface (W/m^2)
        scale_factor (float):
            The uv scale factor (no units)

    Returns:
        uv_index (iris.cube.Cube):
            A cube of the calculated UV index.
    """
    uv_index = (uv_upward + uv_downward) * scale_factor
    uv_index.long_name = "uv index"
    uv_index.units = "1"
    return uv_index
