# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2021 Met Office.
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
"""Script to calculate a hail fraction."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(*cubes: cli.inputcubelist, model_id_attr: str = None):
    """
    Calculates the fraction of precipitation that is forecast to fall as hail.

    Args:
        cubes (iris.cube.CubeList or list):
            Contains cubes of the maximum vertical updraught, hail size,
            cloud condensation level temperature, convective cloud top temperature,
            altitude of hail to rain falling level and the altitude of the orography.
        model_id_attr (str):
            Name of the attribute used to identify the source model for
            blending.

    Returns:
        iris.cube.Cube:
            A single cube containing the hail fraction.

    """
    from collections.abc import Iterable

    from iris.cube import CubeList

    from improver.precipitation_type.hail_fraction import HailFraction

    def flatten(alist):
        for item in alist:
            if isinstance(item, Iterable) and not isinstance(item, (str, bytes)):
                yield from flatten(item)
            else:
                yield item

    cubelist = CubeList(flatten(cubes))

    (
        vertical_updraught,
        hail_size,
        cloud_condensation_level,
        convective_cloud_top,
        hail_melting_level,
        altitude,
    ) = cubelist.extract(
        [
            "maximum_vertical_updraught",
            "diameter_of_hail_stones",
            "air_temperature_at_condensation_level",
            "air_temperature_at_convective_cloud_top",
            "altitude_of_rain_from_hail_falling_level",
            "surface_altitude",
        ]
    )

    return HailFraction(model_id_attr=model_id_attr)(
        vertical_updraught,
        hail_size,
        cloud_condensation_level,
        convective_cloud_top,
        hail_melting_level,
        altitude,
    )
