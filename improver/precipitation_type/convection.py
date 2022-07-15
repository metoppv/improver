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
"""Module containing convection diagnosis utilities."""

from typing import List, Optional, Union

import iris
import numpy as np
from iris.cube import Cube, CubeList
from numpy import ndarray

from improver.bases import BasePlugin
from improver.metadata.utilities import (
    create_new_diagnostic_cube,
    generate_mandatory_attributes,
)


class ConvectionRatioFromComponents(BasePlugin):
    """
    Diagnose the convective precipitation ratio by using differences between
    convective and dynamic components.
    """

    def __init__(self) -> None:
        self.convective = None
        self.dynamic = None

    def _split_input(self, cubes: Union[CubeList, List[Cube]]) -> None:
        """
        Extracts convective and dynamic components from the list as objects on the class
        and ensures units are m s-1
        """
        if not isinstance(cubes, iris.cube.CubeList):
            cubes = iris.cube.CubeList(cubes)
        self.convective = self._get_cube(cubes, "lwe_convective_precipitation_rate")
        self.dynamic = self._get_cube(cubes, "lwe_stratiform_precipitation_rate")

    @staticmethod
    def _get_cube(cubes: CubeList, name: str) -> Cube:
        """
        Get one cube named "name" from the list of cubes and set its units to m s-1.

        Args:
            cubes:
            name:

        Returns:
            Cube with units set
        """
        try:
            (cube,) = cubes.extract(name)
        except ValueError:
            raise ValueError(
                f"Cannot find a cube named '{name}' in {[c.name() for c in cubes]}"
            )
        if cube.units != "m s-1":
            cube = cube.copy()
            try:
                cube.convert_units("m s-1")
            except ValueError:
                raise ValueError(
                    f"Input {name} cube cannot be converted to 'm s-1' from {cube.units}"
                )
        return cube

    def _convective_ratio(self) -> ndarray:
        """
        Calculates the convective ratio from the convective and dynamic precipitation
        rate components, masking data where both are zero. The tolerance for comparing
        with zero is 1e-9 m s-1.
        """
        precipitation = self.convective + self.dynamic
        with np.errstate(divide="ignore", invalid="ignore"):
            convective_ratios = np.ma.masked_where(
                np.isclose(precipitation.data, 0.0, atol=1e-9),
                self.convective.data / precipitation.data,
            )
        return convective_ratios

    def process(self, cubes: List[Cube], model_id_attr: Optional[str] = None) -> Cube:
        """
        Calculate the convective ratio from the convective and dynamic components as:
            convective_ratio = convective / (convective + dynamic)

        If convective + dynamic is zero, then the resulting point is masked.

        Args:
            cubes:
                Both the convective and dynamic components as iris.cube.Cube in a list
                with names 'lwe_convective_precipitation_rate' and
                'lwe_stratiform_precipitation_rate'
            model_id_attr:
                Name of the attribute used to identify the source model for
                blending. This is inherited from the input temperature cube.

        Returns:
            Cube containing the convective ratio.
        """

        self._split_input(cubes)

        attributes = generate_mandatory_attributes(
            [self.convective], model_id_attr=model_id_attr
        )
        output_cube = create_new_diagnostic_cube(
            "convective_ratio",
            "1",
            self.convective,
            attributes,
            data=self._convective_ratio(),
        )

        return output_cube
