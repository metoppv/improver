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
"""Module containing aviation icing classes."""
from datetime import timedelta
from typing import Tuple

import iris
import numpy as np
from iris.cube import Cube, CubeList

from improver import PostProcessingPlugin
from improver.metadata.constants import FLOAT_DTYPE
from improver.metadata.utilities import (
    create_new_diagnostic_cube,
    generate_mandatory_attributes,
)
from improver.utilities.cube_checker import spatial_coords_match

class IcingSeverityMultivariateRegression_USAF2024(PostProcessingPlugin):
    """    
    The algorithm outputs the unitless aircraft icing severity index. 
    This index can be converted directly to categorical icing severity level
    using the category definitions below. Alternatively, the probability of 
    reaching or exceeding these categorical icing severity levels can be 
    calculated in a downstream thresholding operation.

    Inputs:
    Temperature (T) in units of K
    Relative Humidity (RH) in units of %

    Outputs:
    Aircraft icing severity index (AISI) unitless

    Description of the algorithm:

    IF RH is greater than 70% and T is between 250.0K and 273.15K THEN:
        AISI=100*TANH(0.06*RH-4.0)[TANH(0.1(T-247.0)]

    Categorical icing severity levels are defined as
              AISI < 58 : "No Icing"
        58 <= AISI < 85 : "Light Icing"
        85 <= AISI < 92 : "Moderate Icing"
        92 <= AISI      : "Severe Icing"

    """

    @staticmethod
    def _extract_input(cubes: CubeList, cube_name: str) -> Cube:
        """Extract the relevant cube based on the cube name.

        Args: 
            cubes: Cubes from which to extract required input.
            cube_name: Name of cube to extract.

        Returns:
            The extracted cube.
        """
        try:
            cube = cubes.extract_cube(iris.Constraint(cube_name))
        except iris.exceptions.ConstraintMismatchError:
            raise ValueError(f"No cube named {cube_name} found in {cubes}")
        return cube

    def _get_inputs(self, cubes: CubeList) -> Tuple[Cube, Cube]:
        """
        Separates T and RH cubes and checks that the following match: 
        forecast_reference_time, spatial coords, and time.
        """

        output_cubes = iris.cube.CubeList()
        input_names = {
            "air_temperature": ["K"],
            "relative_humidity": ["%"],
        }

        for input_name, units in input_names.items():
            output_cubes.append(self._extract_input(cubes, input_name))
            if not output_cubes[-1].units in units:
                expected_unit_string = " or ".join(map(str, units))
                received_unit_string = str(output_cubes[-1].units)
                raise ValueError(
                    f"The {output_cubes[-1].name()} units are incorrect, expected "
                    f"units as {expected_unit_string} but received {received_unit_string})."
                )

        t,rh = output_cubes

        if t.coord("forecast_reference_time") != rh.coord("forecast_reference_time"):
            raise ValueError(
                f"{t.name()} and {rh.name()} do not have the same forecast reference time"
            )

        if not spatial_coords_match([t, rh]):
            raise ValueError(
                f"{t.name()} and {rh.name()} do not have the same spatial "
                f"coordinates"
            )
        
        if t.coord("time") != rh.coord("time"):
            raise ValueError(
                f"{t.name()} and {rh.name()} do not have the same valid time"
            )

        return t, rh

    def process(self, cubes: CubeList, model_id_attr: str = None) -> Cube:
        """
        From the supplied Air Temperature and Relative Humidity cubes, calculate the Aircraft
        Icing Severity Index.

        Args:
            cubes:
                Cubes of Air Temperature and Relative Humidity.
            model_id_attr:
                The name of the dataset attribute to be used to identify the source
                model when blending data from different models.

        Returns:
            Cube of Aircraft Icing Severity Index

        Raises:
            ValueError:
                If one of the cubes is not found, doesn't match the others, or has incorrect units
        """
        t, rh = self._get_inputs(cubes)

        # Regression equations require math on cubes with incompatible units, so strip data
        template = t.copy()
        t  = t.data
        rh = rh.data
        

        # Regression equation if RH is greater than 70% and T is between 250.0K and 273.15K
        aisi = 100*np.tanh(0.06*rh-4.0)*(np.tanh(0.1*(t-247.0)))
        aisi[np.where(rh<70)]  = 0
        aisi[np.where(t<250.0)]  = 0
        aisi[np.where(t>273.15)] = 0

        cube = create_new_diagnostic_cube(
            name=(
                "aircraft_icing_severity_index"
            ),
            units="1",
            template_cube=template,
            data=aisi.astype(FLOAT_DTYPE),
            mandatory_attributes=generate_mandatory_attributes(
                cubes, model_id_attr=model_id_attr
            ),
        )

        return cube
