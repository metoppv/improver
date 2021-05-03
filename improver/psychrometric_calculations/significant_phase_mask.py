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
"""Module for calculating the significant phase mask."""

from typing import Optional

import numpy as np
from iris.cube import Cube
from numpy import ndarray

from improver import BasePlugin
from improver.metadata.utilities import (
    create_new_diagnostic_cube,
    generate_mandatory_attributes,
)


class SignificantPhaseMask(BasePlugin):
    """
    Derives a categorical field for the specified precipitation phase indicating whether
    that phase is the dominant phase at each point (1 where true, else 0) based on input
    snow-fraction data.
    The decision is: snow-fraction <= 0.01: Rain; snow-fraction >= 0.99: Snow; Sleet
    in between.
    """

    def __init__(self, model_id_attr: Optional[str] = None) -> None:
        """
        Initialise the class

        Args:
            model_id_attr:
                Name of the attribute used to identify the source model for
                blending.
        """
        self.model_id_attr = model_id_attr
        self.lower_threshold = 0.01
        self.upper_threshold = 0.99
        self.phase_operator = {
            "rain": self._rain_phase,
            "sleet": self._sleet_phase,
            "snow": self._snow_phase,
        }

    @staticmethod
    def _validate_snow_fraction(snow_fraction: Cube) -> None:
        """Ensures that the input snow-fraction field has appropriate name
        (snow_fraction), units (1) and data (between 0 and 1 inclusive).

        Args:
            snow_fraction

        Raises
            ValueError:
                If any of the above are not True.
        """
        if snow_fraction.name() != "snow_fraction":
            raise ValueError(
                f"Expected cube named 'snow_fraction', not {snow_fraction.name()}"
            )
        if f"{snow_fraction.units}" != "1":
            raise ValueError(f"Expected cube with units '1', not {snow_fraction.units}")
        if np.ma.is_masked(snow_fraction.data):
            raise NotImplementedError("SignificantPhaseMask cannot handle masked data")
        if np.any((snow_fraction.data < 0) | (snow_fraction.data > 1)):
            raise ValueError(
                f"Expected cube data to be in range 0 <= x <= 1. "
                f"Found max={snow_fraction.data.max()}; min={snow_fraction.data.min()}"
            )

    def _rain_phase(self, snow_fraction_data: Cube) -> ndarray:
        """Calculates the rain_phase data"""
        return np.where(snow_fraction_data <= self.lower_threshold, 1, 0)

    def _snow_phase(self, snow_fraction_data: Cube) -> ndarray:
        """Calculates the snow_phase data"""
        return np.where(snow_fraction_data >= self.upper_threshold, 1, 0)

    def _sleet_phase(self, snow_fraction_data: Cube) -> ndarray:
        """Calculates the sleet_phase data"""
        return np.where(
            (self.lower_threshold < snow_fraction_data)
            & (snow_fraction_data < self.upper_threshold),
            1,
            0,
        )

    def process(self, snow_fraction: Cube, phase: str) -> Cube:
        """
        Make significant-phase-mask cube for the specified phase.

        Args:
            snow_fraction:
                The input snow-fraction data to derive the phase mask from.
            phase:
                One of "rain", "sleet" or "snow". This is the phase mask that will be
                returned.

        Returns:
            The requested phase mask containing 1 where that phase is dominant
            and 0 elsewhere. Dimensions will be identical to snow-fraction.
        """
        self._validate_snow_fraction(snow_fraction)

        try:
            data = self.phase_operator[phase](snow_fraction.data).astype(np.int8)
        except KeyError:
            raise KeyError(
                f"Requested phase mask '{phase}' not in {list(self.phase_operator.keys())}"
            )
        phase_mask = create_new_diagnostic_cube(
            f"{phase}_mask",
            "1",
            snow_fraction,
            generate_mandatory_attributes(
                [snow_fraction], model_id_attr=self.model_id_attr
            ),
            data=data,
        )
        return phase_mask
