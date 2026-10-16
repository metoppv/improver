# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Calculations to produce overall Pollen Index values for a period, Hourly or Daily."""

from copy import deepcopy
from typing import Union

import numpy as np
from iris.cube import Cube, CubeList

from improver import PostProcessingPlugin
from improver.utilities.common_input_handle import as_cubelist


class PollenMaximumIndexForPeriod(PostProcessingPlugin):
    """Plugin to calculate the overall Pollen Index for either a Daily or Hourly period.

    The input cubelist has Pollen Index Values for all pollen taxa
    for the period as specified in the cubes. The maximum value across all
    taxa at a location is saved as the Index for that location.
    """

    # The output cube is a deepcopy of the first input cube (to keep metadata) and is then manipulated in place
    _output_cube = None

    def _calculate(self, cubes: tuple[Cube, ...] | CubeList):
        """Calculate the overall Pollen Index for the period.

        The period, either Hourly or Daily, can be determined by the value in the
        cube attribute "forecast_period" which has the period in seconds. i.e.
        3600 for Hourly and 86400 for Daily.

        For each grid point, determine the maximum pollen value across all taxa,
        and use this as the pollen index for that grid point.

        Args:
            cubes:
                Input cubes for all pollen types
        """
        # Stack the cubes along a new taxa dimension and calculate the maximum across that dimension
        stacked_data = np.stack([cube.data for cube in cubes], axis=0)

        # Create a new numpy array with this shape to hold the pollen index values, and fill it
        # with the maximum values across the taxa dimension
        pollen_index_data = np.max(stacked_data, axis=0)
        self._output_cube.data = pollen_index_data.astype(np.int32)

    def _metadata(self):
        """Change the cube name and other metadata.

        Some attributes are removed because they are not relevant to this aggregated
        cube data which is no longer specific to one pollen taxon.
        """
        self._output_cube.attributes.pop("scaling_factor", None)
        self._output_cube.attributes.pop("taxa", None)
        self._output_cube.attributes.pop("taxa_category", None)
        self._output_cube.attributes.pop("biological_taxon_name", None)
        self._output_cube.rename("pollen_index")

    def process(self, *cubes: Union[Cube, CubeList]) -> Cube:
        """Calculate the Pollen Hourly Index.

        Args:
            cubes (iris.cube.CubeList or list of iris.cube.Cube):
                Input cubes for all pollen types for Pollen Value for 1 hour.

        Returns:
            The calculated output cube.

        Warns:
            UserWarning:
                If output values fall outside typical expected ranges
        """
        cubes = as_cubelist(*cubes)

        # Create output_cube ready to take data from calculations, using the first cube as a template
        template_cube = cubes[0]
        self._output_cube = deepcopy(template_cube)
        self._calculate(cubes)
        self._metadata()
        return self._output_cube
