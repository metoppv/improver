# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Script to linearly interpolate thresholds"""

from typing import List, Optional

import iris
import numpy as np
from iris.cube import Cube
from numpy import ndarray

from improver import PostProcessingPlugin
from improver.calibration.utilities import convert_cube_data_to_2d
from improver.ensemble_copula_coupling.utilities import (
    interpolate_multiple_rows_same_x,
    restore_non_percentile_dimensions,
)
from improver.metadata.probabilistic import (
    find_threshold_coordinate,
)
from improver.utilities.cube_manipulation import (
    collapse_realizations,
    enforce_coordinate_ordering,
)


class ThresholdInterpolation(PostProcessingPlugin):
    def __init__(self, thresholds: List[float]):
        """
        Args:
            thresholds:
                List of the desired output thresholds.

        Raises:
            ValueError:
                If the thresholds list is empty.
        """
        if not thresholds:
            raise ValueError("The thresholds list cannot be empty.")
        self.thresholds = thresholds
        self.threshold_coord = None

    def mask_checking(self, forecast_at_thresholds: Cube) -> Optional[np.ndarray]:
        """
        Check if the mask is consistent across different slices of the threshold coordinate.

        Args:
            forecast_at_thresholds:
                The input cube containing forecast data with a threshold coordinate.

        Returns:
            original_mask:
                The original mask if the data is masked and the mask is consistent across
                different slices of the threshold coordinate, otherwise None.

        Raises:
            ValueError: If the mask varies across different slices of the threshold coordinate.
        """
        original_mask = None
        if np.ma.is_masked(forecast_at_thresholds.data):
            (crd_dim,) = forecast_at_thresholds.coord_dims(self.threshold_coord.name())
            if np.diff(forecast_at_thresholds.data.mask, axis=crd_dim).any():
                raise ValueError(
                    f"The mask is expected to be constant across different slices of the {self.threshold_coord.name()}"
                    f" dimension, however, in the dataset provided, the mask varies across the {self.threshold_coord.name()}"
                    f" dimension. This is not currently supported."
                )
            else:
                original_mask = next(
                    forecast_at_thresholds.slices_over(self.threshold_coord.name())
                ).data.mask

        return original_mask

    def _interpolate_thresholds(
        self,
        forecast_at_thresholds: Cube,
    ) -> np.ndarray:
        """
        Interpolate forecast data to a new set of thresholds.

        This method performs linear interpolation of forecast data from an initial
        set of thresholds to a new set of thresholds. The interpolation is done
        by converting the data to a 2D array, performing the interpolation, and
        then restoring the original dimensions.

        Args:
            forecast_at_thresholds:
                Cube containing forecast data with a threshold coordinate.

        Returns:
            ndarray:
                Interpolated forecast data with the new set of thresholds.
        """
        original_thresholds = self.threshold_coord.points

        # Ensure that the threshold dimension is first, so that the
        # conversion to a 2d array produces data in the desired order.
        enforce_coordinate_ordering(forecast_at_thresholds, self.threshold_coord.name())
        forecast_at_reshaped_thresholds = convert_cube_data_to_2d(
            forecast_at_thresholds, coord=self.threshold_coord.name()
        )

        forecast_at_interpolated_thresholds = interpolate_multiple_rows_same_x(
            np.array(self.thresholds, dtype=np.float64),
            original_thresholds.astype(np.float64),
            forecast_at_reshaped_thresholds.astype(np.float64),
        )

        forecast_at_interpolated_thresholds = np.transpose(
            forecast_at_interpolated_thresholds
        )

        # Restore the original dimensions of the interpolated forecast data.
        forecast_at_thresholds_data = restore_non_percentile_dimensions(
            forecast_at_interpolated_thresholds,
            next(forecast_at_thresholds.slices_over(self.threshold_coord.name())),
            len(self.thresholds),
        )

        return forecast_at_thresholds_data

    def create_cube_with_thresholds(
        self,
        forecast_at_thresholds: Cube,
        cube_data: ndarray,
    ) -> Cube:
        """
        Create a cube with a threshold coordinate based on a template cube extracted
        by slicing over the threshold coordinate.

        The resulting cube will have an extra threshold coordinate compared with
        the template cube. The shape of the cube_data should be the shape of the
        desired output cube.

        Args:
            forecast_at_thresholds:
                Cube containing forecast data with a threshold coordinate.
            cube_data:
                Array containing the interpolated forecast data with the new thresholds.

        Returns:
            Cube containing the new threshold coordinate and the interpolated data.
        """
        template_cube = next(
            forecast_at_thresholds.slices_over(self.threshold_coord.name())
        )
        template_cube.remove_coord(self.threshold_coord)

        # create cube with new threshold dimension
        cubes = iris.cube.CubeList([])
        for point in self.thresholds:
            cube = template_cube.copy()
            coord = iris.coords.DimCoord(
                np.array([point], dtype="float32"), units=self.threshold_coord.units
            )
            coord.rename(self.threshold_coord.name())
            coord.var_name = "threshold"
            coord.attributes = self.threshold_coord.attributes
            cube.add_aux_coord(coord)
            cubes.append(cube)
        result = cubes.merge_cube()
        # replace data
        result.data = cube_data
        return result

    def process(
        self,
        forecast_at_thresholds: Cube,
    ) -> Cube:
        """
        Process the input cube to interpolate forecast data to a new set of thresholds.

        This method performs the following steps:
        1. Identifies the threshold coordinate in the input cube.
        2. Checks if the mask is consistent across different slices of the threshold coordinate.
        3. Collapses the realizations if present.
        4. Interpolates the forecast data to the new set of thresholds.
        5. Creates a new cube with the interpolated threshold data.
        6. Applies the original mask to the new cube if it exists.

        Args:
            forecast_at_thresholds:
                Cube expected to contain a threshold coordinate.

        Returns:
            Cube:
                Cube with forecast values at the desired set of thresholds.
                The threshold coordinate is always the zeroth dimension.
        """
        self.threshold_coord = find_threshold_coordinate(forecast_at_thresholds)

        original_mask = self.mask_checking(forecast_at_thresholds)

        if forecast_at_thresholds.coords("realization"):
            forecast_at_thresholds = collapse_realizations(forecast_at_thresholds)

        forecast_at_thresholds_data = self._interpolate_thresholds(
            forecast_at_thresholds,
        )
        threshold_cube = self.create_cube_with_thresholds(
            forecast_at_thresholds,
            forecast_at_thresholds_data,
        )
        if original_mask is not None:
            original_mask = np.broadcast_to(original_mask, threshold_cube.shape)
            threshold_cube.data = np.ma.MaskedArray(
                threshold_cube.data, mask=original_mask
            )

        return threshold_cube
