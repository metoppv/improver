# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Script to linearly interpolate thresholds"""

import warnings
from typing import List, Optional, Union
from improver import PostProcessingPlugin
import iris
import numpy as np
from iris.cube import Cube
from numpy import ndarray

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

    def __init__(self, thresholds: List):

        """
        Linearly interpolate thresholds.

        Args:
            thresholds:
                List of the desired output thresholds.
        """
        self.thresholds = thresholds
        self.threshold_coord = None

    def mask_checking(self, forecast_at_thresholds: Cube):
        original_mask = None
        if np.ma.is_masked(forecast_at_thresholds.data):
            (crd_dim,) = forecast_at_thresholds.coord_dims(self.threshold_coord)
            if np.diff(forecast_at_thresholds.data.mask, axis=crd_dim).any():
                raise ValueError(
                    f"The mask is expected to be constant across different slices of the {self.threshold_coord}"
                    f" dimension, however, in the dataset provided, the mask varies across the {self.threshold_coord}"
                    f" dimension. This is not currently supported."
                )
            else:
                original_mask = next(
                    forecast_at_thresholds.slices_over(self.threshold_coord)
                ).data.mask

        return original_mask

    def _interpolate_thresholds(
        self,
        forecast_at_thresholds: Cube,
    ) -> Cube:
        """
        Interpolation of forecast for a set of thresholds from an initial
        set of thresholds to a new set of thresholds. This is constructed
        by linearly interpolating between the original set of thresholds
        to a new set of thresholds.

        Args:
            forecast_at_thresholds:
                Cube containing a threshold coordinate.
            desired_thresholds:
                Array of the desired thresholds.
            self.threshold_coord:
                Name of required threshold coordinate.

        Returns:
            Cube containing values for the required diagnostic e.g.
            air_temperature at the required thresholds.
        """
        original_thresholds = forecast_at_thresholds.coord(self.threshold_coord).points

        # Ensure that the threshold dimension is first, so that the
        # conversion to a 2d array produces data in the desired order.
        enforce_coordinate_ordering(forecast_at_thresholds, self.threshold_coord)
        forecast_at_reshaped_thresholds = convert_cube_data_to_2d(
            forecast_at_thresholds, coord=self.threshold_coord
        )

        forecast_at_interpolated_thresholds = interpolate_multiple_rows_same_x(
            np.array(self.thresholds, dtype=np.float64),
            original_thresholds.astype(np.float64),
            forecast_at_reshaped_thresholds.astype(np.float64),
        )

        forecast_at_interpolated_thresholds = np.transpose(
            forecast_at_interpolated_thresholds
        )

        # Reshape forecast_at_percentiles, so the percentiles dimension is
        # first, and any other dimension coordinates follow.
        forecast_at_thresholds_data = restore_non_percentile_dimensions(
            forecast_at_interpolated_thresholds,
            next(forecast_at_thresholds.slices_over(self.threshold_coord)),
            len(self.thresholds),
        )

        return forecast_at_thresholds_data

    def create_cube_with_thresholds(
        self,
        forecast_at_thresholds: Cube,
        cube_data: ndarray,
    ) -> Cube:
        """
        Create a cube with a threshold coordinate based on a template cube.
        The resulting cube will have an extra threshold coordinate compared with
        the template cube. The shape of the cube_data should be the shape of the
        desired output cube.

        Args:
            thresholds:
                Ensemble thresholds. There should be the same number of
                thresholds as the first dimension of cube_data.
            template_cube:
                Cube to copy metadata from.
            cube_data:
                Data to insert into the template cube.
                The shape of the cube_data, excluding the dimension associated with
                the threshold coordinate, should be the same as the shape of
                template_cube.
                For example, template_cube shape is (3, 3, 3), whilst the cube_data
                is (10, 3, 3, 3), where there are 10 thresholds.
            cube_unit:
                The units of the data within the cube, if different from those of
                the template_cube.

        Returns:
            Cube containing a percentile coordinate as the leading dimension (or
            scalar percentile coordinate if single-valued)
        """
        template_cube = next(forecast_at_thresholds.slices_over(self.threshold_coord))
        threshold_units = template_cube.coord(self.threshold_coord).units
        template_cube.remove_coord(self.threshold_coord)

        # create cube with new threshold dimension
        cubes = iris.cube.CubeList([])
        for point in self.thresholds:
            cube = template_cube.copy()
            coord = iris.coords.DimCoord(
                np.array([point], dtype="float32"), units=threshold_units
            )
            coord.rename(self.threshold_coord)
            coord.var_name = "threshold"
            coord.attributes = {'spp__relative_to_threshold': 'less_than'}
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
        1. Creates a list of thresholds, if not provided.
        2. Interpolate the threshold coordinate into an alternative
           set of thresholds using linear interpolation.

        Args:
            forecast_at_thresholds:
                Cube expected to contain a threshold coordinate.
            thresholds:
                List of the desired output thresholds.

        Returns:
            Cube with forecast values at the desired set of thresholds.
            The threshold coordinate is always the zeroth dimension.
        """
        self.threshold_coord = find_threshold_coordinate(forecast_at_thresholds).name()

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
            threshold_cube.data = np.ma.MaskedArray(threshold_cube.data, mask=original_mask)

#        if threshold_cube.coords("realization"):
#            threshold_cube = collapse_realizations(threshold_cube)

        return threshold_cube