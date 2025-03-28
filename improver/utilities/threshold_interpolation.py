# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Script to linearly interpolate thresholds"""

import numbers
from typing import Dict, List, Optional, Union

import iris
import numpy as np
from cf_units import Unit
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
    def __init__(
        self,
        threshold_values: Optional[List[float]] = None,
        threshold_config: Optional[Dict[str, Union[List[float], str]]] = None,
        threshold_units: Optional[str] = None,
    ):
        """
        Args:
            threshold_values:
                List of the desired output thresholds.
            threshold_config:
                Threshold configuration containing threshold values.
                Best used in combination with 'threshold_units'. It should contain
                a dictionary of strings that can be interpreted as floats with the
                structure: "THRESHOLD_VALUE": "None" (no fuzzy bounds).
                Repeated thresholds with different bounds are ignored; only the
                last duplicate will be used.
                Threshold_values and and threshold_config are mutually exclusive
                arguments, defining both will lead to an exception.
            threshold_units:
                Units of the threshold values. If not provided the units are
                assumed to be the same as those of the input cube.

        Raises:
            ValueError: If threshold_config and threshold_values are both set
            ValueError: If neither threshold_config or threshold_values are set
        """
        if threshold_config and threshold_values:
            raise ValueError(
                "threshold_config and threshold_values are mutually exclusive "
                "arguments - please provide one or the other, not both"
            )
        if threshold_config is None and threshold_values is None:
            raise ValueError(
                "One of threshold_config or threshold_values must be provided."
            )
        self.threshold_values = threshold_values
        self.threshold_coord = None
        self.threshold_config = threshold_config

        thresholds = self._set_thresholds(threshold_values, threshold_config)
        self.thresholds = [thresholds] if np.isscalar(thresholds) else thresholds
        self.threshold_units = (
            None if threshold_units is None else Unit(threshold_units)
        )

        self.original_units = None

    @staticmethod
    def _set_thresholds(
        threshold_values: Optional[Union[float, List[float]]],
        threshold_config: Optional[dict],
    ) -> List[float]:
        """
        Interprets a threshold_config dictionary if provided, or ensures that
        a list of thresholds has suitable precision.

        Args:
            threshold_values:
                A list of threshold values or a single threshold value.
            threshold_config:
                A dictionary defining threshold values and optionally upper
                and lower bounds for those values to apply fuzzy thresholding.

        Returns:
            thresholds:
                A list of input thresholds as float64 type.
        """
        if threshold_config:
            thresholds = []
            for key in threshold_config.keys():
                # Ensure thresholds are float64 to avoid rounding errors during
                # possible unit conversion.
                thresholds.append(float(key))
        else:
            # Ensure thresholds are float64 to avoid rounding errors during possible
            # unit conversion.
            if isinstance(threshold_values, numbers.Number):
                threshold_values = [threshold_values]
            thresholds = [float(x) for x in threshold_values]
        return thresholds

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

        if self.threshold_units is not None:
            template_cube.units = self.threshold_units

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
        2. Checks if the mask is consistent across different slices of the threshold
        coordinate.
        3. Convert the threshold coordinate to the specified units if provided.
        4. Collapses the realizations if present.
        5. Interpolates the forself.thresholdecast data to the new set of thresholds.
        6. Creates a new cube with the interpolated threshold data.
        7. Applies the original mask to the new cube if it exists.
        8. Converts the original threshold coordinate units back to the original units.
        9. Restores the original cube units, to combat how iris can set the cube's units
        to the modified dimension's units, when these units should be dimensionless
        ('1').


        Args:
            forecast_at_thresholds:
                Cube expected to contain a threshold coordinate.

        Returns:
            Cube:
                Cube with forecast values at the desired set of thresholds.
                The threshold coordinate is always the zeroth dimension.
        """
        self.threshold_coord = find_threshold_coordinate(forecast_at_thresholds)
        self.threshold_coord_name = self.threshold_coord.name()
        self.original_units = forecast_at_thresholds.units
        self.original_threshold_units = self.threshold_coord.units

        original_mask = self.mask_checking(forecast_at_thresholds)

        if self.threshold_units is not None:
            forecast_at_thresholds.coord(self.threshold_coord_name).convert_units(
                self.threshold_units
            )

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

        # Revert the threshold coordinate's units
        threshold_cube.coord(self.threshold_coord_name).convert_units(
            self.original_threshold_units
        )

        # Ensure the cube's overall units are restored
        threshold_cube.units = self.original_units

        return threshold_cube
