# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Module to contain mathematical operations."""

from typing import List, Optional, Tuple, Union

import iris
import numpy as np
import numpy.ma as ma
from iris.cube import Cube
from numpy import ndarray

from improver import BasePlugin
from improver.metadata.utilities import (
    create_new_diagnostic_cube,
    generate_mandatory_attributes,
)
from improver.utilities.cube_manipulation import (
    enforce_coordinate_ordering,
    get_dim_coord_names,
    sort_coord_in_cube,
)


class Integration(BasePlugin):
    """Perform integration along a chosen coordinate. This class currently
    supports the integration of positive values only, in order to
    support its usage as part of computing the wet-bulb temperature integral.
    Generalisation of this class to support standard numerical integration
    can be undertaken, if required.
    """

    def __init__(
        self,
        coord_name_to_integrate: str,
        start_point: Optional[float] = None,
        end_point: Optional[float] = None,
        positive_integration: bool = False,
    ) -> None:
        """
        Initialise class.

        Args:
            coord_name_to_integrate:
                Name of the coordinate to be integrated.
            start_point:
                Point at which to start the integration.
                Default is None. If start_point is None, integration starts
                from the first available point.
            end_point:
                Point at which to end the integration.
                Default is None. If end_point is None, integration will
                continue until the last available point.
            positive_integration:
                Description of the direction in which to integrate.
                True corresponds to the values within the array
                increasing as the array index increases.
                False corresponds to the values within the array
                decreasing as the array index increases.
        """
        self.coord_name_to_integrate = coord_name_to_integrate
        self.start_point = start_point
        self.end_point = end_point
        self.positive_integration = positive_integration
        self.input_cube = None

    def __repr__(self) -> str:
        """Represent the configured plugin instance as a string."""
        result = (
            "<Integration: coord_name_to_integrate: {}, "
            "start_point: {}, end_point: {}, "
            "positive_integration: {}>".format(
                self.coord_name_to_integrate,
                self.start_point,
                self.end_point,
                self.positive_integration,
            )
        )
        return result

    def ensure_monotonic_increase_in_chosen_direction(self, cube: Cube) -> Cube:
        """Ensure that the chosen coordinate is monotonically increasing in
        the specified direction.

        Args:
            cube:
                The cube containing the coordinate to check.
                Note that the input cube will be modified by this method.

        Returns:
            The cube containing a coordinate that is monotonically
            increasing in the desired direction.
        """
        coord_name = self.coord_name_to_integrate
        increasing_order = np.all(np.diff(cube.coord(coord_name).points) > 0)

        if increasing_order and not self.positive_integration:
            cube = sort_coord_in_cube(cube, coord_name, descending=True)

        if not increasing_order and self.positive_integration:
            cube = sort_coord_in_cube(cube, coord_name)

        return cube

    def prepare_for_integration(self) -> Tuple[Cube, Cube]:
        """Prepare for integration by creating the cubes needed for the
        integration. These are separate cubes for representing the upper
        and lower limits of the integration.

        Returns:
            - Cube containing the upper bounds to be used during the
              integration.
            - Cube containing the lower bounds to be used during the
              integration.
        """
        if self.positive_integration:
            upper_bounds = self.input_cube.coord(self.coord_name_to_integrate).points[
                1:
            ]
            lower_bounds = self.input_cube.coord(self.coord_name_to_integrate).points[
                :-1
            ]
        else:
            upper_bounds = self.input_cube.coord(self.coord_name_to_integrate).points[
                :-1
            ]
            lower_bounds = self.input_cube.coord(self.coord_name_to_integrate).points[
                1:
            ]

        upper_bounds_cube = self.input_cube.extract(
            iris.Constraint(coord_values={self.coord_name_to_integrate: upper_bounds})
        )
        lower_bounds_cube = self.input_cube.extract(
            iris.Constraint(coord_values={self.coord_name_to_integrate: lower_bounds})
        )

        return upper_bounds_cube, lower_bounds_cube

    def _generate_output_name_and_units(self) -> Tuple[str, str]:
        """Gets suitable output name and units from input cube metadata"""
        new_name = f"{self.input_cube.name()}_integral"
        original_units = self.input_cube.units
        integrated_units = self.input_cube.coord(self.coord_name_to_integrate).units
        new_units = "{} {}".format(original_units, integrated_units)
        return new_name, new_units

    def _create_output_cube(
        self,
        template: Cube,
        data: Union[List[float], ndarray],
        points: Union[List[float], ndarray],
        bounds: Union[List[float], ndarray],
    ) -> Cube:
        """
        Populates a template cube with data from the integration

        Args:
            template:
                Copy of upper or lower bounds cube, based on direction of
                integration
            data:
                Integrated data
            points:
                Points values for the integrated coordinate. These will not
                match the template cube if any slices were skipped in the
                integration, and therefore are used to slice the template cube
                to match the data array.
            bounds:
                Bounds values for the integrated coordinate

        Returns:
            Cube with data from integration
        """
        # extract required slices from template cube
        template = template.extract(
            iris.Constraint(
                coord_values={self.coord_name_to_integrate: lambda x: x in points}
            )
        )

        # re-promote integrated coord to dimension coord if need be
        aux_coord_names = [coord.name() for coord in template.aux_coords]
        if self.coord_name_to_integrate in aux_coord_names:
            template = iris.util.new_axis(template, self.coord_name_to_integrate)

        # order dimensions on the template cube so that the integrated
        # coordinate is first (as this is the leading dimension on the
        # data array)
        enforce_coordinate_ordering(template, self.coord_name_to_integrate)

        # generate appropriate metadata for new cube
        attributes = generate_mandatory_attributes([template])
        coord_dtype = template.coord(self.coord_name_to_integrate).dtype
        name, units = self._generate_output_name_and_units()

        # create new cube from template
        integrated_cube = create_new_diagnostic_cube(
            name, units, template, attributes, data=np.array(data)
        )

        integrated_cube.coord(self.coord_name_to_integrate).bounds = np.array(
            bounds
        ).astype(coord_dtype)

        # re-order cube to match dimensions of input cube
        ordered_dimensions = get_dim_coord_names(self.input_cube)
        enforce_coordinate_ordering(integrated_cube, ordered_dimensions)
        return integrated_cube

    def perform_integration(
        self, upper_bounds_cube: Cube, lower_bounds_cube: Cube
    ) -> Cube:
        """Perform the integration.

        Integration is performed by firstly defining the stride as the
        difference between the upper and lower bound. The contribution from
        the uppermost half of the stride is calculated by multiplying the
        upper bound value by 0.5 * stride, and the contribution
        from the lowermost half of the stride is calculated by multiplying the
        lower bound value by 0.5 * stride. The contribution from the
        uppermost half of the stride and the bottom half of the stride is
        summed.

        Integration is performed ONLY over positive values.

        Args:
            upper_bounds_cube:
                Cube containing the upper bounds to be used during the
                integration.
            lower_bounds_cube:
                Cube containing the lower bounds to be used during the
                integration.

        Returns:
            Cube containing the output from the integration.
        """

        def skip_slice(upper_bound, lower_bound, direction, start_point, end_point):
            """Conditions under which a slice should not be included in
            the integrated total.  All inputs (except the string "direction")
            are floats."""
            if start_point:
                if direction and lower_bound < start_point:
                    return True
                if not direction and upper_bound > start_point:
                    return True
            if end_point:
                if direction and upper_bound > end_point:
                    return True
                if not direction and lower_bound < end_point:
                    return True
            return False

        data = []
        coord_points = []
        coord_bounds = []
        integral = 0
        levels_tuple = zip(
            upper_bounds_cube.slices_over(self.coord_name_to_integrate),
            lower_bounds_cube.slices_over(self.coord_name_to_integrate),
        )

        for upper_bounds_slice, lower_bounds_slice in levels_tuple:
            (upper_bound,) = upper_bounds_slice.coord(
                self.coord_name_to_integrate
            ).points
            (lower_bound,) = lower_bounds_slice.coord(
                self.coord_name_to_integrate
            ).points

            if skip_slice(
                upper_bound,
                lower_bound,
                self.positive_integration,
                self.start_point,
                self.end_point,
            ):
                continue

            stride = np.abs(upper_bound - lower_bound)
            upper_half_data = np.where(
                upper_bounds_slice.data > 0, upper_bounds_slice.data * 0.5 * stride, 0.0
            )
            lower_half_data = np.where(
                lower_bounds_slice.data > 0, lower_bounds_slice.data * 0.5 * stride, 0.0
            )
            integral += upper_half_data + lower_half_data

            data.append(integral.copy())
            coord_points.append(
                upper_bound if self.positive_integration else lower_bound
            )
            coord_bounds.append([lower_bound, upper_bound])

        if len(data) == 0:
            msg = (
                "No integration could be performed for "
                "coord_to_integrate: {}, start_point: {}, end_point: {}, "
                "positive_integration: {}. "
                "No usable data was found.".format(
                    self.coord_name_to_integrate,
                    self.start_point,
                    self.end_point,
                    self.positive_integration,
                )
            )
            raise ValueError(msg)

        template = upper_bounds_cube if self.positive_integration else lower_bounds_cube
        integrated_cube = self._create_output_cube(
            template.copy(), data, coord_points, coord_bounds
        )
        return integrated_cube

    def process(self, cube: Cube) -> Cube:
        """Integrate data along a specified coordinate.  Only positive values
        are integrated; zero and negative values are not included in the sum or
        as levels on the integrated cube.

        Args:
            cube:
                Cube containing the data to be integrated.

        Returns:
            The cube containing the result of the integration.
            This will have the same name and units as the input cube (TODO
            same name and units are incorrect - fix this).
        """
        self.input_cube = self.ensure_monotonic_increase_in_chosen_direction(cube)
        upper_bounds_cube, lower_bounds_cube = self.prepare_for_integration()

        integrated_cube = self.perform_integration(upper_bounds_cube, lower_bounds_cube)

        return integrated_cube


def fast_linear_fit(
    x_data: ndarray,
    y_data: ndarray,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
    gradient_only: bool = False,
    with_nan: bool = False,
) -> Tuple[ndarray, ndarray]:
    """Uses a simple linear fit approach to calculate the
    gradient along specified axis (default is to fit all points).
    Uses vectorized operations, so it's much faster than using scipy lstsq
    in a loop. This function does not handle NaNs, but will work with masked arrays.

    Args:
        x_data:
            x axis data.
        y_data:
            y axis data.
        axis:
            Optional argument, specifies the axis to operate on.
            Default is to flatten arrays and fit all points.
        keepdims:
            If this is set to True, the axes which are reduced are left in the
            result as dimensions with size one. With this option, the result
            will broadcast correctly against the input array.
        gradient_only:
            If true only returns the gradient.
        with_nan:
            If true, there are NaNs in your data (that you know about).

    Returns:
        tuple with first element being the gradient between x and y, and
        the second element being the calculated y-intercepts.
    """
    # Check that the positions of nans match in x and y
    if with_nan and not (np.isnan(x_data) == np.isnan(y_data)).all():
        raise ValueError("Positions of NaNs in x and y do not match")

    # Check that there are no mismatched masks (this will mess up the mean).
    if not with_nan and not (ma.getmask(y_data) == ma.getmask(x_data)).all():
        raise ValueError("Mask of x and y do not match.")

    if with_nan:
        mean, sum_func = np.nanmean, np.nansum
    else:
        mean, sum_func = np.mean, np.sum

    x_mean = mean(x_data, axis=axis, keepdims=True)
    y_mean = mean(y_data, axis=axis, keepdims=True)

    x_diff = x_data - x_mean
    y_diff = y_data - y_mean

    xy_cov = sum_func(x_diff * y_diff, axis=axis, keepdims=keepdims)
    x_var = sum_func(x_diff * x_diff, axis=axis, keepdims=keepdims)

    grad = xy_cov / x_var

    if gradient_only:
        return grad

    if not keepdims:
        x_mean = x_mean.squeeze(axis=axis)
        y_mean = y_mean.squeeze(axis=axis)

    intercept = y_mean - grad * x_mean
    return grad, intercept


class CalculateClimateAnomalies(BasePlugin):
    """Converts an input cube of data to a cube containing anomaly data."""

    def __init__(
        self,
        diagnostic_cube: Cube,
        mean_cube: Cube,
        variance_cube: Optional[Cube] = None,
    ) -> None:
        """
        Initialise class.

        Args:
            diagnostic_cube:
                Cube containing the data to be converted to anomalies.
            mean_cube:
                Cube containing the mean data to be used for the
                calculation of anomalies.
            variance_cube:
                Cube containing the variance data to be used for the
                calculation of standardised anomalies. If not provided,
                only anomalies (not standardised anomalies) will be
                calculated.
        """
        self.diagnostic_cube = diagnostic_cube
        self.mean_cube = mean_cube
        self.variance_cube = variance_cube

    def verify_inputs(self) -> None:
        """Verify that the input cubes are compatible for calculating
        anomalies."""

        cubes_to_check = self._get_cubes_to_check()
        self._verify_standard_names_match(cubes_to_check)
        self._verify_units_match(cubes_to_check)
        self._verify_grids_match(cubes_to_check)
        self._verify_time_coords_match(cubes_to_check)

    def _get_cubes_to_check(self) -> List[Cube]:
        cubes_to_check = [self.mean_cube]
        if self.variance_cube:
            cubes_to_check.append(self.variance_cube)
        return cubes_to_check

    def _verify_standard_names_match(self, cubes_to_check: List[Cube]) -> None:
        """Check that all cubes have the same standard name to prevent accidental
        use of cubes with referring to different physical phenomenon
        (e.g. temperature and precipitation).
        """
        for cube in cubes_to_check:
            if cube.standard_name != self.diagnostic_cube.standard_name:
                raise ValueError(
                    f"The diagnostic cube and {cube} must have the same standard name. "
                    f"The supplied diagnostic cube has standard name: {self.diagnostic_cube.standard_name} "
                    f"and the supplied {cube} has standard name: {cube.standard_name}"
                )

    def _verify_units_match(self, cubes_to_check: List[Cube]) -> None:
        """Check that all cubes have the same units. E.g. to prevent accidental
        use of cubes with rate data with cubes with accumulation data"""
        for cube in cubes_to_check:
            if cube.units != self.diagnostic_cube.units:
                raise ValueError(
                    f"The diagnostic cube and {cube} must have the same units. "
                    f"The supplied diagnostic cube has units: {self.diagnostic_cube.units} "
                    f"and the supplied {cube} has units: {cube.units}"
                )

    def _verify_grids_match(self, cubes_to_check: List[Cube]) -> None:
        """Check that all cubes have the same spatial coordinates (i.e. the same grid)"""

        diagnostic_cube_grid = self._get_cube_grid(self.diagnostic_cube)
        for cube in cubes_to_check:
            cube_grid = self._get_cube_grid(cube)
            if not np.array_equal(
                cube_grid[0], diagnostic_cube_grid[0]
            ) or not np.array_equal(cube_grid[1], diagnostic_cube_grid[1]):
                raise ValueError(
                    f"The diagnostic cube and {cube} must have the same grid. "
                    f"The supplied diagnostic cube has grid: {diagnostic_cube_grid} "
                    f"and the supplied {cube} has grid: {cube_grid}"
                )

    def _verify_time_coords_match(self, cubes_to_check: List[Cube]) -> None:
        diagnostic_cube_time_range = self._get_cube_time_range(self.diagnostic_cube)
        for cube in cubes_to_check:
            cube_time_range = self._get_cube_time_range(cube)
            if cube_time_range != diagnostic_cube_time_range:
                raise ValueError(
                    f"The diagnostic cube and {cube} must have the same time points. "
                    f"The supplied diagnostic cube has time points: {diagnostic_cube_time_range} "
                    f"and the supplied {cube} has time points: {cube_time_range}"
                )

    @staticmethod
    def _get_cube_grid(cube: Cube) -> Tuple[ndarray, ndarray]:
        cube_grid = (cube.coord("latitude").points, cube.coord("longitude").points)
        return cube_grid

    @staticmethod
    def _get_cube_time_range(cube: Cube) -> Tuple[float, float]:
        """Get the time range of the input cube"""
        start_time = cube.coord("time").points[0]
        end_time = cube.coord("time").points[-1]
        return start_time, end_time

    def calculate_anomalies(self) -> ndarray:
        """Calculate anomalies from the input cubes."""
        anomalies_data = self.diagnostic_cube.data - self.mean_cube.data
        if self.variance_cube:
            anomalies_data = anomalies_data / np.sqrt(self.variance_cube.data)

        return anomalies_data

    def _generate_output_name_and_units(self) -> str:
        """Gets suitable output name and units from input cube metadata"""
        if self.variance_cube:
            new_name = f"{self.diagnostic_cube.name()}_standardised_anomalies"
            new_units = None
        else:
            new_name = f"{self.diagnostic_cube.name()}_anomalies"
            new_units = self.diagnostic_cube.units
        return new_name, new_units

    def _create_output_cube(self, data: Union[List[float], ndarray]) -> Cube:
        """
        Populates a template cube with data from the anomalies calculation.

        Args:
            data:
                Anomalies data

        Returns:
            Cube with data from anomalies calculation
        """

        name, units = self._generate_output_name_and_units()
        attributes = generate_mandatory_attributes([self.diagnostic_cube])
        output_cube = create_new_diagnostic_cube(
            name, units, self.diagnostic_cube, attributes, data=np.array(data)
        )

        return output_cube

    def process(self) -> Cube:
        """Calculate anomalies from the input cubes."""
        self.verify_inputs()
        anomalies_data = self.calculate_anomalies()
        anomalies_cube = self._create_output_cube(anomalies_data)

        return anomalies_cube
