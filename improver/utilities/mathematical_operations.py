# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Module to contain mathematical operations."""

from typing import List, Optional, Tuple, Union

import iris
import numpy as np
import numpy.ma as ma
from iris.coords import CellMethod
from iris.cube import Cube
from numpy import ndarray

from improver import BasePlugin
from improver.metadata.utilities import (
    create_new_diagnostic_cube,
    generate_mandatory_attributes,
)
from improver.utilities.cube_checker import (
    assert_spatial_coords_match,
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
    """Utility functionality to convert an input cube of data to a cube containing
    anomaly data. If a forecast and a climatological mean are supplied, the anomaly
    calculated will be forecast - mean. If the climatological standard deviation is
    also supplied, then a standardized anomaly is calculated as
    (forecast - mean) / standard deviation"""

    def __init__(
        self,
        ignore_temporal_mismatch: bool = False,
    ) -> None:
        """
        Initialise class.

        Args:
            ignore_temporal_mismatch:
                If True, ignore mismatch in time coordinates between the input cubes.
                Default is False. Set to True when converting to the anomaly of a
                diagnostic cube relative to mean and standard deviation cubes of a
                different period.
        """
        self.ignore_temporal_mismatch = ignore_temporal_mismatch

    @staticmethod
    def verify_units_match(
        diagnostic_cube: Cube, mean_cube: Cube, std_cube: Optional[Cube] = None
    ) -> None:
        """Check that all cubes have the same units. E.g. to prevent accidental use of
        cubes with rate data with cubes with accumulation data.

        Raises:
            ValueError: If the mean and diagnostic cubes have different units.
            ValueError: If the standard deviation (if supplied) and diagnostic cubes
            have different units.
        """
        errors = []
        if mean_cube.units != diagnostic_cube.units:
            errors.append(
                "The mean cube must have the same units as the diagnostic cube. "
                f"The following units were found: {mean_cube.units},"
                f"{diagnostic_cube.units}"
            )
        if std_cube and std_cube.units != str(diagnostic_cube.units):
            errors.append(
                "The standard deviation cube must have the same units as the "
                f"diagnostic cube. The following units were found: {std_cube.units},"
                f"{diagnostic_cube.units}"
            )

        if errors:
            raise ValueError("\n".join(errors))

    @staticmethod
    def calculate_anomalies(
        diagnostic_cube: Cube, mean_cube: Cube, std_cube: Optional[Cube] = None
    ) -> ndarray:
        """Calculate climate anomalies from the input cubes."""
        anomalies_data = diagnostic_cube.data - mean_cube.data
        if std_cube:
            anomalies_data = anomalies_data / std_cube.data
        return anomalies_data

    @staticmethod
    def _update_cube_name_and_units(
        output_cube: Cube, standardized_anomaly: bool = False
    ) -> None:
        """This method updates the name and units of the given output cube based on
        whether it represents a standardized anomaly or not. The cube is modified
        in place.

        Args:
            output_cube:
                The cube to be updated.
            standardized_anomaly:
                Flag indicating if the output is a standardized anomaly. If True,
                a "_standardized_anomaly" suffix is added to the name and units are
                made dimensionless ('1').
                If False, an "_anomaly" suffix is added to the name and units are
                unchanged.
        """
        suffix = "_standardized_anomaly" if standardized_anomaly else "_anomaly"

        try:
            output_cube.standard_name = output_cube.standard_name + suffix
        except ValueError:
            output_cube.long_name = output_cube.standard_name + suffix

        if standardized_anomaly:
            output_cube.units = "1"

    @staticmethod
    def _add_reference_epoch_metadata(output_cube: Cube, mean_cube: Cube) -> None:
        """Add epoch metadata to describe the creation of the anomaly.
        1. Add a scalar coordinate 'reference epoch' to describe the time period over
        which the climatology was calculated. The time bounds are inherited from the
        mean cube.
        2. Add a cell method called 'anomaly' to describe the operation
        that was performed.

        The output_cube is modified in place.
        """
        end_time = mean_cube.coord("time").points[-1]
        bounds = mean_cube.coord("time").bounds if mean_cube.coord("time").has_bounds() else [mean_cube.coord("time").points[0], mean_cube.coord("time").points[-1]]
        reference_epoch = iris.coords.AuxCoord(
            points=end_time,
            bounds=bounds,
            long_name="reference_epoch",
            units=mean_cube.coord("time").units,
        )

        output_cube.add_aux_coord(reference_epoch)

        cell_method = CellMethod(method="anomaly", coords=["reference_epoch"])
        output_cube.add_cell_method(cell_method)

    def _create_output_cube(
        self,
        diagnostic_cube: Cube,
        mean_cube: Cube,
        std_cube: Optional[Cube] = None,
    ) -> Cube:
        """
        Create a final anomalies cube by copying the diagnostic cube, updating its data
        with the anomalies data, and updating its metadata.
        """
        # Use the diagnostic cube as a template for the output cube
        output_cube = diagnostic_cube.copy()

        # Update the cube name and units
        standardized_anomaly = True if std_cube else False
        self._update_cube_name_and_units(output_cube, standardized_anomaly)

        # Create the reference epoch coordinate and cell method
        self._add_reference_epoch_metadata(output_cube, mean_cube)

        anomalies_data = self.calculate_anomalies(diagnostic_cube, mean_cube, std_cube)

        output_cube.data = anomalies_data

        return output_cube

    def process(
        self,
        diagnostic_cube: Cube,
        mean_cube: Cube,
        std_cube: Optional[Cube] = None,
    ) -> Cube:
        """Calculate anomalies from the input cubes.

        Args:
            diagnostic_cube:
                Cube containing the data to be converted to anomalies.
            mean_cube:
                Cube containing the mean data to be used for the calculation of
                anomalies.
            std_cube:
                Cube containing the standard deviation data to be used for the
                calculation of standardized anomalies. If not provided, only
                anomalies (not standardized anomalies) will be calculated.

        Returns:
            Cube containing the result of the calculation with updated metadata:
            - Name is updated to include "_anomaly" or "_standardized_anomaly" suffix.
            - Units are updated to "1" for standardized anomalies.
            - A reference epoch coordinate and an "anomaly" cell method are added."""

        cubes_to_check = [
            cube for cube in [diagnostic_cube, mean_cube, std_cube] if cube is not None
        ]
        self.verify_units_match(diagnostic_cube, mean_cube, std_cube)
        assert_spatial_coords_match(cubes_to_check)
        verify_time_coords_match(
            diagnostic_cube,
            mean_cube,
            std_cube=std_cube,
            ignore_temporal_mismatch=self.ignore_temporal_mismatch,
        )

        anomalies_cube = self._create_output_cube(diagnostic_cube, mean_cube, std_cube)

        return anomalies_cube


class CalculateForecastValueFromClimateAnomaly(BasePlugin):
    """Utility functionality to convert an input cube containing anomaly data to a
    forecast value cube. If a climatological anomaly and mean are supplied, the
    forecast values will be calculated by anomaly + mean. If the anomaly is
    standardized and a standard deviation and mean are supplied, the forecast values
    will be calculated by (standardized anomaly * standard deviation) + mean."""

    def __init__(
        self,
        ignore_temporal_mismatch: bool = False,
    ) -> None:
        """
        Initialise class.

        Args:
            ignore_temporal_mismatch:
                If True, ignore mismatch in time coordinates between the input cubes.
                Default is False. Set to True when converting from the anomaly of a
                diagnostic cube relative to mean and standard deviation cubes of a
                different period.
        """
        self.ignore_temporal_mismatch = ignore_temporal_mismatch

    @staticmethod
    def verify_inputs_for_forecast(
        standardized_anomaly: bool, std_cube: Optional[Cube] = None
    ) -> None:
        """Check that all required inputs are provided for the type of anomaly data
        (standardized or unstandardized) used. If the anomaly cube is standardized,
        the standard deviation cube must be provided. If the anomaly cube is not
        standardized, the standard deviation cube must not be provided. This is to
        prevent accidental use of standardized anomaly data when the standard deviation
        is not available.

        Raises:
            ValueError: If the anomaly cube is standardized and the standard deviation
            cube is not provided.
            ValueError: If the anomaly cube is not standardized and the
            standard deviation cube is provided.
        """
        if standardized_anomaly and not std_cube:
            # require a standard deviation to convert a standardized anomaly
            raise ValueError(
                "The standard deviation cube must be provided to calculate the "
                "forecast values from a standardized anomaly."
            )
        elif not standardized_anomaly and std_cube:
            # should not have a standard deviation if converting an anomaly
            raise ValueError(
                "The standard deviation cube should not be provided to calculate the "
                "forecast values from an unstandardized anomaly."
            )

    @staticmethod
    def verify_units_match(
        anomaly_cube: Cube,
        mean_cube: Cube,
        standardized_anomaly: bool,
        std_cube: Optional[Cube] = None,
    ) -> None:
        """Check that all required inputs are provided and all cubes have the same
        units. E.g. to prevent accidental use of cubes with rate data with cubes with
        accumulation data.

        Raises:
            ValueError: If the units of the cubes do not match.
            ValueError: If the cubes are not compatible with the anomaly cube.
        """
        if not standardized_anomaly and mean_cube.units != anomaly_cube.units:
            raise ValueError(
                "The mean cube must have the same units as the anomaly cube. "
                f"The following units were found: {mean_cube.units},"
                f"{anomaly_cube.units}."
            )
        elif standardized_anomaly and std_cube and mean_cube.units != std_cube.units:
            raise ValueError(
                "The standard deviation cube must have the same units as the mean cube."
                f"The following units were found: {std_cube.units},"
                f"{mean_cube.units}."
            )

    @staticmethod
    def calculate_forecast_value(
        anomaly_cube: Cube, mean_cube: Cube, std_cube: Optional[Cube] = None
    ) -> ndarray:
        """Calculate forecast values from the input cubes."""
        if std_cube:
            forecast_value = (anomaly_cube.data * std_cube.data) + mean_cube.data
        else:
            forecast_value = anomaly_cube.data + mean_cube.data

        return forecast_value

    @staticmethod
    def _update_cube_name_and_units(output_cube: Cube, mean_cube: Cube) -> None:
        """Set standard_name, long_name, and units on the output_cube using the
        mean_cube.

        Args:
            output_cube:
                The cube to be updated.

        """
        output_cube.standard_name = mean_cube.standard_name
        output_cube.long_name = mean_cube.long_name

        output_cube.units = mean_cube.units

    @staticmethod
    def _remove_reference_epoch_metadata(output_cube: Cube) -> None:
        """Remove the reference epoch coordinate and cell method from the output cube
        if present. The cube is modified in place.

        Args:
            output_cube:
                The cube from which to remove the reference epoch metadata.
        """
        if output_cube.coords("reference_epoch"):
            output_cube.remove_coord("reference_epoch")

        output_cube.cell_methods = tuple(
            cm for cm in output_cube.cell_methods if cm.method != "anomaly"
        )

    def _create_output_cube(
        self,
        anomaly_cube: Cube,
        mean_cube: Cube,
        std_cube: Optional[Cube] = None,
    ) -> Cube:
        """
        Create a final forecast value cube by copying the anomaly cube, updating its
        data with the forecast values data, and updating its metadata. The replaced
        metadata would be present on any anomaly cube processed through the plugin
        CalculateClimateAnomalies and is no longer necessary.
        """
        # Use the anomaly cube as a template for the output cube
        output_cube = anomaly_cube.copy()

        # Update the cube name and units
        self._update_cube_name_and_units(output_cube, mean_cube)

        # Remove the reference epoch coordinate and cell method
        self._remove_reference_epoch_metadata(output_cube)

        forecast_value_data = self.calculate_forecast_value(
            anomaly_cube, mean_cube, std_cube
        )

        output_cube.data = forecast_value_data

        return output_cube

    def process(
        self,
        anomaly_cube: Cube,
        mean_cube: Cube,
        std_cube: Cube = None,
    ) -> Cube:
        """Calculate forecast values from the input cubes.

        Args:
            anomaly_cube:
                Cube containing the data to be converted to forecast values.
            mean_cube:
                Cube containing the mean data to be used for the calculation of
                forecast values.
            std_cube:
                Cube containing the standard deviation data to be used for the
                calculation of forecast values from standardized anomalies.
                If not provided, only anomalies (not standardized anomalies) should be
                provided to calculate the forecast values.
        Returns:
            Cube containing the result of the calculation with updated metadata:
            - Name is updated to remove "_anomaly" or "_standardized_anomaly" suffix.
            - Units are converted to the units of the mean cube, if required.
            - The reference epoch coordinate and "anomaly" cell method are removed.
        """
        standardized_anomaly = False
        for name in [anomaly_cube.standard_name, anomaly_cube.long_name]:
            if name and "standardized" in name:
                standardized_anomaly = True
                break

        cubes_to_check = [
            cube for cube in [anomaly_cube, mean_cube, std_cube] if cube is not None
        ]

        self.verify_inputs_for_forecast(standardized_anomaly, std_cube)
        self.verify_units_match(anomaly_cube, mean_cube, standardized_anomaly, std_cube)
        assert_spatial_coords_match(cubes_to_check)
        verify_time_coords_match(
            anomaly_cube,
            mean_cube,
            std_cube=std_cube,
            ignore_temporal_mismatch=self.ignore_temporal_mismatch,
        )

        return self._create_output_cube(anomaly_cube, mean_cube, std_cube)


def verify_time_coords_match(
    diagnostic_cube: Cube,
    mean_cube: Cube,
    std_cube: Optional[Cube] = None,
    ignore_temporal_mismatch: bool = False,
) -> None:
    """Check that all cubes have compatible time coordinates.

    Args:
        diagnostic_cube:
            Cube containing the diagnostic or anomaly data.
        mean_cube:
            Cube containing the mean data.
        std_cube:
            Cube containing the standard deviation data (optional).
        ignore_temporal_mismatch:
            If True, ignore mismatch in time coordinates between the input cubes.

    Raises:
        ValueError: If the time coordinates of the cubes are incompatible.
    """
    errors = []

    # Check if mean and standard deviation cubes have the same bounds
    if std_cube and not np.array_equal(
        mean_cube.coord("time").bounds, std_cube.coord("time").bounds
    ):
        errors.append(
            "The mean and standard deviation cubes must have compatible "
            "time bounds. The following bounds were found: "
            f"mean_cube bounds: {mean_cube.coord('time').bounds}, "
            f"std_cube bounds: {std_cube.coord('time').bounds}."
        )

    # Check if diagnostic cube's time points fall within the bounds of the mean cube
    if not ignore_temporal_mismatch:
        diagnostic_max = diagnostic_cube.coord("time").cell(-1).point
        diagnostic_min = diagnostic_cube.coord("time").cell(0).point
        mean_time_bounds = mean_cube.coord("time").bounds[0]
        mean_time_bounds = [
            mean_cube.coord("time").units.num2date(bound) for bound in mean_time_bounds
        ]
        if not (
            diagnostic_max <= mean_time_bounds[1]
            and diagnostic_min >= mean_time_bounds[0]
        ):
            errors.append(
                "The diagnostic cube's time points must fall within the bounds "
                "of the mean cube. The following was found: "
                f"diagnostic cube maximum: {diagnostic_max}, "
                f"diagnostic cube minimum: {diagnostic_min}, "
                f"mean cube upper bound: {mean_time_bounds[1]}, "
                f"mean cube lower bound: {mean_time_bounds[0]}."
            )

    if errors:
        raise ValueError("\n".join(errors))
