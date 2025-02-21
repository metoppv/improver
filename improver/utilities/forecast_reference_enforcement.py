# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Provides utilities for updating a forecast or forecasts based on a reference."""

import warnings
from typing import Iterable, List, Optional, Union

import iris
import numpy as np
from iris.cube import Cube, CubeList

from improver import PostProcessingPlugin
from improver.metadata.forecast_times import unify_cycletime
from improver.metadata.probabilistic import is_probability


class EnforceConsistentForecasts(PostProcessingPlugin):
    """Enforce that the forecasts provided are no less than, no greater than, or between
    some linear function(s) of a reference forecast. For example, wind speed forecasts
    may be provided as the reference forecast for wind gust forecasts with a requirement
    that wind gusts are not less than 110% of the corresponding wind speed forecast.
    Wind speed forecasts of 10 m/s will mean that the resulting wind gust forecast will
    be at least 11 m/s.
    """

    def __init__(
        self,
        additive_amount: Union[float, List[float]] = 0.0,
        multiplicative_amount: Union[float, List[float]] = 1.0,
        comparison_operator: Union[str, List[str]] = ">=",
        diff_for_warning: Optional[float] = None,
        use_latest_update_time: Optional[bool] = False,
    ) -> None:
        """
        Initialise class for enforcing a forecast to be either greater than or equal to,
        or less than or equal to a linear function of a reference forecast. Can also
        enforce that a forecast is between two bounds created from the reference
        forecast, if this is the case then a list of two elements must be provided for
        additive_amount, multiplicative_amount, and comparison_operator.

        Args:
            additive_amount: The amount to be added to the reference forecast prior to
                enforcing consistency between the forecast and reference forecast. If
                both an additive_amount and multiplicative_amount are specified then
                addition occurs after multiplication. This option cannot be used for
                probability forecasts, if it is then an error will be raised.
            multiplicative_amount: The amount to multiply the reference forecast by
                prior to enforcing consistency between the forecast and reference
                forecast. If both an additive_amount and multiplicative_amount are
                specified then addition occurs after multiplication. This option cannot
                be used for probability forecasts, if it is then an error will be raised.
            comparison_operator: Determines whether the forecast is enforced to be not
                less than or not greater than the reference forecast. Valid choices are
                ">=", for not less than, and "<=" for not greater than. If provided as
                a list then each of ">=" and "<=" must be in the list exactly once.
            diff_for_warning: If assigned, the plugin will raise a warning if any
                absolute change in forecast value is greater than this value.
            use_latest_update_time:
                If True the returned cube that has been enforced will have a
                forecast_reference_time and/or blend_time that is the latest of
                the forecast and reference_forecast.
        """

        self.additive_amount = additive_amount
        self.multiplicative_amount = multiplicative_amount
        self.comparison_operator = comparison_operator
        self.diff_for_warning = diff_for_warning
        self.use_latest_update_time = use_latest_update_time

    @staticmethod
    def calculate_bound(
        cube: Cube, additive_amount: float, multiplicative_amount: float
    ) -> Cube:
        """
        Function to calculate a linear transformation of the reference forecast.

        Args:
            cube: An iris cube.
            additive_amount: The amount to be added to the cube. If both an
                additive_amount and multiplicative_amount are specified then addition
                occurs after multiplication.
            multiplicative_amount: The amount to multiply the cube by. If both an
                additive_amount and multiplicative_amount are specified then addition
                occurs after multiplication.

        Returns:
            A cube with identical metadata to input cube but with transformed data.
        """

        output = cube.copy()
        output.data = multiplicative_amount * output.data
        output.data = additive_amount + output.data

        return output

    def process(self, forecast: Cube, reference_forecast: Cube) -> Cube:
        """
        Function to enforce that the values in the forecast cube are not less than or
        not greater than a linear function of the corresponding values in
        reference_forecast, or between two bounds generated from two different linear
        functions of the reference_forecast.

        Args:
            forecast: A forecast cube
            reference_forecast: A reference forecast cube used to determine the bound/s
                of the forecast cube.

        Returns:
            A forecast cube with identical metadata to forecast but the forecasts are
            enforced to be within the calculated bounds.

        Raises:
            ValueError: If units of forecast and reference cubes are different and
                cannot be converted to match.
            ValueError: If additive_amount and multiplicative_amount are not 0.0 and 1.0,
                respectively, when a probability forecast is input.
            ValueError: If incorrect comparison_operator is input.
            ValueError: If contradictory bounds are generated.
            ValueError: If any of additive_amount, multiplicative_amount, or
                comparison_operator are lists when they are not all lists.

        Warns:
            Warning: If difference between generated bounds and forecast is greater than
                diff_for_warning.
        """

        # check forecast and reference units match
        try:
            reference_forecast.convert_units(forecast.units)
        except ValueError:
            if forecast.units != reference_forecast.units:
                msg = (
                    "The units in the forecast and reference cubes do not match and "
                    "cannot be converted to match. The units of forecast were "
                    f"{forecast.units}, the units of reference_forecast were "
                    f"{reference_forecast.units}."
                )
                raise ValueError(msg)

        # linear transformation cannot be applied to probability forecasts
        if self.additive_amount != 0.0 or self.multiplicative_amount != 1.0:
            if is_probability(forecast):
                msg = (
                    "For probability data, additive_amount must be 0.0 and "
                    "multiplicative_amount must be 1.0. The input additive_amount was "
                    f"{self.additive_amount}, the input multiplicative_amount was "
                    f"{self.multiplicative_amount}."
                )
                raise ValueError(msg)

        # calculate forecast_bound by applying specified linear transformation to
        # reference_forecast
        check_if_list = [
            isinstance(item, list)
            for item in [
                self.additive_amount,
                self.multiplicative_amount,
                self.comparison_operator,
            ]
        ]
        if all(check_if_list):
            lower_bound = self.calculate_bound(
                reference_forecast,
                self.additive_amount[0],
                self.multiplicative_amount[0],
            ).data
            upper_bound = self.calculate_bound(
                reference_forecast,
                self.additive_amount[1],
                self.multiplicative_amount[1],
            ).data
            if self.comparison_operator == ["<=", ">="]:
                upper_bound, lower_bound = lower_bound, upper_bound
            elif self.comparison_operator == [">=", "<="]:
                pass
            else:
                msg = (
                    "When comparison operators are provided as a list, the list must be "
                    f"either ['>=', '<='] or ['<=', '>='], not {self.comparison_operator}."
                )
                raise ValueError(msg)
            if np.any(lower_bound > upper_bound):
                msg = (
                    "The provided reference_cube, additive_amount and "
                    "multiplicative_amount have created contradictory bounds. Some of"
                    "the values in the lower bound are greater than the upper bound."
                )
                raise ValueError(msg)
        elif any(check_if_list):
            msg = (
                "If any of additive_amount, multiplicative_amount, or comparison_operator "
                "are input as a list, then they must all be input as a list of 2 elements. "
            )
            raise ValueError(msg)
        else:
            bound = self.calculate_bound(
                reference_forecast, self.additive_amount, self.multiplicative_amount
            )
            lower_bound = None
            upper_bound = None
            if self.comparison_operator == ">=":
                lower_bound = bound.data
            elif self.comparison_operator == "<=":
                upper_bound = bound.data
            else:
                msg = (
                    "When enforcing consistency with one bound, comparison_operator "
                    f"must be either '>=' or '<=', not {self.comparison_operator}."
                )
                raise ValueError(msg)

        new_forecast = forecast.copy()
        new_forecast.data = np.clip(new_forecast.data, lower_bound, upper_bound)

        if self.use_latest_update_time:
            forecast_cycle_coords = [
                crd
                for crd in ["forecast_reference_time", "blend_time"]
                if new_forecast.coords(crd)
            ]
            ref_cycle_coords = [
                crd
                for crd in ["forecast_reference_time", "blend_time"]
                if reference_forecast.coords(crd)
            ]
            # If one cube has a blend_time and one does not the subsequent
            # tooling will not succeed, so raise an exception here.
            if set(forecast_cycle_coords) != set(ref_cycle_coords):
                raise ValueError(
                    "Cubes do not include the same set of cycle time coordinates "
                    "and cannot be updated to match as part of cube enforcement."
                )
            latest_times = []
            latest_times.extend(
                [new_forecast.coord(crd).cell(0).point for crd in forecast_cycle_coords]
            )
            latest_times.extend(
                [
                    reference_forecast.coord(crd).cell(0).point
                    for crd in ref_cycle_coords
                ]
            )
            latest_time = max(latest_times)

            new_forecast, _ = unify_cycletime(
                [new_forecast, reference_forecast],
                latest_time,
                target_coords=forecast_cycle_coords,
            )

        diff = new_forecast.data - forecast.data
        max_abs_diff = np.max(np.abs(diff))
        if self.diff_for_warning is not None and max_abs_diff > self.diff_for_warning:
            warnings.warn(
                f"Inconsistency between forecast {forecast.name} and "
                f"{reference_forecast.name} is greater than {self.diff_for_warning}. "
                f"Maximum absolute difference reported was {max_abs_diff}"
            )

        return new_forecast


def normalise_to_reference(
    cubes: CubeList, reference: Cube, ignore_zero_total: bool = False
) -> CubeList:
    """Update the data in cubes so that the sum of this data is equal to the reference
    cube. This is done by replacing the data in cubes with a fraction of the data in
    reference based upon the fraction that each cube contributes to the sum total of
    data in cubes.

    Args:
        cubes: Cubelist containing the cubes to be updated. Must contain at least 2
            cubes.
        reference: Cube with data which the sum of cubes will be forced to be equal to.
        ignore_zero_total: If False, an error will be raised if the sum total of data
            in input_cubes is zero where the reference cube contains a non-zero value.
            If True, this case will be ignored, leaving the values in the cubelist as
            zero rather than ensuring their total equals the corresponding value in
            reference cube.

    Raises:
        ValueError: If length of cubes is less than 2.
        ValueError: If any input cubes have a different number of dimensions to
            reference, or if the dimension coordinates in any of the input cubes do not
            match the dimension coordinates in reference.
        ValueError: If there are instances where the total of the input cubes is zero
            but the corresponding value in reference is non-zero. This error can be
            ignored if ignore_zero_total is true.

    Returns:
        Cubelist with length equal to the length of cubes. Each cube in the returned
        cubelist will have metadata matching the cube in the same position in input
        cubes, but containing different data.
    """
    if len(cubes) < 2:
        msg = (
            f"The input cubes must be of at least length 2. The length of the input "
            f"cubes was {len(cubes)}"
        )
        raise ValueError(msg)

    # check cube compatibility
    reference_dim_coords = reference.coords(dim_coords=True)
    n_dims_mismatch = False
    coord_mismatch = False
    n_dims = []
    mismatching_coords = {}
    for cube in cubes:
        cube_dim_coords = cube.coords(dim_coords=True)
        n_dims.append(len(cube_dim_coords))
        if len(cube_dim_coords) != len(reference_dim_coords):
            n_dims_mismatch = True
        if not n_dims_mismatch:
            mismatching_coords[cube.name()] = []
            for dim_coord in cube_dim_coords:
                try:
                    reference_coord = reference.coord(dim_coord.name(), dim_coords=True)
                except iris.exceptions.CoordinateNotFoundError:
                    coord_mismatch = True
                    mismatching_coords[cube.name()].append(dim_coord.name())
                    continue
                if not dim_coord == reference_coord:
                    coord_mismatch = True
                    mismatching_coords[cube.name()].append(dim_coord.name())

    if n_dims_mismatch:
        msg = (
            f"The number of dimensions in input cubes are not all the same as the "
            f"number of dimensions on the reference cube. The number of dimensions in "
            f"the input cubes were {n_dims}. The number of dimensions in the "
            f"reference cube was {len(reference_dim_coords)}."
        )
        raise ValueError(msg)

    if coord_mismatch and not is_probability(reference):
        msg = (
            f"The dimension coordinates on the input cubes and the reference did not "
            f"all match. The following coordinates were found to differ: "
            f"{mismatching_coords}."
        )
        raise ValueError(msg)

    total = cubes[0].data.copy()
    if len(cubes) > 1:
        for cube in cubes[1:]:
            total += cube.data

    # check for zeroes in total when reference is non-zero
    total_zeroes = total == 0.0
    reference_non_zero = reference.data != 0.0
    both_true = np.logical_and(total_zeroes, reference_non_zero)
    if np.any(both_true):
        if not ignore_zero_total:
            msg = (
                "There are instances where the total of input cubes is zero but the "
                "corresponding value in reference is non-zero. The input cubes cannot "
                "be updated so that the total equals the value in the reference in "
                "these instances."
            )
            raise ValueError(msg)

    # update total where zero to avoid dividing by zero later.
    total[total_zeroes] = 1.0

    output = iris.cube.CubeList()
    for index, cube in enumerate(cubes):
        output_cube = cube.copy(data=reference.data * cube.data / total)
        output.append(output_cube)

    return output


def split_cubes_by_name(
    cubes: Union[Iterable[Cube], CubeList], cube_names: Union[str, List[str]] = None
) -> tuple:
    """Split a list of cubes into two lists; one containing all cubes with names which
    match cube_names, and the other containing all the other cubes.

    Args:
        cubes: List of cubes
        cube_names: the name of the cube/s to be used as reference. This can be either
            a single name or a list of names. If None, the first cubelist returned will
            contain all the input cubes and the second cubelist will be empty.

    Returns:
        - A cubelist containing all cubes with names which match cube_names
        - A cubelist containing all cubes with names which do not match cube_names
    """

    desired_cubes = iris.cube.CubeList()
    other_cubes = iris.cube.CubeList()

    if cube_names is None:
        desired_cubes = cubes
    else:
        if isinstance(cube_names, str):
            cube_names = [cube_names]
        for cube in cubes:
            if cube.name() in cube_names:
                desired_cubes.append(cube)
            else:
                other_cubes.append(cube)

    return desired_cubes, other_cubes
