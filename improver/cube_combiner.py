# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Module containing plugins for combining cubes"""

from operator import eq
from typing import Callable, List, Union

import iris
import numpy as np
from iris.coords import AuxCoord, CellMethod, DimCoord
from iris.cube import Cube, CubeList
from iris.exceptions import CoordinateNotFoundError

from improver import BasePlugin
from improver.metadata.amend import update_diagnostic_name
from improver.metadata.check_datatypes import enforce_dtype
from improver.metadata.constants.time_types import TIME_COORDS
from improver.metadata.probabilistic import find_threshold_coordinate
from improver.utilities.cube_manipulation import (
    enforce_coordinate_ordering,
    expand_bounds,
    filter_realizations,
)


class Combine(BasePlugin):
    """Combine input cubes.

    Combine the input cubes into a single cube using the requested operation.
    The first cube in the input list provides the template for output metadata.
    If coordinates are expanded as a result of this combine operation
    (e.g. expanding time for accumulations / max in period) the upper bound of
    the new coordinate will also be used as the point for the new coordinate.
    """

    def __init__(
        self,
        operation: str,
        broadcast: str = None,
        minimum_realizations: Union[str, int, None] = None,
        new_name: str = None,
        cell_method_coordinate: str = None,
        expand_bound: bool = True,
    ):
        r"""
        Args:
            operation (str):
                An operation to use in combining input cubes. One of:
                +, -, \*, add, subtract, multiply, min, max, mean
            broadcast (str):
                If specified, broadcast input cubes to the stated coord prior to combining -
                the coord must already exist on the first input cube.
            minimum_realizations (int):
                If specified, the input cubes will be filtered to ensure that only realizations that
                include all available lead times are combined. If the number of realizations that
                meet this criteria are fewer than this integer, an error will be raised.
                Minimum value is 1.
            new_name (str):
                New name for the resulting dataset.
            cell_method_coordinate (str):
                If specified, a cell method is added to the output with the coordinate
                provided. This is only available for max, min and mean operations.
            expand_bound:
                If True then coord bounds will be extended to represent all cubes being combined.
        """
        try:
            self.minimum_realizations = int(minimum_realizations)
        except TypeError:
            if minimum_realizations is not None:
                raise
            self.minimum_realizations = None
        self.new_name = new_name
        self.broadcast = broadcast
        self.cell_method_coordinate = cell_method_coordinate
        self.expand_bound = expand_bound

        self.plugin = CubeCombiner(
            operation,
            cell_method_coordinate=cell_method_coordinate,
            broadcast=self.broadcast,
            expand_bound=self.expand_bound,
        )

    def process(self, cubes: CubeList) -> Cube:
        """
        Preprocesses the cubes, then passes them to the appropriate plugin

        Args:
            cubes (iris.cube.CubeList or list of iris.cube.Cube):
                An iris CubeList to be combined.

        Returns:
            result (iris.cube.Cube):
                Returns a cube with the combined data.

        Raises:
            TypeError:
                If input list of cubes is empty

            ValueError:
                If minimum_realizations aren't met, or less than one were requested.
        """
        if not cubes:
            raise TypeError("A cube is needed to be combined.")
        if self.new_name is None:
            self.new_name = cubes[0].name()

        if self.minimum_realizations is None:
            filtered_cubes = cubes
        else:
            if self.minimum_realizations < 1:
                raise ValueError(
                    f"Minimum realizations must be at least 1, not {self.minimum_realizations}"
                )

            cube = filter_realizations(cubes)
            realization_count = len(cube.coord("realization").points)
            if realization_count < self.minimum_realizations:
                raise ValueError(
                    f"After filtering, number of realizations {realization_count} "
                    "is less than the minimum number of realizations allowed "
                    f"({self.minimum_realizations})"
                )
            filtered_cubes = cube.slices_over("time")
        return self.plugin(CubeList(filtered_cubes), self.new_name)


class CubeCombiner(BasePlugin):
    """Plugin for combining cubes using linear operators"""

    COMBINE_OPERATORS = {
        "+": np.add,
        "add": np.add,
        "-": np.subtract,
        "subtract": np.subtract,
        "*": np.multiply,
        "multiply": np.multiply,
        "max": np.maximum,
        "min": np.minimum,
        "mean": np.add,
    }  # mean is calculated in two steps: sum and normalise

    def __init__(
        self,
        operation: str,
        cell_method_coordinate: str = None,
        broadcast: str = None,
        expand_bound: bool = True,
    ) -> None:
        """Create a CubeCombiner plugin

        Args:
            operation:
                Operation (+, - etc) to apply to the incoming cubes.
            cell_method_coordinate:
                If specified, a cell method is added to the output with the coordinate
                provided. This is only available for max, min and mean operations.
            broadcast:
                If specified, broadcast input cubes to the stated coord prior to combining -
                the coord must already exist on the first input cube.
            expand_bound:
                If True then coord bounds will be extended to represent all cubes being combined.
        Raises:
            ValueError: if operation is not recognised in dictionary
        """
        self.operation = operation
        self.cell_method_coordinate = cell_method_coordinate
        try:
            self.operator = self.COMBINE_OPERATORS[operation]
        except KeyError:
            msg = "Unknown operation {}".format(operation)
            raise ValueError(msg)
        self.broadcast = broadcast
        self.normalise = operation == "mean"
        self.expand_bound = expand_bound

    @staticmethod
    def _check_dimensions_match(
        cube_list: Union[List[Cube], CubeList], comparators: List[Callable] = [eq],
    ) -> None:
        """
        Check all coordinate dimensions on the input cubes match according to
        the comparators specified.

        Args:
            cube_list:
                List of cubes to compare
            comparators:
                Comparison operators, at least one of which must return "True"
                for each coordinate in order for the match to be valid

        Raises:
            ValueError: If dimension coordinates do not match
        """
        ref_coords = cube_list[0].coords(dim_coords=True)
        for cube in cube_list[1:]:
            coords = cube.coords(dim_coords=True)
            compare = [
                np.any([comp(a, b) for comp in comparators])
                for a, b in zip(coords, ref_coords)
            ]
            if not np.all(compare):
                msg = (
                    "Cannot combine cubes with different dimensions:\n"
                    "{} and {}".format(repr(cube_list[0]), repr(cube))
                )
                raise ValueError(msg)

    @staticmethod
    def _get_expanded_coord_names(cube_list: Union[List[Cube], CubeList]) -> List[str]:
        """
        Get names of coordinates whose bounds need expanding and points
        recalculating after combining cubes. These are the scalar coordinates
        that are present on all input cubes, but have different values.

        Args:
            cube_list:
                List of cubes to that will be combined

        Returns:
            List of coordinate names to expand
        """
        shared_scalar_coords = {
            coord.name() for coord in cube_list[0].coords(dim_coords=False)
        }
        for cube in cube_list[1:]:
            cube_scalar_coords = {
                coord.name() for coord in cube.coords(dim_coords=False)
            }
            shared_scalar_coords = shared_scalar_coords & cube_scalar_coords

        expanded_coords = []
        for cube in cube_list[1:]:
            for coord in shared_scalar_coords:
                if (
                    cube.coord(coord) != cube_list[0].coord(coord)
                    and coord not in expanded_coords
                ):
                    expanded_coords.append(coord)
        return expanded_coords

    def _add_cell_method(self, cube: Cube) -> None:
        """Add a cell method to record the operation undertaken.

        Args:
            cube:
                Cube to which a cell method will be added.

        Raises:
            ValueError: If a cell_method_coordinate is provided and the operation
                is not max, min or mean.
        """
        cell_method_lookup = {"max": "maximum", "min": "minimum", "mean": "mean"}
        if self.operation in ["max", "min", "mean"] and self.cell_method_coordinate:
            cube.add_cell_method(
                CellMethod(
                    cell_method_lookup[self.operation],
                    coords=self.cell_method_coordinate,
                )
            )
        elif self.cell_method_coordinate:
            msg = (
                "A cell method coordinate has been produced with "
                f"operation: {self.operation}. A cell method coordinate "
                "can only be added if the operation is max, min or mean."
            )
            raise ValueError(msg)

    def _combine_cube_data(self, cube_list: Union[List[Cube], CubeList]) -> Cube:
        """
        Perform cumulative operation to combine cube data

        Args:
            cube_list

        Returns:
            Combined cube

        Raises:
            TypeError: if the operation results in an escalated datatype
        """
        result = cube_list[0].copy()

        # Slice over realization if possible to reduce memory usage.
        if (
            "realization" in [crd.name() for crd in result.coords(dim_coords=True)]
            and self.broadcast != "realization"
        ):
            rslices = iris.cube.CubeList(result.slices_over("realization"))
            for cube in cube_list[1:]:
                cslices = cube.slices_over("realization")
                for rslice, cslice in zip(rslices, cslices):
                    rslice.data = self.operator(rslice.data, cslice.data)
            result = rslices.merge_cube()
            enforce_coordinate_ordering(
                result, [d.name() for d in cube_list[0].coords(dim_coords=True)]
            )
        else:
            for cube in cube_list[1:]:
                result.data = self.operator(result.data, cube.data)

        if self.normalise:
            result.data = result.data / len(cube_list)

        enforce_dtype(str(self.operator), cube_list, result)

        return result

    def _setup_coords_for_broadcast(self, cube_list: CubeList) -> CubeList:
        """
        Adds a scalar threshold to any subsequent cube in cube_list so that they all
        match the dimensions, in order, of the first cube in the list

        Args:
            cube_list

        Returns:
            Updated version of cube_list

        Raises:
            CoordinateNotFoundError: if there is no threshold coordinate on the
                first cube in the list
            TypeError: if there is a scalar threshold coordinate on any of the
                later cubes, which would indicate that the cube is only valid for
                a single threshold and should not be broadcast to all thresholds.
        """
        target_cube = cube_list[0]

        if self.broadcast == "threshold":
            self.broadcast = find_threshold_coordinate(cube_list[0]).name()

        try:
            target_coord = target_cube.coord(self.broadcast)
        except CoordinateNotFoundError:
            raise CoordinateNotFoundError(
                f"Cannot find coord {self.broadcast} in {repr(target_cube)} to broadcast to"
            )

        new_list = CubeList([])
        for cube in cube_list:
            try:
                found_coord = cube.coord(target_coord)
            except CoordinateNotFoundError:
                new_coord = target_coord.copy([0], bounds=None)
                cube = cube.copy()
                cube.add_aux_coord(new_coord, None)
                cube = iris.util.new_axis(cube, new_coord)
                enforce_coordinate_ordering(
                    cube, [d.name() for d in target_cube.coords(dim_coords=True)]
                )
            else:
                if found_coord not in cube.dim_coords:
                    msg = "Cannot broadcast to coord threshold as it already exists as an AuxCoord"
                    raise TypeError(msg)
            new_list.append(cube)

        return new_list

    @staticmethod
    def _coords_are_broadcastable(coord1: DimCoord, coord2: DimCoord) -> bool:
        """
        Broadcastable coords will differ only in length, so create a copy of one with
        the points and bounds of the other and compare. Also ensure length of at least
        one of the coords is 1.
        """
        coord_copy = coord1.copy(coord2.points, bounds=coord2.bounds)

        return (coord_copy == coord2) and (
            (len(coord1.points) == 1) or (len(coord2.points) == 1)
        )

    def process(
        self, cube_list: Union[List[Cube], CubeList], new_diagnostic_name: str,
    ) -> Cube:
        """
        Combine data and metadata from a list of input cubes into a single
        cube, using the specified operation to combine the cube data.  The
        first cube in the input list provides the template for the combined
        cube metadata.
        If coordinates are expanded as a result of this combine operation
        (e.g. expanding time for accumulations / max in period) the upper bound
        of the new coordinate will also be used as the point for the new coordinate.

        Args:
            cube_list:
                List of cubes to combine.
            new_diagnostic_name:
                New name for the combined diagnostic.

        Returns:
            Cube containing the combined data.

        Raises:
            ValueError: If the cube_list contains only one cube.
        """
        if len(cube_list) < 2:
            msg = "Expecting 2 or more cubes in cube_list"
            raise ValueError(msg)

        if self.broadcast is not None:
            cube_list = self._setup_coords_for_broadcast(cube_list=cube_list)
            self._check_dimensions_match(
                cube_list, comparators=[eq, self._coords_are_broadcastable]
            )
        else:
            self._check_dimensions_match(cube_list, comparators=[eq])

        result = self._combine_cube_data(cube_list)
        expanded_coord_names = self._get_expanded_coord_names(cube_list)

        if self.operation in ["*", "multiply"]:
            update_diagnostic_name(cube_list[0], new_diagnostic_name, result)
        else:
            if expanded_coord_names and self.expand_bound:
                result = expand_bounds(result, cube_list, expanded_coord_names)
            result.rename(new_diagnostic_name)
            self._add_cell_method(result)

        return result


class MaxInTimeWindow(BasePlugin):
    """Find the maximum within a time window for a period diagnostic. For example,
    find the maximum 3-hour precipitation accumulation within a 24 hour window."""

    def __init__(self, minimum_realizations: Union[str, int, None] = None):
        """Initialise class.

        Args:
            minimum_realizations (int):
                If specified, the input cubes will be filtered to ensure that only realizations that
                include all available lead times are combined. If the number of realizations that
                meet this criteria are fewer than this integer, an error will be raised.
                Minimum value is 1.

        """
        self.minimum_realizations = minimum_realizations
        self.time_units_in_hours = TIME_COORDS["time"].units.replace("seconds", "hours")

    def _get_coords_in_hours(
        self, cubes: List[Cube]
    ) -> List[Union[AuxCoord, DimCoord]]:
        """Get the time coordinates from the input cubes in units of hours
        since 1970-01-01 00:00:00.

        Args:
            cubes: Cubes from which the time coordinates will be extracted.

        Returns:
            The time coordinates extracted from the input cubes.
        """
        coords = [c.coord("time").copy() for c in cubes]
        [c.convert_units(self.time_units_in_hours) for c in coords]
        return coords

    def _check_input_cubes(self, coords: List[Union[AuxCoord, DimCoord]]):
        """Check that the input cubes are period diagnostics i.e. where the time
        coordinate has bounds representing a period and that the bounds represent
        a consistent period.

        Args:
            coords: The time coordinates extracted from the input cubes.

        Raises:
            ValueError: The input cubes do not have bounds.
            ValueError: The input cubes do not all have bounds.
            ValueError: The input cubes have bounds that imply mismatching periods.

        """
        msg = None
        if not all([c.has_bounds() for c in coords]):
            msg = (
                "When computing the maximum over a time window, the inputs "
                "are expected to be diagnostics representing a time period "
                "with bounds. "
            )
            [c.convert_units(self.time_units_in_hours) for c in coords]
            period = np.unique([np.diff(c.bounds) for c in coords if c.has_bounds()])
            if not any([c.has_bounds() for c in coords]):
                msg = msg + ("The cubes provided do not have bounds.")
            else:
                msg = msg + (
                    "The cubes provided do not all have bounds. "
                    f"Period(s) indicated by bounds: {period} hours"
                )
        elif len(np.unique([np.diff(c.bounds) for c in coords])) > 1:
            [c.convert_units(self.time_units_in_hours) for c in coords]
            period = np.unique([np.diff(c.bounds) for c in coords])
            msg = (
                "The bounds on the cubes imply mismatching periods. "
                f"Period(s) indicated by bounds: {period} hours"
            )

        if msg:
            raise ValueError(msg)

    def _correct_metadata(
        self, cube: Cube, coords_in_hours: List[Union[AuxCoord, DimCoord]]
    ) -> Cube:
        """Correct metadata in particular to ensure that the cell methods are
        updated to represent a period for a time window diagnostic.

        Args:
            cube: Cube representating the maximum over a time window for a period
                diagnostic.
            coords_in_hours: List of time coordinates in units of hours since
                1970-01-01 00:00:00.

        Returns:
            Cube representating the maximum over a time window for a period
            diagnostic with appropriate metadata.
        """
        if cube.name().startswith("probability_of"):
            diag_name = cube.coord(var_name="threshold").name()
        else:
            diag_name = cube.name()
        (period,) = np.unique([np.diff(c.bounds) for c in coords_in_hours])
        hour_text = "hour" if round(period) == 1 else "hours"
        sum_comment = (
            f"of {diag_name} over {round(period)} {hour_text} within time window"
        )
        max_comment = f"of {diag_name}"

        # Remove cell methods with the same method and coordinate name as will be added.
        cell_methods = []
        for cm in cube.cell_methods:
            if cm.method in ["sum", "maximum"] and "time" in cm.coord_names:
                continue
            else:
                cell_methods.append(cm)
        cube.cell_methods = tuple(cell_methods)
        # Add cell methods to record that a maximum over time has been computed,
        # as well as some information about the inputs to this value.
        cube.add_cell_method(CellMethod("sum", coords=["time"], comments=sum_comment))
        cube.add_cell_method(
            CellMethod("maximum", coords=["time"], comments=max_comment)
        )
        return cube

    def process(self, cubes: CubeList) -> Cube:
        """Compute the maximum probability or maximum diagnostic value within a
        time window for a period diagnostic using the Combine plugin. The resulting
        cube has a time coordinate with bounds that represent the time window whilst
        the cell method has been updated to represent the period recorded on the input
        cubes. For example, the time window might be 24 hours, whilst the period might
        be 3 hours.

        Args:
            cubes (iris.cube.CubeList or list of iris.cube.Cube):
                An iris CubeList to be combined.

        Returns:
            result (iris.cube.Cube):
                Returns a cube with the combined data.

        """
        coords_in_hours = self._get_coords_in_hours(cubes)
        self._check_input_cubes(coords_in_hours)
        cube = Combine("max", minimum_realizations=self.minimum_realizations)(cubes)
        cube = self._correct_metadata(cube, coords_in_hours)
        return cube
