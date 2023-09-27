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
""" Provides support utilities."""

import copy
from typing import List, Optional, Tuple, Union

import cartopy.crs as ccrs
import iris
import netCDF4
import numpy as np
from cartopy.crs import CRS
from cf_units import Unit
from iris.coords import AuxCoord, CellMethod
from iris.cube import Cube, CubeList
from numpy import ndarray
from scipy.ndimage.filters import maximum_filter

from improver import BasePlugin, PostProcessingPlugin
from improver.metadata.amend import update_diagnostic_name
from improver.metadata.constants.attributes import MANDATORY_ATTRIBUTE_DEFAULTS
from improver.metadata.probabilistic import in_vicinity_name_format, is_probability
from improver.metadata.utilities import create_new_diagnostic_cube
from improver.utilities.cube_checker import check_cube_coordinates, spatial_coords_match
from improver.utilities.cube_manipulation import enforce_coordinate_ordering


def check_if_grid_is_equal_area(
    cube: Cube, require_equal_xy_spacing: bool = True
) -> None:
    """
    Identify whether the grid is an equal area grid, by checking whether points
    are equally spaced along each of the x- and y-axes.  By default this
    function also checks whether the grid spacing is the same in both spatial
    dimensions.

    Args:
        cube:
            Cube with coordinates that will be checked.
        require_equal_xy_spacing:
            Flag to require the grid is equally spaced in the two spatial
            dimensions (not strictly required for equal-area criterion).

    Raises:
        ValueError: If coordinate points are not equally spaced along either
            axis (from calculate_grid_spacing)
        ValueError: If point spacing is not equal for the two spatial axes
    """
    x_diff = calculate_grid_spacing(cube, "metres", axis="x")
    y_diff = calculate_grid_spacing(cube, "metres", axis="y")
    if require_equal_xy_spacing and not np.isclose(x_diff, y_diff):
        raise ValueError("Grid does not have equal spacing in x and y dimensions")


def calculate_grid_spacing(
    cube: Cube, units: Union[Unit, str], axis: str = "x", rtol: float = 1.0e-5
) -> float:
    """
    Returns the grid spacing of a given spatial axis. This will be positive for
    axes that stride negatively.

    Args:
        cube:
            Cube of data on equal area grid
        units:
            Unit in which the grid spacing is required
        axis:
            Axis ('x' or 'y') to use in determining grid spacing
        rtol:
            relative tolerance

    Returns:
        Grid spacing in required unit

    Raises:
        ValueError: If points are not equally spaced
    """
    coord = cube.coord(axis=axis).copy()
    coord.convert_units(units)
    diffs = np.abs(np.diff(coord.points))
    diffs_mean = np.mean(diffs)

    if not np.allclose(diffs, diffs_mean, rtol=rtol, atol=0.0):
        raise ValueError(
            "Coordinate {} points are not equally spaced".format(coord.name())
        )
    else:
        return diffs_mean


def distance_to_number_of_grid_cells(
    cube: Cube, distance: float, axis: str = "x", return_int: bool = True
) -> Union[float, int]:
    """
    Return the number of grid cells in the x and y direction based on the
    input distance in metres.  Requires an equal-area grid on which the spacing
    is equal in the x- and y- directions.

    Args:
        cube:
            Cube containing the x and y coordinates, which will be used for
            calculating the number of grid cells in the x and y direction,
            which equates to the requested distance in the x and y direction.
        distance:
            Distance in metres. Must be positive.
        return_int:
            If true only integer number of grid_cells are returned, rounded
            down. If false the number of grid_cells returned will be a float.
        axis:
            Axis ('x' or 'y') to use in determining grid spacing

    Returns:
        Number of grid cells along the specified (x or y) axis equal to the
        requested distance in metres.

    Raises:
        ValueError: If a non-positive distance is provided.
    """
    d_error = f"Distance of {distance}m"
    if distance <= 0:
        raise ValueError(f"Please specify a positive distance in metres. {d_error}")

    # calculate grid spacing along chosen axis
    grid_spacing_metres = calculate_grid_spacing(cube, "metres", axis=axis)
    grid_cells = distance / abs(grid_spacing_metres)

    if return_int:
        grid_cells = int(grid_cells)
        if grid_cells == 0:
            zero_distance_error = f"{d_error} gives zero cell extent"
            raise ValueError(zero_distance_error)

    return grid_cells


def number_of_grid_cells_to_distance(cube: Cube, grid_points: int) -> float:
    """
    Calculate distance in metres equal to the given number of gridpoints
    based on the coordinates on an input cube.

    Args:
        cube:
            Cube for which the distance is to be calculated.
        grid_points:
            Number of grid points to convert.

    Returns:
        The radius in metres.
    """
    check_if_grid_is_equal_area(cube)
    spacing = calculate_grid_spacing(cube, "metres")
    radius_in_metres = spacing * grid_points
    return radius_in_metres


class DifferenceBetweenAdjacentGridSquares(BasePlugin):
    """
    Calculate the difference between adjacent grid squares within
    a cube. The difference is calculated along the x and y axis
    individually.
    """

    @staticmethod
    def _update_metadata(diff_cube: Cube, coord_name: str, cube_name: str) -> None:
        """Rename cube, add attribute and cell method to describe difference.

        Args:
            diff_cube
            coord_name
            cube_name
        """
        # Add metadata to indicate that a difference has been calculated.
        # TODO: update metadata for difference when
        #  proper conventions have been agreed upon.
        cell_method = CellMethod(
            "difference", coords=[coord_name], intervals="1 grid length"
        )
        diff_cube.add_cell_method(cell_method)
        diff_cube.attributes["form_of_difference"] = "forward_difference"
        diff_cube.rename("difference_of_" + cube_name)

    @staticmethod
    def create_difference_cube(
        cube: Cube, coord_name: str, diff_along_axis: ndarray
    ) -> Cube:
        """
        Put the difference array into a cube with the appropriate
        metadata.

        Args:
            cube:
                Cube from which the differences have been calculated.
            coord_name:
                The name of the coordinate over which the difference
                have been calculated.
            diff_along_axis:
                Array containing the differences.

        Returns:
            Cube containing the differences calculated along the
            specified axis.
        """
        points = cube.coord(coord_name).points
        mean_points = (points[1:] + points[:-1]) / 2

        # Copy cube metadata and coordinates into a new cube.
        # Create a new coordinate for the coordinate along which the
        # difference has been calculated.
        metadata_dict = copy.deepcopy(cube.metadata._asdict())
        diff_cube = Cube(diff_along_axis, **metadata_dict)

        for coord in cube.dim_coords:
            dims = cube.coord_dims(coord)
            if coord.name() in [coord_name]:
                coord = coord.copy(points=mean_points)
            diff_cube.add_dim_coord(coord.copy(), dims)
        for coord in cube.aux_coords:
            dims = cube.coord_dims(coord)
            diff_cube.add_aux_coord(coord.copy(), dims)
        for coord in cube.derived_coords:
            dims = cube.coord_dims(coord)
            diff_cube.add_aux_coord(coord.copy(), dims)
        return diff_cube

    @staticmethod
    def calculate_difference(cube: Cube, coord_name: str) -> ndarray:
        """
        Calculate the difference along the axis specified by the
        coordinate.

        Args:
            cube:
                Cube from which the differences will be calculated.
            coord_name:
                Name of coordinate along which the difference is calculated.

        Returns:
            Array after the differences have been calculated along the
            specified axis.
        """
        diff_axis = cube.coord_dims(coord_name)[0]
        diff_along_axis = np.diff(cube.data, axis=diff_axis)
        return diff_along_axis

    def process(self, cube: Cube) -> Tuple[Cube, Cube]:
        """
        Calculate the difference along the x and y axes and return
        the result in separate cubes. The difference along each axis is
        calculated using numpy.diff.

        Args:
            cube:
                Cube from which the differences will be calculated.

        Returns:
            - Cube after the differences have been calculated along the
              x axis.
            - Cube after the differences have been calculated along the
              y axis.
        """
        diffs = []
        for axis in ["x", "y"]:
            coord_name = cube.coord(axis=axis).name()
            diff_cube = self.create_difference_cube(
                cube, coord_name, self.calculate_difference(cube, coord_name)
            )
            self._update_metadata(diff_cube, coord_name, cube.name())
            diffs.append(diff_cube)
        return tuple(diffs)


class GradientBetweenAdjacentGridSquares(BasePlugin):

    """Calculate the gradient between adjacent grid squares within
    a cube. The gradient is calculated along the x and y axis
    individually."""

    def __init__(self, regrid: bool = False) -> None:
        """Initialise plugin.

        Args:
            regrid:
                If True, the gradient cube is regridded to match the spatial
                dimensions of the input cube. If False, the length of the
                spatial dimensions of the gradient cube are one less than for
                the input cube.
        """
        self.regrid = regrid

    @staticmethod
    def _create_output_cube(
        gradient: ndarray, diff: Cube, cube: Cube, axis: str
    ) -> Cube:
        """
        Create the output gradient cube.

        Args:
            gradient:
                Gradient values used in the data array of the resulting cube.
            diff:
                Cube containing differences along the x or y axis
            cube:
                Cube with correct output dimensions
            axis:
                Short-hand reference for the x or y coordinate, as allowed by
                iris.util.guess_coord_axis.

        Returns:
            A cube of the gradients in the coordinate direction specified.
        """
        grad_cube = create_new_diagnostic_cube(
            "gradient_of_" + cube.name(),
            cube.units / diff.coord(axis=axis).units,
            diff,
            MANDATORY_ATTRIBUTE_DEFAULTS,
            data=gradient,
        )
        return grad_cube

    @staticmethod
    def _gradient_from_diff(diff: Cube, axis: str) -> ndarray:
        """
        Calculate the gradient along the x or y axis from differences between
        adjacent grid squares.

        Args:
            diff:
                Cube containing differences along the x or y axis
            axis:
                Short-hand reference for the x or y coordinate, as allowed by
                iris.util.guess_coord_axis.

        Returns:
            Array of the gradients in the coordinate direction specified.
        """
        grid_spacing = np.diff(diff.coord(axis=axis).points)[0]
        gradient = diff.data / grid_spacing
        return gradient

    def process(self, cube: Cube) -> Tuple[Cube, Cube]:
        """
        Calculate the gradient along the x and y axes and return
        the result in separate cubes. The difference along each axis is
        calculated using numpy.diff.

        Args:
            cube:
                Cube from which the differences will be calculated.

        Returns:
            - Cube after the gradients have been calculated along the
              x axis.
            - Cube after the gradients have been calculated along the
              y axis.
        """
        gradients = []
        diffs = DifferenceBetweenAdjacentGridSquares()(cube)
        for axis, diff in zip(["x", "y"], diffs):
            gradient = self._gradient_from_diff(diff, axis)
            grad_cube = self._create_output_cube(gradient, diff, cube, axis)
            if self.regrid:
                grad_cube = grad_cube.regrid(cube, iris.analysis.Linear())
            gradients.append(grad_cube)

        return tuple(gradients)


class OccurrenceWithinVicinity(PostProcessingPlugin):

    """Calculate whether a phenomenon occurs within the specified radii about
    a point. These radii can be given in metres, or as numbers of grid points.
    Each radius provided will result in a distinct output, with these demarked
    using a `radius_of_vicinity` coordinate on the resulting cube. If a single
    radius is provided, this will be a scalar coordinate.

    Radii in metres may be used with data on a equal areas projection only.
    Grid_point_radii will work with any projection, with caveats.

    .. Further information is available in:
    .. include:: extended_documentation/utilities/spatial/
       occurrence_within_vicinity.rst
    """

    def __init__(
        self,
        radii: Optional[List[Union[float, int]]] = None,
        grid_point_radii: Optional[List[Union[float, int]]] = None,
        land_mask_cube: Cube = None,
    ) -> None:
        """
        Args:
            radii:
                A list of radii in metres used to define the vicinities within
                which to search for occurrences.
            grid_point_radii:
                Alternatively, a list of numbers of grid points that define the
                vicinity radii over which to search for occurrences. Only one of
                radii or grid_point_radii should be set.
            land_mask_cube:
                Binary land-sea mask data. True for land-points, False for sea.
                Restricts in-vicinity processing to only include points of a
                like mask value.

        Raises:
            ValueError: If both radii and grid point radii are set.
            ValueError: If neither radii or grid point radii are set.
            ValueError: If a provided vicinity radius is negative.
            ValueError: Land mask not named land_binary_mask.
        """
        if radii and grid_point_radii:
            raise ValueError(
                "Vicinity processing requires that only one of radii or "
                "grid_point_radii should be set"
            )
        if not radii and not grid_point_radii:
            raise ValueError(
                "Vicinity processing requires that one of radii or "
                "grid_point_radii should be set to a non-zero value"
            )
        if (radii and any(np.array(radii) < 0)) or (
            grid_point_radii and any(np.array(grid_point_radii) < 0)
        ):
            raise ValueError(
                "Vicinity processing requires only positive vicinity radii"
            )

        self.radii = radii if radii else grid_point_radii
        self.native_grid_point_radius = False if radii else True

        if land_mask_cube:
            if land_mask_cube.name() != "land_binary_mask":
                raise ValueError(
                    f"Expected land_mask_cube to be called land_binary_mask, "
                    f"not {land_mask_cube.name()}"
                )
            self.land_mask = np.where(land_mask_cube.data >= 0.5, True, False)
        else:
            self.land_mask = None
        self.land_mask_cube = land_mask_cube

    def maximum_within_vicinity(self, cube: Cube, grid_point_radius: int) -> Cube:
        """
        Find grid points where a phenomenon occurs within a defined radius.
        The occurrences within this vicinity are maximised, such that all
        grid points within the vicinity are recorded as having an occurrence.
        For non-binary fields, if the vicinity of two occurrences overlap,
        the maximum value within the vicinity is chosen.
        If a land-mask cube has been supplied, process land and sea points
        separately.

        Args:
            cube:
                Thresholded cube.
            grid_point_radius:
                The radius in grid points about each point within which to
                determine the maximum value.

        Returns:
            Cube where the occurrences have been spatially spread, so that
            they're equally likely to have occurred anywhere within the
            vicinity defined using the specified radius.
        """
        # Convert the grid_point_radius into a number of points along an edge
        # length, including the central point, e.g. grid_point_radius = 1,
        # points along the edge = 3
        grid_points = (2 * grid_point_radius) + 1

        cube_dtype = cube.data.dtype
        cube_fill_value = netCDF4.default_fillvals.get(cube_dtype.str[1:], np.inf)

        max_cube = cube.copy()
        unmasked_cube_data = cube.data.copy()
        if np.ma.is_masked(cube.data):
            unmasked_cube_data = cube.data.data.copy()
            unmasked_cube_data[cube.data.mask] = -cube_fill_value
        if self.land_mask_cube:
            max_data = np.empty_like(cube.data)
            for match in (True, False):
                matched_data = unmasked_cube_data.copy()
                matched_data[self.land_mask != match] = -cube_fill_value
                matched_max_data = maximum_filter(matched_data, size=grid_points)
                max_data = np.where(self.land_mask == match, matched_max_data, max_data)
        else:
            # The following command finds the maximum value for each grid point
            # from within a square of length "size"
            max_data = maximum_filter(unmasked_cube_data, size=grid_points)
        if np.ma.is_masked(cube.data):
            # Update only the unmasked values
            max_cube.data.data[~cube.data.mask] = max_data[~cube.data.mask]
        else:
            max_cube.data = max_data
        return max_cube

    def process(self, cube: Cube) -> Cube:
        """
        Produces the vicinity processed data. The input data is sliced to
        yield y-x slices to which the maximum_within_vicinity method is applied.
        The different vicinity radii (if multiple) are looped over and a
        coordinate recording the radius used is added to each resulting cube.
        A single cube is returned with the leading coordinates of the input cube
        preserved. If a single vicinity radius is provided, a new scalar
        radius_of_vicinity coordinate will be found on the returned cube. If
        multiple radii are provided, this coordinate will be a dimension
        coordinate following any probabilistic / realization coordinates.

        Args:
            cube:
                Thresholded cube.

        Returns:
            Cube containing the occurrences within a vicinity for each radius,
            calculated for each yx slice, which have been merged to yield a
            single cube.

        Raises:
            ValueError: Cube and land mask have differing spatial coordinates.
        """
        if self.land_mask_cube and not spatial_coords_match(
            [cube, self.land_mask_cube]
        ):
            raise ValueError(
                "Supplied cube do not have the same spatial coordinates and land mask"
            )

        if not self.native_grid_point_radius:
            grid_point_radii = [
                distance_to_number_of_grid_cells(cube, radius) for radius in self.radii
            ]
        else:
            grid_point_radii = self.radii

        radii_cubes = CubeList()

        # List of non-spatial dimensions to restore as leading on the output.
        leading_dimensions = [
            crd.name() for crd in cube.coords(dim_coords=True) if not crd.coord_system
        ]

        for radius, grid_point_radius in zip(self.radii, grid_point_radii):
            max_cubes = CubeList([])
            for cube_slice in cube.slices([cube.coord(axis="y"), cube.coord(axis="x")]):
                max_cubes.append(
                    self.maximum_within_vicinity(cube_slice, grid_point_radius)
                )
            result_cube = max_cubes.merge_cube()

            # Put dimensions back if they were there before.
            result_cube = check_cube_coordinates(cube, result_cube)

            # Add a coordinate recording the vicinity radius applied to the data.
            add_vicinity_coordinate(result_cube, radius, self.native_grid_point_radius)

            radii_cubes.append(result_cube)

        # Merge cubes produced for each vicinity radius.
        result_cube = radii_cubes.merge_cube()

        # Enforce order of leading dimensions on the output to match the input.
        enforce_coordinate_ordering(result_cube, leading_dimensions)

        if is_probability(result_cube):
            result_cube.rename(in_vicinity_name_format(result_cube.name()))
        else:
            result_cube.rename(f"{result_cube.name()}_in_vicinity")

        return result_cube


def lat_lon_determine(cube: Cube) -> Optional[CRS]:
    """
    Test whether a diagnostic cube is on a latitude/longitude grid or uses an
    alternative projection.

    Args:
        cube:
            A diagnostic cube to examine for coordinate system.

    Returns:
        Coordinate system of the diagnostic cube in a cartopy format unless
        it is already a latitude/longitude grid, in which case None is
        returned.
    """
    trg_crs = None
    if (
        not cube.coord(axis="x").name() == "longitude"
        or not cube.coord(axis="y").name() == "latitude"
    ):
        trg_crs = cube.coord_system().as_cartopy_crs()
    return trg_crs


def get_grid_y_x_values(cube: Cube) -> Tuple[ndarray, ndarray]:
    """Extract the y and x coordinate values of each points in the cube.

    The result is defined over the spatial grid, of shape (ny, nx) where
    ny is the length of the y-axis coordinate and nx the length of the
    x-axis coordinate.

    Args:
        cube:
            Cube with points to extract

    Returns:
        - Array of shape (ny, nx) containing y coordinate values
        - Array of shape (ny, nx) containing x coordinate values
    """
    x_points = cube.coord(axis="x").points
    y_points = cube.coord(axis="y").points

    nx = len(x_points)
    ny = len(y_points)

    x_zeros = np.zeros_like(x_points)
    y_zeros = np.zeros_like(y_points)

    # Broadcast x points and y points onto grid
    all_x_points = y_zeros.reshape(ny, 1) + x_points.reshape(1, nx)
    all_y_points = y_points.reshape(ny, 1) + x_zeros.reshape(1, nx)

    return all_y_points, all_x_points


def transform_grid_to_lat_lon(cube: Cube) -> Tuple[ndarray, ndarray]:
    """Calculate the latitudes and longitudes of each points in the cube.

    The result is defined over the spatial grid, of shape (ny, nx) where
    ny is the length of the y-axis coordinate and nx the length of the
    x-axis coordinate.

    Args:
        cube:
            Cube with points to transform

    Returns
        - Array of shape (ny, nx) containing grid latitude values
        - Array of shape (ny, nx) containing grid longitude values
    """
    trg_latlon = ccrs.PlateCarree()
    trg_crs = cube.coord_system().as_cartopy_crs()
    cube = cube.copy()
    # TODO use the proj units that are accesible with later versions of proj
    # to determine the default units to convert to for a given projection.

    # Assuming proj units of metre for all projections not in degrees.
    for axis in ["x", "y"]:
        try:
            cube.coord(axis=axis).convert_units("m")
        except ValueError as err:
            msg = (
                "Cube passed to transform_grid_to_lat_lon does not have an "
                f"{axis} coordinate with units that can be converted to metres. "
            )
            raise ValueError(msg + str(err))

    all_y_points, all_x_points = get_grid_y_x_values(cube)

    # Transform points
    points = trg_latlon.transform_points(trg_crs, all_x_points, all_y_points)
    lons = points[..., 0]
    lats = points[..., 1]

    return lats, lons


def update_name_and_vicinity_coord(cube: Cube, new_name: str, vicinity_radius: float):
    """
    Updates a cube with a new probabilistic-style name and replaces or adds a radius_of_vicinity
    coord with the specified radius.

    Args:
        cube: Cube to be updated in-place
        new_name: The new name to be applied to the Cube, the threshold coord and any related
            cell methods.
        vicinity_radius: The point value for the radius_of_vicinity coord. The units are assumed
            to be the same as the x and y spatial coords of the Cube

    """
    if new_name:
        update_diagnostic_name(cube, new_name, cube)
    if vicinity_radius:
        # The cube blending will drop the radius_of_vicinity coord if the source cubes have
        # differing points. We can use this to determine whether the vicinities matched:
        vicinities_matched = "radius_of_vicinity" in [
            coord.name() for coord in cube.coords()
        ]
        if vicinities_matched:
            cube.remove_coord("radius_of_vicinity")
        add_vicinity_coordinate(
            cube, vicinity_radius, radius_is_max=not vicinities_matched
        )


def add_vicinity_coordinate(
    cube: Cube,
    radius: Union[float, int],
    native_grid_point_radius: bool = False,
    radius_is_max: bool = False,
) -> None:
    """
    Add a coordinate to the cube that records the vicinity radius that
    has been applied to the data.

    Args:
        cube:
            Vicinity processed cube.
        radius:
            The radius as a physical distance (m) or number of grid points, the
            value of which is recorded in the coordinate.
        native_grid_point_radius:
            True if radius is "number of grid points", else False
        radius_is_max:
            True if the specified radius represents a maximum value from the source data. A
            comment is associated with the coord in this case.
    """
    attributes = {}
    if radius_is_max:
        attributes["comment"] = "Maximum"
    if native_grid_point_radius:
        point = np.array(radius, dtype=np.float32)
        units = "1"
        comment = (
            "Units of 1 indicate radius of vicinity is defined "
            "in grid points rather than physical distance"
        )
        attributes["comment"] = "; ".join(
            [n for n in [attributes.get("comment", None), comment] if n]
        )
    else:
        point = np.array(radius, dtype=np.float32)
        units = "m"

    coord = AuxCoord(
        point, units=units, long_name="radius_of_vicinity", attributes=attributes
    )
    cube.add_aux_coord(coord)
