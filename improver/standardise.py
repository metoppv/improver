#!/usr/bin/env python
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
"""Plugin to regrid cube data and standardise metadata"""

import warnings

import iris
import math
import numpy as np
from iris.analysis import Linear, Nearest
from iris.cube import Cube
from iris.exceptions import CoordinateNotFoundError
from scipy.interpolate import griddata
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial.ckdtree import cKDTree as KDTree

from improver import BasePlugin
from improver.metadata.amend import amend_attributes
from improver.metadata.check_datatypes import (
    check_units,
    get_required_dtype,
    get_required_units,
)
from improver.metadata.constants.attributes import MANDATORY_ATTRIBUTE_DEFAULTS
from improver.metadata.constants.mo_attributes import MOSG_GRID_ATTRIBUTES
from improver.metadata.constants.time_types import TIME_COORDS
from improver.threshold import BasicThreshold
from improver.utilities.cube_checker import spatial_coords_match
from improver.utilities.round import round_close
from improver.utilities.spatial import OccurrenceWithinVicinity


def grid_contains_cutout(grid, cutout):
    """
    Check that a spatial cutout is contained within a given grid

    Args:
        grid (iris.cube.Cube):
            A cube defining a data grid
        cutout (iris.cube.Cube):
            The cutout to search for within the grid

    Returns:
        bool:
            True if cutout is contained within grid, False otherwise
    """

    if spatial_coords_match(grid, cutout):
        return True

    # check whether "cutout" coordinate points match a subset of "grid"
    # points on both axes
    for axis in ["x", "y"]:
        grid_coord = grid.coord(axis=axis)
        cutout_coord = cutout.coord(axis=axis)
        # check coordinate metadata
        if (
            cutout_coord.name() != grid_coord.name()
            or cutout_coord.units != grid_coord.units
            or cutout_coord.coord_system != grid_coord.coord_system
        ):
            return False

        # search for cutout coordinate points in larger grid
        cutout_start = cutout_coord.points[0]
        find_start = [
            np.isclose(cutout_start, grid_point) for grid_point in grid_coord.points
        ]
        if not np.any(find_start):
            return False

        start = find_start.index(True)
        end = start + len(cutout_coord.points)
        try:
            if not np.allclose(cutout_coord.points, grid_coord.points[start:end]):
                return False
        except ValueError:
            # raised by np.allclose if "end" index overshoots edge of grid
            # domain - slicing does not raise IndexError
            return False

    return True


class StandardiseMetadata(BasePlugin):
    """Plugin to standardise cube metadata"""

    @staticmethod
    def _collapse_scalar_dimensions(cube):
        """
        Demote any scalar dimensions (excluding "realization") on the input
        cube to auxiliary coordinates.

        Returns:
            iris.cube.Cube
        """
        coords_to_collapse = []
        for coord in cube.coords(dim_coords=True):
            if len(coord.points) == 1 and "realization" not in coord.name():
                coords_to_collapse.append(coord)
        for coord in coords_to_collapse:
            cube = next(cube.slices_over(coord))
        return cube

    @staticmethod
    def _remove_scalar_coords(cube, coords_to_remove):
        """Removes named coordinates from the input cube."""
        for coord in coords_to_remove:
            try:
                cube.remove_coord(coord)
            except CoordinateNotFoundError:
                continue

    @staticmethod
    def _standardise_dtypes_and_units(cube):
        """
        Modify input cube in place to conform to mandatory dtype and unit
        standards.

        Args:
            cube (iris.cube.Cube:
                Cube to be updated in place

        """

        def as_correct_dtype(obj, required_dtype):
            """
            Returns an object updated if necessary to the required dtype

            Args:
                obj (np.ndarray):
                    The object to be updated
                required_dtype (np.dtype):
                    The dtype required

            Returns:
                np.ndarray
            """
            if obj.dtype != required_dtype:
                return obj.astype(required_dtype)
            return obj

        cube.data = as_correct_dtype(cube.data, get_required_dtype(cube))
        for coord in cube.coords():
            if coord.name() in TIME_COORDS and not check_units(coord):
                coord.convert_units(get_required_units(coord))
            req_dtype = get_required_dtype(coord)
            # ensure points and bounds have the same dtype
            if np.issubdtype(req_dtype, np.integer):
                coord.points = round_close(coord.points)
            coord.points = as_correct_dtype(coord.points, req_dtype)
            if coord.has_bounds():
                if np.issubdtype(req_dtype, np.integer):
                    coord.bounds = round_close(coord.bounds)
                coord.bounds = as_correct_dtype(coord.bounds, req_dtype)

    def process(
        self,
        cube,
        new_name=None,
        new_units=None,
        coords_to_remove=None,
        attributes_dict=None,
    ):
        """
        Perform compulsory and user-configurable metadata adjustments.  The
        compulsory adjustments are to collapse any scalar dimensions apart from
        realization (which is expected always to be a dimension); to cast the cube
        data and coordinates into suitable datatypes; and to convert time-related
        metadata into the required units.

        Args:
            cube (iris.cube.Cube):
                Input cube to be standardised
            new_name (str or None):
                Optional rename for output cube
            new_units (str or None):
                Optional unit conversion for output cube
            coords_to_remove (list of str or None):
                Optional list of scalar coordinates to remove from output cube
            attributes_dict (dict or None):
                Optional dictionary of required attribute updates. Keys are
                attribute names, and values are the required value or "remove".

        Returns:
            iris.cube.Cube
        """
        cube = self._collapse_scalar_dimensions(cube)

        if new_name:
            cube.rename(new_name)
        if new_units:
            cube.convert_units(new_units)
        if coords_to_remove:
            self._remove_scalar_coords(cube, coords_to_remove)
        if attributes_dict:
            amend_attributes(cube, attributes_dict)

        # this must be done after unit conversion as if the input is an integer
        # field, unit conversion outputs the new data as float64
        self._standardise_dtypes_and_units(cube)

        return cube


class RegridLandSea(BasePlugin):
    """Regrid a field with the option to adjust the output so that regridded land
    points always take values from a land point on the source grid, and vice versa
    for sea points"""

    REGRID_REQUIRES_LANDMASK = {
        "bilinear": False,
        "nearest": False,
        "nearest-with-mask": True,
        "nearest-2": False,
        "bilinear-2": False,
        "nearest-with-mask-2": True,
        "bilinear-with-mask-2": True,
    }

    def __init__(
        self,
        regrid_mode="bilinear",
        extrapolation_mode="nanmask",
        landmask=None,
        landmask_vicinity=25000,
    ):
        """
        Initialise regridding parameters

        Args:
            regrid_mode (str):
                Mode of interpolation in regridding.  Valid options are "bilinear",
                "nearest" or "nearest-with-mask".  The "nearest-with-mask" option
                triggers adjustment of regridded points to match source points in
                terms of land / sea type.
            extrapolation_mode (str):
                Mode to fill regions outside the domain in regridding.
            landmask (iris.cube.Cube or None):
                Land-sea mask ("land_binary_mask") on the input cube grid, with
                land points set to one and sea points set to zero.  Required for
                "nearest-with-mask" regridding option.
            landmask_vicinity (float):
                Radius of vicinity to search for a coastline, in metres
        """
        if regrid_mode not in self.REGRID_REQUIRES_LANDMASK:
            msg = "Unrecognised regrid mode {}"
            raise ValueError(msg.format(regrid_mode))
        if landmask is None and self.REGRID_REQUIRES_LANDMASK[regrid_mode]:
            msg = "Regrid mode {} requires an input landmask cube"
            raise ValueError(msg.format(regrid_mode))
        self.regrid_mode = regrid_mode
        self.extrapolation_mode = extrapolation_mode
        self.landmask_source_grid = landmask
        self.landmask_vicinity = None if landmask is None else landmask_vicinity
        self.landmask_name = "land_binary_mask"

    def _adjust_landsea(self, cube, target_grid):
        """
        Adjust regridded data using differences between the target landmask
        and that obtained by regridding the source grid landmask, to ensure
        that the "land" or "sea" nature of the points in the regridded cube
        matches that of the target grid.

        Args:
            cube (iris.cube.Cube):
                Cube after initial regridding
            target_grid (iris.cube.Cube):
                Cube containing landmask data on the target grid

        Returns:
            iris.cube.Cube: Adjusted cube
        """
        if self.landmask_name not in self.landmask_source_grid.name():
            msg = "Expected {} in input_landmask cube but found {}".format(
                self.landmask_name, repr(self.landmask_source_grid)
            )
            warnings.warn(msg)

        if self.landmask_name not in target_grid.name():
            msg = "Expected {} in target_grid cube but found {}".format(
                self.landmask_name, repr(target_grid)
            )
            warnings.warn(msg)

        return AdjustLandSeaPoints(vicinity_radius=self.landmask_vicinity)(
            cube, self.landmask_source_grid, target_grid
        )

    def _regrid_landsea_2(self, cube, target_grid, regrid_mode):
        """
        Adjust regridded data using differences between the target landmask
        and that obtained by regridding the source grid landmask, to ensure
        that the "land" or "sea" nature of the points in the regridded cube
        matches that of the target grid.

        Args:
            cube (iris.cube.Cube):
                Cube to be regridded
            target_grid (iris.cube.Cube):
                Cube containing landmask data on the target grid

        Returns:
            iris.cube.Cube: Adjusted cube
        """
        if regrid_mode in ("nearest-with-mask-2", "bilinear-with-mask-2"):
            if self.landmask_name not in self.landmask_source_grid.name():
                msg = "Expected {} in input_landmask cube but found {}".format(
                    self.landmask_name, repr(self.landmask_source_grid)
                )
                warnings.warn(msg)

            # print(self.landmask_source_grid, self.landmask_name)
            if self.landmask_name not in target_grid.name():
                msg = "Expected {} in target_grid cube but found {}".format(
                    self.landmask_name, repr(target_grid)
                )
                warnings.warn(msg)

        return RegridWithLandSeaMask(
            regrid_mode=regrid_mode, vicinity_radius=self.landmask_vicinity
        )(cube, self.landmask_source_grid, target_grid)

    def _regrid_to_target(self, cube, target_grid, regridded_title, regrid_mode):
        """
        Regrid cube to target_grid, inherit grid attributes and update title

        Args:
            cube (iris.cube.Cube):
                Cube to be regridded 
            target_grid (iris.cube.Cube):
                Data on the target grid. If regridding with mask, this cube
                should contain land-sea mask data to be used in adjusting land
                and sea points after regridding.
            regridded_title (str or None):
                New value for the "title" attribute to be used after
                regridding. If not set, a default value is used.
            regrid_mode (str):
                "bilinear","nearest","nearest-with-mask",
                "nearest-2","nearest-with-mask-2","bilinear-2","bilinear-with-mask-2"

        Returns:
            iris.cube.Cube: Regridded cube with updated attributes
        """
        # basic categories (1) Iris-based (2) new nearest based  (3) new bilinear-based
        if regrid_mode in ("bilinear", "nearest", "nearest-with-mask"):
            if "nearest" in regrid_mode:
                regridder = Nearest(extrapolation_mode=self.extrapolation_mode)
            else:
                regridder = Linear(extrapolation_mode=self.extrapolation_mode)
            cube = cube.regrid(target_grid, regridder)

            # Iris regridding is used, and then adjust if land_sea mask is considered
            if self.REGRID_REQUIRES_LANDMASK[regrid_mode]:
                cube = self._adjust_landsea(cube, target_grid)

            # identify grid-describing attributes on source cube that need updating
            # MOSG_GRID_ATTRIBUTES={"mosg__grid_type","mosg__grid_version", "mosg__grid_domain"}
            required_grid_attributes = [
                attr for attr in cube.attributes if attr in MOSG_GRID_ATTRIBUTES
            ]

            # update attributes if available on target grid, otherwise remove
            for key in required_grid_attributes:
                if key in target_grid.attributes:
                    cube.attributes[key] = target_grid.attributes[key]
                else:
                    cube.attributes.pop(key)

        # new version of nearest/bilinear option with/without land-sea mask
        elif regrid_mode in (
            "nearest-2",
            "nearest-with-mask-2",
            "bilinear-2",
            "bilinear-with-mask-2",
        ):
            cube = self._regrid_landsea_2(cube, target_grid, regrid_mode)

        cube.attributes["title"] = (
            MANDATORY_ATTRIBUTE_DEFAULTS["title"]
            if regridded_title is None
            else regridded_title
        )

        return cube

    def process(self, cube, target_grid, regridded_title=None):
        """
        Regrids cube onto spatial grid provided by target_grid

        Args:
            cube (iris.cube.Cube):
                Cube to be regridded
            target_grid (iris.cube.Cube):
                Data on the target grid. If regridding with mask, this cube
                should contain land-sea mask data to be used in adjusting land
                and sea points after regridding.
            regridded_title (str or None):
                New value for the "title" attribute to be used after
                regridding. If not set, a default value is used.

        Returns:
            iris.cube.Cube: Regridded cube with updated attributes
        """
        # if regridding using a land-sea mask, check this covers the source
        # grid in the required coordinates
        if self.REGRID_REQUIRES_LANDMASK[self.regrid_mode]:
            if not grid_contains_cutout(self.landmask_source_grid, cube):
                raise ValueError("Source landmask does not match input grid")
        return self._regrid_to_target(
            cube, target_grid, regridded_title, self.regrid_mode
        )


class AdjustLandSeaPoints(BasePlugin):
    """
    Replace data values at points where the nearest-regridding technique
    selects a source grid-point with an opposite land-sea-mask value to the
    target grid-point.
    The replacement data values are selected from a vicinity of points on the
    source-grid and the closest point of the correct mask is used.
    Where no match is found within the vicinity, the data value is not changed.
    """

    def __init__(self, extrapolation_mode="nanmask", vicinity_radius=25000.0):
        """
        Initialise class

        Args:
            extrapolation_mode (str):
                Mode to use for extrapolating data into regions
                beyond the limits of the source_data domain.
                Available modes are documented in
                `iris.analysis <https://scitools.org.uk/iris/docs/latest/iris/
                iris/analysis.html#iris.analysis.Nearest>`_
                Defaults to "nanmask".
            vicinity_radius (float):
                Distance in metres to search for a sea or land point.
        """
        self.input_land = None
        self.nearest_cube = None
        self.output_land = None
        self.output_cube = None
        self.regridder = Nearest(extrapolation_mode=extrapolation_mode)
        self.vicinity = OccurrenceWithinVicinity(vicinity_radius)

    def __repr__(self):
        """
        Print a human-readable representation of the instantiated object.
        """
        return "<AdjustLandSeaPoints: regridder: {}; vicinity: {}>".format(
            self.regridder, self.vicinity
        )

    def correct_where_input_true(self, selector_val):
        """
        Replace points in the output_cube where output_land matches the
        selector_val and the input_land does not match, but has matching
        points in the vicinity, with the nearest matching point in the
        vicinity in the original nearest_cube.

        Updates self.output_cube.data

        Args:
            selector_val (int):
                Value of mask to replace if needed.
                Intended to be 1 for filling land points near the coast
                and 0 for filling sea points near the coast.
        """
        # Find all points on output grid matching selector_val
        use_points = np.where(self.input_land.data == selector_val)

        # If there are no matching points on the input grid, no alteration can
        # be made. This tests the size of the y-coordinate of use_points.
        if use_points[0].size is 0:
            return

        # Get shape of output grid
        ynum, xnum = self.output_land.shape

        # Using only these points, extrapolate to fill domain using nearest
        # neighbour. This will generate a grid where the non-selector_val
        # points are filled with the nearest value in the same mask
        # classification.
        (y_points, x_points) = np.mgrid[0:ynum, 0:xnum]
        selector_data = griddata(
            use_points,
            self.nearest_cube.data[use_points],
            (y_points, x_points),
            method="nearest",
        )

        # Identify nearby points on regridded input_land that match the
        # selector_value
        if selector_val > 0.5:
            thresholder = BasicThreshold(0.5)
        else:
            thresholder = BasicThreshold(0.5, comparison_operator="<=")
        in_vicinity = self.vicinity(thresholder(self.input_land))

        # Identify those points sourced from the opposite mask that are
        # close to a source point of the correct mask
        (mismatch_points,) = np.logical_and(
            np.logical_and(
                self.output_land.data == selector_val,
                self.input_land.data != selector_val,
            ),
            in_vicinity.data > 0.5,
        )

        # Replace these points with the filled-domain data
        self.output_cube.data[mismatch_points] = selector_data[mismatch_points]

    def process(self, cube, input_land, output_land):
        """
        Update cube.data so that output_land and sea points match an input_land
        or sea point respectively so long as one is present within the
        specified vicinity radius. Note that before calling this plugin the
        input land mask MUST be checked against the source grid, to ensure
        the grids match.

        Args:
            cube (iris.cube.Cube):
                Cube of data to be updated (on same grid as output_land).
            input_land (iris.cube.Cube):
                Cube of land_binary_mask data on the grid from which "cube" has
                been reprojected (it is expected that the iris.analysis.Nearest
                method would have been used). Land points should be set to one
                and sea points set to zero.
                This is used to determine where the input model data is
                representing land and sea points.
            output_land (iris.cube.Cube):
                Cube of land_binary_mask data on target grid.
        """

        # Check cube and output_land are on the same grid:
        if not spatial_coords_match(cube, output_land):
            raise ValueError(
                "X and Y coordinates do not match for cubes {}"
                "and {}".format(repr(cube), repr(output_land))
            )
        self.output_land = output_land

        # Regrid input_land to output_land grid.
        self.input_land = input_land.regrid(self.output_land, self.regridder)

        # Slice over x-y grids for multi-realization data.
        result = iris.cube.CubeList()

        for xyslice in cube.slices([cube.coord(axis="y"), cube.coord(axis="x")]):

            # Store and copy cube ready for the output data
            self.nearest_cube = xyslice
            self.output_cube = self.nearest_cube.copy()

            # Update sea points that were incorrectly sourced from land points
            self.correct_where_input_true(0)

            # Update land points that were incorrectly sourced from sea points
            self.correct_where_input_true(1)

            result.append(self.output_cube)

        result = result.merge_cube()
        return result


class RegridWithLandSeaMask(BasePlugin):
    """
    Replace data values at points where the nearest-regridding technique
    selects a source grid-point with an opposite land-sea-mask value to the
    target grid-point.
    The replacement data values are selected from a vicinity of points on the
    source-grid and the closest point of the correct mask is used.
    Where no match is found within the vicinity, the data value is not changed.
    """

    def __init__(self, regrid_mode="bilinear-2", vicinity_radius=25000.0):
        """
        Initialise class
        """
        self.regrid_mode = regrid_mode
        self.vicinity = vicinity_radius

    def __repr__(self):
        """
        Print a human-readable representation of the instantiated object.
        """
        return "<Regridder: {}; vicinity: {}>".format(self.regrid_mode, self.vicinity)

    def _get_cube_coord_names(self, cube):
        """
        get all coord names for a cube
        Args:
             cube (iris.cube.Cube):
                input cube 
        Return:
            cube_coord_names (list):
                return a list of coord names     
        """
        cube_coord_names = []
        for coord in cube.dim_coords:
            cube_coord_names.append(coord.standard_name)
        return cube_coord_names

    def _variable_name(self, cube, names):
        """
        Identify the name of a variable from a list of possible candidates.
        Args:
             cube (iris.cube.Cube):
                input cube 
             names(list): 
                possible name list
        Return:
             matching name of the variable (str) 
        """
        coord_names = self._get_cube_coord_names(cube)
        found_name = None
        for name in names:
            if name in coord_names:
                found_name = name
                break
        if found_name is None:
            raise ValueError("Unable to find a variable matching {}", str(names))
        return found_name

    def _latlon_names(self, cube):
        """
        Identify the names of the latitude and longitude dimensions of cube
        Args:
            cube (iris.cube.Cube):
                input cube 
        Return:
             names of latitude and laongitude (str) 
        """
        COMMON_LAT_NAMES = ["latitude", "lat", "lats"]
        COMMON_LON_NAMES = ["longitude", "lon", "lons"]
        lats_name = self._variable_name(cube, COMMON_LAT_NAMES)
        lons_name = self._variable_name(cube, COMMON_LON_NAMES)
        return lats_name, lons_name

    def _latlon_from_cube(self, cube):
        """
        Produce an array of latitude-longitude coordinates used by an Iris cube
        Args:
           cube(iris.cube.Cube):
               cube information 
        Return: 
           latlon(numpy ndarray):
               latitude-longitude pairs (N x 2)
        """
        lats_name, lons_name = self._latlon_names(cube)
        lats_data = cube.coord(lats_name).points
        lons_data = cube.coord(lons_name).points
        lats_mesh, lons_mesh = np.meshgrid(lats_data, lons_data, indexing="ij")
        lats = lats_mesh.flatten()
        lons = lons_mesh.flatten()
        latlon = np.dstack((lats, lons)).squeeze()
        return latlon

    def _convert_from_projection_to_latlons(self, cube_out, cube_in):
        """
        convert cube_out's LambertAzimuthalEqualArea's coord to GeogCS's lats/lons
        output grid (cube_out) could be in LambertAzimuthalEqualArea system
        cube_in is in GeogCS's lats/lons system. 
        Args:
            cube_out (iris.cube.Cube):
                target cube with LambertAzimuthalEqualArea's coord system
            cube_in (iris.cube.Cube):
                source cube with GeorCS (as reference coord system for conversion)

        Returns:
            out_latlons(numpy ndarray):
                latitude-longitude pairs for target grid points
        """

        if (
            cube_out.coord(axis="x").standard_name != "projection_x_coordinate"
            or cube_out.coord(axis="y").standard_name != "projection_y_coordinate"
        ):
            return

        # get coordinate points in native projection & transfer into xx,yy(1D)
        proj_x = cube_out.coord("projection_x_coordinate").points
        proj_y = cube_out.coord("projection_y_coordinate").points
        yy, xx = np.meshgrid(proj_y, proj_x, indexing="ij")
        yy = yy.flatten()
        xx = xx.flatten()

        # extract the native projection and convert it to a cartopy projection:
        cs_nat = cube_out.coord_system()
        cs_nat_cart = cs_nat.as_cartopy_projection()

        # find target projection,convert it to a cartopy projection
        cs_tgt = cube_in.coord("latitude").coord_system
        cs_tgt_cart = cs_tgt.as_cartopy_projection()

        # use cartopy's transform to convert coord.in native proj to coord in target proj
        lons, lats, _ = cs_tgt_cart.transform_points(cs_nat_cart, xx, yy).T

        out_latlons = np.dstack((lats, lons)).squeeze()

        return out_latlons

    def _ecef_coords(self, lats, lons, alts=np.array(0.0)):
        """
        Transforms the coordinates to Earth Centred Earth Fixed coordinates
        with WGS84 parameters. used in function _nearest_input_pts
        Args:
            lats(numpy.ndarray):
                latitude coordinates
            lons(numpy.ndarray):
                longitude coordinates
            alts(numpy.ndarray):
                altitudes coordinates
        Return:
            tuple (x, y, z) transformed coordinates
        """
        WGS84_A = 6378137.0
        WGS84_IF = 298.257223563
        WGS84_F = 1.0 / WGS84_IF
        WGS84_B = WGS84_A * (1.0 - WGS84_F)
        # eccentricity
        WGS84_E = math.sqrt((2.0 * WGS84_F) - (WGS84_F * WGS84_F))

        rlats = np.deg2rad(lats)
        rlons = np.deg2rad(lons)
        clats = np.cos(rlats)
        clons = np.cos(rlons)
        slats = np.sin(rlats)
        slons = np.sin(rlons)
        n = WGS84_A / np.sqrt(1.0 - (WGS84_E * WGS84_E * slats * slats))
        x = (n + alts) * clats * clons
        y = (n + alts) * clats * slons
        z = (n * (1.0 - (WGS84_E * WGS84_E)) + alts) * slats
        return x, y, z

    def _nearest_input_pts(self, in_latlons, out_latlons, k):
        """
        Find k nearest source (input) points to each target(output) point, using a KDtree
        Args:
            in_latlons(numpy ndarray):
                source grid points' latitude-longitudes (N x 2)
            out_latlons(numpy ndarray)
                target grid points' latitude-longitudes (M x 2)
            k: number of points surrounding each output point
        Return: 
            tuple of (distances, indexes) numpy ndarray of distances from target grid point 
            to source grid points and indexes of those points (M x K)           
        """
        in_x, in_y, in_z = self._ecef_coords(
            in_latlons[:, 0].flat, in_latlons[:, 1].flat
        )
        in_coords = np.c_[in_x, in_y, in_z]
        in_kdtree = KDTree(in_coords)

        out_x, out_y, out_z = self._ecef_coords(
            out_latlons[:, 0].flat, out_latlons[:, 1].flat
        )
        out_coords = np.c_[out_x, out_y, out_z]
        distances, indexes = in_kdtree.query(out_coords, k)

        if distances.ndim == 1:
            distances = np.expand_dims(distances, axis=1)
        if indexes.ndim == 1:
            indexes = np.expand_dims(indexes, axis=1)
        return distances, indexes

    def _land_classify_out(self, cube_out_mask):
        """
        Classify surface types of target grid points based on a binary True/False land mask
        Args:
            cube_out_mask (iris.cube.Cube):
                land_sea mask information cube for target grid(land=>1)
        Return:
            1D numpy.ndarray land-sea mask information for 1D-ordered target grid points
        """
        # cube y-axis => latitude or projection-y
        cube_out_dim0 = cube_out_mask.coord(axis="y").shape[0]
        cube_out_dim1 = cube_out_mask.coord(axis="x").shape[0]
        out_classified = cube_out_mask.data.reshape(cube_out_dim0 * cube_out_dim1)
        return out_classified

    def _land_classify_in(self, cube_in_mask, classify_latlons):
        """
        Classify surface types of source grid points based on a binary True/False land mask
        cube_in_mask's grid could be different from input source grid of NWP results  
        Args:
            cube_in_mask (iris.cube.Cube):
                land_sea mask information cube for input source grid(land=>1)
                should in GeogCS's lats/lons coordinate system
            classify_latlons(numpy.ndarray): 
                latitude and longitude source grid points to classify (N x 2)
        Return:
            numpy ndarray of classifications (N) for 1D-ordered source grid points
        """
        in_land_mask = cube_in_mask.data
        lats_name, lons_name = self._latlon_names(cube_in_mask)
        in_land_mask_lats = cube_in_mask.coord(lats_name).points
        in_land_mask_lons = cube_in_mask.coord(lons_name).points

        mask_rg_interp = RegularGridInterpolator(
            (in_land_mask_lats, in_land_mask_lons),
            in_land_mask,
            method="nearest",
            bounds_error=False,
            fill_value=0.0,
        )
        is_land = np.bool_(mask_rg_interp(classify_latlons))
        return is_land

    def _nearest_distance(self, distances, indexes, in_values):
        """
        main regridding function for the nearest distance option
        Args:
            distnaces(numpy.ndarray):
                distnace array from each target grid point to its source grid points        
            indexes(numpy.ndarray):
                array of source grid point number for each target grid points 
            in_values(numpy.ndarray):
                input values (multidimensional, reshaped in function _reshape_data_cube)
        Return:
            results(numpy.ndarray):
                output values (multidimensional)           
        """
        k = distances.shape[1]
        min_index = np.argmin(distances, axis=1)
        index0 = np.arange(min_index.shape[0])
        index_in = indexes[index0, min_index]
        results = in_values[index_in]
        return results

    def _nearest_distance_with_mask(
        self,
        distances,
        indexes,
        surface_type_mask,
        weights,
        in_latlons,
        out_latlons,
        in_classified,
        out_classified,
        in_values,
        cube_in_dim1,
    ):
        """               
        main regridding function for the nearest distance option
        some input just for handling island-like points
        Args:
            distnaces(numpy.ndarray):
                distnace array from each target grid point to its source grid points        
            indexes(numpy.ndarray):
                array of source grid point number for each target grid points 
            surface_type_mask(numpy.ndarray):
                numpy ndarray of bool, true if source points' surface type matches target point's 
            in_latlons(numpy.ndarray):
                tource points's latitude-longitudes 
            out_latlons(numpy.ndarray):
                target points's latitude-longitudes 
            weights(numpy.ndarray):
                array of source grid point weighting for all target grid points
            in_classified(numpy.ndarray):
                land_sea type for source grid points (land =>True)
            out_classified(numpy.ndarray):
                land_sea type for terget grid points (land =>True)               
            in_values(numpy.ndarray):
                input values (maybe multidimensional, reshaped in function _reshape_data_cube)
        Return:
            results(numpy.ndarray):
                output values (multidimensional)     
        """
        if distances.shape != surface_type_mask.shape:
            raise ValueError("Distance and mask arrays must be same shape")
        k = distances.shape[1]

        # check if there are false points, update nearest points using KD Tree
        p_in_1 = np.sum(surface_type_mask, axis=1)
        p_with_false = (np.where(p_in_1 < 4))[0]

        indexes, distances, surface_type_mask = self._update_nearest_points_info(
            p_with_false,
            in_latlons,
            out_latlons,
            indexes,
            distances,
            surface_type_mask,
            in_classified,
            out_classified,
        )

        # check if there are island-like points, if so special treatment
        p_in_2 = np.sum(surface_type_mask, axis=1)
        p_with_4false = (np.where(p_in_2 == 0))[0]

        if p_with_4false.shape[0] > 0:
            weights, indexes, surface_type_mask = self._handle_4false_source_p(
                p_with_4false,
                weights,
                indexes,
                surface_type_mask,
                in_latlons,
                out_latlons,
                in_classified,
                out_classified,
                cube_in_dim1,
            )

        # convert mask to be true where input points should not be considered
        not_mask = np.logical_not(surface_type_mask)

        # replace distances with infinity where they should not be used
        masked_distances = np.where(not_mask, np.float64("inf"), distances)
        results = self._nearest_distance(masked_distances, indexes, in_values)

        return results

    def _update_nearest_points_info(
        self,
        p_with_false,
        in_latlons,
        out_latlons,
        indexes,
        distances,
        surface_type_mask,
        in_classified,
        out_classified,
    ):
        """
        updating nearest source points and distances/surface_type for selective target points
        Args:
            p_with_false(numpy.ndarray):
                selected target points which will use Inverse Distance Weighting(idw) approach
            in_latlons(numpy.ndarray):
                Source points's latitude-longitudes 
            out_latlons(numpy.ndarray):
                Target points's latitude-longitudes 
            indexes(numpy.ndarray):
                array of source grid point number for all target grid points 
            distnaces(numpy.ndarray):
                distnace array from each target grid point to its source grid points 
            surface_type_mask(numpy.ndarray)):
                numpy ndarray of bool, true if source point type matches target point type
            in_classified(numpy.ndarray):
                land_sea type for source grid points (land =>True)
            out_classified(numpy.ndarray):
                land_sea type for terget grid points (land =>True)
        Returns:
            indexes(numpy.ndarray):
                updated array of source grid point number for all target grid points
            weights(numpy.ndarray):     
                updated array of source grid point weighting for all target grid points
            surface_type_mask(numpy.ndarray)):
                updated ndarray, true if source point type matches target point type
        """

        out_latlons_1 = out_latlons[p_with_false]
        k_nearest = 4
        distances_1, indexes_1 = self._nearest_input_pts(
            in_latlons, out_latlons_1, k_nearest
        )

        out_classified_1 = out_classified[p_with_false]
        surface_type_mask_1 = self._similar_surface_classify(
            in_classified, out_classified_1, indexes_1
        )

        indexes[p_with_false] = indexes_1
        distances[p_with_false] = distances_1
        surface_type_mask[p_with_false] = surface_type_mask_1
        return indexes, distances, surface_type_mask

    def _similar_surface_classify(self, in_is_land, out_is_land, nearest_in_indexes):
        """
        Classify surface types as matched (True) or unmatched(False) between target points 
        and their source point 
        Args:
            in_is_land(numpy ndarray):
                source point classifications (N)
            out_is_land(numpy ndarray):
                target point classifications (M)
            nearest_in_indexes (numpy ndarray)
                indexes of input points nearby output points (M x K)
        Return:
            numpy ndarray of bool, True if input surface type matches output or no matches(M x K)
        """
        k = nearest_in_indexes.shape[1]
        out_is_land_bcast = np.broadcast_to(
            out_is_land, (k, out_is_land.shape[0])
        ).transpose()  # dimensions M x K

        # classify the input points surrounding each output point
        nearest_is_land = in_is_land[nearest_in_indexes]  # dimensions M x K

        # these input points surrounding output points have the same surface type
        nearest_same_type = np.logical_not(
            np.logical_xor(nearest_is_land, out_is_land_bcast)
        )  # dimensions M x K

        return nearest_same_type

    def _get_grid_size(self, cube):
        """
        get cube grid size (cube in even lats/lons system)
        
        Args:
            cube (iris.cube.Cube):
                input cube 
        Return:
            lat_d,lon_d (float):
                latitude/logitude grid size
        """
        lats_name, lons_name = self._latlon_names(cube)
        lat_d = cube.coord(lats_name).points[1] - cube.coord(lats_name).points[0]
        lon_d = cube.coord(lons_name).points[1] - cube.coord(lons_name).points[0]
        return lat_d, lon_d

    def _slice_mask_cube_by_domain(self, cube_in, cube_in_mask, output_domain):
        """
        extract cube domain to be consistent as cube_reference's domain
        
        Args:
            cube_in (iris.cube.Cube): 
                input data cube to be sliced
            cube_in_mask (iris.cube.Cube): 
                input maskcube to be sliced    
            output_domain(tuple):
                lat_max,lon_max,lat_min,lon_min
        Returns:
            cube_in (iris.cube.Cube):
                data cube after slicing
            cube_in_mask (iris.cube.Cube):
                mask cube after slicing
        """
        # extract output grid domain
        lat_max, lon_max, lat_min, lon_min = output_domain
        lat_d_1, lon_d_1 = self._get_grid_size(cube_in)
        lat_d_2, lon_d_2 = self._get_grid_size(cube_in_mask)
        lat_d = lat_d_1 if lat_d_1 > lat_d_2 else lat_d_2
        lon_d = lon_d_1 if lon_d_1 > lon_d_2 else lon_d_2

        # lats_name, lons_name = self._latlon_names(cube_in)  #how to use lats_name ??

        domain = iris.Constraint(
            latitude=lambda val: lat_min - 2.0 * lat_d < val < lat_max + 2.0 * lat_d
        ) & iris.Constraint(
            longitude=lambda val: lon_min - 2.0 * lon_d < val < lon_max + 2.0 * lon_d
        )

        cube_in = cube_in.extract(domain)
        cube_in_mask = cube_in_mask.extract(domain)

        return cube_in, cube_in_mask

    def _slice_cube_by_domain(self, cube_in, output_domain):
        """
        extract cube domain to be consistent as cube_reference's domain
        
        Args:
            cube_in (iris.cube.Cube): 
                input data cube to be sliced
            output_domain(tuple):
                lat_max,lon_max,lat_min,lon_min
        Returns:
            cube_in (iris.cube.Cube):
                data cube after slicing
        """
        # extract output grid domain
        lat_max, lon_max, lat_min, lon_min = output_domain
        lat_d, lon_d = self._get_grid_size(cube_in)

        # lats_name, lons_name = self._latlon_names(cube_in)  #how to use lats_name ??

        domain = iris.Constraint(
            latitude=lambda val: lat_min - 2.0 * lat_d < val < lat_max + 2.0 * lat_d
        ) & iris.Constraint(
            longitude=lambda val: lon_min - 2.0 * lon_d < val < lon_max + 2.0 * lon_d
        )

        cube_in = cube_in.extract(domain)

        return cube_in

    def _reshape_data_cube(self, cube):
        """
        Reshape data cube from (....,lat,lon) into data (lat*lon,...)
        
        Args:
            cube (iris.cube.Cube): 
                original data cube
        Return:
            numpy.ndarray or numpy.ma.core.MaskedArray
            reshaped data array
        """
        in_values = cube.data
        coord_names = self._get_cube_coord_names(cube)
        lats_name, lons_name = self._latlon_names(cube)
        lats_index = coord_names.index(lats_name)
        lons_index = coord_names.index(lons_name)

        in_values = np.swapaxes(in_values, 0, lats_index)
        in_values = np.swapaxes(in_values, 1, lons_index)

        lats_len = int(in_values.shape[0])
        lons_len = int(in_values.shape[1])
        latlon_shape = [lats_len * lons_len] + list(in_values.shape[2:])
        in_values = np.reshape(in_values, latlon_shape)
        return in_values, lats_index, lons_index

    def _reshape_data_back(
        self, regrid_result, cube_out_mask, in_values, lats_index, lons_index
    ):
        """
        Reshape numpy array regrid_result from (lat*lon,...) to (....,lat,lon)
        or from (projy*projx,...) to (...,projy,projx)
        
        Args:
            regrid_result (numpy.ndarray):
                array of regridded result in (lat*lon,....) or (projy*projx,...)
            cube_out_mask (iris.cube.Cube): 
                target grid cube (for getting grid dimension here)
            in_values (numpy.ndarray):
                reshaped source data (in _reshape_data_cube)
            lats_index(int):
                index of lats or projy coord in reshaped array 
            lons_index(int):
                index of lons or projx coord in reshaped array
        Return:
            numpy.ndarray or numpy.ma.core.MaskedArray
            reshaped data array
        """
        cube_out_dim0 = cube_out_mask.coord(axis="y").shape[0]
        cube_out_dim1 = cube_out_mask.coord(axis="x").shape[0]
        latlon_shape = [cube_out_dim0, cube_out_dim1] + list(in_values.shape[1:])

        regrid_result = np.reshape(regrid_result, latlon_shape)
        regrid_result = np.swapaxes(regrid_result, 1, lons_index)
        regrid_result = np.swapaxes(regrid_result, 0, lats_index)
        return regrid_result

    def _create_regrid_cube(self, cube_array, cube_in, cube_out):
        """
        create a regridded cube from regridded value(numpy array) 
        source cube cube_in must be in  GeogCS's lats/lons system
        tergat cube_out either lats/lons system or LambertAzimuthalEqualArea system
        args:
            cube_array (numpy ndarray):
                regridded value (multidimensional)
 
            cube_in (iris.cube.Cube):
                source cube (for value's non-grid dimensions and attributes)
            cube_out (iris.cube.Cube):
                target cube (for target grid information)
        return:
             cube_v(iris.cube.Cube):
                 regridded result cube
        """
        cube_coord_names = self._get_cube_coord_names(cube_in)
        lats_name, lons_name = self._latlon_names(cube_in)
        cube_coord_names.remove(lats_name)
        cube_coord_names.remove(lons_name)

        cube_v = Cube(cube_array)
        cube_v.attributes = cube_in.attributes

        # cube_v.add_aux_coord(cube_in.aux_coords)
        cube_v.var_name = cube_in.var_name
        cube_v.units = cube_in.units

        ndim = len(cube_coord_names)
        for i, val in enumerate(cube_coord_names):
            cube_v.add_dim_coord(cube_in.coord(val), i)

        cube_coord_names = self._get_cube_coord_names(cube_out)
        if "projection_y_coordinate" in cube_coord_names:
            cord_1 = "projection_y_coordinate"
            cord_2 = "projection_x_coordinate"
        else:
            cord_1, cord_2 = self._latlon_names(cube_out)

        cube_v.add_dim_coord(cube_out.coord(cord_1), ndim)
        cube_v.add_dim_coord(cube_out.coord(cord_2), ndim + 1)

        for coord in cube_in.aux_coords:
            dims = np.array(cube_in.coord_dims(coord)) + 1
            cube_v.add_aux_coord(coord.copy(), dims)

        return cube_v

    def _get_related_points_for_target_bilinear(
        self, out_latlons, in_latlons, cube_in_dim1
    ):
        """
        updating source points and weighting for 2-false-source-point cases
        Args:
            p_with_2false(numpy.ndarray):
                selected target points which have 2 false source points
            in_latlons(numpy.ndarray):
                tource points's latitude-longitudes 
            out_latlons(numpy.ndarray):
                target points's latitude-longitudes 
        Returns:
            indexes(numpy.ndarray):
                updated array of source grid point number for all target grid points
            weights(numpy.ndarray):     
                updated array of source grid point weighting for all target grid points
        """
        # get source(in) edge length
        edge_lat = in_latlons[cube_in_dim1, 0] - in_latlons[0, 0]
        edge_lon = in_latlons[1, 1] - in_latlons[0, 1]

        # get m_lon1,n_lat1: source point 1 indelon (lon/lat) each target
        n_lat = (out_latlons[:, 0] - in_latlons[0, 0]) // edge_lat
        m_lon = (out_latlons[:, 1] - in_latlons[0, 1]) // edge_lon

        n_lat = n_lat.astype(int)
        m_lon = m_lon.astype(int)
        area = edge_lon * edge_lat

        node0 = n_lat * cube_in_dim1 + m_lon
        node3 = node0 + 1
        node1 = node0 + cube_in_dim1
        node2 = node1 + 1
        # note: lat (X) but ordering  (lat0, lon0)(lat0,lon1)....(lat0,lon_last),(lat1,lon0),
        indexes = np.transpose([node0, node1, node2, node3])

        return indexes

    def _get_weight_four_valid_points_bilinear(
        self, p_no, indexes, out_latlons, in_latlons, cube_in_dim1
    ):
        """
        updating source points and weighting for 2-false-source-point cases
        Args:
            p_with_2false(numpy.ndarray):
                selected target points which have 2 false source points
            in_latlons(numpy.ndarray):
                tource points's latitude-longitudes 
            out_latlons(numpy.ndarray):
                target points's latitude-longitudes 
        Returns:
            indexes(numpy.ndarray):
                updated array of source grid point number for all target grid points
            weights(numpy.ndarray):     
                updated array of source grid point weighting for all target grid points
        """
        # get source(in) edge length
        edge_lat = in_latlons[cube_in_dim1, 0] - in_latlons[0, 0]
        edge_lon = in_latlons[1, 1] - in_latlons[0, 1]
        area = edge_lat * edge_lon

        # left bottom node
        lat_1 = in_latlons[indexes[p_no, 0], 0]
        lon_1 = in_latlons[indexes[p_no, 0], 1]

        # lats/lons for target
        out_lats = out_latlons[p_no, 0]
        out_lons = out_latlons[p_no, 1]

        # lat2_lon: right_bottom source's lat - target lat
        # lat-x,lon-y for formulation. different from plot
        lat2_lat = lat_1 + edge_lat - out_lats
        # lat3_lon: right_top source's lon - target lon
        lon3_lon = lon_1 + edge_lon - out_lons

        lon_lon1 = out_lons - lon_1
        lat_lat1 = out_lats - lat_1

        w1 = lat2_lat * lon3_lon / area
        w2 = lat_lat1 * lon3_lon / area
        w3 = lat_lat1 * lon_lon1 / area
        w4 = lat2_lat * lon_lon1 / area

        weights_1 = np.transpose([w1, w2, w3, w4])

        return weights_1

    def _modify_weights_indexes_bilinear(
        self,
        in_latlons,
        out_latlons,
        in_classified,
        out_classified,
        weights,
        indexes,
        surface_type_mask,
        cube_in_dim1,
    ):
        """
        updating source points and weighting for 2-false-source-point cases
        Args:
            p_with_2false(numpy.ndarray):
                selected target points which have 2 false source points
            in_latlons(numpy.ndarray):
                tource points's latitude-longitudes 
            out_latlons(numpy.ndarray):
                target points's latitude-longitudes 
            surface_type_mask(numpy.ndarray)):
                numpy ndarray of bool, true if source point type matches target point type
            indexes(numpy.ndarray):
                array of source grid point number for all target grid points 
            weights(numpy.ndarray):
                array of source grid point weighting for all target grid points
            in_classified(numpy.ndarray):
                land_sea type for source grid points (land =>True)
            out_classified(numpy.ndarray):
                land_sea type for terget grid points (land =>True)
        Returns:
            indexes(numpy.ndarray):
                updated array of source grid point number for all target grid points
            weights(numpy.ndarray):     
                updated array of source grid point weighting for all target grid points
        """
        p_in = np.sum(surface_type_mask, axis=1)

        #  zero weigting for false source points
        false_points = np.where(surface_type_mask == False)
        weights[false_points] = 0.00

        ## Handle 3 True/1 False cases
        p_with_1false = (np.where(p_in == 3))[0]

        weights, node_1false = self._handle_1false_source_p_bilinear(
            p_with_1false,
            surface_type_mask,
            indexes,
            weights,
            out_latlons,
            in_latlons,
            cube_in_dim1,
        )

        p_with_3false = (np.where(p_in == 1))[0]
        p_with_2false = (np.where(p_in == 2))[0]

        # use inverse distance to handle leftover cases
        p_idw_stack = np.concatenate((node_1false, p_with_2false, p_with_3false))
        indexes, weights, node_4false = self._get_inverse_distance_weight(
            p_idw_stack,
            in_latlons,
            out_latlons,
            indexes,
            weights,
            in_classified,
            out_classified,
        )

        p_with_4false = (np.where(p_in == 0))[0]

        if p_with_4false.shape[0] > 0 or node_4false.shape[0] > 0:
            p_with_4false = np.concatenate((p_with_4false, node_4false))
            weights, indexes, surface_type_mask = self._handle_4false_source_p(
                p_with_4false,
                weights,
                indexes,
                surface_type_mask,
                in_latlons,
                out_latlons,
                in_classified,
                out_classified,
                cube_in_dim1,
            )

        return weights, indexes

    # function for changing weighting for 3 True/1 False cases
    def _handle_1false_source_p_bilinear(
        self,
        p_with_1false,
        surface_type_mask,
        indexes,
        weights,
        out_latlons,
        in_latlons,
        cube_in_dim1,
    ):
        """
        updating source points and weighting for 1-false-source-point cases
        Args:
            p_with_1false(numpy.ndarray):
                selected target points which have 1 false source point
            surface_type_mask(numpy.ndarray)):
                numpy ndarray of bool, true if source point type matches target point type
            indexes(numpy.ndarray):
                array of source grid point number for all target grid points 
            weights(numpy.ndarray):
                array of source grid point weighting for all target grid points
            in_latlons(numpy.ndarray):
                tource points's latitude-longitudes 
            out_latlons(numpy.ndarray):
                target points's latitude-longitudes 
            cube_in_dim1 (int):
                source grid's latitutde dimension  
        Returns:
             weights(numpy.ndarray):     
                updated array of source grid point weighting for target grid points
        """

        # get source(in) edge length
        edge_lat = in_latlons[cube_in_dim1, 0] - in_latlons[0, 0]
        edge_lon = in_latlons[1, 1] - in_latlons[0, 1]
        area = edge_lat * edge_lon
        node_1false = np.array([], dtype=int)

        # loop for one false point order
        for i in range(4):
            p_with_1false_i = (np.where(surface_type_mask[p_with_1false, i] == False))[
                0
            ]
            node = p_with_1false[p_with_1false_i]

            # lats/lons for target
            out_lats = out_latlons[node, 0]
            out_lons = out_latlons[node, 1]

            # left bottom node
            lat_1 = in_latlons[indexes[node, 0], 0]
            lon_1 = in_latlons[indexes[node, 0], 1]

            # change weights
            if i == 0:
                lat2_lat = lat_1 + edge_lat - out_lats
                lon3_lon = lon_1 + edge_lon - out_lons

                w2 = edge_lat * lon3_lon / area
                w4 = lat2_lat * edge_lon / area

                w1 = np.zeros(w2.shape[0], dtype=np.float32)
                w3 = np.ones(w2.shape[0], dtype=np.float32)
                w3 -= w2 + w4

            elif i == 1:
                lon3_lon = lon_1 + edge_lon - out_lons
                lat_lat1 = out_lats - lat_1

                w1 = edge_lat * lon3_lon / area
                w3 = lat_lat1 * edge_lon / area

                w2 = np.zeros(w1.shape[0], dtype=np.float32)
                w4 = np.ones(w1.shape[0], dtype=np.float32)
                w4 -= w1 + w3

            elif i == 2:
                lon_lon1 = out_lons - lon_1
                lat_lat1 = out_lats - lat_1

                w2 = lat_lat1 * edge_lon / area
                w4 = edge_lat * lon_lon1 / area

                w3 = np.zeros(w2.shape[0], dtype=np.float32)
                w1 = np.ones(w2.shape[0], dtype=np.float32)
                w1 -= w2 + w4

            elif i == 3:
                lat2_lat = lat_1 + edge_lat - out_lats
                lon_lon1 = out_lons - lon_1
                w3 = edge_lat * lon_lon1 / area
                w1 = lat2_lat * edge_lon / area
                w4 = np.zeros(w1.shape[0], dtype=np.float32)
                w2 = np.ones(w1.shape[0], dtype=np.float32)
                w2 -= w1 + w3

            # put back to weights
            weights_i = np.transpose([w1, w2, w3, w4])

            # exclude extrapolated part
            wt_pos = weights_i > -1.0e-6
            kkk = np.all(wt_pos, axis=1)
            node_1false_id = (np.where(kkk == False))[0]

            if node_1false_id.shape[0] > 0:
                node_kp_id = (np.where(kkk == True))[0]
                weights[node[node_kp_id]] = weights_i[node_kp_id]
                node_1false_i = node[node_1false_id]
                node_1false = np.concatenate((node_1false, node_1false_i))
            else:
                weights[node] = weights_i

        return weights, node_1false

    def _handle_4false_source_p(
        self,
        p_with_4false,
        weights,
        indexes,
        surface_type_mask,
        in_latlons,
        out_latlons,
        in_classified,
        out_classified,
        cube_in_dim1,
    ):
        """
        updating source points and weighting for 4-false-source-point cases
        this function used for 
        Args:
            p_with_4false(numpy.ndarray):
                selected target points which have 4 false source points
            in_latlons(numpy.ndarray):
                tource points's latitude-longitudes 
            out_latlons(numpy.ndarray):
                target points's latitude-longitudes 
            surface_type_mask(numpy.ndarray)):
                numpy ndarray of bool, true if source point type matches target point type
            indexes(numpy.ndarray):
                array of source grid point number for all target grid points 
            weights(numpy.ndarray):
                array of source grid point weighting for all target grid points
            in_classified(numpy.ndarray):
                land_sea type for source grid points (land =>True)
            out_classified(numpy.ndarray):
                land_sea type for terget grid points (land =>True)
        Returns:
            indexes(numpy.ndarray):
                updated array of source grid point number for all target grid points
            weights(numpy.ndarray):     
                updated array of source grid point weighting for all target grid points
        """

        # increase 4 points to 8 points
        out_latlons_1 = out_latlons[p_with_4false]
        k_nearest = 8
        circle_limit = self.vicinity

        distances_1, indexes_1 = self._nearest_input_pts(
            in_latlons, out_latlons_1, k_nearest
        )

        out_classified_1 = out_classified[p_with_4false]
        surface_type_mask_1 = self._similar_surface_classify(
            in_classified, out_classified_1, indexes_1
        )

        # if True,but distance > circle, change it to false
        not_in_circle = distances_1 > circle_limit
        surface_type_mask_1 = np.where(not_in_circle, False, surface_type_mask_1)

        # from here, we should judge if all falses. put these points into normal bilinear weights
        # ignore its false or true, and use old indexes, and not change!!
        sum_true = np.sum(surface_type_mask_1, axis=1)
        island_4false = (np.where(sum_true == 0))[0]
        if island_4false.shape[0] > 0:
            # let it alone, use original indexes and weights. Must remember, not assigned them to zero
            island_4false_ext = p_with_4false[island_4false]
            # indexes still uses old indexes,unchanged
            if self.regrid_mode == "bilinear-with-mask-2":
                weights[
                    island_4false_ext
                ] = self._get_weight_four_valid_points_bilinear(
                    island_4false_ext, indexes, out_latlons, in_latlons, cube_in_dim1
                )
            else:  # "nearest-with-mask-2" change to true
                surface_type_mask[island_4false_ext, :] = True

            # for other case, use inverse distance weighting
            island_true = (np.where(sum_true > 0))[0]  # internal no
            no_island_true = island_true.shape[0]

            if no_island_true > 0:
                distances_2 = np.zeros([no_island_true, 4])
                for p_id in range(island_true.shape[0]):

                    island_4false_ext = p_with_4false[island_true[p_id]]
                    weights[island_4false_ext, :] = 0.000
                    surface_type_mask[island_4false_ext, :] = False

                    idx = 0
                    for i in range(8):  # distance orderes from small to large indexes
                        if surface_type_mask_1[island_true[p_id], i] == True:
                            indexes[island_4false_ext, idx] = indexes_1[
                                island_true[p_id], i
                            ]
                            surface_type_mask[
                                island_4false_ext, idx
                            ] = True  # other mask =false
                            distances_2[p_id, idx] = distances_1[island_true[p_id], i]
                            idx += 1
                            # for nearest option, just need nearest true point.
                            if idx == 1 and self.regrid_mode == "nearest-with-mask-2":
                                break
                            # for bilinear case, we get points up to 4. if not, false point okay!
                            elif (
                                idx == 4 and self.regrid_mode == "bilinear-with-mask-2"
                            ):
                                break

                # for bilinear cases, do inverse distance weight
                if self.regrid_mode == "bilinear-with-mask-2":

                    island_true_ext = p_with_4false[island_true]

                    # convert mask to be true where input points should not be considered
                    not_mask = np.logical_not(surface_type_mask[island_true_ext])

                    # replace distances with infinity where they should not be used
                    masked_distances = np.where(
                        not_mask, np.float64("inf"), distances_2
                    )

                    # add a small amount to all distances to avoid division by zero when taking the inverse
                    masked_distances += np.finfo(np.float64).eps
                    # invert the distances, sum the k surrounding points, scale to produce weights
                    inv_distances = 1.0 / masked_distances
                    inv_distances_sum = np.sum(inv_distances, axis=1)
                    inv_distances_sum = 1.0 / inv_distances_sum

                    weights_idw = inv_distances * inv_distances_sum.reshape(-1, 1)
                    weights[island_true_ext] = weights_idw

        return weights, indexes, surface_type_mask

    def _get_inverse_distance_weight(
        self,
        p_idw_stack,
        in_latlons,
        out_latlons,
        indexes,
        weights,
        in_classified,
        out_classified,
    ):
        """
        Locating source points and calculating inverse distance weights for selective target points
        Args:
            p_idw_stack(numpy.ndarray):
                selected target points which will use Inverse Distance Weighting(idw) approach
            in_latlons(numpy.ndarray):
                Source points's latitude-longitudes 
            out_latlons(numpy.ndarray):
                Target points's latitude-longitudes 
            indexes(numpy.ndarray):
                array of source grid point number for all target grid points 
            weights(numpy.ndarray):
                array of source grid point weighting for all target grid points
            in_classified(numpy.ndarray):
                land_sea type for source grid points (land =>True)
            out_classified(numpy.ndarray):
                land_sea type for terget grid points (land =>True)
        Returns:
            indexes(numpy.ndarray):
                updated array of source grid point number for all target grid points
            weights(numpy.ndarray):     
                updated array of source grid point weighting for all target grid points
        """

        out_latlons_1 = out_latlons[p_idw_stack]
        k_nearest = 4
        distances_1, indexes_1 = self._nearest_input_pts(
            in_latlons, out_latlons_1, k_nearest
        )

        out_classified_1 = out_classified[p_idw_stack]
        surface_type_mask_1 = self._similar_surface_classify(
            in_classified, out_classified_1, indexes_1
        )

        # possibly some islands generated, leave it out for next step
        pt_in = np.sum(surface_type_mask_1, axis=1)
        pt_with_4false = (np.where(pt_in == 0))[0]
        node_4false_a = p_idw_stack[pt_with_4false]

        # so only handle with true cases
        pt_with_true = (np.where(pt_in > 0))[0]
        node_with_true = p_idw_stack[pt_with_true]

        # convert mask to be true where input points should not be considered
        not_mask = np.logical_not(surface_type_mask_1[pt_with_true])

        # replace distances with infinity where they should not be used
        masked_distances = np.where(
            not_mask, np.float64("inf"), distances_1[pt_with_true]
        )

        # add a small amount to all distances to avoid division by zero when taking the inverse
        masked_distances += np.finfo(np.float64).eps
        # invert the distances, sum the k surrounding points, scale to produce weights
        inv_distances = 1.0 / masked_distances
        inv_distances_sum = np.sum(inv_distances, axis=1)
        inv_distances_sum = 1.0 / inv_distances_sum

        weights_idw = inv_distances * inv_distances_sum.reshape(-1, 1)

        indexes[node_with_true] = indexes_1[pt_with_true]
        weights[node_with_true] = weights_idw

        return indexes, weights, node_4false_a

    def _apply_weight_bilinear(self, indexes, in_values, weights):
        """
        Apply bilinear weight of source points for target value
        Args:
            indexes(numpy.ndarray):
                array of source grid point number for target grid points
            weights(numpy.ndarray):
                array of source grid point weighting for target grid points
            in_values(numpy.ndarray):
                input values (maybe multidimensional)
        Returns:
            out_values(numpy.ndarray):
            regridded value for target points 
        """
        in_values_expanded = (np.ma.filled(in_values, np.nan))[indexes]
        weighted = np.transpose(
            np.multiply(np.transpose(weights), np.transpose(in_values_expanded))
        )
        out_values = np.sum(weighted, axis=1)
        return out_values

    def process(self, cube_in, cube_in_mask, cube_out_mask):
        """
        Regridding considering land_sea mask. please note cube_in must use
        lats/lons rectlinear system(GeogCS). cube_in_mask and cube_in could be 
        different  resolution. cube_our could be either in lats/lons rectlinear 
        system or LambertAzimuthalEqualArea system.         

        Args:
            cube (iris.cube.Cube):
                Cube of data to be regridded 
            cube_in_mask (iris.cube.Cube):
                Cube of land_binary_mask data ((land:1, sea:0). used to determine
                where the input model data is representing land and sea points.
            cube_out_mask (iris.cube.Cube):
                Cube of land_binary_mask data on target grid (land:1, sea:0).
            regrid_mode(str):
                "nearest-2","bilinear-2", "nearest-with-mask-2", "bilinear-with-mask-2"
        Returns: 
            cube_result (iris.cube.Cube):
                regridded result in cube
        """

        # if different coordinate system, transformation into lats/lons
        if (
            cube_out_mask.coord(axis="x").standard_name == "projection_x_coordinate"
            and cube_out_mask.coord(axis="y").standard_name == "projection_y_coordinate"
        ):
            out_latlons = self._convert_from_projection_to_latlons(
                cube_out_mask, cube_in
            )
        else:
            out_latlons = self._latlon_from_cube(cube_out_mask)

        # extract input cube based on output grid
        lat_max, lon_max = out_latlons.max(axis=0)
        lat_min, lon_min = out_latlons.min(axis=0)

        # extract cube_in to right-domain
        if self.regrid_mode in ("nearest-2", "bilinear-2"):
            cube_in = self._slice_cube_by_domain(
                cube_in, (lat_max, lon_max, lat_min, lon_min)
            )
        elif self.regrid_mode in ("nearest-with-mask-2", "bilinear-with-mask-2"):
            cube_in, cube_in_mask = self._slice_mask_cube_by_domain(
                cube_in, cube_in_mask, (lat_max, lon_max, lat_min, lon_min)
            )

        # extract lats/lons from cube_in
        in_latlons = self._latlon_from_cube(cube_in)
        cube_in_dim0 = cube_in.coord(axis="y").shape[0]  # latitude
        cube_in_dim1 = cube_in.coord(axis="x").shape[0]  # longitude

        if self.regrid_mode in (
            "bilinear-2",
            "bilinear-with-mask-2",
            "nearest-with-mask-2",
        ):
            weights = np.zeros((out_latlons.shape[0], 4), dtype=np.float32)

        # locate source points for target points
        indexes = self._get_related_points_for_target_bilinear(
            out_latlons, in_latlons, cube_in_dim1
        )

        # find k nearest input points to each output point, using a KDtree
        if self.regrid_mode in ("nearest-2", "nearest-with-mask-2"):
            distances = np.zeros((out_latlons.shape[0], 4))
            for i in range(4):
                distances[:, i] = np.square(
                    in_latlons[indexes[:, i], 0] - out_latlons[:, 0]
                ) + np.square(in_latlons[indexes[:, i], 1] - out_latlons[:, 1])

        elif self.regrid_mode in ("bilinear-2", "bilinear-with-mask-2"):

            # for bilinear, define weights (assume all valid source points first)
            p_no = np.arange(out_latlons.shape[0])
            weights[p_no] = self._get_weight_four_valid_points_bilinear(
                p_no, indexes, out_latlons, in_latlons, cube_in_dim1
            )

        # classify if source point surface type is the same as surface type
        if self.regrid_mode in ("nearest-with-mask-2", "bilinear-with-mask-2"):
            # Classify surface types of a source data grid based on a land mask
            in_classified = self._land_classify_in(cube_in_mask, in_latlons)

            # Classify surface type for output grid
            out_classified = self._land_classify_out(cube_out_mask)

            # Classify surface types as being included in  regridding or not.
            surface_type_mask = self._similar_surface_classify(
                in_classified, out_classified, indexes
            )

        # for bilinear-with-mask, lots of correction on wieghts
        if self.regrid_mode == "bilinear-with-mask-2":
            weights, indexes = self._modify_weights_indexes_bilinear(
                in_latlons,
                out_latlons,
                in_classified,
                out_classified,
                weights,
                indexes,
                surface_type_mask,
                cube_in_dim1,
            )

        # Put input nwp data in right shape for regridding function
        in_values, lats_index, lons_index = self._reshape_data_cube(cube_in)

        # main regridding function
        if self.regrid_mode == "nearest-2":
            regrid_result = self._nearest_distance(distances, indexes, in_values)
        elif self.regrid_mode == "nearest-with-mask-2":
            regrid_result = self._nearest_distance_with_mask(
                distances,
                indexes,
                surface_type_mask,
                weights,
                in_latlons,
                out_latlons,
                in_classified,
                out_classified,
                in_values,
                cube_in_dim1,
            )
        elif self.regrid_mode in ("bilinear-2", "bilinear-with-mask-2"):
            regrid_result = self._apply_weight_bilinear(indexes, in_values, weights)

        # reshape results array in right shape & put it into cube
        regrid_result = self._reshape_data_back(
            regrid_result, cube_out_mask, in_values, lats_index, lons_index
        )
        cube_result = self._create_regrid_cube(regrid_result, cube_in, cube_out_mask)

        return cube_result
