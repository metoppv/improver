#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2019 Met Office.
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
import numpy as np
from iris.analysis import Linear, Nearest
from iris.exceptions import CoordinateNotFoundError
from scipy.interpolate import griddata

from improver import BasePlugin
from improver.metadata.amend import amend_attributes
from improver.metadata.check_datatypes import (
    check_units, get_required_dtype, get_required_units)
from improver.metadata.constants.attributes import MANDATORY_ATTRIBUTE_DEFAULTS
from improver.metadata.constants.mo_attributes import MOSG_GRID_ATTRIBUTES
from improver.metadata.constants.time_types import TIME_COORDS
from improver.threshold import BasicThreshold
from improver.utilities.cube_checker import spatial_coords_match
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
    for axis in ['x', 'y']:
        grid_coord = grid.coord(axis=axis)
        cutout_coord = cutout.coord(axis=axis)
        # check coordinate metadata
        if (cutout_coord.name() != grid_coord.name() or
                cutout_coord.units != grid_coord.units or
                cutout_coord.coord_system != grid_coord.coord_system):
            return False

        # search for cutout coordinate points in larger grid
        cutout_start = cutout_coord.points[0]
        find_start = [np.isclose(cutout_start, grid_point)
                      for grid_point in grid_coord.points]
        if not np.any(find_start):
            return False

        start = find_start.index(True)
        end = start + len(cutout_coord.points)
        try:
            if not np.allclose(cutout_coord.points,
                               grid_coord.points[start:end]):
                return False
        except ValueError:
            # raised by np.allclose if "end" index overshoots edge of grid
            # domain - slicing does not raise IndexError
            return False

    return True


class StandardiseGridAndMetadata(BasePlugin):

    """Plugin to regrid cube data and standardise metadata"""

    REGRID_REQUIRES_LANDMASK = {"bilinear": False,
                                "nearest": False,
                                "nearest-with-mask": True}

    def __init__(self, regrid_mode='bilinear', extrapolation_mode='nanmask',
                 landmask=None, landmask_vicinity=25000):
        """
        Initialise regridding parameters

        Args:
            regrid_mode (str):
                Mode of interpolation in regridding.
            extrapolation_mode (str):
                Mode to fill regions outside the domain in regridding.
            landmask (iris.cube.Cube or None):
                Land-sea mask ("land_binary_mask") on the input cube grid.
                Required for "nearest-with-mask" regridding option,
                with land points set to one and sea points set to zero.
            landmask_vicinity (float):
                Radius of vicinity to search for a coastline, in metres

        Raises:
            ValueError: If a landmask is required but not passed in
        """
        if landmask is None and "nearest-with-mask" in regrid_mode:
            msg = ("Regrid mode {} requires an input landmask cube")
            raise ValueError(msg.format(regrid_mode))
        self.regrid_mode = regrid_mode
        self.extrapolation_mode = extrapolation_mode
        self.landmask_source_grid = landmask
        self.landmask_vicinity = (
            None if landmask is None else landmask_vicinity)
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
            msg = ("Expected {} in input_landmask cube but found {}".format(
                self.landmask_name, repr(self.landmask_source_grid)))
            warnings.warn(msg)

        if self.landmask_name not in target_grid.name():
            msg = ("Expected {} in target_grid cube but found {}".format(
                self.landmask_name, repr(target_grid)))
            warnings.warn(msg)

        plugin = AdjustLandSeaPoints(vicinity_radius=self.landmask_vicinity)
        return plugin.process(cube, self.landmask_source_grid, target_grid)

    def _regrid_to_target(self, cube, target_grid, regridded_title):
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

        Returns:
            iris.cube.Cube: Regridded cube with updated attributes
        """
        regridder = Linear(extrapolation_mode=self.extrapolation_mode)
        if "nearest" in self.regrid_mode:
            regridder = Nearest(extrapolation_mode=self.extrapolation_mode)
        cube = cube.regrid(target_grid, regridder)

        if self.REGRID_REQUIRES_LANDMASK[self.regrid_mode]:
            cube = self._adjust_landsea(cube, target_grid)

        # identify grid-describing attributes on source cube that need updating
        required_grid_attributes = [attr for attr in cube.attributes
                                    if attr in MOSG_GRID_ATTRIBUTES]
        # update attributes if available on target grid, otherwise remove
        for key in required_grid_attributes:
            if key in target_grid.attributes:
                cube.attributes[key] = target_grid.attributes[key]
            else:
                cube.attributes.pop(key)

        cube.attributes["title"] = (
            MANDATORY_ATTRIBUTE_DEFAULTS["title"]
            if regridded_title is None else regridded_title)

        return cube

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
                coord.points = np.around(coord.points)
            coord.points = as_correct_dtype(coord.points, req_dtype)
            if coord.has_bounds():
                if np.issubdtype(req_dtype, np.integer):
                    coord.bounds = np.around(coord.bounds)
                coord.bounds = as_correct_dtype(coord.bounds, req_dtype)

    def process(self, cube, target_grid=None, new_name=None, new_units=None,
                regridded_title=None, coords_to_remove=None,
                attributes_dict=None):
        """
        Perform regridding and metadata adjustments

        Args:
            cube (iris.cube.Cube):
                Input cube to be standardised
            target_grid (iris.cube.Cube or None):
                Cube on the required grid. For "nearest-with-mask" regridding,
                this cube should contain a binary land-sea mask
                ("land_binary_mask"). If target_grid is None, no regridding is
                performed.
            new_name (str or None):
                Optional rename for output cube
            new_units (str or None):
                Optional unit conversion for output cube
            regridded_title (str or None):
                New title attribute to be applied after regridding. If not set,
                the title attribute is set to a default value if the field is
                regridded, as "title" may contain grid information.
            coords_to_remove (list of str or None):
                Optional list of scalar coordinates to remove from output cube
            attributes_dict (dict or None):
                Optional dictionary of required attribute updates. Keys are
                attribute names, and values are the required value or "remove".

        Returns:
            iris.cube.Cube
        """
        # regridding
        if target_grid:
            # if regridding using a land-sea mask, check this covers the source
            # grid in the required coordinates
            if self.REGRID_REQUIRES_LANDMASK[self.regrid_mode]:
                if not grid_contains_cutout(self.landmask_source_grid, cube):
                    raise ValueError(
                        "Source landmask does not match input grid")
            cube = self._regrid_to_target(cube, target_grid, regridded_title)

        # standard metadata updates
        cube = self._collapse_scalar_dimensions(cube)

        # optional metadata updates
        if new_name:
            cube.rename(new_name)
        if new_units:
            cube.convert_units(new_units)
        if coords_to_remove:
            self._remove_scalar_coords(cube, coords_to_remove)
        if attributes_dict:
            amend_attributes(cube, attributes_dict)

        # ensure dtypes follow IMPROVER conventions
        self._standardise_dtypes_and_units(cube)

        return cube


class AdjustLandSeaPoints:
    """
    Replace data values at points where the nearest-regridding technique
    selects a source grid-point with an opposite land-sea-mask value to the
    target grid-point.
    The replacement data values are selected from a vicinity of points on the
    source-grid and the closest point of the correct mask is used.
    Where no match is found within the vicinity, the data value is not changed.
    """

    def __init__(self, extrapolation_mode="nanmask", vicinity_radius=25000.):
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
            self.regridder, self.vicinity)

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
        selector_data = griddata(use_points,
                                 self.nearest_cube.data[use_points],
                                 (y_points, x_points), method="nearest")

        # Identify nearby points on regridded input_land that match the
        # selector_value
        if selector_val > 0.5:
            thresholder = BasicThreshold(0.5)
        else:
            thresholder = BasicThreshold(0.5, comparison_operator='<=')
        in_vicinity = self.vicinity(thresholder(self.input_land))

        # Identify those points sourced from the opposite mask that are
        # close to a source point of the correct mask
        mismatch_points, = np.logical_and(
            np.logical_and(self.output_land.data == selector_val,
                           self.input_land.data != selector_val),
            in_vicinity.data > 0.5)

        # Replace these points with the filled-domain data
        self.output_cube.data[mismatch_points] = (
            selector_data[mismatch_points])

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
            raise ValueError('X and Y coordinates do not match for cubes {}'
                             'and {}'.format(repr(cube), repr(output_land)))
        self.output_land = output_land

        # Regrid input_land to output_land grid.
        self.input_land = input_land.regrid(self.output_land, self.regridder)

        # Slice over x-y grids for multi-realization data.
        result = iris.cube.CubeList()
        for xyslice in cube.slices(
                [cube.coord(axis='y'), cube.coord(axis='x')]):

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
