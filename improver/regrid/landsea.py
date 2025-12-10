# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Plugin to regrid with land-sea awareness"""

import functools
import warnings
from typing import Optional, Tuple

import iris
import numpy as np
from iris.analysis import Linear, Nearest
from iris.cube import Cube
from numpy import ndarray
from scipy.spatial import cKDTree

from improver import PostProcessingPlugin
from improver.metadata.constants.attributes import MANDATORY_ATTRIBUTE_DEFAULTS
from improver.metadata.constants.mo_attributes import MOSG_GRID_ATTRIBUTES
from improver.regrid.landsea2 import RegridWithLandSeaMask
from improver.threshold import Threshold
from improver.utilities.cube_checker import spatial_coords_match
from improver.utilities.spatial import OccurrenceWithinVicinity


class RegridLandSea(PostProcessingPlugin):
    """Nearest-neighbour and bilinear regridding with or without land-sea mask
    awareness. When land-sea mask considered, surface-type-mismatched source
    points are excluded from field regridding calculation for target points.
    For example, for regridding a field using nearest-neighbour approach with
    land-sea awareness, regridded land points always take values from a land
    point on the source grid, and vice versa for sea points."""

    REGRID_REQUIRES_LANDMASK = {
        "bilinear": False,
        "nearest": False,
        "esmf-area-weighted": False,
        "nearest-with-mask": True,
        "nearest-2": False,
        "bilinear-2": False,
        "nearest-with-mask-2": True,
        "bilinear-with-mask-2": True,
    }

    def __init__(
        self,
        regrid_mode: str = "bilinear",
        extrapolation_mode: str = "nanmask",
        landmask: Optional[Cube] = None,
        landmask_vicinity: float = 25000,
        mdtol: float = 1,
    ):
        """
        Initialise regridding parameters.

        Args:
            regrid_mode:
                Mode of interpolation in regridding.  Valid options are "bilinear",
                "nearest", "nearest-with-mask", "bilinear-2","nearest-2",
                "nearest-with-mask-2" or "bilinear-with-mask-2".  "***-with-mask**"
                option triggers adjustment of regridded points to match source points
                in terms of land / sea type.
            extrapolation_mode:
                Mode to fill regions outside the domain in regridding.
            landmask:
                Land-sea mask ("land_binary_mask") on the input cube grid, with
                land points set to one and sea points set to zero.  Required for
                "nearest-with-mask" regridding option.
            landmask_vicinity:
                Radius of vicinity to search for a coastline, in metres.
            mdtol:
                Tolerance of missing data for area-weighted regridding. The value
                returned in each element will be masked if the fraction of missing
                data exceeds mdtol. mdtol=0 means no masked data is tolerated while
                mdtol=1 means the element will be masked only if all overlapping
                source elements are masked. Only used for esmf-area-weighted
                regridding. Default is 1.
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
        self.mdtol = mdtol

    def _regrid_to_target(
        self,
        cube: Cube,
        target_grid: Cube,
        regridded_title: Optional[str],
        regrid_mode: str,
    ) -> Cube:
        """
        Regrid cube to target_grid, inherit grid attributes and update title

        Args:
            cube:
                Cube to be regridded
            target_grid:
                Data on the target grid. If regridding with mask, this cube
                should contain land-sea mask data to be used in adjusting land
                and sea points after regridding.
            regridded_title:
                New value for the "title" attribute to be used after
                regridding. If not set, a default value is used.
            regrid_mode:
                "bilinear","nearest","nearest-with-mask",
                "nearest-2","nearest-with-mask-2","bilinear-2","bilinear-with-mask-2"

        Returns:
            Regridded cube with updated attributes.
        """
        if regrid_mode in (
            "nearest-with-mask",
            "nearest-with-mask-2",
            "bilinear-with-mask-2",
        ):
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

        # basic categories (1) Iris-based (2) new nearest based  (3) new bilinear-based
        if regrid_mode in (
            "bilinear",
            "nearest",
            "nearest-with-mask",
            "esmf-area-weighted",
        ):
            if "nearest" in regrid_mode:
                regridder = Nearest(extrapolation_mode=self.extrapolation_mode)
            elif "linear" in regrid_mode:
                regridder = Linear(extrapolation_mode=self.extrapolation_mode)
            elif regrid_mode == "esmf-area-weighted":
                from esmf_regrid.schemes import ESMFAreaWeighted

                regridder = ESMFAreaWeighted(mdtol=self.mdtol)

            cube = cube.regrid(target_grid, regridder)

            # Iris regridding is used, and then adjust if land_sea mask is considered
            if self.REGRID_REQUIRES_LANDMASK[regrid_mode]:
                cube = AdjustLandSeaPoints(
                    vicinity_radius=self.landmask_vicinity,
                    extrapolation_mode=self.extrapolation_mode,
                )(cube, self.landmask_source_grid, target_grid)

        # new version of nearest/bilinear option with/without land-sea mask
        elif regrid_mode in (
            "nearest-2",
            "nearest-with-mask-2",
            "bilinear-2",
            "bilinear-with-mask-2",
        ):
            cube = RegridWithLandSeaMask(
                regrid_mode=regrid_mode, vicinity_radius=self.landmask_vicinity
            )(cube, self.landmask_source_grid, target_grid)

        # identify grid-describing attributes on source cube that need updating
        required_grid_attributes = [
            attr for attr in cube.attributes if attr in MOSG_GRID_ATTRIBUTES
        ]

        # update attributes if available on target grid, otherwise remove
        for key in required_grid_attributes:
            if key in target_grid.attributes:
                cube.attributes[key] = target_grid.attributes[key]
            else:
                cube.attributes.pop(key)

        cube.attributes["title"] = (
            MANDATORY_ATTRIBUTE_DEFAULTS["title"]
            if regridded_title is None
            else regridded_title
        )

        return cube

    def process(
        self, cube: Cube, target_grid: Cube, regridded_title: Optional[str] = None
    ) -> Cube:
        """
        Regrids cube onto spatial grid provided by target_grid.

        Args:
            cube:
                Cube to be regridded.
            target_grid:
                Data on the target grid. If regridding with mask, this cube
                should contain land-sea mask data to be used in adjusting land
                and sea points after regridding.
            regridded_title:
                New value for the "title" attribute to be used after
                regridding. If not set, a default value is used.

        Returns:
            Regridded cube with updated attributes.
        """
        # if regridding using a land-sea mask, check this covers the source
        # grid in the required coordinates
        if self.REGRID_REQUIRES_LANDMASK[self.regrid_mode]:
            if not grid_contains_cutout(self.landmask_source_grid, cube):
                raise ValueError("Source landmask does not match input grid")
        return self._regrid_to_target(
            cube, target_grid, regridded_title, self.regrid_mode
        )


class AdjustLandSeaPoints(PostProcessingPlugin):
    """
    Replace data values at points where the nearest-regridding technique
    selects a source grid-point with an opposite land-sea-mask value to the
    target grid-point.
    The replacement data values are selected from a vicinity of points on the
    source-grid and the closest point of the correct mask is used.
    Where no match is found within the vicinity, the data value is not changed.
    """

    class _NoMatchesError(ValueError):
        """Raise when there are no matches for the specified selector."""

        pass

    def __init__(
        self, extrapolation_mode: str = "nanmask", vicinity_radius: float = 25000.0
    ):
        """
        Initialise class

        Args:
            extrapolation_mode:
                Mode to use for extrapolating data into regions
                beyond the limits of the source_data domain.
                Available modes are documented in
                `iris.analysis <https://scitools.org.uk/iris/docs/latest/iris/
                iris/analysis.html#iris.analysis.Nearest>`_
                Defaults to "nanmask".
            vicinity_radius:
                Distance in metres to search for a sea or land point.
        """
        self.input_land = None
        self.nearest_cube = None
        self.output_land = None
        self.output_cube = None
        self.regridder = Nearest(extrapolation_mode=extrapolation_mode)
        self.vicinity = OccurrenceWithinVicinity(radii=[vicinity_radius])

    @functools.lru_cache(maxsize=2)
    def _get_matches(
        self, selector_val: int
    ) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
        # Find all points on output grid matching selector_val
        use_points = np.where(self.input_land.data == selector_val)
        no_use_points = np.where(self.input_land.data != selector_val)

        # If there are no matching points on the input grid, no alteration can
        # be made.
        if use_points[0].size == 0:
            raise self._NoMatchesError

        # Using only these points, extrapolate to fill domain using nearest
        # neighbour. This will generate a grid where the non-selector_val
        # points are filled with the nearest value in the same mask
        # classification.
        tree = cKDTree(np.c_[use_points[0], use_points[1]])
        _, indices = tree.query(np.c_[no_use_points[0], no_use_points[1]])

        # Identify nearby points on regridded input_land that match the
        # selector_value
        if selector_val > 0.5:
            thresholder = Threshold(threshold_values=0.5)
        else:
            thresholder = Threshold(threshold_values=0.5, comparison_operator="<=")
        in_vicinity = self.vicinity(thresholder(self.input_land))

        # Identify those points sourced from the opposite mask that are
        # close to a source point of the correct mask
        mismatch_points = np.logical_and(
            np.logical_and(
                self.output_land.data == selector_val,
                self.input_land.data != selector_val,
            ),
            in_vicinity.data > 0.5,
        )
        return mismatch_points, indices, use_points, no_use_points

    def correct_where_input_true(self, selector_val: int) -> None:
        """
        Replace points in the output_cube where output_land matches the
        selector_val and the input_land does not match, but has matching
        points in the vicinity, with the nearest matching point in the
        vicinity in the original nearest_cube.
        Updates self.output_cube.data.

        Args:
            selector_val:
                Value of mask to replace if needed.
                Intended to be 1 for filling land points near the coast
                and 0 for filling sea points near the coast.
        """
        try:
            mismatch_points, indices, use_points, no_use_points = self._get_matches(
                selector_val
            )
        except self._NoMatchesError:
            return
        selector_data = self.nearest_cube.data.copy()
        selector_data[no_use_points] = selector_data[use_points][indices]

        # Replace these points with the filled-domain data
        self.output_cube.data[mismatch_points] = selector_data[mismatch_points]

    def process(self, cube: Cube, input_land: Cube, output_land: Cube) -> Cube:
        """
        Update cube.data so that output_land and sea points match an input_land
        or sea point respectively so long as one is present within the
        specified vicinity radius. Note that before calling this plugin the
        input land mask MUST be checked against the source grid, to ensure
        the grids match.

        Args:
            cube:
                Cube of data to be updated (on same grid as output_land).
            input_land:
                Cube of land_binary_mask data on the grid from which "cube" has
                been reprojected (it is expected that the iris.analysis.Nearest
                method would have been used). Land points should be set to one
                and sea points set to zero.
                This is used to determine where the input model data is
                representing land and sea points.
            output_land:
                Cube of land_binary_mask data on target grid.

        Returns:
            Cube of regridding results.
        """
        # Check cube and output_land are on the same grid:
        if not spatial_coords_match([cube, output_land]):
            raise ValueError(
                "X and Y coordinates do not match for cubes {} and {}".format(
                    repr(cube), repr(output_land)
                )
            )
        self.output_land = output_land

        # Regrid input_land to output_land grid.
        self.input_land = input_land.regrid(self.output_land, self.regridder)

        # Slice over x-y grids for multi-realization data.
        result = iris.cube.CubeList()

        # Reset cache as input_land and output_land have changed
        self._get_matches.cache_clear()
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


def grid_contains_cutout(grid: Cube, cutout: Cube) -> bool:
    """
    Check that a spatial cutout is contained within a given grid

    Args:
        grid:
            A cube defining a data grid.
        cutout:
            The cutout to search for within the grid.

    Returns:
        True if cutout is contained within grid, False otherwise.
    """

    if spatial_coords_match([grid, cutout]):
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
            bool(np.isclose(cutout_start, grid_point))
            for grid_point in grid_coord.points
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
