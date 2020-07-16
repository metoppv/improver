# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2020 Met Office.
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
"""Module for generating timezone masks."""

import iris
import numpy as np

from datetime import datetime
from pytz import timezone
from timezonefinder import TimezoneFinder
from iris.exceptions import CoordinateNotFoundError

from improver.metadata.utilities import (
    create_new_diagnostic_cube,
    generate_mandatory_attributes,
)
from improver.utilities.spatial import lat_lon_determine, transform_grid_to_lat_lon

from improver import BasePlugin


class GenerateTimezoneMask(BasePlugin):

    """
    A plugin to create masks for regions of a geographic grid that are on different
    timezones. The resulting masks can be used to isolate data for specific
    timezones from a grid of the same shape as the cube used in this plugin
    to generate the masks.
    """

    def __init__(self, ignore_dst=True, time=None, groupings=None):
        """
        Configure plugin options to generate the desired ancillary.

        Args:
            ignore_dst (bool):
                If True, find and use the UTC offset to a grid point ignoring
                daylights savings.
            time (str):
                A datetime specified in the format YYYYMMDDTHHMMZ at which to
                calculate the mask. If daylights savings are ignored this will
                have no impact on the resulting masks.
            groupings (dict):
                A dictionary specifying how timezones should be grouped if so
                desired. This dictionary takes the form::

                    {0: [-12, -5], 1:[-4, 4], 2: [5, 12]}

                The numbers in the lists denote the inclusive limits of the
                groups.
        """
        self.tf = TimezoneFinder()
        self.time = time
        self.ignore_dst = ignore_dst
        self.groupings = groupings

    def _set_time(self, cube):
        """
        Set self.time to a datetime object specifying the date and time for
        which the masks should be created.

        Args:
            cube (iris.cube.Cube):
                The cube from which the validity time should be taken is one
                has not been explicitly provided by the user.
        """
        if self.time:
            self.time = datetime.strptime(self.time, "%Y%m%dT%H%MZ")
        else:
            try:
                self.time = cube.coord("time").cell(0).point
            except CoordinateNotFoundError:
                msg = (
                    "The input cube does not contain a 'time' coordinate. "
                    "As such a time must be provided by the user."
                )
                raise ValueError(msg)

    @staticmethod
    def _get_coordinate_pairs(cube):
        """
        Create an array containing all the pairs of coordinates that describe
        y-x points in the grid.

        Args:
            cube (iris.cube.Cube):
                The cube from which the y-x grid is being taken.
        Returns:
            numpy.array:
                A numpy array containing all the pairs of coordinates that describe
                the y-x points in the grid. This array is 2-dimensional, being
                2 by the product of the grid's y-x dimension lengths.
        """
        if lat_lon_determine(cube) is not None:
            yy, xx = transform_grid_to_lat_lon(cube)
        else:
            latitudes = cube.coord("latitude").points
            longitudes = cube.coord("longitude").points

            # timezone finder works using -180 to 180 longitudes.
            if (longitudes > 180).any():
                longitudes[longitudes > 180] -= 360
                if (longitudes > 180 or longitudes < -180).any():
                    raise ValueError("Nope")
            yy, xx = np.meshgrid(latitudes, longitudes, indexing='ij')

        return np.array([yy.flatten(), xx.flatten()])

    def _calculate_tz_offsets(self, coordinate_pairs):
        """
        Loop over all the coordinates provided and for each calculate the
        offset from UTC in seconds.

        Args:
            coordinate_pairs (numpy.array):
                A numpy array containing all the pairs of coordinates that describe
                the y-x points in the grid. This array is 2-dimensional, being
                2 by the product of the grid's y-x dimension lengths.
        Returns:
            numpy.array:
                A 1-dimensional array of grid offsets with a length equal
                to the product of the grid's y-x dimension lengths.
        """
        grid_offsets = []
        for ii, (latitude, longitude) in enumerate(coordinate_pairs.T):
            point_tz = self._get_timezone(latitude, longitude)

            if point_tz is None:
                grid_offsets.append(3600 * longitude / 15)
            else:
                grid_offsets.append(self._calculate_offset(point_tz))

        return np.array(grid_offsets)

    def _get_timezone(self, latitude, longitude):
        """
        Args:
            latitude, longitude (float):
                The latitude and longitude for which a timezone should be identified.
        Returns:
            str or None:
                The string representation of the timezone for a point if the
                point has a defined timezone (land), e.g. "America/Chicago",
                otherwise return None.
        """
        point_tz = self.tf.certain_timezone_at(lng=longitude, lat=latitude)
        return point_tz

    def _calculate_offset(self, point_tz):
        """
        Calculates the offset in seconds from UTC for a given timezone, either
        with or without consideration of daylights savings.

        Args:
            point_tz (str):
                The string representation of the timezone for a point
                e.g. "America/Chicago",
        Returns:
            int:
                Timezone offset from UTC in seconds.
        """
        target = timezone(point_tz)
        offset = target.utcoffset(self.time)

        dst = 0
        if self.ignore_dst:
            dst = target.dst(self.time)
            # The timezone for Ireland does not capture DST:
            # https://github.com/regebro/tzlocal/issues/80
            if point_tz == "Europe/Dublin":
                dst = timezone("Europe/London").dst(self.time)

        return int((offset - dst).total_seconds())

    def _group_timezones(self, timezone_mask):
        """
        If the ancillary will be used with data that is not available at hourly
        intervals, the masks can be grouped to match the intervals of the data.
        For example, 3-hourly interval data might group UTC offsets:

            [-12, -11], [-10, -9, -8], [-7, -6, -5], etc.

        Args:
            timezone_mask (iris.cube.CubeList):
                A cube list containing a mask cube for each UTC offset that
                has been found necessary.
        Returns:
            iris.cube.CubeList:
                A cube list containing cubes created by blending together
                different UTC offset cubes to create larger masked regions.
        """
        grouped_timezone_masks = iris.cube.CubeList()
        for group in self.groupings.values():
            constraint = iris.Constraint(
                UTC_offset=lambda cell: group[0] <= cell <= group[-1]
            )
            subset = timezone_mask.extract(constraint)
            subset = subset.merge_cube()
            grouped_timezone_masks.append(
                subset.collapsed("UTC_offset", iris.analysis.MIN)
            )
        return grouped_timezone_masks

    def process(self, cube):
        """
        Use the grid from the provided cube to create masks that correspond to
        all the timezones that exist within the cube. These masks are then
        returned in a single cube with a leading UTC_offset coordinate that
        differentiates between them.

        Args:
            cube (iris.cube.Cube):
                A cube with the desired grid. If no 'time' is specified in
                the plugin configuration the time on this cube will be used
                for determining the UTC offsets (this is only relevant if
                daylights savings times are being included).
        Returns:
            iris.cube.Cube:
                A timezone mask cube.
        """
        self._set_time(cube)
        coordinate_pairs = self._get_coordinate_pairs(cube)
        grid_offsets = self._calculate_tz_offsets(coordinate_pairs)
        # Model data is hourly, so we need offsets at hourly fidelity
        grid_offsets = np.around(grid_offsets / 3600).astype(np.int32)

        # Reshape the flattened array back into the original cube shape.
        grid_offsets = grid_offsets.reshape(cube.shape)

        min_offset = grid_offsets.min()
        max_offset = grid_offsets.max()

        attributes = generate_mandatory_attributes([cube])
        template_cube = create_new_diagnostic_cube("timezone_mask", 1, cube, attributes)

        # Create a cube containing the timezone UTC offset information.
        timezone_mask = iris.cube.CubeList()
        for offset in range(min_offset, max_offset + 1):
            zone = (grid_offsets != offset).astype(np.int32)
            coord = iris.coords.DimCoord([offset], long_name="UTC_offset")
            tz_slice = template_cube.copy(data=zone)
            tz_slice.add_aux_coord(coord)
            timezone_mask.append(tz_slice)

        if self.groupings:
            timezone_mask = self._group_timezones(timezone_mask)

        timezone_mask = timezone_mask.merge_cube()
        return timezone_mask
