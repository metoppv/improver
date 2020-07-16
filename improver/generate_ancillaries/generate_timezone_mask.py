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
import json
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

    """

    def __init__(self, ignore_dst=True, time=None, groupings=None):

        self.tf = TimezoneFinder()
        self.time = time
        self.ignore_dst = ignore_dst
        self.groupings = groupings

    def _set_time(self, cube):
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

    def get_timezone(self, latitude, longitude):
        """
        Args:
            latitude, longitude (float):
                The latitude and longitude for which a timezone should be identified.
        Returns:
            int:
                The UTC offset in integer hours of the latitude and longitude provided.
                This offset is not date dependent, i.e. it ignores daylights savings.
        """
        point_tz = self.tf.certain_timezone_at(lng=longitude, lat=latitude)
        return point_tz

    def calculate_offset(self, point_tz):
        """
        calculates the offset in seconds for a given timezone, either with or
        without consideration of daylights savings.
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

    def calculate_tz_offsets(self, coordinate_pairs):

        grid_offsets = []
        for ii, (latitude, longitude) in enumerate(coordinate_pairs.T):
            point_tz = self.get_timezone(latitude, longitude)

            if point_tz is None:
                grid_offsets.append(3600 * longitude / 15)
            else:
                grid_offsets.append(self.calculate_offset(point_tz))

        return np.array(grid_offsets)

    @staticmethod
    def get_coordinate_pairs(cube):

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
            yy, xx = np.meshgrid(latitudes, longitudes)

        return np.array([yy.flatten(), xx.flatten()])

    def group_timezones(self, timezone_mask):

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
        Args:
            cube (iris.cube.Cube):
                A cube with the desired grid.
        Returns:
            iris.cube.Cube:
                A timezone mask cube.
        """
        self._set_time(cube)
        coordinate_pairs = self.get_coordinate_pairs(cube)
        grid_offsets = self.calculate_tz_offsets(coordinate_pairs)

        # Model data is hourly, so we need offsets at hourly fidelity
        grid_offsets = np.around(grid_offsets / 3600).astype(np.int32)

        # Reshape the flattened array back into the original cube shape.
        grid_offsets = grid_offsets.reshape(cube.shape, order="F")

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
            timezone_mask = self.group_timezones(timezone_mask)

        timezone_mask = timezone_mask.merge_cube()
        return timezone_mask
