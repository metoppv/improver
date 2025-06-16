# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""A module for creating orographic smoothing coefficients"""

from geopandas import GeoSeries, clip
from numpy import array, min
from shapely.geometry import Point

from improver import BasePlugin


class DistanceTo(BasePlugin):
    """
    Plugin to calculate the distance to the nearest feature in a shapefile from
    a selection of sites.

    If requested the shapefile will be clipped to the bounds of the site locations with a buffer
    to improve performance. This is useful when the shapefile is large and it would be expensive to calculate
    the distance to all features in the shapefile but information may be lost at the edges of the domain.
    """

    def __init__(self, new_name: str = None, buffer: float = 30000):
        """Initialise the DistanceTo plugin.
        Args:
            new_name:
                The name of the output cube".
            buffer:
                A buffer distance in m. If the shapefile is clipped this distance will be added onto
                the outermost site locations to define the domain to clip the shapefile to.
        """
        self.new_name = new_name
        self.buffer = buffer

    def clip_coordinates(self, points: list):
        """Get the maximum and minimum coordinate point from a list of points. Add/subtract a
        buffer to the max/min to reduce the information lost outside the domain.

        Args:
            points:
                A list of points to find the min and max from.
        Returns:
            A list containing the minimum and maximum points with the buffer subtracted/added."""

        ordered_points = sorted(set(points))

        min_point = ordered_points[0] - self.buffer
        max_point = ordered_points[-1] + self.buffer

        return [min_point, max_point]

    def clip_shapefile(self, shapefile, bounds_x, bounds_y):
        """Clip the shapefile to the provided bounds.

        Args:
            shapefile:
                The shapefile to clip.
            bounds_x:
                A list containing the minimum and maximum x coordinates.
            bounds_y:
                A list containing the minimum and maximum y coordinates.
        Returns:
            The clipped shapefile."""
        return clip(
            shapefile, mask=[bounds_x[0], bounds_y[0], bounds_x[1], bounds_y[1]]
        )

    def reproject_shapefile(self, shapefile, site_cube):
        """Reproject the shapefile and site cube to Lambert azimuthal
        equal-area projection (EPSG:3035).

        Args:
            shapefile:
                The shapefile to reproject.
            site_cube:
                The cube containing the site locations. It is assumed that the site
                corrdinates are either Latitude/Longitude or Lambert azimuthal equal-area projection.
        Returns:
            A tuple containing the reprojected site points and shapefile."""
        x_points = site_cube.coord(axis="x").points
        y_points = site_cube.coord(axis="y").points

        points = [Point(x, y) for x, y in zip(x_points, y_points)]

        projection_dict = {"latitude": 4326, "projection_y_coordinate": 3035}

        site_coord = site_cube.coord(axis="y").name()

        site_points = GeoSeries(points, crs=projection_dict[site_coord])
        shapefile_reprojection = shapefile.to_crs(3035)
        site_points = site_points.to_crs(3035)
        return site_points, shapefile_reprojection

    def distance_to(self, site_points, shapefile):
        distance_results = []
        for point in site_points:
            distance_results.append(min(point.distance(shapefile.geometry)))

        return distance_results

    def create_output_cube(self, site_cube, data):
        """Create an output cube with the distance data."""
        output_cube = site_cube.copy(data=array(data))
        output_cube.rename(self.new_name)
        output_cube.units = "m"

        return output_cube

    def process(self, site_cube, shapefile, clip=False):
        site_points, shapefile_reprojection = self.reproject_shapefile(
            shapefile, site_cube
        )

        if clip:
            x_bounds = self.clip_coordinates(site_points.x)
            y_bounds = self.clip_coordinates(site_points.y)

            clipped_shapefile = self.clip_shapefile(
                shapefile_reprojection, x_bounds, y_bounds
            )
        else:
            clipped_shapefile = shapefile_reprojection

        # Calculate the distance to the nearest feature in the shapefile
        distance_to_results = self.distance_to(site_points, clipped_shapefile)

        # Create a new cube to hold the distance results
        distance_to_cube = self.create_output_cube(site_cube, distance_to_results)
        return distance_to_cube
