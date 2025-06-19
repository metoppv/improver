# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""A module for generating a distance to feature ancillary cube."""

from typing import List, Tuple

from geopandas import GeoDataFrame, GeoSeries, clip
from iris.cube import Cube
from numpy import array, min, round
from shapely.geometry import Point

from improver import BasePlugin


class DistanceTo(BasePlugin):
    """
    Plugin to calculate the distance to the nearest feature in a geometry from
    sites to the nearest metre.

    Given a cube containing site locations and a GeoDataFrame the distance to each site from
    the neareast point of the geometry is caluculated by converting the geometry and site cube
    to the Lambert azuthermal equal area projection and using the distance method from Shapeley
    to find the distance from each site to every point in the geometry. The minimum of these
    distances is returned as the distance to the nearest feature in the geometry and this is rounded
    to the nearest metre.

    If requested the provided geometry will be clipped to the bounds of the site locations with a buffer
    to improve performance. This is useful when the geometry is large and it would be expensive to calculate
    the distance to all features in the geometry but information may be lost at the edges of the domain.
    """

    def __init__(self, new_name: str = None, buffer: float = 30000, clip: bool = False):
        """Initialise the DistanceTo plugin.
        Args:
            new_name:
                The name of the output cube".
            buffer:
                A buffer distance in m. If the geometry is clipped this distance will be added onto
                the outermost site locations to define the domain to clip the geometry to.
        """
        self.new_name = new_name
        self.buffer = buffer
        self.clip = clip

    def clip_coordinates(self, points: List[float]) -> List[float]:
        """Get the coordinates to use when clipping the geometry. This is determined by finding the
        maximum and minimum coordinate points from a list. A buffer distance is then added/subtracted
        to the max/min .

        Args:
            points:
                A list of points to find the min and max from.
        Returns:
            A list containing the minimum and maximum points with the buffer subtracted or added respectively."""

        ordered_points = sorted(set(points))

        min_point = ordered_points[0] - self.buffer
        max_point = ordered_points[-1] + self.buffer

        return [min_point, max_point]

    def clip_geometry(
        self, geometry: GeoDataFrame, bounds_x: List[float], bounds_y: List[float]
    ) -> GeoDataFrame:
        """Clip the geometry to the provided bounds.

        Args:
            geometry:
                The geometry to clip.
            bounds_x:
                A list containing the minimum and maximum x coordinates.
            bounds_y:
                A list containing the minimum and maximum y coordinates.
        Returns:
            The clipped geometry.

        Raises:
            ValueError: If the clipped geometry is empty after clipping with the provided bounds."""

        clipped_geometry = clip(
            geometry, mask=[bounds_x[0], bounds_y[0], bounds_x[1], bounds_y[1]]
        )
        if clipped_geometry.empty:
            raise ValueError(f"""Clipping the geometry with a buffer size of {self.buffer}m has produced an empty geometry. Either
                             increase the buffer size or set clip to False to use the full geometry.""")

        return clipped_geometry

    def project_geometry(
        self, geometry: GeoDataFrame, site_cube: Cube
    ) -> Tuple[GeoSeries, GeoDataFrame]:
        """Project the geometry and site cube to Lambert azimuthal
        equal-area projection (EPSG:3035).

        Args:
            geometry:
                The geometry to reproject.
            site_cube:
                The cube containing the site locations. It is assumed that the site
                coordinates are defined as latitude and longitude.
        Returns:
            A tuple containing the projected site locations and geometry."""

        x_points = site_cube.coord(axis="x").points
        y_points = site_cube.coord(axis="y").points

        points = [Point(x, y) for x, y in zip(x_points, y_points)]

        projection_dict = {"latitude": 4326, "projection_y_coordinate": 3035}

        site_coord = site_cube.coord(axis="y").name()

        site_points = GeoSeries(points, crs=projection_dict[site_coord])
        geometry_reprojection = geometry.to_crs(3035)
        site_points = site_points.to_crs(3035)
        return site_points, geometry_reprojection

    def distance_to(self, site_points: GeoSeries, geometry: GeoDataFrame) -> List[int]:
        """Calculate the distance from each site point to the nearest feature in the geometry.
        Args:
            site_points:
                A GeoSeries containing the site points in a Lambert azimuthal equal-area projection.
            geometry:
                A GeoDataFrame containing the geometry geometry in a Lambert azimuthal equal-area projection.
        Returns:
            A list of distances from each site point to the nearest feature in the geometry rounded to the
            nearest metre."""
        distance_results = []
        for point in site_points:
            distance_to_nearest = min(point.distance(geometry.geometry))
            distance_results.append(round(distance_to_nearest))

        return distance_results

    def create_output_cube(self, site_cube: Cube, data: List[int]) -> Cube:
        """Create an output cube that will have the same metatdata as the input site cube except the units
        are changed to meters and, if requested, the name of the output cube will be changed
        Args:
           site_cube:
               The input cube containing site locations.
           data:
               A list of distances from each site point to the nearest feature in the geometry.
           Returns:
               A new cube containing the distances with the same metadata as input site cube but with
               updated units and name."""

        output_cube = site_cube.copy(data=array(data))
        if self.new_name:
            output_cube.rename(self.new_name)
        output_cube.units = "m"

        return output_cube

    def process(self, site_cube: Cube, geometry: GeoDataFrame) -> Cube:
        """Generate a cube of the distance from sites in site_cube to the nearest point in geometry.

        The latitude, longitude coordinates in the site_cube are extracted to define the location of the sites
        and these are projected to the Lambert azimuthal equal-area projection. The geometry is also projected to the same
        projection.

        If requested the geometry will be clipped to smallest square possible such that all sites in site_cube are included. A buffer
        distance is then added to each edge of the square which defines the size the geometry will be clipped to. This is useful when
        the geometry size is large and it would be expensive to calculate the distance to all features in the geometryor where the domain
        of the geometry is much larger than the site locations. Information may be lost at the edges of the domain if the feature is sparsely
        located in the geometry.

        The distance from each site to every point in the geometry is then calculated and the minimum of these
        distances is returned. The distances are rounded to the nearest metre.

        The output cube will have the same metadata as the input site_cube except the units will be changed to meters
        and, if requested, the name of the output cube will be updated.

        Args:
            site_cube:
                The input cube containing site locations. This cube must have x and y axis which contain
                the site coordinates in latitude and longitude.
            geometry:
                The GeoDataFrame containing the geometry to calculate distances to.
        Returns:
            A new cube containing the distances from each site to the nearest feature in the geometry rounded to the nearest metre."""

        # Project the geometry and site cube coordinates to Lambert azimuthal equal-area projection.
        site_coords, geometry_projection = self.project_geometry(geometry, site_cube)

        if self.clip:
            # Clip the geometry to the bounds of the site coordinates with a buffer if requested
            x_bounds = self.clip_coordinates(site_coords.x)
            y_bounds = self.clip_coordinates(site_coords.y)

            clipped_geometry = self.clip_geometry(
                geometry_projection, x_bounds, y_bounds
            )
        else:
            clipped_geometry = geometry_projection

        # Calculate the distance to the nearest feature in the geometry
        distance_to_results = self.distance_to(site_coords, clipped_geometry)

        return self.create_output_cube(site_cube, distance_to_results)
