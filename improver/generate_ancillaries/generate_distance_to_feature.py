# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""A module for generating a distance to feature ancillary cube."""

from typing import List, Optional, Tuple

import pyproj
from geopandas import GeoDataFrame, GeoSeries, clip
from iris.cube import Cube
from numpy import array, min, round
from shapely.geometry import Point

from improver import BasePlugin


class DistanceTo(BasePlugin):
    """
    Plugin to calculate the distance to the nearest feature in a geometry from
    sites to the nearest metre.

    Given a cube containing site locations and a GeoDataFrame the distance to each site
    from the nearest point of the feature geometry is calculated. This is done by converting
    the feature geometry and sites to a common target orography that must be specified using a
    European Petroleum Survey Group (EPSG) code that identifies the projection. For the
    UK code 3035, that provides a Lambert Azimuthal Equal Areas projection across the
    region might be used. The chosen projection should match the projection on which the
    ancillary will be used. The distance method from Shapely is used to find the distance
    from each site to every point in the feature geometry. The minimum of these distances is
    returned as the distance to the nearest feature in the feature geometry and this is rounded
    to the nearest metre.

    If requested, the provided geometry will be clipped to the bounds of the site
    locations with a buffer to improve performance. This is useful when the geometry is
    large and it would be expensive to calculate the distance to all features in the
    geometry but information may be lost at the edges of the domain.
    """

    def __init__(
        self,
        epsg_projection: int,
        new_name: Optional[str] = None,
        buffer: float = 30000,
        clip_geometry_flag: bool = False,
    ) -> None:
        """
        Initialise the DistanceTo plugin.

        Args:
            epsg_projection:
                The EPSG code of the coordinate reference system on to which latitude
                and longitudes will be projected to calculate distances. This is
                a projected coordinate system in which distances are measured in metres,
                for example a Lambert Azimuthal Equal Areas projection across the UK,
                code 3035.
            new_name:
                The name of the output cube.
            buffer:
                A buffer distance in m. If the geometry is clipped, this distance will
                be added onto the outermost site locations to define the domain to clip
                the geometry to.
            clip_geometry_flag:
                A flag to indicate whether the geometry should be clipped to the bounds of
                the site locations with a buffer distance added to the bounds. If set to
                False, the full geometry will be used to calculate the distance to the
                nearest feature.
        """
        self.epsg_projection = epsg_projection
        self.new_name = new_name
        self.buffer = buffer
        self.clip_geometry_flag = clip_geometry_flag

    @staticmethod
    def get_clip_values(points: List[float], buffer: float) -> List[float]:
        """Get the coordinates to use when clipping the geometry. This is determined by
        finding the maximum and minimum coordinate points from a list. A buffer distance
        may then be added/subtracted to the max/min.

        Args:
            points:
                A list of points to find the min and max from.
            buffer:
                The buffer distance to add/subtract to the max/min points.
        Returns:
            A list containing the maximum and minimum points with the buffer added
            or subtracted respectively."""

        ordered_points = sorted(set(points))
        min_point = ordered_points[0] - buffer
        max_point = ordered_points[-1] + buffer

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
            ValueError: If the clipped geometry is empty after clipping with the
            provided bounds."""

        clipped_geometry = clip(
            geometry, mask=[bounds_x[0], bounds_y[0], bounds_x[1], bounds_y[1]]
        )
        if clipped_geometry.empty:
            raise ValueError(
                "Clipping the geometry with a buffer size of "
                f"{self.buffer}m has produced an empty geometry. Either "
                "increase the buffer size or set clip_geometry_flag to "
                "False to use the full geometry."
            )

        return clipped_geometry

    def check_target_crs(self, site_points: GeoSeries):
        """Check that the provided target projection is suitable for the sites
        being used.

        Args:
            site_points:
                A GeoSeries containing the site points.

        Raises:
            ValueError: If the provided target coordinate reference system is not
                        suitable for the site points.
        """
        x_bounds = self.get_clip_values(site_points.x, 0)
        y_bounds = self.get_clip_values(site_points.y, 0)

        target_crs = pyproj.CRS.from_epsg(self.epsg_projection)
        x_min, y_min, x_max, y_max = target_crs.area_of_use.bounds

        valid_target = (
            x_bounds[0] > x_min
            and x_bounds[1] < x_max
            and y_bounds[0] > y_min
            and y_bounds[1] < y_max
        )
        if not valid_target:
            raise ValueError(
                "The provided projection defined by EPSG code "
                f"{self.epsg_projection} is not suitable for the site "
                "locations provided. Limits of this domain are: "
                f"x: {x_min} to {x_max}, y: {y_min} to {y_max}, whilst "
                f"the site locations are bounded by x: {x_bounds[0]} to {x_bounds[1]}, "
                f"y: {y_bounds[0]} to {y_bounds[1]}."
            )

    def project_geometry(
        self, geometry: GeoDataFrame, site_cube: Cube
    ) -> Tuple[GeoSeries, GeoDataFrame]:
        """Project the geometry and site cube to the target projection.

        Args:
            geometry:
                The geometry to reproject.
            site_cube:
                The cube containing the site locations. It is assumed that the site
                coordinates are defined as latitude and longitude on a WGS84
                coordinate system (EPSG:4326).
        Returns:
            A tuple containing the projected site locations and geometry."""

        x_points = site_cube.coord(axis="x").points
        y_points = site_cube.coord(axis="y").points

        site_points_list = [Point(x, y) for x, y in zip(x_points, y_points)]
        # Assumes site coordinates are defined as latitude and longitude
        # defaulting to an EPSG:4326 coordinate system, which is WGS84.
        site_points = GeoSeries(site_points_list, crs=4326)

        # Check that the provided target projection is suitable for the site points
        self.check_target_crs(site_points)

        geometry_reprojection = geometry.to_crs(self.epsg_projection)
        site_points = site_points.to_crs(self.epsg_projection)
        return site_points, geometry_reprojection

    def distance_to(self, site_points: GeoSeries, geometry: GeoDataFrame) -> List[int]:
        """Calculate the distance from each site point to the nearest feature in the
        geometry.

        Args:
            site_points:
                A GeoSeries containing the site points in the target projection.
            geometry:
                A GeoDataFrame containing the geometry in the target projection.
        Returns:
            A list of distances from each site point to the nearest feature in the
            geometry rounded to the nearest metre."""
        distance_results = []
        for point in site_points:
            distance_to_nearest = min(point.distance(geometry.geometry))
            distance_results.append(round(distance_to_nearest))

        return distance_results

    def create_output_cube(self, site_cube: Cube, data: List[int]) -> Cube:
        """Create an output cube that will have the same metadata as the input site
        cube except the units are changed to metres and, if requested, the name of the
        output cube will be changed.

        Args:
           site_cube:
               The input cube containing site locations that are defined by latitude
               and longitude coordinates.
           data:
               A list of distances from each site point to the nearest feature in the
               geometry.
           Returns:
               A new cube containing the distances with the same metadata as input site
               cube but with updated units and name."""

        output_cube = site_cube.copy(data=array(data))
        if self.new_name:
            output_cube.rename(self.new_name)
        output_cube.units = "m"

        return output_cube

    def process(self, site_cube: Cube, geometry: GeoDataFrame) -> Cube:
        """Generate a cube of the distance from sites in site_cube to the
        nearest point in geometry.

        The latitude, longitude coordinates in the site_cube are extracted
        to define the location of the sites. The sites and feature geometry
        are reprojected to the target projection.

        If requested the feature geometry will be clipped to the smallest square possible
        such that all sites in site_cube are included. A buffer distance is then added to each
        edge of the square which defines the size the feature geometry will be clipped to. This
        is useful when the feature geometry size is large and it would be expensive to calculate
        the distance to all features in the geometry or where the domain of the feature geometry
        is much larger than the area containing the site locations. Information may be lost at the edges of
        the domain if the feature is sparsely located in the geometry.

        The distance from each site to every point in the feature geometry is then calculated
        and the minimum of these distances is returned. The distances are rounded to the
        nearest metre.

        The output cube will have the same metadata as the input site_cube except the
        units will be changed to meters and, if requested, the name of the output cube
        will be updated.

        Args:
            site_cube:
                The input cube containing site locations. This cube must have x and y
                axis which contain the site coordinates in latitude and longitude.
            geometry:
                The GeoDataFrame containing the geometry to calculate distances to.

        Returns:
            A new cube containing the distances from each site to the nearest feature
            in the geometry rounded to the nearest metre."""

        # Project the geometry and site cube coordinates to the target projection.
        site_coords, geometry_projection = self.project_geometry(geometry, site_cube)

        if self.clip_geometry_flag:
            # Clip the geometry to the bounds of the site coordinates with a buffer if
            # requested
            x_bounds = self.get_clip_values(site_coords.x, self.buffer)
            y_bounds = self.get_clip_values(site_coords.y, self.buffer)

            clipped_geometry = self.clip_geometry(
                geometry_projection, x_bounds, y_bounds
            )
        else:
            clipped_geometry = geometry_projection

        # Calculate the distance to the nearest feature in the geometry
        distance_to_results = self.distance_to(site_coords, clipped_geometry)

        return self.create_output_cube(site_cube, distance_to_results)
