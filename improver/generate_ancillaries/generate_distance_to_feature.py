# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""A module for generating a distance to feature ancillary cube."""

import os
from typing import List, Optional, Tuple, Union

import numpy as np
import pyproj
from geopandas import GeoDataFrame, GeoSeries, clip
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube
from numpy import array, min, round
from shapely.geometry import Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from improver import BasePlugin


class DistanceTo(BasePlugin):
    """
    Plugin to calculate the distance to the nearest feature in a geometry from
    sites to the nearest metre.

    Given a cube containing site locations and a GeoDataFrame, the distance to each site
    from the nearest point of the feature geometry is calculated. This is done by converting
    the feature geometry and sites to a common target orography that must be specified using a
    European Petroleum Survey Group (EPSG) code that identifies the projection. For the
    UK, EPSG code 3035 may be used to provide a Lambert Azimuthal Equal Area projection that
    is suitable for the region. The chosen projection should match the projection on which the
    ancillary will be used. The distance method from Shapely is used to find the distance
    from each site to every point in the feature geometry. The minimum of these distances is
    returned as the distance to the nearest feature in the feature geometry and this is rounded
    to the nearest metre.

    If requested, the provided geometry will be clipped to the bounds of the site
    locations with a buffer to improve performance by reducing computation. This is useful when
    the geometry is large and it would be expensive to calculate the distance to all features
    in the geometry but information may be lost at the edges of the domain.
    """

    def __init__(
        self,
        epsg_projection: int,
        new_name: Optional[str] = None,
        angle_pairs: Optional[List[Tuple[float, float]]] = None,
        buffer: float = 30000,
        clip_geometry_flag: bool = False,
        parallel: bool = False,
        n_parallel_jobs: Optional[int] = len(os.sched_getaffinity(0)),
    ) -> None:
        """
        Initialise the DistanceTo plugin.

        Args:
            epsg_projection:
                The EPSG code of the coordinate reference system on to which latitudes
                and longitudes will be projected to calculate distances. This is
                a projected coordinate system in which distances are measured in metres,
                for example, EPSG code 3035, which defines a Lambert Azimuthal Equal
                Areas projection suitable for the UK.
            new_name:
                The name of the output cube.
            angle_pairs:
                Optional list of angular sector bounds in degrees clockwise from
                true north. Each element should be (start_angle, end_angle).
                If provided, the shortest distance to geometry in each sector is
                returned as an additional leading dimension on the output cube.
            buffer:
                A buffer distance in m. If the geometry is clipped, this distance will
                be added onto the outermost site locations to define the domain to clip
                the geometry to.
            clip_geometry_flag:
                A flag to indicate whether the geometry should be clipped to the bounds of
                the site locations with a buffer distance added to the bounds. If set to
                False, the full geometry will be used to calculate the distance to the
                nearest feature.
            parallel:
                A flag to indicate whether to use parallel processing when calculating
                distances.
            n_parallel_jobs:
                The number of parallel jobs to use when calculating distances.
                By default, os.sched_getaffinity(0) is used to give the number of cores
                that the process is eligible to use.
        """
        self.epsg_projection = epsg_projection
        self.new_name = new_name
        self.angle_pairs = self.validate_angle_pairs(angle_pairs)
        self.buffer = buffer
        self.clip_geometry_flag = clip_geometry_flag
        self.parallel = parallel
        self.n_parallel_jobs = n_parallel_jobs

    @staticmethod
    def validate_angle_pairs(
        angle_pairs: Optional[List[Tuple[float, float]]],
    ) -> Optional[List[Tuple[float, float]]]:
        """Validate sector angle pairs.

        Args:
            angle_pairs:
                Optional list of 2-tuples of angles in degrees.

        Returns:
            Validated angle pairs.

        Raises:
            ValueError:
                If any angle pair is malformed.
        """

        if angle_pairs is None:
            return None

        validated_pairs = []
        for pair in angle_pairs:
            if len(pair) != 2:
                raise ValueError(
                    "Each item in angle_pairs must contain exactly two angles."
                )
            start_angle, end_angle = pair
            if start_angle < 0 or start_angle > 360 or end_angle < 0 or end_angle > 360:
                raise ValueError("Angles in angle_pairs must be between 0 and 360.")
            if start_angle == end_angle and not (start_angle == 0 and end_angle == 360):
                raise ValueError(
                    "Identical start and end angles are invalid unless using [0, 360]."
                )
            validated_pairs.append((float(start_angle), float(end_angle)))

        return validated_pairs

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

    @staticmethod
    def _create_sector_geometry(
        point: Point,
        angle_pair: Tuple[float, float],
        max_radius: float,
        n_arc_points: int = 180,
    ) -> Optional[BaseGeometry]:
        """Create a sector polygon for the given site point.

        Args:
            point:
                Site location in projected coordinates.
            angle_pair:
                (start_angle, end_angle) in degrees clockwise from true north.
            max_radius:
                Radius used to create the sector polygon in metres.
            n_arc_points:
                Number of arc points used to approximate curved boundaries.

        Returns:
            A Polygon representing the sector, or None for full-circle sectors.
        """

        start_angle, end_angle = angle_pair
        if start_angle == 0 and end_angle == 360:
            return None

        start_angle = start_angle % 360
        end_angle = end_angle % 360

        sector_parts = (
            [(start_angle, end_angle)]
            if end_angle > start_angle
            else [(start_angle, 360.0), (0.0, end_angle)]
        )

        polygons = []
        for part_start, part_end in sector_parts:
            angles = np.linspace(part_start, part_end, n_arc_points)
            arc_points = [
                (
                    point.x + max_radius * np.sin(np.deg2rad(angle)),
                    point.y + max_radius * np.cos(np.deg2rad(angle)),
                )
                for angle in angles
            ]
            polygons.append(
                Polygon([(point.x, point.y), *arc_points, (point.x, point.y)])
            )

        return unary_union(polygons)

    def distance_to(
        self,
        site_points: GeoSeries,
        geometry: GeoDataFrame,
        angle_pair: Optional[Tuple[float, float]] = None,
    ) -> List[float]:
        """Calculate the distance from each site point to the nearest feature in the
        geometry.

        Args:
            site_points:
                A GeoSeries containing the site points in the target projection.
            geometry:
                A GeoDataFrame containing the geometry in the target projection.
            angle_pair:
                Optional angular sector bounds, clockwise from true north.
        Returns:
            A list of distances from each site point to the nearest feature in the
            geometry rounded to the nearest metre."""

        bounds = geometry.total_bounds
        corners = np.array(
            [
                [bounds[0], bounds[1]],
                [bounds[0], bounds[3]],
                [bounds[2], bounds[1]],
                [bounds[2], bounds[3]],
            ]
        )

        def _distance_to_nearest(
            point: Point,
            geometry: GeoDataFrame,
            angle_pair: Optional[Tuple[float, float]] = None,
        ) -> float:
            """Calculate the distance from a point to the nearest feature in the
            geometry.
            Args:
                point:
                    A shapely Point object representing the site location.
                geometry:
                    A GeoDataFrame containing the geometry in the target projection.
                angle_pair:
                    Optional angular sector bounds, clockwise from true north.
            Returns:
                The distance from the point to the nearest feature in the geometry
                rounded to the nearest metre.
            """

            geometry_to_search = geometry.geometry
            if angle_pair is not None:
                max_radius = (
                    np.max(np.hypot(corners[:, 0] - point.x, corners[:, 1] - point.y))
                    + 1.0
                )
                sector_polygon = self._create_sector_geometry(
                    point, angle_pair, max_radius=max_radius
                )

                if sector_polygon is not None:
                    intersected = geometry_to_search.intersection(sector_polygon)
                    geometry_to_search = intersected[
                        ~intersected.is_empty & intersected.notnull()
                    ]
                    if geometry_to_search.empty:
                        return np.nan

            return round(min(point.distance(geometry_to_search)))

        if self.parallel:
            from joblib import Parallel, delayed

            parallel = Parallel(n_jobs=self.n_parallel_jobs, prefer="threads")
            output_generator = parallel(
                delayed(_distance_to_nearest)(point, geometry, angle_pair)
                for point in site_points
            )
            distance_results = list(output_generator)
        else:
            distance_results = []
            for point in site_points:
                distance_results.append(
                    _distance_to_nearest(point, geometry, angle_pair)
                )

        return distance_results

    def create_output_cube(
        self, site_cube: Cube, data: Union[List[float], List[List[float]]]
    ) -> Cube:
        """Create an output cube that will have the same metadata as the input site
        cube except the units are changed to metres and, if requested, the name of the
        output cube will be changed.

        Args:
           site_cube:
               The input cube containing site locations that are defined by latitude
               and longitude coordinates.
           data:
               Distances from each site point to the nearest feature in the geometry,
               optionally grouped by sector.
           Returns:
               A new cube containing the distances with the same metadata as input site
               cube but with updated units and name."""

        if self.angle_pairs is None:
            output_cube = site_cube.copy(data=array(data))
        else:
            sector_data = array(data)
            n_sectors = len(self.angle_pairs)
            sector_coord = DimCoord(
                np.arange(n_sectors, dtype=np.int32), long_name="sector", units="1"
            )
            dim_coords_and_dims = [(sector_coord, 0)]
            for coord in site_cube.dim_coords:
                (dim,) = site_cube.coord_dims(coord)
                dim_coords_and_dims.append((coord.copy(), dim + 1))

            aux_coords_and_dims = []
            for coord in site_cube.aux_coords:
                dims = site_cube.coord_dims(coord)
                if dims:
                    aux_coords_and_dims.append(
                        (coord.copy(), tuple(dim + 1 for dim in dims))
                    )
                else:
                    aux_coords_and_dims.append((coord.copy(), None))

            angle_means = []
            for pair in self.angle_pairs:
                if pair[0] > pair[1]:
                    angle_means.append((((pair[0] + pair[1]) / 2) - 180) % 360)
                else:
                    angle_means.append(np.mean(pair))

            aux_coords_and_dims.append(
                (
                    AuxCoord(
                        np.array(angle_means, dtype=np.float32),
                        bounds=np.array(self.angle_pairs, dtype=np.float32),
                        long_name="sector_angle_from_true_north",
                        units="degrees",
                    ),
                    0,
                )
            )

            output_kwargs = {
                "units": site_cube.units,
                "attributes": site_cube.attributes.copy(),
                "cell_methods": site_cube.cell_methods,
                "dim_coords_and_dims": dim_coords_and_dims,
                "aux_coords_and_dims": aux_coords_and_dims,
            }
            if site_cube.standard_name is not None:
                output_kwargs["standard_name"] = site_cube.standard_name
            else:
                output_kwargs["long_name"] = site_cube.long_name
            if site_cube.var_name is not None:
                output_kwargs["var_name"] = site_cube.var_name

            output_cube = Cube(sector_data, **output_kwargs)

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
        if self.angle_pairs is None:
            distance_to_results = self.distance_to(site_coords, clipped_geometry)
        else:
            distance_to_results = [
                self.distance_to(site_coords, clipped_geometry, angle_pair=angle_pair)
                for angle_pair in self.angle_pairs
            ]

        return self.create_output_cube(site_cube, distance_to_results)
