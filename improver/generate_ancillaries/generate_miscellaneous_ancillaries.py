# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""A module for functions that generate ancillary cubes."""

import numpy as np
from geopandas import GeoDataFrame
from iris.cube import Cube, CubeList

from improver.generate_ancillaries.generate_distance_to_feature import DistanceTo
from improver.nbhood.nbhood import MetaNeighbourhood
from improver.regrid.landsea import RegridLandSea
from improver.spotdata.spot_extraction import SpotExtraction


def generate_distance_to_ocean(
    epsg_projection: int, coastline: GeoDataFrame, land: GeoDataFrame, site_cube: Cube
) -> Cube:
    """Generate a distance to ocean ancillary cube. The DistanceTo plugin can't be used
    directly because there isn't a GeoDataFrame for the ocean.

    The DistanceTo class is used with the coastline GeoDataframe to calculate the
    distance of each site to the coastline. The DistanceTo class is also used to
    calculate the distance to land for each site. This identifies which sites are
    land and which are ocean. Sites in the ocean can then be set to 0m as they are in
    the ocean.

    Args:
        epsg_projection:
            The EPSG code of the coordinate reference system on to which latitude
            and longitudes will be projected to calculate distances. This is
            a projected coordinate system in which distances are measured in metres,
            for example a Lambert Azimuthal Equal Areas projection across the UK,
            code 3035.
        coastline:
            A GeoDataFrame containing the coastline geometry.
        land:
            A GeoDataFrame containing the land geometry.
        site_cube:
            A cube containing the site locations. There must be latitude and longitude
            coordinates.
    Returns:
        A cube containing the distance to ocean ancillary data.
    """

    distance_to_coastline = DistanceTo(epsg_projection, new_name="distance_to_coastline")(
        site_cube, coastline
    )

    # As we only care about identifying sites on land (i.e. 0m) we can use a small buffer
    # to speed up the calculation.
    distance_to_land = DistanceTo(
        epsg_projection, new_name="distance_to_land", buffer=10, clip_geometry_flag=True
    )(site_cube, land)

    # Set the distance to ocean to 0 for sites that are in the ocean
    distance_to_ocean = distance_to_coastline.copy(data=distance_to_coastline.data)
    distance_to_ocean.data[distance_to_land.data != 0] = 0
    distance_to_ocean.rename("distance_to_ocean")

    return distance_to_ocean


def generate_distance_to_water(distance_to_water_feature: CubeList) -> Cube:
    """Generate a distance to water ancillary cube. The distance to water is the minimum
    of all the provided distance to water features, such as rivers, lakes, and oceans.

    The first cube in distance_to_water_feature is used as a template for the output
    metadata with the name updated to "distance_to_water".

    Args:
        distance_to_water_feature:
            A CubeList containing distance to water features from sites (i.e. rivers,
            lakes, and oceans).
            Each cube should have the same set of sites defined.
    Returns:
        A cube containing the distance to water ancillary data.
    """

    # Calculate the minimum distance to water

    distances_to_features = np.stack([cube.data for cube in distance_to_water_feature])
    min_distance = np.min(distances_to_features, axis=0)

    # Create a new cube for the distance to water
    distance_to_water = distance_to_water_feature[0].copy(data=min_distance)
    distance_to_water.rename("distance_to_water")

    return distance_to_water


def generate_roughness_length_at_sites(
    roughness_length: Cube, neighbour_cube: Cube
) -> Cube:
    """Generate a roughness length ancillary cube at the site locations. This performs a
    spot extraction of the roughness length data at the site locations and removes time
    related coordinates.

    Args:
        roughness_length:
            A cube containing the roughness length data.
        neighbour_cube:
            A cube containing information about the spot data sites and
            their grid point neighbours.
    Returns:
        A cube containing the roughness length at the site locations.
    """
    roughness_length_spot = SpotExtraction(neighbour_selection_method="nearest")(
        neighbour_cube, roughness_length
    )

    # Update metadata to remove any time coordinates
    cube_coord = [coord.name() for coord in roughness_length_spot.coords()]

    time_coordinates = ["time", "forecast_reference_time", "forecast_period"]
    for coord in time_coordinates:
        if coord in cube_coord:
            roughness_length_spot.remove_coord(coord)
    return roughness_length_spot


def generate_land_area_fraction_at_sites(
    land_cover_cube: Cube, gridded_template_cube: Cube, neighbour_cube: Cube
) -> Cube:
    """Generate a land area fraction ancillary cube at the site locations by utilising
    the Corine Land cover.

    The Corine Land cover is available from
    https://doi.org/10.2909/960998c1-1870-4e82-8051-6485205ebbac. This
    function requires the land cover file is provided as an iris cube.

    The land_cover cube is first regridded to be on the same projection as the provided
    gridded template. A neighbourhood is then applied to the regridded land cover cube
    to calculate the land area fraction at each grid point. The neighbourhood radius is
    set to 2500m. The land area fraction is then extracted at the site locations using
    the neighbour_cube.

    Args:
        land_cover_cube:
            A cube containing the Corine Land cover data. The data values should be
            integers representing
            different land cover types.
        gridded_template_cube:
            A cube containing the gridded template to regrid the land cover cube to.
        neighbour_cube:
            A cube containing information about the spot data sites and their grid point
            neighbours.

    Returns:
        A cube containing the land area fraction at the site locations.
    """

    # 41-44 is the key for water in the Corine Land cover dataset.
    land_mask = land_cover_cube.copy(data=np.where(land_cover_cube.data > 40, 0, 1))
    # Oceans far from the coast have data value -128 to represent no data
    land_mask.data = np.where(land_cover_cube.data < 0, 0, land_mask.data)
    # 48 is the key for complex land surfaces in Corine
    land_mask.data = np.where(land_cover_cube.data == 48, 1, land_mask.data)

    # regrid the land mask to the same projection as the gridded template. This may lead
    # to grid squares not being 1 or 0 but as the land mask will be neighbourhooded this
    # will not be an issue.
    land_cover_cube_uk_regrid = RegridLandSea()(
        cube=land_mask, target_grid=gridded_template_cube
    )

    # Neighbourhood around each grid point to get a land area fraction
    land_mask_neighbourhood = MetaNeighbourhood(
        neighbourhood_output="probabilities", radii=2500
    )(cube=land_cover_cube_uk_regrid)

    spot_extracted_land_fraction = SpotExtraction(neighbour_selection_method="nearest")(
        neighbour_cube=neighbour_cube, diagnostic_cube=land_mask_neighbourhood
    )

    # Update metadata
    spot_extracted_land_fraction.rename("land_area_fraction")

    spot_extracted_land_fraction.attributes = {
        k: v
        for k, v in spot_extracted_land_fraction.attributes.items()
        if k == "conventions"
    }

    cube_coord = [coord.name() for coord in spot_extracted_land_fraction.coords()]

    coordinates_to_remove = ["band", "spatial_ref"]
    for coord in coordinates_to_remove:
        if coord in cube_coord:
            spot_extracted_land_fraction.remove_coord(coord)

    return spot_extracted_land_fraction
