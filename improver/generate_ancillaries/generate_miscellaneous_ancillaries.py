# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""A module for functions that generate ancillary cubes."""

import numpy as np
from geopandas import GeoDataFrame
from iris import Constraint
from iris.cube import Cube, CubeList

from improver.generate_ancillaries.generate_distance_to_feature import DistanceTo
from improver.spotdata.build_spotdata_cube import build_spotdata_cube
from improver.spotdata.neighbour_finding import NeighbourSelection
from improver.spotdata.spot_extraction import SpotExtraction
from improver.spotdata.utilities import extract_site_json
from improver.utilities.spatial import (
    check_if_grid_is_equal_area,
    distance_to_number_of_grid_cells,
)


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

    distance_to_coastline = DistanceTo(
        epsg_projection, new_name="distance_to_coastline"
    )(site_cube, coastline)

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
    land_cover_cube: Cube, neighbour_cube: Cube, radius: int = 2500
) -> Cube:
    """Generate a land area fraction ancillary cube at the site locations by utilising
    the Corine Land cover.

    The Corine Land cover is available from
    https://doi.org/10.2909/960998c1-1870-4e82-8051-6485205ebbac. This
    function requires the land cover file is provided as an iris cube.

    A neighbour cube is generated on the native grid of the input land cover cube.
    This allows us to select the cell, on what ever resolution this data is provided,
    that is closest to each site location. A box can then be formed about this cell
    of a given size and the fraction of points within the box that are classified as
    land can be counted. The returned value is the fraction of the total box size
    that is classified as land on the input land cover grid.

    Args:
        land_cover_cube:
            A cube containing the Corine Land cover data. The data values should be
            integers representing
            different land cover types.
        neighbour_cube:
            A cube containing information about the spot data sites. We use this rather
            than a site list as it contains a completed set of altitudes which have
            been extracted from orography data on the model domain. These fill in
            where the site source data may be missing altitude information.
        radius:
            The radius in metres of the box about each site location to use to calculate
            the land area fraction. The default value of 2500m gives a box of
            approximately 5km x 5km.

    Returns:
        A cube containing the land area fraction at the site locations.
    """
    # Determine the grid resolution of the land cover data in metres
    check_if_grid_is_equal_area(land_cover_cube)
    cell_radius = distance_to_number_of_grid_cells(land_cover_cube, radius)

    # Ensure we are working with an x/y grid only
    xaxis, yaxis = land_cover_cube.coord(axis="x"), land_cover_cube.coord(axis="y")
    land_cover_cube = next(land_cover_cube.slices([xaxis, yaxis]))

    # 41-44 is the key for water in the Corine Land cover dataset.
    land_mask = land_cover_cube.copy(data=np.where(land_cover_cube.data > 40, 0, 1))
    # Oceans far from the coast have data value -128 to represent no data
    land_mask.data = np.where(land_cover_cube.data < 0, 0, land_mask.data)
    # 48 is the key for complex land surfaces in Corine
    land_mask.data = np.where(land_cover_cube.data == 48, 1, land_mask.data)

    # Extract the site definitions from the provided neighbour cube.
    site_definitions = extract_site_json(neighbour_cube)

    # If a unique site id is present, we need to extract the name for reuse.
    default_keys = ["altitude", "latitude", "longitude", "wmo_id"]
    try:
        (unique_site_id_key,) = [
            key for key in site_definitions[0].keys() if key not in default_keys
        ]
    except ValueError:
        unique_site_id_key = None

    # Find nearest grid point to each site on the land cover grid.
    neighbour_plugin = NeighbourSelection(unique_site_id_key=unique_site_id_key)
    # We are using the land cover data here as both orography and landmask,
    # but we don't need any orographic information for a nearest neighbour
    # selection and with a guaranteed set of complete site altitudes provided
    # by the neighbour cube, so this is okay.
    neighbours = neighbour_plugin(site_definitions, land_mask, land_mask)

    kwargs = (
        {"unique_site_id": 0, "unique_site_id_key": unique_site_id_key}
        if unique_site_id_key is not None
        else {}
    )
    template = build_spotdata_cube(
        0, "land_area_fraction", 1, 0, 0, 0, "00000", **kwargs
    )
    # Iris does not concatenate variable length strings, so we need to set
    # types explicitly in each site cube to ensure these can be concatenated.
    crd_types = {crd.name(): neighbours.coord(crd).dtype for crd in template.coords()}

    # Constraints to allow us to extract the x and y grid indices.
    x_index_con = Constraint(grid_attributes_key="x_index")
    y_index_con = Constraint(grid_attributes_key="y_index")

    land_fraction = CubeList()
    for site in neighbours.slices_over("spot_index"):
        ix, iy = (
            site.extract(x_index_con).data.astype(int).item(),
            site.extract(y_index_con).data.astype(int).item(),
        )
        subset = land_mask.data[
            max(0, ix - cell_radius) : ix + cell_radius + 1,
            max(0, iy - cell_radius) : iy + cell_radius + 1,
        ]
        fraction = subset.sum() / subset.size

        site_frac = template.copy(data=np.array([fraction], dtype=np.float32))
        for crd, crd_type in crd_types.items():
            site_frac.coord(crd).points = site.coord(crd).points.astype(crd_type)
        land_fraction.append(site_frac)

    return land_fraction.concatenate_cube()
