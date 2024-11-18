# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Module to contain standard grid projection definitions."""

from iris.coord_systems import GeogCS, LambertAzimuthalEqualArea

# Global grid coordinate reference system
GLOBAL_GRID_CCRS = GeogCS(6371229.0)

# Reference ellipsoid for Earth's geoid
ELLIPSOID = GeogCS(semi_major_axis=6378137.0, semi_minor_axis=6356752.314140356)

# Projection settings for UKVX standard grid
STANDARD_GRID_CCRS = LambertAzimuthalEqualArea(
    latitude_of_projection_origin=54.9,
    longitude_of_projection_origin=-2.5,
    false_easting=0.0,
    false_northing=0.0,
    ellipsoid=ELLIPSOID,
)

# Metadata for different spatial grids
GRID_COORD_ATTRIBUTES = {
    "latlon": {
        "xname": "longitude",
        "yname": "latitude",
        "units": "degrees",
        "coord_system": GLOBAL_GRID_CCRS,
        "default_grid_spacing": 10,
    },
    "equalarea": {
        "xname": "projection_x_coordinate",
        "yname": "projection_y_coordinate",
        "units": "metres",
        "coord_system": STANDARD_GRID_CCRS,
        "default_grid_spacing": 2000,
    },
}
