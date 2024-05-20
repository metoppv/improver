# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of improver and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Module containing ancillary generation utilities for Improver"""

from typing import Any, Dict, List, Optional, Union

import iris
import numpy as np
from cf_units import Unit
from iris.cube import Cube, CubeList
from numpy import ndarray
from numpy.ma.core import MaskedArray

from improver import BasePlugin

# The following dictionary defines the default orography altitude bands in
# metres above/below sea level for which masks are required.
THRESHOLDS_DICT = {
    "bounds": [
        [-500.0, 50.0],
        [50.0, 100.0],
        [100.0, 150.0],
        [150.0, 200.0],
        [200.0, 250.0],
        [250.0, 300.0],
        [300.0, 400.0],
        [400.0, 500.0],
        [500.0, 650.0],
        [650.0, 800.0],
        [800.0, 950.0],
        [950.0, 6000.0],
    ],
    "units": "m",
}


def _make_mask_cube(
    mask_data: MaskedArray,
    coords: List,
    topographic_bounds: List[float],
    topographic_units: str,
    sea_points_included: bool = False,
) -> Cube:
    """
    Makes cube from numpy masked array generated from orography fields.

    Args:
        mask_data:
            The numpy array to make a cube from.
        coords:
            Dictionary of coordinate on the model ancillary file.
        topographic_bounds:
            List containing the lower and upper thresholds defining the mask
        topographic_units:
            Name of the units of the topographic zone coordinate of the output
            cube.
        sea_points_included:
            Default is False. Value for the output cube attribute
            'topographic_zones_include_seapoints', signifying whether sea
            points have been included when the ancillary is generated.

    Returns:
        Cube containing the mask_data array, with appropriate coordinate
        and attribute information.
    """
    mask_data = mask_data.astype(np.int32)
    mask_cube = iris.cube.Cube(mask_data, long_name="topography_mask")
    if any(item is None for item in topographic_bounds):
        msg = (
            "The topographic bounds variable should have both an "
            "upper and lower limit: "
            "Your topographic_bounds are {}"
        )
        raise TypeError(msg.format(topographic_bounds))

    if len(topographic_bounds) != 2:
        msg = (
            "The topographic bounds variable should have only an "
            "upper and lower limit: "
            "Your topographic_bounds variable has length {}"
        )
        raise TypeError(msg.format(len(topographic_bounds)))

    coord_name = "topographic_zone"
    central_point = np.mean(topographic_bounds)
    threshold_coord = iris.coords.AuxCoord(
        central_point,
        bounds=topographic_bounds,
        long_name=coord_name,
        units=Unit(topographic_units),
    )
    mask_cube.add_aux_coord(threshold_coord)
    # We can't save attributes with boolean values so convert to string.
    mask_cube.attributes.update(
        {"topographic_zones_include_seapoints": str(sea_points_included)}
    )
    for coord in coords:
        if coord.name() in ["projection_y_coordinate", "latitude"]:
            mask_cube.add_dim_coord(coord, 0)
        elif coord.name() in ["projection_x_coordinate", "longitude"]:
            mask_cube.add_dim_coord(coord, 1)
        else:
            mask_cube.add_aux_coord(coord)
    mask_cube = iris.util.new_axis(mask_cube, scalar_coord=coord_name)
    return mask_cube


class CorrectLandSeaMask(BasePlugin):
    """
    Round landsea mask to binary values

    Corrects interpolated land sea masks to boolean values of
    False [sea] and True [land].
    """

    def __init__(self) -> None:
        pass

    def __repr__(self) -> str:
        """Represent the configured plugin instance as a string"""
        result = "<CorrectLandSeaMask>"
        return result

    @staticmethod
    def process(standard_landmask: Cube) -> Cube:
        """Read in the interpolated landmask and round values < 0.5 to False
             and values >=0.5 to True.

        Args:
            standard_landmask:
                input landmask on standard grid.

        Returns:
            output landmask of boolean values.
        """
        mask_sea = standard_landmask.data < 0.5
        standard_landmask.data[mask_sea] = False
        mask_land = standard_landmask.data > 0.0
        standard_landmask.data[mask_land] = True
        standard_landmask.data = standard_landmask.data.astype(np.int8)
        standard_landmask.rename("land_binary_mask")
        return standard_landmask


class GenerateOrographyBandAncils(BasePlugin):
    """
    Generate topographic band ancillaries for the standard grids.

    Reads orography files, then generates binary mask
    of land points within the orography band specified.
    """

    def __init__(self) -> None:
        pass

    def __repr__(self) -> str:
        """Represent the configured plugin instance as a string."""
        result = "<GenerateOrographyBandAncils>"
        return result

    @staticmethod
    def sea_mask(
        landmask: ndarray, orog_band: ndarray, sea_fill_value: Optional[int] = None
    ) -> Union[MaskedArray, ndarray]:
        """
        Function to mask sea points and substitute the default numpy
        fill value behind this mask_cube.

        Args:
            landmask:
                The landmask generated by gen_landmask.
            orog_band:
                The binary array to which the landmask will be applied.
            sea_fill_value:
                A fill value to set sea points to and leave the output
                unmasked, rather than the default behaviour of returning a
                masked array with a default fill value.

        Returns:
            An array where the sea points have been masked out and filled
            with a default fill value, or just filled with the given
            sea_fill_value and not masked.
        """
        points_to_mask = np.logical_not(landmask)

        if sea_fill_value is None:
            sea_fill_value = np.ma.default_fill_value(orog_band)
            orog_data = orog_band.copy()
            orog_data[points_to_mask] = sea_fill_value
            mask_data = np.ma.masked_array(orog_data, mask=points_to_mask)
        else:
            mask_data = orog_band
            mask_data[points_to_mask] = sea_fill_value
        return mask_data

    def gen_orography_masks(
        self,
        standard_orography: Cube,
        standard_landmask: Optional[Cube],
        thresholds: List[float],
        units: str = "m",
    ) -> Cube:
        """
        Function to generate topographical band masks.

        For each threshold defined in 'thresholds', a cube with 0 over sea
        points and 1 for land points within the topography band will be
        generated.
        The lower threshold is exclusive to the band whilst the upper
        threshold is inclusive i.e:
        lower_threshold < band <= upper_threshold


        For example, for threshold pair [1,3] with orography::

                     [[0 0 2]    and      sea mask: [[0 0 1]
                      [2 2 3]                        [1 1 1]
                      [0 1 4]]                       [0 1 1]]

        the resultant array will be::

                     [[0 0 1]
                      [0 1 1]
                      [0 0 0]]

        Args:
            standard_orography:
                The standard orography.
            standard_landmask:
                The landmask generated by gen_landmask.
            thresholds:
                Upper and/or lower thresholds of the current topographical
                band.
            units:
                Units to be fed to CF_units to create a unit for the cube.
                The unit must be convertable to meters. If no unit is given
                this will default to meters.

        Returns:
            Cube containing topographical band mask.

        Raises:
            KeyError: if the key does not match any in THRESHOLD_DICT.
        """
        thresholds = np.array(thresholds, dtype=np.float32)
        thresholds = Unit(units).convert(thresholds, standard_orography.units)
        coords = standard_orography.coords()

        lower_threshold, upper_threshold = thresholds

        orog_band = (
            (standard_orography.data > lower_threshold)
            & (standard_orography.data <= upper_threshold)
        ).astype(int)

        # If we didn't find any points to mask, set all points to zero i.e
        # masked.
        if not isinstance(orog_band, np.ndarray):
            orog_band = np.zeros(standard_orography.data.shape).astype(int)

        if standard_landmask is not None:
            mask_data = self.sea_mask(
                standard_landmask.data, orog_band, sea_fill_value=0
            )
            mask_cube = _make_mask_cube(
                mask_data,
                coords,
                topographic_bounds=thresholds,
                topographic_units=standard_orography.units,
            )
        else:
            mask_cube = _make_mask_cube(
                orog_band,
                coords,
                topographic_bounds=thresholds,
                topographic_units=standard_orography.units,
                sea_points_included=True,
            )

        mask_cube.units = Unit("1")
        return mask_cube

    def process(
        self,
        orography: Cube,
        thresholds_dict: Dict[str, Any],
        landmask: Optional[Cube] = None,
    ) -> CubeList:
        """Loops over the supplied orographic bands, adding a cube
           for each band to the mask cubelist.

        Args:
            orography:
                orography on standard grid.
            thresholds_dict:
                Definition of orography bands required. Has key-value pairs of
                "bounds": list of list of pairs of bounds for each band and
                "units":"string containing units of bounds", for example::

                    {'bounds':[[0,100], [100,200]], 'units': "m"}

            landmask:
                land mask on standard grid, with land points set to one and
                sea points set to zero. If provided sea points are set to
                zero in every band.

        Returns:
            list of orographic band mask cubes.
        """
        cubelist = iris.cube.CubeList()
        if len(thresholds_dict) == 0:
            msg = "No threshold(s) found for topographic bands."
            raise ValueError(msg)

        for limits in thresholds_dict["bounds"]:
            oro_band = self.gen_orography_masks(
                orography, landmask, limits, thresholds_dict["units"]
            )
            cubelist.append(oro_band)
        return cubelist
