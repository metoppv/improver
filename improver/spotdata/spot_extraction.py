# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.

"""Spot data extraction from diagnostic fields using neighbour cubes."""

from typing import List, Optional, Tuple, Union

import iris
import numpy as np
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube
from numpy import ndarray

from improver import BasePlugin
from improver.metadata.constants.attributes import MANDATORY_ATTRIBUTE_DEFAULTS
from improver.metadata.constants.mo_attributes import MOSG_GRID_ATTRIBUTES
from improver.metadata.utilities import check_grid_match
from improver.spotdata.build_spotdata_cube import build_spotdata_cube
from improver.utilities.cube_manipulation import enforce_coordinate_ordering

from . import UNIQUE_ID_ATTRIBUTE


class SpotExtraction(BasePlugin):
    """
    For the extraction of diagnostic data using neighbour cubes that contain
    spot-site information and the appropriate grid point from which to source
    data.
    """

    def __init__(self, neighbour_selection_method: str = "nearest") -> None:
        """
        Args:
            neighbour_selection_method:
                The neighbour cube may contain one or several sets of grid
                coordinates that match a spot site. These are determined by
                the neighbour finding method employed. This keyword is used to
                extract the desired set of coordinates from the neighbour cube.
        """
        self.neighbour_selection_method = neighbour_selection_method

    def __repr__(self) -> str:
        """Represent the configured plugin instance as a string."""
        return "<SpotExtraction: neighbour_selection_method: {}>".format(
            self.neighbour_selection_method
        )

    def extract_coordinates(self, neighbour_cube: Cube) -> Cube:
        """
        Extract the desired set of grid coordinates that correspond to spot
        sites from the neighbour cube.

        Args:
            neighbour_cube:
                A cube containing information about the spot data sites and
                their grid point neighbours.

        Returns:
            A cube containing only the x and y grid coordinates for the
            grid point neighbours given the chosen neighbour selection
            method. The neighbour cube contains the indices stored as
            floating point values, so they are converted to integers
            in this cube.

        Raises:
            ValueError if the neighbour_selection_method expected is not found
            in the neighbour cube.
        """
        method = iris.Constraint(
            neighbour_selection_method_name=self.neighbour_selection_method
        )
        index_constraint = iris.Constraint(grid_attributes_key=["x_index", "y_index"])
        coordinate_cube = neighbour_cube.extract(method & index_constraint)
        if coordinate_cube:
            coordinate_cube.data = np.rint(coordinate_cube.data).astype(int)
            return coordinate_cube

        available_methods = neighbour_cube.coord(
            "neighbour_selection_method_name"
        ).points
        raise ValueError(
            'The requested neighbour_selection_method "{}" is not available in'
            " this neighbour_cube. Available methods are: {}.".format(
                self.neighbour_selection_method, available_methods
            )
        )

    @staticmethod
    def check_for_unique_id(neighbour_cube: Cube) -> Optional[Tuple[ndarray, str]]:
        """
        Determine if there is a unique ID coordinate, and if so return
        the values and name of that coordinate.

        Args:
            neighbour_cube:
                This cube on which to look for a unique site ID coordinate.

        Returns:
            - array of unique site IDs
            - name of unique site ID coordinate
        """
        try:
            (unique_id_coord,) = [
                crd
                for crd in neighbour_cube.coords()
                if UNIQUE_ID_ATTRIBUTE in crd.attributes
            ]
        except ValueError:
            pass
        else:
            return (unique_id_coord.points, unique_id_coord.name())

    def get_aux_coords(
        self, diagnostic_cube: Cube, x_indices: ndarray, y_indices: ndarray
    ) -> Tuple[List[AuxCoord], List[AuxCoord]]:
        """
        Extract scalar and non-scalar auxiliary coordinates from the diagnostic
        cube. Multi-dimensional auxiliary coordinates have the relevant points
        and bounds extracted for each site and a 1-dimensional coordinate is
        constructed that can be associated with the spot-index coordinate.

        Args:
            diagnostic_cube:
                The cube from which auxiliary coordinates will be taken.
            x_indices, y_indices:
                The array indices that correspond to sites for which coordinate
                data is to be extracted.

        Returns:
            - list of scalar coordinates
            - list of non-scalar, multi-dimensional coordinates reshaped into
              1-dimensional coordinates.
        """
        scalar_coords = []
        nonscalar_coords = []
        for coord in diagnostic_cube.aux_coords:
            if coord.ndim > 1:
                coord_points, coord_bounds = self.get_coordinate_data(
                    diagnostic_cube, x_indices, y_indices, coord.name()
                )
                nonscalar_coords.append(
                    coord.copy(points=coord_points, bounds=coord_bounds)
                )
            elif coord.points.size == 1:
                scalar_coords.append(coord)
        return scalar_coords, nonscalar_coords

    @staticmethod
    def get_coordinate_data(
        diagnostic_cube: Cube, x_indices: ndarray, y_indices: ndarray, coordinate: str
    ) -> Union[ndarray, List[Union[ndarray, None]]]:
        """
        Extracts coordinate points from 2-dimensional coordinates for
        association with sites.

        diagnostic_cube:
            The cube from which auxiliary coordinates will be taken.
        x_indices, y_indices:
            The array indices that correspond to sites for which coordinate
            data is to be extracted.
        coordinate:
            The name of the coordinate from which to extract data.

        Returns:
            A list containing an array of coordinate and bound values, with the
            later instead being None if no bounds exist.
        """
        coord_data = []
        coord = diagnostic_cube.coord(coordinate)
        coord_data.append(coord.points[..., y_indices, x_indices])
        if coord.has_bounds():
            coord_data.append(coord.bounds[..., y_indices, x_indices, :])
        else:
            coord_data.append(None)
        return coord_data

    @staticmethod
    def build_diagnostic_cube(
        neighbour_cube: Cube,
        diagnostic_cube: Cube,
        spot_values: ndarray,
        additional_dims: Optional[List[DimCoord]] = [],
        scalar_coords: Optional[List[AuxCoord]] = None,
        auxiliary_coords: Optional[List[AuxCoord]] = None,
        unique_site_id: Optional[Union[List[str], ndarray]] = None,
        unique_site_id_key: Optional[str] = None,
    ) -> Cube:
        """
        Builds a spot data cube containing the extracted diagnostic values.

        Args:
            neighbour_cube:
                This cube is needed as a source for information about the spot
                sites which needs to be included in the spot diagnostic cube.
            diagnostic_cube:
                The cube is needed to provide the name and units of the
                diagnostic that is being processed.
            spot_values:
                An array containing the diagnostic values extracted for the
                required spot sites.
            additional_dims:
                Optional list containing iris.coord.DimCoords with any leading
                dimensions required before spot data.
            scalar_coords:
                Optional list containing iris.coord.AuxCoords with all scalar coordinates
                relevant for the spot sites.
            auxiliary_coords:
                Optional list containing iris.coords.AuxCoords which are non-scalar.
            unique_site_id:
                Optional list of 8-digit unique site identifiers.
            unique_site_id_key:
                String to name the unique_site_id coordinate. Required if
                unique_site_id is in use.

        Returns:
            A spot data cube containing the extracted diagnostic data.
        """
        # Find any AuxCoords associated with the additional_dims so these can be copied too
        additional_dims_aux = []
        for dim_coord in additional_dims:
            dim_coord_dim = diagnostic_cube.coord_dims(dim_coord)
            aux_coords = [
                aux_coord
                for aux_coord in diagnostic_cube.aux_coords
                if diagnostic_cube.coord_dims(aux_coord) == dim_coord_dim
            ]
            additional_dims_aux.append(aux_coords if aux_coords else [])

        spot_diagnostic_cube = build_spotdata_cube(
            spot_values,
            diagnostic_cube.name(),
            diagnostic_cube.units,
            neighbour_cube.coord("altitude").points,
            neighbour_cube.coord(axis="y").points,
            neighbour_cube.coord(axis="x").points,
            neighbour_cube.coord("wmo_id").points,
            unique_site_id=unique_site_id,
            unique_site_id_key=unique_site_id_key,
            scalar_coords=scalar_coords,
            auxiliary_coords=auxiliary_coords,
            additional_dims=additional_dims,
            additional_dims_aux=additional_dims_aux,
        )
        return spot_diagnostic_cube

    def process(
        self,
        neighbour_cube: Cube,
        diagnostic_cube: Cube,
        new_title: Optional[str] = None,
    ) -> Cube:
        """
        Create a spot data cube containing diagnostic data extracted at the
        coordinates provided by the neighbour cube.

        .. See the documentation for more details about the inputs and output.
        .. include:: /extended_documentation/spotdata/spot_extraction/
           spot_extraction_examples.rst

        Args:
            neighbour_cube:
                A cube containing information about the spot data sites and
                their grid point neighbours.
            diagnostic_cube:
                A cube of diagnostic data from which spot data is being taken.
            new_title:
                New title for spot-extracted data.  If None, this attribute is
                reset to a default value, since it has no prescribed standard
                and may therefore contain grid information that is no longer
                correct after spot-extraction.

        Returns:
            A cube containing diagnostic data for each spot site, as well
            as information about the sites themselves.
        """
        # Check we are using a matched neighbour/diagnostic cube pair
        check_grid_match([neighbour_cube, diagnostic_cube])

        # Get the unique_site_id if it is present on the neighbour cbue
        unique_site_id_data = self.check_for_unique_id(neighbour_cube)
        if unique_site_id_data:
            unique_site_id = unique_site_id_data[0]
            unique_site_id_key = unique_site_id_data[1]
        else:
            unique_site_id, unique_site_id_key = None, None

        # Ensure diagnostic cube is y-x order as neighbour cube expects.
        enforce_coordinate_ordering(
            diagnostic_cube,
            [
                diagnostic_cube.coord(axis="y").name(),
                diagnostic_cube.coord(axis="x").name(),
            ],
            anchor_start=False,
        )

        coordinate_cube = self.extract_coordinates(neighbour_cube)
        x_indices, y_indices = coordinate_cube.data
        spot_values = diagnostic_cube.data[..., y_indices, x_indices]

        additional_dims = []
        if len(spot_values.shape) > 1:
            additional_dims = diagnostic_cube.dim_coords[:-2]
        scalar_coords, nonscalar_coords = self.get_aux_coords(
            diagnostic_cube, x_indices, y_indices
        )

        spotdata_cube = self.build_diagnostic_cube(
            neighbour_cube,
            diagnostic_cube,
            spot_values,
            scalar_coords=scalar_coords,
            auxiliary_coords=nonscalar_coords,
            additional_dims=additional_dims,
            unique_site_id=unique_site_id,
            unique_site_id_key=unique_site_id_key,
        )

        # Copy attributes from the diagnostic cube that describe the data's
        # provenance
        spotdata_cube.attributes = diagnostic_cube.attributes
        spotdata_cube.attributes["model_grid_hash"] = neighbour_cube.attributes[
            "model_grid_hash"
        ]

        # Remove the unique_site_id coordinate attribute as it is internal
        # metadata only
        if unique_site_id is not None:
            spotdata_cube.coord(unique_site_id_key).attributes.pop(UNIQUE_ID_ATTRIBUTE)

        # Remove grid attributes and update title
        for attr in MOSG_GRID_ATTRIBUTES:
            spotdata_cube.attributes.pop(attr, None)
        spotdata_cube.attributes["title"] = (
            MANDATORY_ATTRIBUTE_DEFAULTS["title"] if new_title is None else new_title
        )

        # Copy cell methods
        spotdata_cube.cell_methods = diagnostic_cube.cell_methods

        return spotdata_cube
