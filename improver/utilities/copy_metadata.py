# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
import warnings
from typing import List, Union

from dateutil import parser as dparser
from iris.cube import Cube, CubeList

from improver import BasePlugin
from improver.metadata.amend import amend_attributes
from improver.utilities.common_input_handle import as_cubelist


class CopyMetadata(BasePlugin):
    """Copy attribute or auxilary coordinate values from template_cube to cube,
    overwriting any existing values."""

    def __init__(
        self,
        attributes: List = [],
        aux_coord: List = [],
        ancillary_variables: List = [],
    ):
        """
        Initialise the plugin with a list of attributes to copy.

        Args:
            attributes:
                List of names of attributes to copy. If any are not present on
                template_cube, a KeyError will be raised.
            aux_coord:
                List of names of auxilary coordinates to copy. If any are not
                present on template_cube, a KeyError will be raised. If the
                aux_coord is already present in the cube, it will be overwritten.
            ancillary_variables:
                List of names of ancillary variables to copy. If any are not
                present on template_cube, a KeyError will be raised. If the
                ancillary variable is already present in the cube, it will be overwritten.
        """
        self.attributes = attributes
        self.aux_coord = aux_coord
        self.ancillary_variables = ancillary_variables

    @staticmethod
    def get_most_recent_history(datelist: list) -> list:
        """
        Gets the most recent history attribute from the list of provided dates.

        Args:
            datelist:
                A list of dates to find the most recent calue from.

        Returns:
            The most recent history attribute.
        """
        prev_time = None

        for date in datelist:
            new_time = dparser.parse(date, fuzzy=True)
            if not prev_time:
                prev_time = new_time
                str_time = date
            elif new_time > prev_time:
                prev_time = new_time
                str_time = date

        return str_time

    def find_common_attributes(self, cubes: CubeList, attributes: List) -> dict:
        """
        Find the common attribute values between the cubes. If the attribute is history, the most recent
        value will be returned.

        Args:
            cubes:
                A list of template cubes to extract common attributes from.
            attributes:
                A list of attributes to be copied.
        Returns:
            A dictionary of common attributes.
        Raises:
            ValueError: If the attribute is not found in any of the template cubes
            ValueError: If the attribute has different values in the provided template cubes.
        """
        common_attributes = {}
        for attribute in attributes:
            attribute_value = [
                cube.attributes.get(attribute)
                for cube in cubes
                if cube.attributes.get(attribute) is not None
            ]
            if attribute == "history":
                # We expect the history attribute to differ between cubes, so we will only keep the most recent one
                common_attributes[attribute] = self.get_most_recent_history(
                    attribute_value
                )
            elif len(attribute_value) == 0:
                raise ValueError(
                    f"Attribute {attribute} not found in any of the template cubes"
                )
            elif any(attr != attribute_value[0] for attr in attribute_value):
                raise ValueError(
                    f"Attribute {attribute} has different values in the provided template cubes"
                )
            else:
                common_attributes[attribute] = attribute_value[0]

        return common_attributes

    def process(self, *cubes: Union[Cube, CubeList]) -> Cube:
        """
        Copy attribute or auxilary coordinate values from template_cube to cube,
        overwriting any existing values. If the history attribute is present in
        the list of requested attributes, the most recent value will be used. If an
        auxilary coordinate or ancillary variable needs to be copied then all
        template cubes must have the auxilary coordinate or ancillary variable
        present.

        Operation is performed in-place on provided inputs.

        Args:
            cubes:
                List of cubes. First cube provided represents the cube to be updated. All
                other cubes are treated as template cubes.

        Returns:
            A cube with attributes copied from the template cubes

        """
        cubes_proc = as_cubelist(*cubes)
        if len(cubes_proc) < 2:
            raise RuntimeError(
                f"At least two cubes are required for this operation, got {len(cubes_proc)}"
            )
        cube = cubes_proc.pop(0)
        template_cubes = cubes_proc
        new_attributes = self.find_common_attributes(template_cubes, self.attributes)
        amend_attributes(cube, new_attributes)

        for coord in self.aux_coord:
            # If the template cube has the auxiliary coordinate, copy it
            if template_cubes[0].coords(coord):
                # If coordinate is already present in the cube, remove it
                if cube.coords(coord):
                    cube.remove_coord(coord)
                cube.add_aux_coord(
                    template_cubes[0].coord(coord),
                    data_dims=template_cubes[0].coord_dims(coord=coord),
                )
            else:
                warnings.warn(
                    f"Auxiliary Coordinate '{coord}' not found in cube '{template_cubes[0].name()}'.",
                    UserWarning,
                )

        for ancillary_var in self.ancillary_variables:
            # If the template cube has the ancillary variable, copy it
            if template_cubes[0].ancillary_variables(ancillary_var):
                # If ancillary variable is already present in the cube, remove it
                if cube.ancillary_variables(ancillary_var):
                    cube.remove_ancillary_variable(ancillary_var)
                cube.add_ancillary_variable(
                    template_cubes[0].ancillary_variable(ancillary_var),
                    data_dims=template_cubes[0].ancillary_variable_dims(ancillary_var),
                )
            else:
                warnings.warn(
                    f"Ancillary variable '{ancillary_var}' not found in cube '{template_cubes[0].name()}'.",
                    UserWarning,
                )

        return cube
