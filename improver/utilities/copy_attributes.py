# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
from typing import List, Union

from dateutil import parser as dparser
from iris.cube import Cube, CubeList

from improver import BasePlugin
from improver.metadata.amend import amend_attributes
from improver.utilities.common_input_handle import as_cubelist


class CopyAttributes(BasePlugin):
    """Copy attribute values from template_cube to cube, overwriting any existing values."""

    def __init__(self, attributes: List):
        """
        Initialise the plugin with a list of attributes to copy.

        Args:
            attributes:
                List of names of attributes to copy. If any are not present on template_cube, a
                KeyError will be raised.
        """
        self.attributes = attributes

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
        Copy attribute values from template_cube to cube, overwriting any existing values. If the history
        attribute is present in the list of requested attributes, the most recent value will be used.

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
        print(new_attributes)
        amend_attributes(cube, new_attributes)

        return cube
