# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
from typing import List, Tuple, Union

from iris.cube import Cube, CubeList

from improver import BasePlugin
from improver.metadata.amend import amend_attributes
from improver.utilities.common_input_handle import as_cube, as_cubelist


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

    def process(
        self, *cubes: Union[Cube, CubeList], template_cube: Union[Cube, CubeList]
    ) -> Union[Tuple[Union[Cube, CubeList]], Cube, CubeList]:
        """
        Copy attribute values from template_cube to cube, overwriting any existing values.

        Operation is performed in-place on provided inputs.

        Args:
            cubes:
                Source cube(s) to be updated.
            template_cube:
                Source cube to get attribute values from.

        Returns:
            Updated cube(s).

        """
        cubes_proc = as_cubelist(*cubes)
        template_cube = as_cube(template_cube)

        for cube in cubes_proc:
            new_attributes = {k: template_cube.attributes[k] for k in self.attributes}
            amend_attributes(cube, new_attributes)
        return cubes if len(cubes) > 1 else cubes[0]
