# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown copyright. The Met Office.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
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

        Raises:
            KeyError: If any of the attributes are not present on the template_cube.

        Returns:
            Updated cube(s).

        """
        cubes_proc = as_cubelist(*cubes)
        template_cube = as_cube(template_cube)

        for cube in cubes_proc:
            new_attributes = {k: template_cube.attributes[k] for k in self.attributes}
            amend_attributes(cube, new_attributes)
        return cubes if len(cubes) > 1 else cubes[0]
