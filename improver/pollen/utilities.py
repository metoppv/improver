# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
from copy import deepcopy

from iris.coords import AuxCoord
from iris.cube import Cube


def build_output_cube_with_new_units(self, cube: Cube, new_units) -> Cube:
    """Build the output cube as a copy of the input cube, adjusting units.

    There are two places in the Pollen workflow where units are changed and the
    iris function Cube.convert_units() doesn't work, so the cube needs to be rebuilt
    with the new units and appropriate metadata. This function handles that process.

    The places where this is required are:
    - In the PollenHourlyConcentration plugin, the input cube has units of g/m3 and
      the output cube needs to have units of grains/m3 after a conversion.
    - In the PollenIndexForPeriod plugin, the input cube has units grains/m3 and
      the output cube has unitless index values (0 to 4) that are derived from
      the input cube data values.
    Args:
        cube:
            The input cube from which to copy data and metadata.
        new_units:
            The units to set on the output cube.
    Returns:
        Cube:
            The output cube with updated units and copied data and metadata.
    """
    standard_name = cube.standard_name
    long_name = cube.long_name
    var_name = cube.var_name
    attributes = cube.attributes
    cell_methods = cube.cell_methods
    dim_coords = cube.dim_coords
    dim_coord_dims = [cube.coord_dims(coord.name()) for coord in dim_coords]
    new_dim_coords_and_dims = list(zip(dim_coords, dim_coord_dims))

    new_aux_coords_list = []
    aux_coords = cube.aux_coords
    for aux_coord in aux_coords:
        new_aux_coord = AuxCoord.from_coord(aux_coord)
        new_aux_coords_list.append(new_aux_coord)
    new_aux_coords = tuple(new_aux_coords_list)

    aux_factories = cube.aux_factories
    if len(aux_factories) == 0:
        aux_factories = None
    cell_measures = cube.cell_measures()
    if len(cell_measures) == 0:
        cell_measures = None
    ancillary_variables = cube.ancillary_variables()
    if len(ancillary_variables) == 0:
        ancillary_variables = None

    output_cube = Cube(
        deepcopy(cube.data),
        standard_name=standard_name,
        long_name=long_name,
        var_name=var_name,
        units=new_units,
        attributes=attributes,
        cell_methods=cell_methods,
        dim_coords_and_dims=new_dim_coords_and_dims,
        aux_factories=aux_factories,
        cell_measures_and_dims=cell_measures,
        ancillary_variables_and_dims=ancillary_variables,
    )
    for coord in new_aux_coords:
        output_cube.add_aux_coord(coord)
    return output_cube
