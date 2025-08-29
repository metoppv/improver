# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.

"""Spot data utilities."""

from iris.cube import Cube


def get_neighbour_finding_method_name(land_constraint: bool, minimum_dz: bool) -> str:
    """
    Create a name to describe the neighbour method based on the constraints
    provided.

    Returns:
        A string that describes the neighbour finding method employed.
        This is essentially a concatenation of the options.
    """
    method_name = "{}{}{}".format(
        "nearest",
        "_land" if land_constraint else "",
        "_minimum_dz" if minimum_dz else "",
    )
    return method_name


def extract_site_json(neighbours: Cube):
    """
    Extract the site definition JSON from a neighbour cube. This
    produces a JSON file that can be used to recreate a neighbour cube,
    for example on a different domain / projection.

    Args:
        neighbours:
            A cube containing the neighbour information.

    Returns:
        A list of JSON dictionaries containing the site definitions.
    """
    neighbour_slice = next(neighbours.slices("spot_index"))
    n_sites = neighbour_slice.shape[0]
    site_specific_crds = [
        crd for crd in neighbour_slice.aux_coords if crd.shape[0] == n_sites
    ]
    keys = [crd.name() for crd in site_specific_crds]

    site_definitions = []
    for site_index in range(neighbour_slice.shape[0]):
        site_definition = {}
        for crd in keys:
            site_definition[crd] = neighbour_slice.coord(crd).points[site_index].item()
            if site_definition[crd] == "None":
                site_definition[crd] = None
            if isinstance(site_definition[crd], str):
                site_definition[crd] = int(site_definition[crd])
        site_definitions.append(site_definition)

    return site_definitions
