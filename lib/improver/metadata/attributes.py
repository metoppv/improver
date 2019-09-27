# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2019 Met Office.
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
"""
Module defining attributes for IMPROVER blended data and the nowcast
"""

import re
import warnings

from improver.utilities.cube_metadata import amend_metadata


STANDARD_GRID_TITLE_STRING = "UK 2 km Standard Grid"
SPOT_TITLE_STRING = "UK Spot Values"

DATASET_ATTRIBUTES = {
    "nowcast": {
        "source": "IMPROVER",
        "title": "MONOW Extrapolation Nowcast",
        "institution": "Met Office"
    },
    "multi-model blend": {
        "source": "IMPROVER",
        "title": "IMPROVER Post-Processed Multi-Model Blend",
        "institution": "Met Office"
    }
}


def set_product_attributes(cube, product):
    """
    Set attributes on an output cube of type matching a key string in the
    DATASET_ATTRIBUTES dictionary.

    Args:
        cube (iris.cube.Cube):
            Cube containing product data
        product (str):
            String describing product type

    Returns:
        updated_cube (iris.cube.Cube):
            Cube with updated attributes
    """
    try:
        original_title = cube.attributes["title"]
    except KeyError:
        original_title = ""

    try:
        updated_cube = amend_metadata(
            cube, attributes=DATASET_ATTRIBUTES[product])
    except KeyError:
        options = list(DATASET_ATTRIBUTES.keys())
        raise ValueError(
            "product '{}' not available (options: {})".format(
                product, options))

    if STANDARD_GRID_TITLE_STRING in original_title:
        updated_cube.attributes["title"] += " on {}".format(
            STANDARD_GRID_TITLE_STRING)

    return updated_cube


def update_spot_title_attribute(cube):
    """
    Update the "title" attribute on a spot data cube IF there is an existing
    attribute of the form "<description>_on_<grid>" or containing the UK
    standard grid definition string.  Modifies cube in place.

    Args:
        cube (iris.cube.Cube):
            Spot data cube
    """
    regex = re.compile(
        '(?P<desc>)'   # title describing field
        '( on )'       # expected joining statement
        '(?P<grid>)')  # eg STANDARD_GRID_TITLE_STRING
    try:
        title_regex = regex.match(cube.attributes["title"])
    except KeyError:
        # if there is no original title, we cannot update it sensibly
        return

    if title_regex is None:
        # try a straight replacement
        if STANDARD_GRID_TITLE_STRING in cube.attributes["title"]:
            new_title = cube.attributes["title"].replace(
                STANDARD_GRID_TITLE_STRING, SPOT_TITLE_STRING)
            cube.attributes["title"] = new_title
        else:
            # if title does not match expected pattern we cannot update it
            # sensibly - raise a warning here
            warnings.warn(
                "'title' attribute does not match expected pattern - "
                "unable to replace grid description")
            return
    else:
        cube.attributes["title"] = (
            title_regex.group(1) + title_regex.group(2) + SPOT_TITLE_STRING)
