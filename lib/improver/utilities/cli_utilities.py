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
"""Provides support utilities for cli scripts."""

import json


def load_json_or_none(file_path):
    """If there is a path, runs json.load and returns it. Else returns None.

    Args:
        file_path (str or None):
            File path to the json file to load.

    Returns:
        dict or None:
            A dictionary loaded from a json file.
            or
            None
    """
    metadata_dict = None
    if file_path:
        # Load JSON file for metadata amendments.
        with open(file_path, 'r') as input_file:
            metadata_dict = json.load(input_file)
    return metadata_dict


def radius_or_radii_and_lead(radius=None, radii_by_lead_time=None):
    """Takes either argument and returns radius/radii and lead time.

    Args:
        radius (float or None):
            If it exists it returns it as a radius
        radii_by_lead_time (list of str or None):
            If radius doesn't exist and this does, it splits by a comma
            and gives radius_or_radii [0] and lead_times [1].

    Returns:
        (tuple): tuple containing:
                **radius_or_radii** (float):
                    Radius or radii.
                **lead_times** (list):
                    If radii, list of lead times. Else None.

    Raises:
        TypeError:
            When both radius and radii_by_lead_time are None.
        TypeError:
            When both radius and radii_by_lead_time are not None.
    """
    if radius is None and radii_by_lead_time is None:
        msg = ("Neither radius or radii_by_lead_time have been set. "
               "One option should be specified.")
        raise TypeError(msg)
    elif radius is not None and radii_by_lead_time is not None:
        msg = ("Both radius and radii_by_lead_time have been set. "
               "Only one option should be specified.")
        raise TypeError(msg)
    if radius:
        radius_or_radii = radius
        lead_times = None
    elif radii_by_lead_time:
        radius_or_radii = radii_by_lead_time[0].split(",")
        lead_times = radii_by_lead_time[1].split(",")

    return radius_or_radii, lead_times
