# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Provides support utilities for cli scripts."""

import json
from typing import Dict, Optional

import joblib


def load_json_or_none(file_path: Optional[str]) -> Optional[Dict]:
    """If there is a path, runs json.load and returns it. Else returns None.

    Args:
        file_path:
            File path to the json file to load.

    Returns:
        A dictionary loaded from a json file, or None.
    """
    metadata_dict = None
    if file_path:
        # Load JSON file for metadata amendments.
        with open(file_path, "r") as input_file:
            metadata_dict = json.load(input_file)
    return metadata_dict


def load_pickle_or_none(file_path: Optional[str]) -> Optional[list]:
    """If there is a path, load the pickled object and returns it. Else returns None.

    Args:
        file_path:
            File path to the pickled object file to load.

    Returns:
        A list of contained pickled objects, or None.
    """
    if file_path:
        with open(file_path, "rb") as input_file:
            object = joblib.load(input_file)
    return object if file_path else None
