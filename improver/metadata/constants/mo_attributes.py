# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Module defining Met Office specific attributes"""

GRID_TYPE = "standard"
STAGE_VERSION = "1.3.0"

MOSG_GRID_ATTRIBUTES = {"mosg__grid_type", "mosg__grid_version", "mosg__grid_domain"}

# Define current StaGE and MONOW grid metadata
MOSG_GRID_DEFINITION = {
    "uk_ens": {
        "mosg__grid_type": GRID_TYPE,
        "mosg__model_configuration": "uk_ens",
        "mosg__grid_domain": "uk_extended",
        "mosg__grid_version": STAGE_VERSION,
    },
    "gl_ens": {
        "mosg__grid_type": GRID_TYPE,
        "mosg__model_configuration": "gl_ens",
        "mosg__grid_domain": "global",
        "mosg__grid_version": STAGE_VERSION,
    },
    "uk_det": {
        "mosg__grid_type": GRID_TYPE,
        "mosg__model_configuration": "uk_det",
        "mosg__grid_domain": "uk_extended",
        "mosg__grid_version": STAGE_VERSION,
    },
    "gl_det": {
        "mosg__grid_type": GRID_TYPE,
        "mosg__model_configuration": "gl_det",
        "mosg__grid_domain": "global",
        "mosg__grid_version": STAGE_VERSION,
    },
    "nc_det": {"mosg__model_configuration": "nc_det"},
}
